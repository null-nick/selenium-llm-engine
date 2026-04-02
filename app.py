import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Set

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    StreamingResponse,
)

from db.db import (
    get_logged_engines,
    get_prompt_logs,
    get_response_time_stats,
    get_stats,
    init_database,
    log_prompt,
    inc_errors,
    inc_requests,
    inc_responses,
)
from core.engine_manager import EngineManager

# ---------------------------------------------------------------------------
# In-memory application log buffer (survives for the process lifetime)
# ---------------------------------------------------------------------------

_LOG_BUFFER: deque[Dict[str, Any]] = deque(maxlen=500)
_LOG_SEQ = 0
_LOG_BUFFER_LOCK = threading.Lock()


class _BufferHandler(logging.Handler):
    """Appends log records to the in-memory ring buffer."""

    def emit(self, record: logging.LogRecord) -> None:
        global _LOG_SEQ
        try:
            with _LOG_BUFFER_LOCK:
                _LOG_SEQ += 1
                _LOG_BUFFER.append(
                    {
                        "seq": _LOG_SEQ,
                        "time": self.formatTime(record, "%H:%M:%S"),
                        "level": record.levelname,
                        "name": record.name,
                        "msg": self.format(record),
                    }
                )
        except Exception:
            self.handleError(record)


_buf_handler = _BufferHandler()
_buf_handler.setLevel(logging.DEBUG)

logging.basicConfig(level=logging.INFO)
# Attach after basicConfig so the root logger already exists
logging.getLogger().addHandler(_buf_handler)

logger = logging.getLogger("selenium-llm-api")

app = FastAPI(title="Selenium LLM Engine", version="0.1")

# Rate limiting (per ip, sliding window)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 20
rate_limit_store: Dict[str, list[float]] = defaultdict(list)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _rate_limit_exceeded(request: Request) -> bool:
    key = _client_ip(request)
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    entries = rate_limit_store[key]
    rate_limit_store[key] = [t for t in entries if t >= window_start]
    if len(rate_limit_store[key]) >= RATE_LIMIT_MAX:
        return True
    rate_limit_store[key].append(now)
    return False


# Reset/state coordination helpers
RESET_IN_PROGRESS = False
IN_FLIGHT_TASKS: Set[asyncio.Task] = set()


def _register_task(task: asyncio.Task) -> None:
    IN_FLIGHT_TASKS.add(task)


def _unregister_task(task: asyncio.Task) -> None:
    IN_FLIGHT_TASKS.discard(task)


async def _cancel_inflight_tasks() -> None:
    tasks = list(IN_FLIGHT_TASKS)
    if not tasks:
        return
    for t in tasks:
        t.cancel()
    try:
        # Wait for cancel to propagate (including exceptions raised on cancellation)
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception:
        pass
    IN_FLIGHT_TASKS.clear()


async def _safe_parse_json(request: Request) -> Dict[str, Any]:
    try:
        return await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def _openai_response(
    engine_name: str, model_name: str, prompt: str, response_text: str, elapsed_ms: int
) -> Dict[str, Any]:
    prompt_tokens = _estimate_tokens(prompt)
    completion_tokens = _estimate_tokens(response_text)
    return {
        "id": f"llm_{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "engine": engine_name,
        "prompt": prompt,
        "elapsed_ms": elapsed_ms,
    }


def _openai_chunk(chunk_id: str, model_name: str, content: str, finish_reason: Any) -> str:
    """Format a single SSE chunk in OpenAI chat.completion.chunk format."""
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


@app.on_event("startup")
async def startup_event() -> None:
    init_database()
    EngineManager.get()  # initialize manager
    _register_engine_routes(app)  # dynamic per-engine /name/prompt routes


@app.get("/")
async def root() -> RedirectResponse:
    # Redirect to the web UI for convenience
    return RedirectResponse(url="/ui")


@app.get("/api/ping")
async def ping() -> Dict[str, str]:
    return {"status": "ok", "service": "selenium-llm-engine"}


@app.get("/api/engines")
async def api_engines() -> Dict[str, Any]:
    """List all discovered engines with metadata (no browser started)."""
    mgr = EngineManager.get()
    return {"data": mgr.list_engines()}


@app.post("/api/engines/reload")
async def api_engines_reload() -> Dict[str, Any]:
    """Re-scan the engines/ directory and refresh the registry."""
    mgr = EngineManager.get()
    updated = mgr.reload_engines()
    return {"status": "ok", "data": updated}


@app.get("/models")
async def models() -> Dict[str, Any]:
    """Legacy endpoint — returns OpenAI-compatible model list (same as /v1/models).
    Also includes legacy 'name'/'limits'/'supported_models' fields for backward
    compatibility with older clients."""
    mgr = EngineManager.get()
    created = int(time.time())
    data = []
    for desc in mgr.list_engines():
        engine_name = desc["name"]
        entry: Dict[str, Any] = {
            # OpenAI-compatible fields (required by clients like Alpaca)
            "id": engine_name,
            "object": "model",
            "created": created,
            "owned_by": "selenium-llm-engine",
            # Legacy extra fields (kept for backward compat)
            "name": engine_name,
        }
        try:
            eng = mgr.get_engine(engine_name)
            entry["limits"] = eng.get_interface_limits()
            entry["supported_models"] = eng.get_supported_models()
        except Exception:
            pass
        data.append(entry)
    return {"object": "list", "data": data}


@app.get("/models/{engine_name}")
async def model_info(engine_name: str) -> Dict[str, Any]:
    try:
        engine = EngineManager.get().get_engine(engine_name)
    except ValueError:
        raise HTTPException(status_code=404, detail="Engine not found")
    return {
        "engine": engine_name,
        "limits": engine.get_interface_limits(),
        "models": engine.get_supported_models(),
    }


@app.post("/login/{engine_name}")
async def login_engine(engine_name: str) -> Dict[str, Any]:
    try:
        engine = EngineManager.get().set_active_engine(engine_name)
    except ValueError:
        raise HTTPException(status_code=404, detail="Engine not found")

    result = await engine.start_login_flow()
    return result


@app.get("/login/{engine_name}/state")
async def login_state(engine_name: str) -> Dict[str, Any]:
    try:
        engine = EngineManager.get().get_engine(engine_name)
    except ValueError:
        raise HTTPException(status_code=404, detail="Engine not found")

    state = await engine.check_login_state()
    return state


@app.post("/engine/{engine_name}/prompt")
async def engine_prompt(engine_name: str, req: Request) -> Any:
    """Dynamic prompt endpoint — works for any discovered engine."""
    if _rate_limit_exceeded(req):
        raise HTTPException(status_code=429, detail="Too many requests")
    mgr = EngineManager.get()
    try:
        canonical = mgr._resolve(engine_name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Engine not found: {engine_name}")
    data = await _safe_parse_json(req)
    return await _prompt(
        canonical,
        req,
        explicit_prompt=data.get("prompt") or data.get("messages"),
        model_name=data.get("model", canonical),
        stream=bool(data.get("stream", False)),
    )


# ---------------------------------------------------------------------------
# Legacy per-engine prompt endpoints — generated dynamically at startup
# ---------------------------------------------------------------------------


def _register_engine_routes(application: FastAPI) -> None:
    """Create /{engine_name}/prompt routes for every discovered engine."""
    mgr = EngineManager.get()
    for desc in mgr.list_engines():
        engine_name = desc["name"]

        # Build a closure that captures the canonical engine name
        def _make_handler(name: str):
            async def handler(req: Request) -> Any:
                if _rate_limit_exceeded(req):
                    raise HTTPException(status_code=429, detail="Too many requests")
                data = await _safe_parse_json(req)
                return await _prompt(
                    name,
                    req,
                    explicit_prompt=data.get("prompt") or data.get("messages"),
                    model_name=name,
                    stream=bool(data.get("stream", False)),
                )
            handler.__name__ = f"{name}_prompt"
            return handler

        application.add_api_route(
            f"/{engine_name}/prompt",
            _make_handler(engine_name),
            methods=["POST"],
        )
        logger.info(f"[app] Registered route POST /{engine_name}/prompt")


@app.get("/v1/models")
async def v1_models() -> Dict[str, Any]:
    """OpenAI-compatible model list. Returns one entry per provider (canonical name).
    Clients that send model='chatgpt' or model='gemini' will be routed correctly.
    Aliases and per-variant ids are intentionally excluded to maximise client compatibility."""
    mgr = EngineManager.get()
    created = int(time.time())
    entries: list[Dict[str, Any]] = []
    for desc in mgr.list_engines():
        entries.append(
            {
                "id": desc["name"],
                "object": "model",
                "created": created,
                "owned_by": "selenium-llm-engine",
            }
        )
    return {"object": "list", "data": entries}


@app.get("/v1/models/{model_id:path}")
async def v1_model_detail(model_id: str) -> Dict[str, Any]:
    """OpenAI-compatible single model lookup. Supports 'provider' or 'provider:variant'."""
    mgr = EngineManager.get()
    provider = model_id.split(":")[0]
    try:
        mgr._resolve(provider)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "selenium-llm-engine",
    }


@app.post("/v1/chat/completions")
async def openai_chat(req: Request) -> Any:
    if _rate_limit_exceeded(req):
        raise HTTPException(status_code=429, detail="Too many requests")

    data = await _safe_parse_json(req)
    model = data.get("model") or "chatgpt"
    # Support provider:variant notation (e.g. "chatgpt:gpt-4o" -> engine="chatgpt")
    engine_hint = model.split(":")[0]
    mgr = EngineManager.get()
    try:
        engine = mgr._resolve(engine_hint)
    except ValueError:
        # Fall back to chatgpt for unrecognised model names (OpenAI compat behaviour)
        engine = "chatgpt"

    if "prompt" in data:
        prompt_payload = data.get("prompt")
    elif "messages" in data:
        prompt_payload = data.get("messages")
    else:
        raise HTTPException(status_code=400, detail="Missing prompt/messages")

    stream = bool(data.get("stream", False))

    return await _prompt(
        engine, req, explicit_prompt=prompt_payload, model_name=model, stream=stream
    )


# Legacy alias — some clients call /chat/completions without the /v1 prefix
@app.post("/chat/completions")
async def openai_chat_legacy(req: Request) -> Any:
    return await openai_chat(req)


async def _prompt(
    engine_name: str,
    req: Request,
    explicit_prompt: Any = None,
    model_name: str = "default",
    stream: bool = False,
) -> Any:
    if RESET_IN_PROGRESS:
        raise HTTPException(
            status_code=503,
            detail="Service is resetting; please retry after a moment",
        )

    if _rate_limit_exceeded(req):
        raise HTTPException(status_code=429, detail="Too many requests")

    if explicit_prompt is None:
        payload = await _safe_parse_json(req)
        prompt_text = payload.get("prompt") or payload.get("messages")
    else:
        prompt_text = explicit_prompt

    if not prompt_text:
        raise HTTPException(status_code=400, detail="Missing prompt/messages")

    if isinstance(prompt_text, list):
        prompt_text = "\n".join(
            x.get("content", "") if isinstance(x, dict) else str(x) for x in prompt_text
        )

    if not isinstance(prompt_text, str):
        prompt_text = str(prompt_text)

    current_task = asyncio.current_task()
    if current_task is not None:
        _register_task(current_task)

    inc_requests()
    start = time.time()
    try:
        engine = EngineManager.get().set_active_engine(engine_name)
        effective_model = engine.get_current_model()

        if stream:

            async def generate_stream():
                try:
                    result = await engine.generate_response(prompt_text)
                    elapsed_ms = int((time.time() - start) * 1000)
                    log_prompt(
                        engine_name,
                        effective_model,
                        prompt_text,
                        result,
                        "ok",
                        elapsed_ms,
                    )
                    inc_responses()
                    chunk_id = f"llm_{int(time.time())}"
                    yield _openai_chunk(chunk_id, effective_model, result, None)
                    yield _openai_chunk(chunk_id, effective_model, "", "stop")
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    elapsed_ms = int((time.time() - start) * 1000)
                    log_prompt(
                        engine_name, "unknown", prompt_text, str(e), "error", elapsed_ms
                    )
                    inc_errors()
                    raise HTTPException(status_code=500, detail=str(e))

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        response = await engine.generate_response(prompt_text)
        duration_ms = int((time.time() - start) * 1000)
        log_prompt(
            engine_name,
            effective_model,
            prompt_text,
            response,
            "ok",
            duration_ms,
        )
        inc_responses()

        return _openai_response(
            engine_name,
            effective_model,
            prompt_text,
            response,
            duration_ms,
        )

    except asyncio.CancelledError:
        duration_ms = int((time.time() - start) * 1000)
        log_prompt(
            engine_name,
            "unknown",
            prompt_text,
            "cancelled due to reset",
            "error",
            duration_ms,
        )
        inc_errors()
        raise HTTPException(status_code=503, detail="Request cancelled due to reset")
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        log_prompt(engine_name, "unknown", prompt_text, str(e), "error", duration_ms)
        inc_errors()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if current_task is not None:
            _unregister_task(current_task)


@app.get("/stats")
async def stats() -> Dict[str, Any]:
    return {
        "stats": get_stats(),
        "logged_engines": get_logged_engines(),
        "response_time": get_response_time_stats(),
    }


@app.get("/api/logs/app")
async def app_logs(since: int = 0) -> Dict[str, Any]:
    """Return application log entries with seq > since (incremental polling)."""
    with _LOG_BUFFER_LOCK:
        entries = [e for e in _LOG_BUFFER if e["seq"] > since]
    return {"entries": entries}


@app.get("/api/engines/selector-hints")
async def selector_hints() -> Dict[str, Any]:
    """Return runtime-discovered best selectors for all active engine instances.

    Only engines that have processed at least one prompt will have cached
    selector data.  The UI can use this to suggest JSON reordering.
    """
    mgr = EngineManager.get()
    data: Dict[str, Any] = {}
    for name, engine in mgr.engines.items():
        data[name] = {
            "prompt_selector": getattr(engine, "_cached_prompt_selector", None),
            "send_selector": getattr(engine, "_cached_send_selector", None),
            "prompt_area_selectors": getattr(engine, "prompt_area_selectors", []),
            "send_button_selectors": getattr(engine, "send_button_selectors", []),
        }
    return {"data": data}


@app.post("/reset")
async def reset_state() -> Dict[str, Any]:
    global RESET_IN_PROGRESS
    manager = EngineManager.get()
    errors: list[str] = []

    RESET_IN_PROGRESS = True
    try:
        # Cancel in-flight prompt handlers (soft cancellation)
        try:
            await _cancel_inflight_tasks()
        except Exception as e:
            logger.warning(f"[reset] cancel_inflight_tasks error: {e}")
            errors.append(f"cancel: {e}")

        try:
            await manager.stop_all()
        except Exception as e:
            logger.warning(f"[reset] stop_all error (continuing): {e}")
            errors.append(f"stop_all: {e}")

        manager.engines.clear()
        manager.active_engine = None
        rate_limit_store.clear()

        message = (
            "Engine state cleared"
            if not errors
            else f"Engine state cleared (with errors: {'; '.join(errors)})"
        )
        return {"status": "ok", "message": message}

    finally:
        RESET_IN_PROGRESS = False


@app.post("/api/reset")
async def api_reset_state() -> Dict[str, Any]:
    return await reset_state()


@app.get("/logs")
async def logs(
    limit: int = 50,
    offset: int = 0,
    engine: str | None = None,
    model: str | None = None,
    status: str | None = None,
):
    return get_prompt_logs(
        limit=limit,
        offset=offset,
        engine=engine,
        model=model,
        status=status,
    )


@app.get("/api/history")
async def history(
    limit: int = 50,
    offset: int = 0,
    engine: str | None = None,
    model: str | None = None,
    status: str | None = None,
):
    return get_prompt_logs(
        limit=limit,
        offset=offset,
        engine=engine,
        model=model,
        status=status,
    )


@app.get("/ui", response_class=HTMLResponse)
async def ui() -> Any:
    html = Path("./web/index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)
