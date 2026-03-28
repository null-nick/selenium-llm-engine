import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from db.db import get_prompt_logs, get_stats, init_database, log_prompt, inc_errors, inc_requests, inc_responses
from engine.engine_manager import EngineManager

logging.basicConfig(level=logging.INFO)
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


def _openai_response(engine_name: str, model_name: str, prompt: str, response_text: str, elapsed_ms: int) -> Dict[str, Any]:
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
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "engine": engine_name,
        "prompt": prompt,
        "elapsed_ms": elapsed_ms,
    }


@app.on_event("startup")
async def startup_event() -> None:
    init_database()
    EngineManager.get()  # initialize manager

@app.get("/")
async def root() -> RedirectResponse:
    # Redirect to the web UI for convenience
    return RedirectResponse(url="/ui")

@app.get("/api/ping")
async def ping() -> Dict[str, str]:
    return {"status": "ok", "service": "selenium-llm-engine"}

@app.get("/models")
async def models() -> Dict[str, Any]:
    mgr = EngineManager.get()
    engines = ["chatgpt", "gemini"]
    return {
        "data": [
            {
                "name": e,
                "limits": mgr.get_engine(e).get_interface_limits(),
                "supported_models": mgr.get_engine(e).get_supported_models(),
            }
            for e in engines
        ]
    }

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

@app.post("/chatgpt/prompt")
async def chatgpt_prompt(req: Request) -> Any:
    if _rate_limit_exceeded(req):
        raise HTTPException(status_code=429, detail="Too many requests")
    return await _prompt("chatgpt", req, model_name="chatgpt", stream=bool((await req.json()).get("stream", False)))

@app.post("/gemini/prompt")
async def gemini_prompt(req: Request) -> Any:
    if _rate_limit_exceeded(req):
        raise HTTPException(status_code=429, detail="Too many requests")
    return await _prompt("gemini", req, model_name="gemini", stream=bool((await req.json()).get("stream", False)))

@app.post("/v1/chat/completions")
async def openai_chat(req: Request) -> Any:
    if _rate_limit_exceeded(req):
        raise HTTPException(status_code=429, detail="Too many requests")

    data = await req.json()
    model = data.get("model", "chatgpt")
    engine = "gemini" if "gemini" in model.lower() else "chatgpt"

    if "prompt" in data:
        prompt_payload = data.get("prompt")
    elif "messages" in data:
        prompt_payload = data.get("messages")
    else:
        raise HTTPException(status_code=400, detail="Missing prompt/messages")

    stream = bool(data.get("stream", False))

    return await _prompt(engine, req, explicit_prompt=prompt_payload, model_name=model, stream=stream)


async def _prompt(
    engine_name: str,
    req: Request = None,
    explicit_prompt: Any = None,
    model_name: str = "default",
    stream: bool = False,
) -> Any:
    if _rate_limit_exceeded(req):
        raise HTTPException(status_code=429, detail="Too many requests")

    if explicit_prompt is None:
        payload = await req.json()
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

    inc_requests()
    start = time.time()
    try:
        engine = EngineManager.get().set_active_engine(engine_name)

        if stream:
            async def generate_stream():
                try:
                    result = await engine.generate_response(prompt_text)
                    elapsed_ms = int((time.time() - start) * 1000)
                    log_prompt(engine_name, engine.get_current_model(), prompt_text, result, "ok", elapsed_ms)
                    inc_responses()
                    payload = _openai_response(engine_name, model_name, prompt_text, result, elapsed_ms)
                    yield f"data: {payload}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    elapsed_ms = int((time.time() - start) * 1000)
                    log_prompt(engine_name, "unknown", prompt_text, str(e), "error", elapsed_ms)
                    inc_errors()
                    raise HTTPException(status_code=500, detail=str(e))

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        response = await engine.generate_response(prompt_text)
        duration_ms = int((time.time() - start) * 1000)
        log_prompt(engine_name, engine.get_current_model(), prompt_text, response, "ok", duration_ms)
        inc_responses()

        return _openai_response(engine_name, model_name, prompt_text, response, duration_ms)

    except HTTPException:
        raise
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        log_prompt(engine_name, "unknown", prompt_text, str(e), "error", duration_ms)
        inc_errors()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats() -> Dict[str, Any]:
    return {"stats": get_stats(), "latest_logs": get_prompt_logs(20)}

@app.get("/logs")
async def logs(limit: int = 50, offset: int = 0):
    return get_prompt_logs(limit=limit, offset=offset)


@app.get("/ui", response_class=HTMLResponse)
async def ui() -> Any:
    html = Path("./web/index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)
