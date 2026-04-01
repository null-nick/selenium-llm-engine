from fastapi.testclient import TestClient
import json
import pytest

from app import app
from core.engine_manager import EngineManager

client = TestClient(app)


class DummyEngine:
    def __init__(self):
        self.model = "default"

    def get_interface_limits(self):
        return {"max_prompt_chars": 1234, "model_name": "default"}

    def get_supported_models(self):
        return ["default"]

    async def start_login_flow(self):
        return {"logged_in": False, "login_state": "unlogged"}

    async def check_login_state(self):
        return {"logged_in": False, "login_state": "unlogged"}

    async def generate_response(self, prompt):
        return "dummy response"

    def get_current_model(self):
        return "default"


@pytest.fixture(autouse=True)
def setup_engine_manager(monkeypatch):
    """Replace the EngineManager singleton with a pre-loaded test instance."""
    # Mock DB calls so no filesystem writes happen during tests
    monkeypatch.setattr("app.inc_requests", lambda: None)
    monkeypatch.setattr("app.inc_responses", lambda: None)
    monkeypatch.setattr("app.inc_errors", lambda: None)
    monkeypatch.setattr("app.log_prompt", lambda *a, **kw: None)

    mgr = EngineManager.get()
    mgr.engines.clear()
    mgr.active_engine = None

    # Inject two synthetic descriptors so /models and /api/engines work
    from core.engine_manager import EngineDescriptor

    chatgpt_desc = EngineDescriptor(
        name="chatgpt",
        aliases=["chatgpt", "openai", "gpt"],
        display_name="ChatGPT (test)",
        service_url="https://chat.openai.com",
        models={"default": 51000},
        default_model="default",
        source="builtin",
        source_path="<test>",
    )
    gemini_desc = EngineDescriptor(
        name="gemini",
        aliases=["gemini", "google"],
        display_name="Gemini (test)",
        service_url="https://gemini.google.com",
        models={"default": 32000},
        default_model="default",
        source="builtin",
        source_path="<test>",
    )
    mgr._descriptors = {"chatgpt": chatgpt_desc, "gemini": gemini_desc}
    mgr._alias_map = {
        "chatgpt": "chatgpt",
        "openai": "chatgpt",
        "gpt": "chatgpt",
        "gemini": "gemini",
        "google": "gemini",
    }

    # Pre-populate with DummyEngine instances so no real Selenium init happens
    mgr.engines["chatgpt"] = DummyEngine()
    mgr.engines["gemini"] = DummyEngine()

    yield

    mgr.engines.clear()
    mgr.active_engine = None


def test_ping():
    response = client.get("/api/ping")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models():
    """Legacy /models must return OpenAI-compatible format (id field required by clients like Alpaca)."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert "data" in data
    for entry in data["data"]:
        assert "id" in entry, "Each model entry must have an 'id' field"
        assert entry["id"] is not None
        assert entry["object"] == "model"
        # Legacy extra fields still present
        assert "name" in entry


def test_legacy_chat_completions():
    """POST /chat/completions (without /v1) must work as alias."""
    response = client.post(
        "/chat/completions",
        json={"model": "chatgpt", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"


def test_login_state():
    response = client.post("/login/chatgpt")
    assert response.status_code == 200
    assert response.json()["login_state"] == "unlogged"


def test_prompt_legacy_chatgpt():
    response = client.post("/chatgpt/prompt", json={"prompt": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "dummy response"


def test_prompt_invalid_json_body():
    response = client.post(
        "/chatgpt/prompt",
        data="http://localhost:14848/v1/chat/completions",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert "Invalid JSON body" in response.json()["detail"]


def test_prompt_dynamic_endpoint():
    response = client.post("/engine/chatgpt/prompt", json={"prompt": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "dummy response"


def test_prompt_dynamic_endpoint_alias():
    """Engine aliases should work on the dynamic endpoint too."""
    response = client.post("/engine/openai/prompt", json={"prompt": "Hello"})
    assert response.status_code == 200


def test_prompt_unknown_engine():
    response = client.post("/engine/nonexistent/prompt", json={"prompt": "Hello"})
    assert response.status_code == 404


def test_api_engines():
    response = client.get("/api/engines")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    names = [e["name"] for e in data["data"]]
    assert "chatgpt" in names
    assert "gemini" in names


def test_api_engines_reload():
    """Reload endpoint must return 200 and a valid data list."""
    response = client.post("/api/engines/reload")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["data"], list)


def test_unlogged_flag_behavior():
    from pathlib import Path
    from core.json_engine import JsonEngine
    from core.selenium_llm_base import SeleniumLLMBase

    base_engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"unlogged": 20000, "default": 50000},
        default_model="default",
    )
    base_engine.is_user_logged_in = lambda: False
    assert base_engine.get_current_model() == "default"

    unlogged_engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"unlogged": 20000, "default": 50000},
        default_model="default",
        allow_unlogged=True,
    )
    unlogged_engine.is_user_logged_in = lambda: False
    assert unlogged_engine.get_current_model() == "unlogged"

    chatgpt_engine = JsonEngine(Path("engines/chatgpt.json"))
    chatgpt_engine.is_user_logged_in = lambda: False
    assert chatgpt_engine.get_current_model() == "unlogged"


def test_reset_state():
    manager = EngineManager.get()
    # set active engine then verify reset clears it
    manager.active_engine = manager.engines.get("chatgpt")
    assert manager.active_engine is not None

    response = client.post("/reset")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert manager.engines == {}
    assert manager.active_engine is None


def test_logs_history_endpoint():
    # ensure prompt logging endpoint is accessible and returns a list
    response = client.get("/logs?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_api_history_endpoint():
    response = client.get("/api/history?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


# ---------------------------------------------------------------------------
# OpenAI-compatible /v1/* endpoint tests
# ---------------------------------------------------------------------------


def test_v1_models_list():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    ids = [m["id"] for m in data["data"]]
    assert "chatgpt" in ids
    assert "gemini" in ids
    # only canonical names — no aliases, no provider:variant
    assert not any(":" in mid for mid in ids)
    for entry in data["data"]:
        assert entry["object"] == "model"
        assert entry["owned_by"] == "selenium-llm-engine"


def test_v1_models_single():
    response = client.get("/v1/models/chatgpt")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "chatgpt"
    assert data["object"] == "model"


def test_v1_models_variant():
    response = client.get("/v1/models/chatgpt:default")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "chatgpt:default"


def test_v1_models_unknown():
    response = client.get("/v1/models/nonexistent_engine")
    assert response.status_code == 404


def test_v1_chat_completions_messages():
    response = client.post(
        "/v1/chat/completions",
        json={"model": "chatgpt", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "dummy response"


def test_v1_chat_null_model():
    """model=null must not crash — falls back to chatgpt."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": None, "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 200


def test_v1_chat_provider_variant_model():
    """provider:variant notation must resolve to the correct engine."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": "chatgpt:gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 200


def test_token_count_nonzero():
    response = client.post("/chatgpt/prompt", json={"prompt": "Hello world"})
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_v1_streaming_sse_format():
    """stream=True must return SSE with chat.completion.chunk objects."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "chatgpt", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line.startswith("data:")]
    assert any("[DONE]" in line for line in lines)
    data_lines = [line for line in lines if "[DONE]" not in line]
    assert len(data_lines) >= 1
    for line in data_lines:
        chunk = json.loads(line.removeprefix("data:").strip())
        assert chunk["object"] == "chat.completion.chunk"
        assert "choices" in chunk
