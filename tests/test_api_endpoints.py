from fastapi.testclient import TestClient
import asyncio
import json
import threading
import time
import types
import sys
import pytest

# Some environments may not have distutils installed for undetected_chromedriver.
# Use a minimal fake module so unit tests can import core modules safely.
if "undetected_chromedriver" not in sys.modules:
    sys.modules["undetected_chromedriver"] = types.SimpleNamespace(
        Chrome=lambda *args, **kwargs: None
    )

from app import app, _register_engine_routes
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

    # Re-register dynamic per-engine routes for this test fixture
    _register_engine_routes(app)

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
    class FakeBaseEngine:
        def __init__(self, model_limits_map, default_model, allow_unlogged=False):
            self.model_limits_map = model_limits_map
            self.default_model = default_model
            self.allow_unlogged = allow_unlogged
            self._logged_in = True

        def is_user_logged_in(self):
            return self._logged_in

        def set_logged_in(self, value):
            self._logged_in = value

        def get_current_model(self):
            if not self.is_user_logged_in() and self.allow_unlogged and "unlogged" in self.model_limits_map:
                return "unlogged"
            return self.default_model

    base_engine = FakeBaseEngine(
        model_limits_map={"unlogged": 20000, "default": 50000},
        default_model="default",
    )
    base_engine.set_logged_in(False)
    assert base_engine.get_current_model() == "default"

    unlogged_engine = FakeBaseEngine(
        model_limits_map={"unlogged": 20000, "default": 50000},
        default_model="default",
        allow_unlogged=True,
    )
    unlogged_engine.set_logged_in(False)
    assert unlogged_engine.get_current_model() == "unlogged"


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

    stats_res = client.get("/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert "stats" in stats_data
    # if DB is writable/clearable, stats may be empty; if readonly they may persist
    assert isinstance(stats_data["stats"], dict)
    assert "response_time" in stats_data
    assert "global_avg_ms" in stats_data["response_time"]
    assert "per_engine_avg_ms" in stats_data["response_time"]
    assert isinstance(stats_data["response_time"]["per_engine_avg_ms"], dict)


def test_api_reset_alias():
    manager = EngineManager.get()
    manager.active_engine = manager.engines.get("chatgpt")

    response = client.post("/api/reset")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert manager.engines == {}
    assert manager.active_engine is None


def test_reset_cancels_inflight_requests():
    class SlowDummyEngine:
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
            await asyncio.sleep(3)
            return "slow response"

        async def stop(self):
            return

        def get_current_model(self):
            return "default"

    manager = EngineManager.get()
    manager.engines["chatgpt"] = SlowDummyEngine()

    results = {}

    def call_prompt():
        try:
            r = client.post("/engine/chatgpt/prompt", json={"prompt": "hello"})
            results["response"] = r
        except Exception as e:
            results["error"] = e

    thread = threading.Thread(target=call_prompt)
    thread.start()

    # wait a moment for request to be in-flight
    time.sleep(0.1)

    response = client.post("/reset")
    assert response.status_code == 200

    thread.join(timeout=10)
    assert not thread.is_alive()

    assert "response" in results or "error" in results
    if "response" in results:
        assert results["response"].status_code in (503, 500)


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


def test_captcha_detection_short_circuit(monkeypatch):
    from core.selenium_llm_base import SeleniumLLMBase

    class FakeCaptchaDriver:
        current_url = "https://chat.openai.com"

        def find_elements(self, by, selector):
            if selector == "iframe#cf-chl-widget-ezspn":
                return [object()]
            return []

    engine = SeleniumLLMBase(
        service_url="https://chat.openai.com",
        model_limits_map={"default": 50000},
        default_model="default",
    )
    engine._ensure_ready = lambda: None
    engine.driver = FakeCaptchaDriver()

    result = engine._sync_generate_response_once("Hello")
    assert "CAPTCHA" in result or "captcha" in result
    assert "completa" in result


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


# ---------------------------------------------------------------------------
# Selector hints endpoint and selector caching regression tests
# ---------------------------------------------------------------------------


def test_selector_hints_empty_when_no_prompts():
    """GET /api/engines/selector-hints returns an empty data dict before any prompt is sent."""
    mgr = EngineManager.get()
    mgr.engines.clear()
    response = client.get("/api/engines/selector-hints")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert data["data"] == {}


def test_selector_hints_structure_after_engine_loaded():
    """Once an engine instance is in the manager the hints endpoint must expose its selector lists."""
    engine = DummyEngine()
    engine.prompt_area_selectors = ["textarea", "div[contenteditable='true']"]
    engine.send_button_selectors = ["button[type='submit']", "button[aria-label*='Send']"]
    engine._cached_prompt_selector = None
    engine._cached_send_selector = None

    mgr = EngineManager.get()
    mgr.engines["chatgpt"] = engine

    response = client.get("/api/engines/selector-hints")
    assert response.status_code == 200
    data = response.json()["data"]
    assert "chatgpt" in data
    hints = data["chatgpt"]
    assert "prompt_selector" in hints
    assert "send_selector" in hints
    assert "prompt_area_selectors" in hints
    assert "send_button_selectors" in hints
    assert hints["prompt_selector"] is None
    assert hints["send_selector"] is None
    assert hints["prompt_area_selectors"] == engine.prompt_area_selectors
    assert hints["send_button_selectors"] == engine.send_button_selectors


def test_selector_hints_reflect_cached_values():
    """Cached selectors are included in the hints response after being set."""
    engine = DummyEngine()
    engine.prompt_area_selectors = ["textarea", "div[contenteditable='true']"]
    engine.send_button_selectors = ["button[type='submit']", "button[aria-label*='Send']"]
    engine._cached_prompt_selector = "div[contenteditable='true']"
    engine._cached_send_selector = "button[aria-label*='Send']"

    mgr = EngineManager.get()
    mgr.engines["gemini"] = engine

    response = client.get("/api/engines/selector-hints")
    assert response.status_code == 200
    hints = response.json()["data"]["gemini"]
    assert hints["prompt_selector"] == "div[contenteditable='true']"
    assert hints["send_selector"] == "button[aria-label*='Send']"


def test_find_interactable_element_caches_selector():
    """_find_interactable_element sets cache_attr to the found selector."""
    try:
        from core.selenium_llm_base import SeleniumLLMBase
    except ModuleNotFoundError:
        pytest.skip("undetected_chromedriver not compatible with this Python version")

    from unittest.mock import MagicMock, patch

    base = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    assert base._cached_prompt_selector is None

    mock_driver = MagicMock()
    fake_el = MagicMock()

    winning_selector = "div[contenteditable='true']"

    def fake_wait_until(condition):
        # Simulate: first selector times out, second succeeds
        sel = condition.locator[1]
        if sel == winning_selector:
            return fake_el
        from selenium.common.exceptions import TimeoutException
        raise TimeoutException()

    mock_wait = MagicMock()
    mock_wait.until.side_effect = fake_wait_until

    def make_wait(driver, timeout):
        return mock_wait

    with patch("core.selenium_llm_base.WebDriverWait", side_effect=make_wait):
        selectors = ["textarea", winning_selector]
        result = base._find_interactable_element(
            mock_driver, selectors, timeout=3.0, cache_attr="_cached_prompt_selector"
        )

    assert result == fake_el
    assert base._cached_prompt_selector == winning_selector


def test_find_interactable_element_tries_cached_first():
    """When a cached selector exists it is tried before others."""
    try:
        from core.selenium_llm_base import SeleniumLLMBase
    except ModuleNotFoundError:
        pytest.skip("undetected_chromedriver not compatible with this Python version")

    from unittest.mock import MagicMock, patch

    base = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    cached_sel = "div[contenteditable='true']"
    base._cached_prompt_selector = cached_sel

    tried_order: list[str] = []
    fake_el = MagicMock()

    def fake_wait_until(condition):
        sel = condition.locator[1]
        tried_order.append(sel)
        if sel == cached_sel:
            return fake_el
        from selenium.common.exceptions import TimeoutException
        raise TimeoutException()

    mock_wait = MagicMock()
    mock_wait.until.side_effect = fake_wait_until

    with patch("core.selenium_llm_base.WebDriverWait", return_value=mock_wait):
        selectors = ["textarea", cached_sel, "input"]
        base._find_interactable_element(
            MagicMock(), selectors, timeout=3.0, cache_attr="_cached_prompt_selector"
        )

    assert tried_order[0] == cached_sel, "Cached selector must be tried first"


# ---------------------------------------------------------------------------
# New endpoints: /api/logs/app and updated /stats
# ---------------------------------------------------------------------------


def test_app_logs_endpoint_returns_list():
    """GET /api/logs/app must return a JSON object with an 'entries' list."""
    response = client.get("/api/logs/app")
    assert response.status_code == 200
    data = response.json()
    assert "entries" in data
    assert isinstance(data["entries"], list)


def test_app_logs_since_parameter():
    """Passing since=<large_int> must return only newer entries (or an empty list)."""
    response = client.get("/api/logs/app?since=999999")
    assert response.status_code == 200
    data = response.json()
    assert data["entries"] == []


def test_stats_includes_logged_engines():
    """GET /stats must include a 'logged_engines' list instead of 'latest_logs'."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    assert "logged_engines" in data
    assert isinstance(data["logged_engines"], list)
    assert "latest_logs" not in data


def test_stats_includes_response_time():
    """GET /stats must include response time averages."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "response_time" in data
    assert isinstance(data["response_time"], dict)
    assert "global_avg_ms" in data["response_time"]
    assert "per_engine_avg_ms" in data["response_time"]
    assert isinstance(data["response_time"]["per_engine_avg_ms"], dict)
