import asyncio
from fastapi.testclient import TestClient
import pytest

from app import app
from engine.engine_manager import EngineManager

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_engine_manager(monkeypatch):
    # avoid real selenium driver operations in tests
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

    manager = EngineManager.get()
    manager.engines.clear()
    monkeypatch.setattr("engine.engine_manager.SeleniumChatGPT", lambda **kwargs: DummyEngine())
    monkeypatch.setattr("engine.engine_manager.SeleniumGemini", lambda **kwargs: DummyEngine())

    yield

def test_ping():
    response = client.get("/api/ping")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_models():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data

def test_login_state():
    response = client.post("/login/chatgpt")
    assert response.status_code == 200
    assert response.json()["login_state"] == "unlogged"

def test_prompt():
    response = client.post("/chatgpt/prompt", json={"prompt": "Hello"})
    assert response.status_code == 200
    assert response.json()["response"] == "dummy response"
