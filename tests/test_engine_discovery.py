from core.engine_manager import EngineManager


def test_engine_manager_loads_custom_engines():
    # Bypass existing singleton state to verify descriptor scanning from engines/ directory.
    orig_instance = EngineManager._instance
    EngineManager._instance = None
    try:
        mgr = EngineManager.get()
        engine_names = [desc["name"] for desc in mgr.list_engines()]
        assert "claude" in engine_names
        assert "stepfun" in engine_names
    finally:
        EngineManager._instance = orig_instance
