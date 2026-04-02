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
        assert "copilot" in engine_names
        assert "grok" in engine_names
        assert "perplexity" in engine_names
    finally:
        EngineManager._instance = orig_instance


def test_engine_manager_reports_copilot_notes():
    orig_instance = EngineManager._instance
    EngineManager._instance = None
    try:
        mgr = EngineManager.get()
        engine_data = mgr.list_engines()
        copilot = next((e for e in engine_data if e['name'] == 'copilot'), None)
        assert copilot is not None
        assert 'notes' in copilot
        assert isinstance(copilot['notes'], str)
        assert len(copilot['notes']) > 0
    finally:
        EngineManager._instance = orig_instance
