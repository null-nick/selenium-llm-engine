# engine/ — legacy package stub (kept for backward compatibility)
#
# The engine implementation has been moved to core/:
#   core/selenium_llm_base.py  — base class
#   core/json_engine.py        — JSON-driven engine
#   core/engine_manager.py     — discovery & registry
#
# Engine definitions (JSON and Python) now live in engines/:
#   engines/chatgpt.json
#   engines/gemini.json
#   engines/_example_custom.py.dist  — template for new engines
