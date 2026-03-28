from threading import Lock
from typing import Dict, Optional

from engine.selenium_chatgpt import SeleniumChatGPT
from engine.selenium_gemini import SeleniumGemini
from engine.selenium_llm_base import SeleniumLLMBase


class EngineManager:
    _instance = None
    _lock = Lock()

    def __init__(self):
        self.engines: Dict[str, SeleniumLLMBase] = {}
        self.active_engine: Optional[SeleniumLLMBase] = None

    @classmethod
    def get(cls) -> "EngineManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = EngineManager()
        return cls._instance

    def get_engine(self, name: str) -> SeleniumLLMBase:
        normalized = name.strip().lower()
        if normalized not in self.engines:
            if normalized in ("chatgpt", "openai", "gpt"):
                self.engines[normalized] = SeleniumChatGPT()
            elif normalized in ("gemini", "google", "gemini"):
                self.engines[normalized] = SeleniumGemini()
            else:
                raise ValueError(f"Unsupported engine: {name}")
        return self.engines[normalized]

    def set_active_engine(self, name: str) -> SeleniumLLMBase:
        engine = self.get_engine(name)
        self.active_engine = engine
        return engine

    def get_active_engine(self) -> SeleniumLLMBase:
        if not self.active_engine:
            raise RuntimeError("No active engine set")
        return self.active_engine

    async def stop_all(self) -> None:
        for engine in self.engines.values():
            try:
                await engine.stop()
            except Exception:
                pass
