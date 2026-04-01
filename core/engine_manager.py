"""EngineManager — discovers, registers, and manages Selenium LLM engines.

Engines are loaded from the ``engines/`` directory (relative to this file's
project root) and can be defined in two ways:

JSON definition (recommended for simple engines)
    Any ``*.json`` file inside the ``engines/`` directory is loaded as a
    :class:`~core.json_engine.JsonEngine`.  See ``DEVELOPERS.md`` for the
    full schema.

Python definition (required for complex / custom logic)
    Any ``*.py`` file (excluding files whose name starts with ``_``) inside the
    ``engines/`` directory is imported dynamically.  The module must expose
    **exactly one** class that:

    * inherits from :class:`~core.selenium_llm_base.SeleniumLLMBase`
    * defines a class-level ``ENGINE_NAME: str`` attribute
    * optionally defines ``ENGINE_ALIASES: list[str]``

Both file types can coexist — a ``.py`` engine with the same name as a
``.json`` engine is ignored (JSON takes precedence).

Backward compatibility
    If no ``engines/`` directory exists the manager falls back to the
    hard-coded ChatGPT / Gemini engines so that existing deployments continue
    to work without any changes.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

from core.selenium_llm_base import SeleniumLLMBase

logger = logging.getLogger("engine_manager")

# Root of the project (parent of the ``core/`` package directory).
_PROJECT_ROOT = Path(__file__).parent.parent
_ENGINES_DIR = _PROJECT_ROOT / "engines"


# ---------------------------------------------------------------------------
# Descriptor (lightweight — no browser started)
# ---------------------------------------------------------------------------


@dataclass
class EngineDescriptor:
    """Metadata about an engine without instantiating it."""

    name: str
    aliases: list[str]
    display_name: str
    service_url: str
    models: dict[str, int]
    default_model: str
    source: str  # "json" | "python" | "builtin"
    source_path: str  # filesystem path or "<builtin>"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "aliases": self.aliases,
            "service_url": self.service_url,
            "models": self.models,
            "default_model": self.default_model,
            "source": self.source,
            "source_path": self.source_path,
        }


# ---------------------------------------------------------------------------
# Engine scanning helpers
# ---------------------------------------------------------------------------


def _scan_json(path: Path) -> Optional[EngineDescriptor]:
    try:
        with path.open(encoding="utf-8") as fh:
            cfg = json.load(fh)
        name = cfg.get("name")
        if not name:
            logger.warning(f"[engine_manager] JSON engine without 'name': {path}")
            return None
        return EngineDescriptor(
            name=name,
            aliases=list(cfg.get("aliases", [name])),
            display_name=cfg.get("display_name", name),
            service_url=cfg.get("service_url", ""),
            models=dict(cfg.get("models", {"default": 10000})),
            default_model=cfg.get("default_model", "default"),
            source="json",
            source_path=str(path),
        )
    except Exception as exc:
        logger.warning(f"[engine_manager] Failed to scan JSON engine {path}: {exc}")
        return None


def _scan_python(path: Path) -> Optional[EngineDescriptor]:
    if path.name.startswith("_"):
        return None  # private / template files
    try:
        spec = importlib.util.spec_from_file_location(f"engines._dyn.{path.stem}", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj is not SeleniumLLMBase
                and issubclass(obj, SeleniumLLMBase)
                and hasattr(obj, "ENGINE_NAME")
            ):
                engine_name: str = obj.ENGINE_NAME
                return EngineDescriptor(
                    name=engine_name,
                    aliases=list(getattr(obj, "ENGINE_ALIASES", [engine_name])),
                    display_name=getattr(obj, "ENGINE_DISPLAY_NAME", engine_name),
                    service_url=getattr(obj, "ENGINE_SERVICE_URL", ""),
                    models=dict(getattr(obj, "ENGINE_MODELS", {"default": 10000})),
                    default_model=getattr(obj, "ENGINE_DEFAULT_MODEL", "default"),
                    source="python",
                    source_path=str(path),
                )
        logger.warning(
            f"[engine_manager] No SeleniumLLMBase subclass with ENGINE_NAME in {path}"
        )
        return None
    except Exception as exc:
        logger.warning(f"[engine_manager] Failed to scan Python engine {path}: {exc}")
        return None


def scan_engines(engines_dir: Path) -> dict[str, EngineDescriptor]:
    """Scan *engines_dir* and return a name→descriptor mapping.

    JSON files take precedence over Python files with the same engine name.
    """
    descriptors: dict[str, EngineDescriptor] = {}

    if not engines_dir.is_dir():
        logger.warning(
            f"[engine_manager] engines/ directory not found at {engines_dir} — "
            "falling back to built-in engines"
        )
        return descriptors

    # --- pass 1: JSON ---
    for path in sorted(engines_dir.glob("*.json")):
        desc = _scan_json(path)
        if desc and desc.name not in descriptors:
            descriptors[desc.name] = desc
            for alias in desc.aliases:
                if alias != desc.name:
                    logger.debug(
                        f"[engine_manager] Registered JSON engine '{desc.name}' "
                        f"(alias: {alias})"
                    )
            logger.info(
                f"[engine_manager] Loaded JSON engine '{desc.name}' from {path.name}"
            )

    # --- pass 2: Python ---
    for path in sorted(engines_dir.glob("*.py")):
        desc = _scan_python(path)
        if desc and desc.name not in descriptors:
            descriptors[desc.name] = desc
            logger.info(
                f"[engine_manager] Loaded Python engine '{desc.name}' from {path.name}"
            )
        elif desc:
            logger.debug(
                f"[engine_manager] Python engine '{desc.name}' skipped "
                f"(already registered from JSON)"
            )

    return descriptors


# ---------------------------------------------------------------------------
# Engine instantiation
# ---------------------------------------------------------------------------


def _instantiate(descriptor: EngineDescriptor, **kwargs) -> SeleniumLLMBase:
    """Create a live engine instance from its descriptor."""
    if descriptor.source == "json":
        from core.json_engine import JsonEngine

        return JsonEngine(Path(descriptor.source_path), **kwargs)

    if descriptor.source == "python":
        spec = importlib.util.spec_from_file_location(
            f"engines._dyn.{descriptor.name}", descriptor.source_path
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj is not SeleniumLLMBase
                and issubclass(obj, SeleniumLLMBase)
                and getattr(obj, "ENGINE_NAME", None) == descriptor.name
            ):
                return obj(**kwargs)
        raise RuntimeError(
            f"Python engine class for '{descriptor.name}' not found in "
            f"{descriptor.source_path}"
        )

    if descriptor.source == "builtin":
        return _builtin_instance(descriptor.name, **kwargs)

    raise ValueError(f"Unknown engine source type: {descriptor.source!r}")


def _builtin_descriptors() -> dict[str, EngineDescriptor]:
    """Hard-coded fallback used when no ``engines/`` directory exists."""
    chatgpt = EngineDescriptor(
        name="chatgpt",
        aliases=["chatgpt", "openai", "gpt"],
        display_name="ChatGPT (OpenAI)",
        service_url="https://chat.openai.com",
        models={
            "gpt-4o": 60000,
            "gpt-4o-mini": 60000,
            "gpt-4-turbo": 50000,
            "gpt-4": 40000,
            "gpt-3.5-turbo": 30000,
            "unlogged": 20000,
            "default": 51000,
        },
        default_model="gpt-4o",
        source="builtin",
        source_path="<builtin>",
    )
    gemini = EngineDescriptor(
        name="gemini",
        aliases=["gemini", "google"],
        display_name="Gemini (Google)",
        service_url="https://gemini.google.com",
        models={
            "2.5-flash": 32000,
            "2.0-flash": 32000,
            "1.5-flash": 100000,
            "1.5-pro": 500000,
            "unlogged": 21500,
            "default": 32000,
        },
        default_model="2.5-flash",
        source="builtin",
        source_path="<builtin>",
    )
    result: dict[str, EngineDescriptor] = {"chatgpt": chatgpt, "gemini": gemini}
    for alias in chatgpt.aliases:
        result.setdefault(alias, chatgpt)
    for alias in gemini.aliases:
        result.setdefault(alias, gemini)
    return result


# Hardcoded JSON configs for built-in engines (used when engines/ dir missing).
_BUILTIN_CONFIGS: dict[str, dict] = {
    "chatgpt": {
        "name": "chatgpt",
        "display_name": "ChatGPT (OpenAI)",
        "aliases": ["chatgpt", "openai", "gpt"],
        "service_url": "https://chat.openai.com",
        "default_model": "gpt-4o",
        "models": {
            "gpt-4o": 60000,
            "gpt-4o-mini": 60000,
            "gpt-4-turbo": 50000,
            "gpt-4": 40000,
            "gpt-3.5-turbo": 30000,
            "unlogged": 20000,
            "default": 51000,
        },
        "selectors": {
            "prompt_area": [
                "div[data-testid='prompt-textarea'][contenteditable='true']",
                "div.ProseMirror[contenteditable='true']",
                "div.ProseMirror",
                "#prompt-textarea",
                "textarea[data-testid='prompt-textarea']",
                "div[contenteditable='true'][data-placeholder]",
                "textarea",
                "div[contenteditable='true']",
            ],
            "send_button": [
                "button[data-testid='send-button']",
                "#composer-submit-button",
                "button[aria-label='Send prompt']",
                "button[aria-label*='Send']",
            ],
            "response_area": [
                "[data-message-author-role='assistant']",
                "div.markdown.prose",
                "div.markdown",
                ".agent-turn",
            ],
            "stop": [
                "button[data-testid='stop-button']",
                "button[aria-label='Stop generating']",
                "button[aria-label*='Stop']",
            ],
        },
        "login_detection": {
            "url_prefix": "https://chat.openai.com",
            "url_deny_keywords": ["login", "auth", "signin"],
            "login_button_xpath": "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'log in') or contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in')]",
            "authenticated_css_selectors": [
                "div[data-testid='conversation-panel']",
                "div[data-testid='chat-history']",
                "div[data-testid='disabled-service']",
            ],
        },
    },
    "gemini": {
        "name": "gemini",
        "display_name": "Gemini (Google)",
        "aliases": ["gemini", "google"],
        "service_url": "https://gemini.google.com",
        "default_model": "2.5-flash",
        "models": {
            "2.5-flash": 32000,
            "2.0-flash": 32000,
            "1.5-flash": 100000,
            "1.5-pro": 500000,
            "unlogged": 21500,
            "default": 32000,
        },
        "selectors": {
            "prompt_area": [
                "div[contenteditable='true'][data-placeholder]",
                "div[contenteditable='true'][aria-label*='Ask']",
                "div[contenteditable='true'][aria-label*='Message']",
                "div.ql-editor[contenteditable='true']",
                "textarea[placeholder*='Ask']",
                "div[contenteditable='true']",
                "textarea",
            ],
            "send_button": [
                "button[aria-label='Send message']",
                "button[aria-label*='Send message']",
                "button[data-testid='send-button']",
                "button[aria-label*='Send']",
                "button[type='submit']",
            ],
            "response_area": [
                "model-response .markdown",
                "model-response",
                ".model-response",
                ".response-container",
                "message-content",
                ".message-content",
                "div[class*='response-text']",
            ],
            "stop": [
                "button[aria-label='Stop response']",
                "button[aria-label='Cancel']",
                "button[aria-label*='Stop']",
                ".stop-button",
                "[data-testid='stop-button']",
            ],
        },
        "login_detection": {
            "url_prefix": "https://gemini.google.com",
            "url_deny_keywords": ["signin", "login"],
            "login_button_xpath": "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in') or contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'login')]",
            "authenticated_css_selectors": [
                "div.assistant-message",
                ".gemini-response",
                ".chat-message.ai",
            ],
        },
    },
}


def _builtin_instance(name: str, **kwargs) -> SeleniumLLMBase:
    """Instantiate a built-in engine without requiring the engines/ directory."""
    from core.json_engine import JsonEngine

    canonical = name.strip().lower()
    # Resolve any well-known alias
    alias_map = {
        "openai": "chatgpt",
        "gpt": "chatgpt",
        "google": "gemini",
    }
    canonical = alias_map.get(canonical, canonical)
    if canonical not in _BUILTIN_CONFIGS:
        raise ValueError(f"Unknown built-in engine: {name}")
    return JsonEngine(_BUILTIN_CONFIGS[canonical], **kwargs)


# ---------------------------------------------------------------------------
# EngineManager singleton
# ---------------------------------------------------------------------------


class EngineManager:
    """Thread-safe singleton that manages engine lifecycle and discovery."""

    _instance: Optional["EngineManager"] = None
    _lock: Lock = Lock()

    def __init__(self) -> None:
        self.engines: dict[str, SeleniumLLMBase] = {}
        self.active_engine: Optional[SeleniumLLMBase] = None
        self._descriptors: dict[str, EngineDescriptor] = {}
        self._alias_map: dict[str, str] = {}  # alias → canonical name
        self._load_descriptors()

    @classmethod
    def get(cls) -> "EngineManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = EngineManager()
        return cls._instance

    # ---------------------------------------------------------------------- discovery

    def _load_descriptors(self) -> None:
        """Scan engines/ and build the descriptor + alias maps."""
        self._descriptors.clear()
        self._alias_map.clear()

        raw = scan_engines(_ENGINES_DIR)
        if not raw:
            logger.info(
                "[engine_manager] No engines found in engines/ — using built-in fallback"
            )
            raw = _builtin_descriptors()

        for desc in raw.values():
            if desc.name in self._descriptors:
                continue  # already registered
            self._descriptors[desc.name] = desc
            for alias in desc.aliases:
                self._alias_map[alias.strip().lower()] = desc.name

    def reload_engines(self) -> list[dict]:
        """Re-scan the engines/ directory and refresh the registry.

        Already-running engine instances are **not** stopped — they
        remain available until :meth:`stop_all` is called.
        """
        logger.info("[engine_manager] Reloading engine registry…")
        self._load_descriptors()
        return self.list_engines()

    def list_engines(self) -> list[dict]:
        """Return metadata for all registered engines (no browser started)."""
        seen: set[str] = set()
        result = []
        for desc in self._descriptors.values():
            if desc.name not in seen:
                result.append(desc.to_dict())
                seen.add(desc.name)
        return result

    # ---------------------------------------------------------------------- access

    def _resolve(self, name: str) -> str:
        """Resolve an alias or name to the canonical engine name."""
        key = name.strip().lower()
        if key in self._alias_map:
            return self._alias_map[key]
        # Also try direct name match
        if key in self._descriptors:
            return key
        raise ValueError(f"Unknown engine: '{name}'")

    def get_engine(self, name: str) -> SeleniumLLMBase:
        """Return (and lazy-instantiate) the engine identified by *name* or an alias."""
        canonical = self._resolve(name)
        if canonical not in self.engines:
            desc = self._descriptors[canonical]
            logger.info(f"[engine_manager] Instantiating engine '{canonical}'…")
            self.engines[canonical] = _instantiate(desc)
        return self.engines[canonical]

    def set_active_engine(self, name: str) -> SeleniumLLMBase:
        engine = self.get_engine(name)
        self.active_engine = engine
        return engine

    def get_active_engine(self) -> SeleniumLLMBase:
        if not self.active_engine:
            raise RuntimeError("No active engine set")
        return self.active_engine

    # ---------------------------------------------------------------------- lifecycle

    async def stop_all(self) -> None:
        for engine in self.engines.values():
            try:
                await engine.stop()
            except Exception:
                pass
