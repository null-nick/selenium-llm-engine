from __future__ import annotations

import logging
import time
from typing import Any

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait

from core.selenium_llm_base import SeleniumLLMBase

logger = logging.getLogger("pizzagpt")

ENGINE_NAME = "pizzagpt"
ENGINE_ALIASES = ["pizzagpt", "pizza", "pizza-gpt"]
ENGINE_DISPLAY_NAME = "PizzaGPT"
ENGINE_SERVICE_URL = "https://www.pizzagpt.it/"
ENGINE_MODELS: dict[str, int] = {
    "gpt-5-mini": 12000,
    "gpt-oss-120b": 12000,
    "llama-3.3-70b": 12000,
    "qwen-3-32b": 12000,
    "openai/gpt-oss-safeguard-20b": 12000,
    "meta-llama/llama-3.2-1b-instruct": 12000,
}
ENGINE_DEFAULT_MODEL = "gpt-5-mini"
ENGINE_ALLOW_UNLOGGED = True


class PizzaGPTEngine(SeleniumLLMBase):
    ENGINE_NAME = ENGINE_NAME
    ENGINE_ALIASES = ENGINE_ALIASES
    ENGINE_DISPLAY_NAME = ENGINE_DISPLAY_NAME
    ENGINE_SERVICE_URL = ENGINE_SERVICE_URL
    ENGINE_MODELS = ENGINE_MODELS
    ENGINE_DEFAULT_MODEL = ENGINE_DEFAULT_MODEL
    ENGINE_ALLOW_UNLOGGED = ENGINE_ALLOW_UNLOGGED

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            service_url=ENGINE_SERVICE_URL,
            model_limits_map=ENGINE_MODELS,
            default_model=ENGINE_DEFAULT_MODEL,
            allow_unlogged=ENGINE_ALLOW_UNLOGGED,
            **kwargs,
        )
        self.prompt_area_selectors = [
            "textarea[placeholder='Scrivi qualcosa...']",
            "textarea.resize-none.w-full.textarea",
            "textarea",
        ]
        self.send_button_selectors = [
            "button[type='submit']",
            "button.btn-circle[type='button']",
            "div.flex.items-center.gap-2 button:last-of-type",
        ]
        self.response_area_selectors = [
            ".chat.group.chat-start .chat-bubble",
            ".chat-start .chat-bubble",
            ".chat-bubble .prose",
        ]
        self.stop_selectors = [
            "button[aria-label*='Stop']",
            "button[title*='Stop']",
        ]
        self._model_selectors = [
            "select.select.rounded-full.text-xs.border-none",
            "div.flex.items-center.gap-2 select",
            "select",
        ]
        self._model_aliases = {
            "gpt-5": "gpt-5-mini",
            "gpt5": "gpt-5-mini",
            "gpt-5-mini": "gpt-5-mini",
            "gpt-oss": "gpt-oss-120b",
            "oss": "gpt-oss-120b",
            "gpt-oss-120b": "gpt-oss-120b",
            "llama": "llama-3.3-70b",
            "llama-3.3": "llama-3.3-70b",
            "llama-3.3-70b": "llama-3.3-70b",
            "qwen": "qwen-3-32b",
            "qwen-3": "qwen-3-32b",
            "qwen-3-32b": "qwen-3-32b",
            "safeguard": "openai/gpt-oss-safeguard-20b",
            "gpt-oss-safeguard": "openai/gpt-oss-safeguard-20b",
            "openai/gpt-oss-safeguard-20b": "openai/gpt-oss-safeguard-20b",
            "llama-3.2": "meta-llama/llama-3.2-1b-instruct",
            "meta-llama/llama-3.2-1b-instruct": "meta-llama/llama-3.2-1b-instruct",
        }
        self._last_detected_model: str | None = None

    def _ensure_logged_in(self, driver: Any) -> bool:
        try:
            if not (driver.current_url or "").startswith(ENGINE_SERVICE_URL):
                driver.get(ENGINE_SERVICE_URL)
            for css in self.prompt_area_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, css)
                if any(self._element_is_displayed(el) for el in elements):
                    return True
            return True
        except Exception as exc:
            logger.warning("[pizzagpt] login check failed: %s", exc)
            return False

    def get_current_model(self) -> str:
        driver = getattr(self, "driver", None)
        if driver is not None:
            detected = self._detect_current_model(driver)
            if detected:
                self._last_detected_model = detected
                return detected
        return self._last_detected_model or super().get_current_model()

    def _prepare_for_prompt(
        self,
        driver: Any,
        model_name: str | None,
        reasoning_mode: str | None = None,
    ) -> None:
        del reasoning_mode
        requested_model = self._normalize_pizzagpt_model(model_name)
        current_model = self._detect_current_model(driver)
        if current_model:
            self._last_detected_model = current_model

        if not requested_model:
            return
        if requested_model not in self.ENGINE_MODELS:
            raise RuntimeError(
                f"pizzagpt_unknown_model: {requested_model}. Supported models: {', '.join(self.ENGINE_MODELS)}"
            )
        if current_model == requested_model:
            return

        logger.info("[pizzagpt] Switching model from %s to %s", current_model or "<unknown>", requested_model)
        select_el = self._find_model_select(driver)
        try:
            driver.execute_script("arguments[0].removeAttribute('disabled');", select_el)
        except Exception:
            pass
        Select(select_el).select_by_value(requested_model)
        try:
            driver.execute_script(
                "arguments[0].dispatchEvent(new Event('input', { bubbles: true }));"
                "arguments[0].dispatchEvent(new Event('change', { bubbles: true }));",
                select_el,
            )
        except Exception:
            pass

        deadline = time.time() + 8.0
        while time.time() < deadline:
            selected = self._detect_current_model(driver)
            if selected == requested_model:
                self._last_detected_model = selected
                return
            time.sleep(0.2)
        raise RuntimeError(f"pizzagpt_model_switch_timeout: failed to activate {requested_model}")

    def _detect_current_model(self, driver: Any) -> str | None:
        try:
            select_el = self._find_model_select(driver)
        except Exception:
            return self._last_detected_model

        value = (select_el.get_attribute("value") or "").strip()
        if value:
            return self._normalize_pizzagpt_model(value) or value

        try:
            option = select_el.find_element(By.CSS_SELECTOR, "option:checked")
            value = (option.get_attribute("value") or "").strip()
            if value:
                return self._normalize_pizzagpt_model(value) or value
        except Exception:
            pass
        return self._last_detected_model

    def _find_model_select(self, driver: Any) -> Any:
        for selector in self._model_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
            except Exception:
                elements = []
            for element in elements:
                try:
                    options = element.find_elements(By.CSS_SELECTOR, "option[value]")
                except Exception:
                    options = []
                values = {(opt.get_attribute("value") or "").strip() for opt in options}
                if values & set(self.ENGINE_MODELS.keys()):
                    return element
        raise RuntimeError("pizzagpt_model_select_not_found")

    def _normalize_pizzagpt_model(self, model_name: str | None) -> str | None:
        requested = self._canonicalize_requested_model(model_name)
        if requested is None:
            return None
        lowered = requested.strip().lower()
        if lowered in self._model_aliases:
            return self._model_aliases[lowered]
        return requested

    def _extract_response_text_from_element(self, driver: Any, element: Any) -> str:
        try:
            result = driver.execute_script(
                "const root = arguments[0];"
                "if (!root) return '';"
                "const prose = root.matches('.prose') ? root : root.querySelector('.prose');"
                "if (!prose) return '';"
                "return (prose.innerText || prose.textContent || '').trim();",
                element,
            )
            if isinstance(result, str) and result.strip():
                return result.strip()
        except Exception:
            pass
        return super()._extract_response_text_from_element(driver, element)
