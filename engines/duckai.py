from __future__ import annotations

import logging
import time
from typing import Any

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from core.selenium_llm_base import SeleniumLLMBase

logger = logging.getLogger("duckai")

ENGINE_NAME = "duckai"
ENGINE_ALIASES = ["duckai", "duck", "duckduckgo"]
ENGINE_DISPLAY_NAME = "Duck.ai"
ENGINE_SERVICE_URL = "https://duck.ai"
ENGINE_MODELS: dict[str, int] = {
    "gpt-5-mini": 12000,
    "gpt-4o-mini": 12000,
    "tinfoil/gpt-oss-120b": 12000,
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": 12000,
    "claude-haiku-4-5": 12000,
    "mistral-small-2603": 12000,
}
ENGINE_DEFAULT_MODEL = "gpt-5-mini"
ENGINE_ALLOW_UNLOGGED = True


class DuckAIEngine(SeleniumLLMBase):
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
            "textarea[name='user-prompt']",
            "div[data-testid='duckai-chat-input'] textarea",
            "form[data-chat-footer='true'] textarea",
            "textarea[placeholder*='Rispondi']",
            "textarea",
        ]
        self.send_button_selectors = [
            "button[type='submit'][aria-label='Invia']",
            "form[data-chat-footer='true'] button[type='submit']",
            "button[type='submit']",
        ]
        self.response_area_selectors = [
            "[data-activeresponse='true']",
            "[id*='assistant-message']",
            "div[id*='assistant-message']",
            "[data-dark-theme] .space-y-4.whitespace-normal",
            "div[data-dark-theme]",
        ]
        self.stop_selectors = [
            "button[aria-label='Interrompi la generazione']",
            "button[aria-label*='Interrompi']",
            "button[aria-label*='Stop']",
        ]
        self.media_config = {
            "image": {
                "limits": {"unlogged": 1, "base": 1, "paid": 1},
                "supported_models": ["all"],
                "upload_selectors": [
                    "input[type='file'][name='upload'][accept*='image']",
                    "input[type='file'][name='upload']",
                    "input[type='file']",
                ],
            },
            "document": {
                "limits": {"unlogged": 1, "base": 1, "paid": 1},
                "supported_models": ["all"],
                "upload_selectors": [
                    "input[type='file'][name='upload'][accept*='pdf']",
                    "input[type='file'][name='upload']",
                    "input[type='file']",
                ],
            },
        }
        self._model_button_selector = "button[data-testid='model-select-button']"
        self._model_group_selector = "ul[role='radiogroup']"
        self._last_detected_model: str | None = None
        self._model_aliases = {
            "gpt-5": "gpt-5-mini",
            "gpt5": "gpt-5-mini",
            "gpt-5-mini": "gpt-5-mini",
            "gpt-4o": "gpt-4o-mini",
            "gpt4o": "gpt-4o-mini",
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-oss": "tinfoil/gpt-oss-120b",
            "oss": "tinfoil/gpt-oss-120b",
            "gpt-oss-120b": "tinfoil/gpt-oss-120b",
            "tinfoil/gpt-oss-120b": "tinfoil/gpt-oss-120b",
            "llama": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "llama-4": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "llama-4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "claude": "claude-haiku-4-5",
            "haiku": "claude-haiku-4-5",
            "claude-haiku": "claude-haiku-4-5",
            "claude-haiku-4-5": "claude-haiku-4-5",
            "mistral": "mistral-small-2603",
            "mistral-small": "mistral-small-2603",
            "mistral-small-2603": "mistral-small-2603",
        }

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
            logger.warning("[duckai] login check failed: %s", exc)
            return False

    def get_current_model(self) -> str:
        driver = getattr(self, "driver", None)
        if driver is not None:
            detected = self._detect_current_model(driver)
            if detected:
                self._last_detected_model = detected
                return detected
        return self._last_detected_model or super().get_current_model()

    def _prepare_for_prompt(self, driver: Any, model_name: str | None) -> None:
        requested = self._normalize_duckai_model(model_name)
        current = self._detect_current_model(driver)
        if current:
            self._last_detected_model = current
        if not requested:
            return
        if requested not in self.ENGINE_MODELS:
            raise RuntimeError(
                f"duckai_unknown_model: {requested}. Supported models: {', '.join(self.ENGINE_MODELS)}"
            )
        if current == requested:
            return
        logger.info("[duckai] Switching model from %s to %s", current or "<unknown>", requested)
        self._open_model_picker(driver)
        self._click_model_option(driver, requested)

        deadline = time.time() + 8.0
        while time.time() < deadline:
            selected = self._detect_current_model(driver)
            if selected == requested:
                self._last_detected_model = selected
                return
            time.sleep(0.2)
        raise RuntimeError(f"duckai_model_switch_timeout: failed to activate {requested}")

    def _normalize_duckai_model(self, model_name: str | None) -> str | None:
        requested = self._canonicalize_requested_model(model_name)
        if requested is None:
            return None
        lowered = requested.strip().lower()
        if lowered in self._model_aliases:
            return self._model_aliases[lowered]
        return requested

    def _detect_current_model(self, driver: Any) -> str | None:
        selectors = [
            "input[name='model'][aria-checked='true']",
            "input[name='model']:checked",
        ]
        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
            except Exception:
                elements = []
            for element in elements:
                value = (element.get_attribute("value") or "").strip()
                if value:
                    return self._normalize_duckai_model(value) or value

        try:
            buttons = driver.find_elements(By.CSS_SELECTOR, self._model_button_selector)
        except Exception:
            buttons = []
        for button in buttons:
            if not self._element_is_displayed(button):
                continue
            label = " ".join((button.text or "").split())
            mapped = self._map_button_label(label)
            if mapped:
                return mapped
        return None

    def _map_button_label(self, text: str) -> str | None:
        normalized = " ".join(text.lower().split())
        mapping = {
            "gpt-5": "gpt-5-mini",
            "gpt-4o": "gpt-4o-mini",
            "claude haiku 4.5": "claude-haiku-4-5",
            "mistral small 4": "mistral-small-2603",
        }
        if normalized in mapping:
            return mapping[normalized]
        if "haiku" in normalized:
            return "claude-haiku-4-5"
        if "mistral" in normalized:
            return "mistral-small-2603"
        if "llama" in normalized:
            return "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        if "oss" in normalized:
            return "tinfoil/gpt-oss-120b"
        if "gpt-4o" in normalized:
            return "gpt-4o-mini"
        if "gpt-5" in normalized:
            return "gpt-5-mini"
        return None

    def _open_model_picker(self, driver: Any) -> None:
        def _find_button(d: Any) -> Any:
            for element in d.find_elements(By.CSS_SELECTOR, self._model_button_selector):
                if self._element_is_displayed(element):
                    return element
            return False

        button = WebDriverWait(driver, 10).until(_find_button)
        driver.execute_script("arguments[0].click();", button)
        WebDriverWait(driver, 5).until(
            lambda d: d.find_elements(By.CSS_SELECTOR, self._model_group_selector)
        )

    def _click_model_option(self, driver: Any, model_slug: str) -> None:
        label_selector = f'label[for="{model_slug}"]'
        input_selector = f'input[name="model"][value="{model_slug}"]'
        labels = driver.find_elements(By.CSS_SELECTOR, label_selector)
        if labels:
            driver.execute_script("arguments[0].click();", labels[0])
            return
        inputs = driver.find_elements(By.CSS_SELECTOR, input_selector)
        if inputs:
            driver.execute_script("arguments[0].click();", inputs[0])
            return
        available = []
        for element in driver.find_elements(By.CSS_SELECTOR, "input[name='model']"):
            value = (element.get_attribute("value") or "").strip()
            if value:
                available.append(value)
        raise RuntimeError(
            f"duckai_model_not_found: {model_slug}. Available: {', '.join(available) or '<none>'}"
        )
