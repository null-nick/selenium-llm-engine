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
ENGINE_REASONING_MODES = ["fast", "reasoning", "extended-reasoning"]


class DuckAIEngine(SeleniumLLMBase):
    ENGINE_NAME = ENGINE_NAME
    ENGINE_ALIASES = ENGINE_ALIASES
    ENGINE_DISPLAY_NAME = ENGINE_DISPLAY_NAME
    ENGINE_SERVICE_URL = ENGINE_SERVICE_URL
    ENGINE_MODELS = ENGINE_MODELS
    ENGINE_DEFAULT_MODEL = ENGINE_DEFAULT_MODEL
    ENGINE_ALLOW_UNLOGGED = ENGINE_ALLOW_UNLOGGED
    ENGINE_REASONING_MODES = ENGINE_REASONING_MODES

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
        self._model_confirm_selectors = [
            "button[type='submit']",
        ]
        self._reasoning_button_selector = [
            "button[aria-label='Reasoning mode']",
        ]
        self._last_detected_model: str | None = None
        self._last_detected_reasoning_mode: str | None = None
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
        self._reasoning_aliases = {
            "fast": "fast",
            "veloce": "fast",
            "reasoning": "reasoning",
            "ragionamento": "reasoning",
            "thinking": "reasoning",
            "extended": "extended-reasoning",
            "extended-reasoning": "extended-reasoning",
            "ragionamento-esteso": "extended-reasoning",
            "ragionamento esteso": "extended-reasoning",
        }
        self._reasoning_labels = {
            "fast": ["Veloce", "Fast"],
            "reasoning": ["Ragionamento", "Reasoning", "Thinking"],
            "extended-reasoning": ["Ragionamento esteso", "Extended reasoning", "Deep reasoning"],
        }
        self._reasoning_supported_models = {
            "gpt-5-mini",
            "gpt-4o-mini",
            "claude-haiku-4-5",
            "tinfoil/gpt-oss-120b",
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

    def _prepare_for_prompt(
        self,
        driver: Any,
        model_name: str | None,
        reasoning_mode: str | None = None,
    ) -> None:
        requested_model = self._normalize_duckai_model(model_name)
        current_model = self._detect_current_model(driver)
        if current_model:
            self._last_detected_model = current_model

        requested_reasoning = self._normalize_reasoning_mode(reasoning_mode)
        current_reasoning = self._detect_reasoning_mode(driver)
        if current_reasoning:
            self._last_detected_reasoning_mode = current_reasoning

        if requested_model:
            if requested_model not in self.ENGINE_MODELS:
                raise RuntimeError(
                    f"duckai_unknown_model: {requested_model}. Supported models: {', '.join(self.ENGINE_MODELS)}"
                )
            if current_model != requested_model:
                logger.info("[duckai] Switching model from %s to %s", current_model or "<unknown>", requested_model)
                driver.get(f"{ENGINE_SERVICE_URL}/new")
                time.sleep(0.8)
                self._open_model_picker(driver)
                self._click_model_option(driver, requested_model)
                self._confirm_model_selection(driver, requested_model)
                deadline = time.time() + 8.0
                while time.time() < deadline:
                    selected = self._detect_current_model(driver)
                    if selected == requested_model:
                        self._last_detected_model = selected
                        break
                    time.sleep(0.2)
                else:
                    raise RuntimeError(f"duckai_model_switch_timeout: failed to activate {requested_model}")

        active_model_for_reasoning = self._last_detected_model or current_model or requested_model
        if requested_reasoning and active_model_for_reasoning in self._reasoning_supported_models:
            if current_reasoning != requested_reasoning:
                logger.info(
                    "[duckai] Switching reasoning mode from %s to %s",
                    current_reasoning or "<unknown>",
                    requested_reasoning,
                )
                self._open_reasoning_picker(driver)
                self._click_reasoning_option(driver, requested_reasoning)
                deadline = time.time() + 8.0
                while time.time() < deadline:
                    selected = self._detect_reasoning_mode(driver)
                    if selected == requested_reasoning:
                        self._last_detected_reasoning_mode = selected
                        break
                    time.sleep(0.2)
                else:
                    raise RuntimeError(
                        f"duckai_reasoning_switch_timeout: failed to activate {requested_reasoning}"
                    )
        elif requested_reasoning:
            logger.info(
                "[duckai] Skipping reasoning mode %s because model %s does not expose reasoning controls",
                requested_reasoning,
                active_model_for_reasoning or "<unknown>",
            )

    def _normalize_duckai_model(self, model_name: str | None) -> str | None:
        requested = self._canonicalize_requested_model(model_name)
        if requested is None:
            return None
        lowered = requested.strip().lower()
        if lowered in self._model_aliases:
            return self._model_aliases[lowered]
        return requested

    def _normalize_reasoning_mode(self, reasoning_mode: str | None) -> str | None:
        if reasoning_mode is None:
            return None
        lowered = str(reasoning_mode).strip().lower()
        if not lowered:
            return None
        return self._reasoning_aliases.get(lowered, lowered)

    def _detect_reasoning_mode(self, driver: Any) -> str | None:
        buttons = []
        for selector in self._reasoning_button_selector:
            try:
                buttons.extend(driver.find_elements(By.CSS_SELECTOR, selector))
            except Exception:
                pass
        for button in buttons:
            if not self._element_is_displayed(button):
                continue
            label = " ".join((button.text or "").split())
            mapped = self._map_reasoning_label(label)
            if mapped:
                return mapped
        return self._last_detected_reasoning_mode

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

    def _map_reasoning_label(self, text: str) -> str | None:
        normalized = " ".join(text.lower().split())
        for mode, labels in self._reasoning_labels.items():
            for label in labels:
                if normalized == " ".join(label.lower().split()):
                    return mode
        if "esteso" in normalized or "extended" in normalized or "deep" in normalized:
            return "extended-reasoning"
        if "ragion" in normalized or "think" in normalized:
            return "reasoning"
        if "veloc" in normalized or "fast" in normalized:
            return "fast"
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

    def _open_reasoning_picker(self, driver: Any) -> None:
        def _find_button(d: Any) -> Any:
            for selector in self._reasoning_button_selector:
                for element in d.find_elements(By.CSS_SELECTOR, selector):
                    if self._element_is_displayed(element):
                        return element
            return False

        button = WebDriverWait(driver, 10).until(_find_button)
        driver.execute_script("arguments[0].click();", button)
        time.sleep(0.5)

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

    def _click_reasoning_option(self, driver: Any, reasoning_mode: str) -> None:
        labels = self._reasoning_labels.get(reasoning_mode, [])
        xpaths = []
        for label in labels:
            xpaths.extend([
                f"//button[normalize-space(.)='{label}']",
                f"//label[normalize-space(.)='{label}']",
                f"//*[self::button or self::label or self::div or self::span][normalize-space(.)='{label}']",
                f"//button[contains(normalize-space(.), '{label}')]",
                f"//label[contains(normalize-space(.), '{label}')]",
            ])
        for xpath in xpaths:
            try:
                elements = driver.find_elements(By.XPATH, xpath)
            except Exception:
                elements = []
            for element in elements:
                if not self._element_is_displayed(element):
                    continue
                driver.execute_script("arguments[0].click();", element)
                return
        raise RuntimeError(
            f"duckai_reasoning_not_found: {reasoning_mode}. Labels tried: {', '.join(labels) or '<none>'}"
        )

    def _confirm_model_selection(self, driver: Any, model_slug: str) -> None:
        confirm_xpaths = [
            "//div[@role='dialog']//button[normalize-space(.)='Start New Chat']",
            "//div[@role='dialog']//button[normalize-space(.)='Start a new chat']",
            "//div[@role='dialog']//button[normalize-space(.)='Avvia una nuova chat']",
            "//div[@role='dialog']//button[contains(normalize-space(.), 'Start New Chat')]",
            "//div[@role='dialog']//button[contains(normalize-space(.), 'new chat')]",
            "//div[@role='dialog']//button[contains(normalize-space(.), 'nuova chat')]",
        ]
        for xpath in confirm_xpaths:
            try:
                elements = driver.find_elements(By.XPATH, xpath)
            except Exception:
                elements = []
            for element in elements:
                if not self._element_is_displayed(element):
                    continue
                driver.execute_script("arguments[0].click();", element)
                return

        for selector in self._model_confirm_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
            except Exception:
                elements = []
            for element in elements:
                if not self._element_is_displayed(element):
                    continue
                text_value = " ".join((element.text or "").split()).lower()
                if any(
                    token in text_value
                    for token in [
                        "start new chat",
                        "start a new chat",
                        "new chat",
                        "nuova chat",
                        "start",
                        "avvia",
                    ]
                ):
                    driver.execute_script("arguments[0].click();", element)
                    return

        raise RuntimeError(
            f"duckai_model_confirm_not_found: {model_slug}. Model dialog confirm button was not visible."
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
