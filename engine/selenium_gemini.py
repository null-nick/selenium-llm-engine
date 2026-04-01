import logging
from selenium.webdriver.common.by import By
from engine.selenium_llm_base import SeleniumLLMBase

logger = logging.getLogger("selenium_gemini")

SERVICE_URL = "https://gemini.google.com"
MODEL_LIMITS_MAP = {
    "2.5-flash": 32000,
    "2.0-flash": 32000,
    "1.5-flash": 100000,
    "1.5-pro": 500000,
    "unlogged": 21500,
    "default": 32000,
}


class SeleniumGemini(SeleniumLLMBase):
    def __init__(self, **kwargs):
        super().__init__(
            service_url=SERVICE_URL,
            model_limits_map=MODEL_LIMITS_MAP,
            default_model="2.5-flash",
            **kwargs,
        )

        # Gemini chat input (contenteditable rich-textarea or plain div).
        self.prompt_area_selectors = [
            "div[contenteditable='true'][data-placeholder]",
            "div[contenteditable='true'][aria-label*='Ask']",
            "div[contenteditable='true'][aria-label*='Message']",
            "div[contenteditable='true'][aria-label*='Enter']",
            "div.ql-editor[contenteditable='true']",
            "textarea[placeholder*='Ask']",
            "textarea[placeholder*='Message']",
            "div[contenteditable='true']",
            "textarea",
        ]
        self.send_button_selectors = [
            "button[aria-label='Send message']",
            "button[aria-label*='Send message']",
            "button[data-testid='send-button']",
            "button[aria-label*='Send']",
            "button[type='submit']",
        ]
        # Gemini response area — custom elements and class-based selectors.
        self.response_area_selectors = [
            "model-response .markdown",
            "model-response",
            ".model-response",
            ".response-container",
            "message-content",
            ".message-content",
            "div[class*='response-text']",
        ]
        self.stop_selectors = [
            "button[aria-label='Stop response']",
            "button[aria-label='Cancel']",
            "button[aria-label*='Stop']",
            ".stop-button",
            "[data-testid='stop-button']",
        ]

    def _ensure_logged_in(self, driver) -> bool:
        try:
            if not driver.current_url.startswith("https://gemini.google.com"):
                driver.get(SERVICE_URL)

            current_url = driver.current_url.lower()
            if "signin" in current_url or "login" in current_url:
                return False

            login_buttons = driver.find_elements(
                By.XPATH,
                "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in') or contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in') or contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'login')]",
            )
            if login_buttons:
                return False

            # Check for Gemini converse panel or assistant output
            assistant = driver.find_elements(
                By.CSS_SELECTOR,
                "div.assistant-message, .gemini-response, .chat-message.ai",
            )
            if assistant:
                return True

            return True

        except Exception as e:
            logger.warning(f"[selenium_gemini] login check failed: {e}")
            return False
