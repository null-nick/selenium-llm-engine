import logging
from selenium.webdriver.common.by import By
from engine.selenium_llm_base import SeleniumLLMBase

logger = logging.getLogger("selenium_chatgpt")

SERVICE_URL = "https://chat.openai.com"
MODEL_LIMITS_MAP = {
    "gpt-4o": 60000,
    "gpt-4o-mini": 60000,
    "gpt-4-turbo": 50000,
    "gpt-4": 40000,
    "gpt-3.5-turbo": 30000,
    "unlogged": 20000,
    "default": 51000,
}


class SeleniumChatGPT(SeleniumLLMBase):
    def __init__(self, **kwargs):
        super().__init__(
            service_url=SERVICE_URL,
            model_limits_map=MODEL_LIMITS_MAP,
            default_model="gpt-4o",
            **kwargs,
        )

    def _ensure_logged_in(self, driver) -> bool:
        try:
            if not driver.current_url.startswith("https://chat.openai.com"):
                driver.get(SERVICE_URL)

            current_url = driver.current_url.lower()
            if (
                "login" in current_url
                or "auth" in current_url
                or "signin" in current_url
            ):
                return False

            # Check possible login buttons by XPath (case-insensitive)
            login_buttons = driver.find_elements(
                By.XPATH,
                "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'log in') or contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in')]",
            )
            if login_buttons:
                return False

            # Check chat area presence (authenticated state)
            chat_area = driver.find_elements(
                By.CSS_SELECTOR,
                "div[data-testid='conversation-panel'], div[data-testid='chat-history'], div[data-testid='disabled-service']",
            )
            if chat_area:
                return True

            # fallback: if no login button found then assume logged in
            return True

        except Exception as e:
            logger.warning(f"[selenium_chatgpt] login check failed: {e}")
            return False
