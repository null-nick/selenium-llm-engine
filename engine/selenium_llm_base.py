import asyncio
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import undetected_chromedriver as uc
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger("selenium_llm_base")


class SeleniumLLMBase:
    def __init__(
        self,
        service_url: str,
        model_limits_map: Dict[str, int],
        default_model: str,
        headless: bool = True,
        profile_dir: Optional[str] = None,
    ):
        self.service_url = service_url
        self.model_limits_map = model_limits_map
        self.default_model = default_model
        self.driver = None
        self.headless = headless
        self._initialized = False
        self.profile_dir = profile_dir or "/config/.config/chromium-synth"
        self._last_login_state: Optional[bool] = None

        os.makedirs(self.profile_dir, exist_ok=True)

    def get_supported_models(self) -> list[str]:
        return list(self.model_limits_map.keys())

    def get_current_model(self) -> str:
        return self.default_model

    def _get_model_limit(self, model_name: str) -> int:
        model_name = model_name.lower().strip()
        if model_name in self.model_limits_map:
            return self.model_limits_map[model_name]
        if "default" in self.model_limits_map:
            return self.model_limits_map["default"]
        return 10000

    def get_interface_limits(self) -> dict[str, Any]:
        return {
            "max_prompt_chars": self._get_model_limit(self.get_current_model()),
            "model_name": self.get_current_model(),
        }

    def _locate_chromium_binary(self) -> Optional[str]:
        possible = [
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/opt/google/chrome/chrome",
        ]
        for path in possible:
            if os.path.exists(path):
                return path
        return None

    def _build_options(self) -> Options:
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-software-rasterizer")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--user-data-dir=%s" % self.profile_dir)
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--window-size=1280,900")
        return options

    def _init_driver(self) -> Any:
        if self.driver is not None:
            return self.driver

        chromium_binary = self._locate_chromium_binary()
        if chromium_binary:
            logger.info(f"Using Chromium binary at {chromium_binary}")
        else:
            logger.warning("Chromium binary not found in static locations, uc will auto-resolve")

        options = self._build_options()
        if chromium_binary:
            options.binary_location = chromium_binary

        self.driver = uc.Chrome(options=options)
        self.driver.set_page_load_timeout(120)
        self.driver.set_script_timeout(120)
        self._initialized = True
        return self.driver

    def _ensure_ready(self) -> None:
        if not self._initialized or self.driver is None:
            self._init_driver()

    def _ensure_logged_in(self, driver) -> bool:
        # Implemented by subclasses.
        raise NotImplementedError()

    def is_user_logged_in(self) -> bool:
        try:
            self._ensure_ready()
            logged = self._ensure_logged_in(self.driver)
            self._last_login_state = logged
            return logged
        except Exception as e:
            logger.warning(f"Unable to determine login state: {e}")
            return False

    async def start_login_flow(self, timeout: int = 60) -> dict[str, Any]:
        self._ensure_ready()
        try:
            self.driver.get(self.service_url)
            await asyncio.sleep(2)
            return await self.check_login_state()
        except Exception as e:
            logger.error(f"start_login_flow error: {e}")
            return {"logged_in": False, "login_state": "unknown", "error": str(e)}

    async def check_login_state(self) -> dict[str, Any]:
        try:
            logged = self.is_user_logged_in()
            state = "logged" if logged else "unlogged"
            return {"logged_in": logged, "login_state": state}
        except Exception as e:
            logger.error(f"check_login_state error: {e}")
            return {"logged_in": False, "login_state": "unknown", "error": str(e)}

    async def generate_response(self, prompt: str) -> str:
        self._ensure_ready()
        if not self.is_user_logged_in():
            raise RuntimeError("Not logged in to the engine")

        self.driver.get(self.service_url)
        # wait for prompt area and then send
        time.sleep(1)

        # This is generic; subclasses may override with stronger logic
        try:
            input_area = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "textarea, div[contenteditable='true']"))
            )
            # send prompt via JS to avoid stale states
            self.driver.execute_script(
                "arguments[0].focus(); arguments[0].innerText = arguments[1];",
                input_area,
                prompt,
            )
            input_area.send_keys("\n")
            time.sleep(2)
            # get last assistant output
            messages = self.driver.find_elements(By.CSS_SELECTOR, "div[data-testid='assistant-response'], .assistant-message, div.markdown")
            if messages:
                text = messages[-1].text.strip()
                return text if text else ""
            return ""
        except Exception as e:
            logger.error(f"generate_response failed: {e}")
            raise

    async def stop(self) -> None:
        try:
            if self.driver is not None:
                self.driver.quit()
        except Exception as e:
            logger.warning(f"driver close error: {e}")
        finally:
            self.driver = None
            self._initialized = False
