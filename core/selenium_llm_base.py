import asyncio
import glob
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional, cast

import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger("selenium_llm_base")


class SeleniumLLMBase:
    def __init__(
        self,
        service_url: str,
        model_limits_map: Dict[str, int],
        default_model: str,
        headless: Optional[bool] = None,
        profile_dir: Optional[str] = None,
    ):
        self.service_url = service_url
        self.model_limits_map = model_limits_map
        self.default_model = default_model
        self.driver = None

        if headless is None:
            env_headless = os.getenv("CHROMIUM_HEADLESS", "0")
            try:
                self.headless = bool(int(env_headless))
            except Exception:
                self.headless = False
        else:
            self.headless = headless
        self._initialized = False
        self.profile_dir = profile_dir or os.getenv(
            "CHROMIUM_PROFILE_DIR", "/config/.config/chromium-synth"
        )
        self._last_login_state: Optional[bool] = None

        os.makedirs(self.profile_dir, exist_ok=True)

        # Selector lists used by _sync_generate_response — override in subclasses.
        self.prompt_area_selectors: list[str] = [
            "textarea",
            "div[contenteditable='true']",
        ]
        self.send_button_selectors: list[str] = [
            "button[type='submit']",
            "button[aria-label*='Send']",
        ]
        self.response_area_selectors: list[str] = [
            ".assistant-message",
            "div.markdown",
        ]
        self.stop_selectors: list[str] = [
            "button[aria-label*='Stop']",
            "[data-testid='stop-button']",
        ]

    def get_supported_models(self) -> list[str]:
        return list(self.model_limits_map.keys())

    def get_current_model(self) -> str:
        # Return 'unlogged' when the engine is not logged in and has this model
        if not self.is_user_logged_in() and "unlogged" in self.model_limits_map:
            return "unlogged"
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
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
        ]
        for path in possible:
            if os.path.exists(path):
                logger.info(f"[selenium] Found Chromium at: {path}")
                return path
        logger.warning("[selenium] Chromium binary not found in common locations")
        return None

    def _locate_chromedriver_binary(self) -> Optional[str]:
        candidates = [
            "/usr/bin/chromedriver",
            "/usr/bin/chromium-driver",
            "/usr/local/bin/chromedriver",
            "/usr/local/bin/chromium-driver",
            "/opt/chromedriver/chromedriver",
            shutil.which("chromedriver") or "",
            shutil.which("chromium-driver") or "",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                logger.info(f"[selenium] Found ChromeDriver binary: {path}")
                return path
        try:
            from webdriver_manager.chrome import ChromeDriverManager

            logger.warning(
                "[selenium] ChromeDriver not found, attempting webdriver-manager install"
            )
            path = ChromeDriverManager().install()
            logger.info(
                f"[selenium] webdriver-manager installed ChromeDriver at {path}"
            )
            return path
        except Exception as e:
            logger.warning(
                f"[selenium] webdriver-manager ChromeDriver install failed: {e}"
            )
        logger.warning("[selenium] ChromeDriver binary not found")
        return None

    def _get_chromium_major_version(
        self, chromium_binary: Optional[str] = None
    ) -> Optional[int]:
        binary = chromium_binary or self._locate_chromium_binary()
        if not binary:
            return None
        try:
            result = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # output: "Chromium 130.0.6723.58 ..." or "Google Chrome 130.0.6723.58 ..."
            version_str = result.stdout.strip()
            for part in version_str.split():
                if "." in part:
                    try:
                        major = int(part.split(".")[0])
                        if major > 50:  # sanity check
                            logger.info(f"[selenium] Chromium major version: {major}")
                            return major
                    except ValueError:
                        continue
        except Exception as e:
            logger.warning(f"[selenium] Could not get Chromium version: {e}")
        return None

    def _build_options(self) -> Options:
        """Build Chrome options matching SyntH's working configuration.

        Uses standard selenium Options (not uc.ChromeOptions) — this is what
        the working Synthetic Heart implementation uses.
        """
        options = Options()

        essential_args = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-extensions",
            "--disable-plugins",
            "--disable-images",
            "--disable-javascript",
            "--disable-web-security",
            "--allow-running-insecure-content",
            "--disable-features=VizDisplayCompositor",
            "--user-data-dir=%s" % self.profile_dir,
            "--profile-directory=Default",
            "--remote-debugging-port=0",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
        ]
        if self.headless:
            essential_args.append("--headless=new")

        for arg in essential_args:
            options.add_argument(arg)

        options.add_argument("--window-size=1280,900")
        return options

    def _init_driver(self) -> Any:
        """Initialize Chrome driver using the same approach as SyntH's _create_shared_driver."""
        if self.driver is not None:
            return self.driver

        logger.info("[selenium] Initializing Chrome driver...")
        self._cleanup_chromium_remnants()

        chromium_binary = self._locate_chromium_binary() or "/usr/bin/chromium"
        chromedriver_path = (
            self._locate_chromedriver_binary() or "/usr/bin/chromedriver"
        )

        # Get Chromium major version for uc compatibility
        chromium_major = self._get_chromium_major_version(chromium_binary)

        # Clear undetected-chromedriver cache to avoid stale patched binaries
        uc_cache_dir = os.path.join(tempfile.gettempdir(), "undetected_chromedriver")
        if os.path.exists(uc_cache_dir):
            shutil.rmtree(uc_cache_dir, ignore_errors=True)
            logger.info("[selenium] Cleared undetected-chromedriver cache")

        max_retries = 3
        self.driver = None
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            options = self._build_options()
            options.binary_location = chromium_binary
            try:
                logger.info(
                    f"[selenium] Driver initialization attempt {attempt + 1}/{max_retries}"
                )
                # Match SyntH's working _create_shared_driver() pattern exactly:
                # - Use standard Options() (set in _build_options)
                # - Pass service=Service(executable_path=...) for chromedriver
                # - Pass version_main for uc version compatibility
                uc_kwargs: dict[str, Any] = {
                    "options": options,
                    "service": Service(executable_path=chromedriver_path),
                }
                if chromium_major is not None:
                    uc_kwargs["version_main"] = chromium_major
                self.driver = uc.Chrome(**uc_kwargs)

                # Clean up extra windows (SyntH pattern)
                if len(self.driver.window_handles) > 1:
                    logger.info(
                        f"[selenium] Driver created with {len(self.driver.window_handles)} windows, cleaning up..."
                    )
                    for handle in self.driver.window_handles[1:]:
                        try:
                            self.driver.switch_to.window(handle)
                            self.driver.close()
                        except Exception:
                            pass
                    self.driver.switch_to.window(self.driver.window_handles[0])

                logger.info(
                    f"[selenium] Driver created with {len(self.driver.window_handles)} window(s)"
                )
                break
            except Exception as err:
                last_err = err
                logger.warning(
                    f"[selenium] Attempt {attempt + 1}/{max_retries} failed: {err}"
                )
                self._cleanup_chromium_remnants()
                if attempt < max_retries - 1:
                    time.sleep(3)

        if self.driver is None:
            # Fallback: standard webdriver (no anti-detection patching)
            logger.warning(
                "[selenium] uc.Chrome failed after all retries, trying webdriver.Chrome fallback"
            )
            try:
                fallback_options = self._build_options()
                fallback_options.binary_location = chromium_binary
                self.driver = webdriver.Chrome(
                    service=Service(executable_path=chromedriver_path),
                    options=fallback_options,
                )
                logger.info("[selenium] webdriver.Chrome fallback succeeded")
            except Exception as fallback_err:
                logger.error(
                    f"[selenium] webdriver.Chrome fallback also failed: {fallback_err}"
                )
                raise RuntimeError(
                    f"Driver initialization failed (uc: {last_err!r}, fallback: {fallback_err!r})"
                ) from fallback_err

        self.driver.set_page_load_timeout(120)
        self.driver.set_script_timeout(120)
        self._initialized = True
        logger.info("[selenium] Driver initialized successfully")
        return self.driver

    def _cleanup_chromium_remnants(self) -> None:
        """Aggressively clean up Chromium processes and lock files (SyntH pattern)."""
        try:
            logger.info("[selenium] Cleaning up Chromium remnants...")

            # Kill processes aggressively with -9 (SyntH pattern)
            for pattern in [
                "chromium",
                "chrome",
                "chromedriver",
                "undetected_chromedriver",
            ]:
                try:
                    subprocess.run(
                        ["pkill", "-9", "-f", pattern],
                        check=False,
                        capture_output=True,
                        timeout=5,
                    )
                except Exception:
                    pass

            # Wait for processes to terminate (SyntH waits 5s)
            time.sleep(5)
            logger.info("[selenium] Chromium processes killed")

            # Clean up temp dir lock files
            temp_dir = tempfile.gettempdir()
            lock_patterns = [
                os.path.join(temp_dir, ".org.chromium.Chromium.*"),
                os.path.join(temp_dir, "selenium_*_profile", "SingletonLock"),
                os.path.join(temp_dir, "selenium_*_profile", "SingletonCookie"),
                os.path.join(
                    temp_dir, "selenium_*_profile", ".org.chromium.Chromium.*"
                ),
            ]
            for pattern in lock_patterns:
                for lock_file in glob.glob(pattern):
                    try:
                        os.remove(lock_file)
                        logger.info(f"[selenium] Removed lock file: {lock_file}")
                    except Exception:
                        pass

            # Clean up profile directory lock files
            if os.path.exists(self.profile_dir):
                for lock_pat in [
                    "SingletonLock",
                    "SingletonCookie",
                    ".org.chromium.Chromium.*",
                ]:
                    for lock_file in glob.glob(
                        os.path.join(self.profile_dir, lock_pat)
                    ):
                        try:
                            os.remove(lock_file)
                            logger.info(
                                f"[selenium] Removed profile lock file: {lock_file}"
                            )
                        except Exception:
                            pass

            time.sleep(2)
            logger.info("[selenium] Chromium cleanup completed")
        except Exception as e:
            logger.warning(f"[selenium] Error during Chromium cleanup: {e}")

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
        """Open the service URL in the browser (non-blocking) and return login state."""

        def _sync_start_login() -> dict[str, Any]:
            self._ensure_ready()
            assert self.driver is not None, (
                "_ensure_ready() must have initialized driver"
            )
            drv = cast(webdriver.Chrome, self.driver)
            drv.get(self.service_url)
            time.sleep(2)
            logged = self.is_user_logged_in()
            state = "logged" if logged else "unlogged"
            return {"logged_in": logged, "login_state": state}

        try:
            return await asyncio.to_thread(_sync_start_login)
        except Exception as e:
            logger.error(f"start_login_flow error: {e}")
            return {"logged_in": False, "login_state": "unknown", "error": str(e)}

    async def check_login_state(self) -> dict[str, Any]:
        """Return current login state without navigating."""
        try:
            logged = await asyncio.to_thread(self.is_user_logged_in)
            state = "logged" if logged else "unlogged"
            return {"logged_in": logged, "login_state": state}
        except Exception as e:
            logger.error(f"check_login_state error: {e}")
            return {"logged_in": False, "login_state": "unknown", "error": str(e)}

    async def generate_response(self, prompt: str) -> str:
        """Send prompt to the LLM service and return the response text.

        All Selenium calls are executed in a thread pool via asyncio.to_thread
        so that blocking I/O never stalls the FastAPI event loop.
        """
        return await asyncio.to_thread(self._sync_generate_response, prompt)

    # ------------------------------------------------------------------ session health

    def _is_dead_session(self, exc: Exception) -> bool:
        """Return True if *exc* signals a crashed/dead chromedriver session."""
        msg = str(exc).lower()
        return any(
            marker in msg
            for marker in (
                "connection refused",
                "errno 111",
                "errno 113",
                "failed to establish a new connection",
                "max retries exceeded",
                "no such session",
                "invalid session id",
            )
        )

    def _reset_driver(self) -> None:
        """Kill the existing (dead) driver and reset state so _ensure_ready re-inits."""
        logger.warning("[selenium] Resetting dead driver session…")
        if self.driver is not None:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
        self._initialized = False
        self._cleanup_chromium_remnants()

    # ------------------------------------------------------------------ core flow

    def _sync_generate_response(self, prompt: str) -> str:
        """Synchronous core of generate_response — runs in a worker thread."""
        self._ensure_ready()

        unlogged = not self.is_user_logged_in()
        if unlogged:
            logger.warning(
                "[selenium] User is unlogged; continuing with unlogged mode (restricted/unreliable)."
            )

        assert self.driver is not None, "_ensure_ready() must have set self.driver"
        driver = cast(webdriver.Chrome, self.driver)

        try:
            driver.get(self.service_url)
        except Exception as nav_err:
            if self._is_dead_session(nav_err):
                self._reset_driver()
                raise RuntimeError(
                    f"Driver session died during navigation: {nav_err}"
                ) from nav_err
            raise
        time.sleep(2)

        try:
            input_el = self._find_interactable_element(
                driver, self.prompt_area_selectors, timeout=20.0
            )
            if input_el is None:
                raise RuntimeError("Could not find prompt input area")

            self._fill_input(driver, input_el, prompt)
            self._click_send(driver, input_el)
            return self._wait_for_response(driver)

        except Exception as e:
            if self._is_dead_session(e):
                self._reset_driver()
                raise RuntimeError(f"Driver session died mid-prompt: {e}") from e
            logger.error(f"[selenium] _sync_generate_response failed: {e}")
            if unlogged:
                return f"⚠️ Unlogged session: could not run full prompt flow. Error: {e}"
            raise

    # ------------------------------------------------------------------ helpers

    def _find_interactable_element(
        self,
        driver: Any,
        selectors: list[str],
        timeout: float = 20.0,
    ) -> Optional[Any]:
        """Try CSS selectors in order; return first element that is clickable."""
        per = max(1.5, timeout / max(len(selectors), 1))
        for sel in selectors:
            try:
                el = WebDriverWait(driver, per).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
                )
                logger.debug(f"[selenium] Found clickable element: {sel}")
                return el
            except Exception as e:
                if self._is_dead_session(e):
                    raise  # propagate to _sync_generate_response for driver reset
        logger.warning("[selenium] No interactable element found")
        return None

    def _fill_input(self, driver: Any, element: Any, text: str) -> None:
        """Type *text* into a textarea or contenteditable element."""
        try:
            driver.execute_script(
                "arguments[0].scrollIntoView({block:'center'});", element
            )
            time.sleep(0.2)
        except Exception:
            pass

        try:
            element.click()
        except Exception:
            driver.execute_script("arguments[0].click();", element)
        time.sleep(0.1)

        tag = (element.tag_name or "").lower()
        if tag in ("textarea", "input"):
            # Standard form inputs: clear via clear() then type
            try:
                element.clear()
            except Exception:
                try:
                    element.send_keys(Keys.CONTROL + "a")
                    element.send_keys(Keys.DELETE)
                except Exception:
                    pass
            element.send_keys(text)
        else:
            # contenteditable (ProseMirror, Quill, …): select-all then type
            try:
                element.send_keys(Keys.CONTROL + "a")
                time.sleep(0.05)
                element.send_keys(text)
            except Exception:
                # JS fallback: execCommand fires proper DOM events
                try:
                    driver.execute_script(
                        "arguments[0].focus();"
                        "document.execCommand('selectAll', false, null);"
                        "document.execCommand('insertText', false, arguments[1]);",
                        element,
                        text,
                    )
                except Exception as e:
                    logger.error(f"[selenium] fill_input JS fallback failed: {e}")
                    raise
        logger.debug(f"[selenium] Filled input ({len(text)} chars)")

    def _click_send(self, driver: Any, input_el: Any) -> None:
        """Click the send button, or fall back to the Enter key."""
        for sel in self.send_button_selectors:
            try:
                btn = WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
                )
                try:
                    btn.click()
                except Exception:
                    driver.execute_script("arguments[0].click();", btn)
                logger.debug(f"[selenium] Sent via button: {sel}")
                return
            except Exception:
                pass

        # Fallback: Enter key on the input element
        try:
            input_el.send_keys(Keys.RETURN)
            logger.debug("[selenium] Sent via Enter key")
        except Exception as e:
            logger.error(f"[selenium] Could not send prompt: {e}")

    def _wait_for_response(self, driver: Any, max_wait: int = 120) -> str:
        """Wait for the LLM response to fully stream, then return its text.

        Strategy:
        1. Wait up to 30 s for any response element to appear.
        2. Poll until the stop button is gone AND the text has been stable
           for at least 3 s (no change between polls).
        """
        # Phase 1: wait for a response element to appear
        for sel in self.response_area_selectors:
            try:
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, sel))
                )
                logger.debug(f"[selenium] Response area appeared: {sel}")
                break
            except Exception:
                pass

        # Phase 2: wait for streaming to finish (text stability + no stop button)
        start = time.time()
        prev = ""
        stable_since = time.time()

        while time.time() - start < max_wait:
            # --- stop-button check ---
            generating = False
            for sel in self.stop_selectors:
                try:
                    btns = driver.find_elements(By.CSS_SELECTOR, sel)
                    for b in btns:
                        try:
                            if b.is_displayed():
                                generating = True
                                break
                        except Exception:
                            pass
                    if generating:
                        break
                except Exception as e:
                    if self._is_dead_session(e):
                        logger.error("[selenium] Driver died during stop-button check")
                        raise

            if generating:
                time.sleep(1)
                stable_since = time.time()
                continue

            # --- grab latest response text ---
            cur = ""
            for sel in self.response_area_selectors:
                try:
                    els = driver.find_elements(By.CSS_SELECTOR, sel)
                    if els:
                        cur = els[-1].text.strip()
                        if cur:
                            break
                except Exception as e:
                    if self._is_dead_session(e):
                        logger.error("[selenium] Driver died while reading response")
                        raise

            if cur:
                if cur == prev:
                    if time.time() - stable_since >= 3.0:
                        logger.debug("[selenium] Response stable — done")
                        return cur
                else:
                    stable_since = time.time()
                    prev = cur

            time.sleep(0.5)

        logger.warning("[selenium] Response wait timed out, returning partial result")
        return prev

    # ------------------------------------------------------------------ /helpers

    async def stop(self) -> None:
        """Quit the driver and clean up."""

        def _sync_stop() -> None:
            if self.driver is not None:
                try:
                    self.driver.quit()
                except Exception as e:
                    logger.warning(f"[selenium] driver.quit() error: {e}")
            self._cleanup_chromium_remnants()

        try:
            await asyncio.wait_for(asyncio.to_thread(_sync_stop), timeout=10)
        except TimeoutError:
            logger.warning("[selenium] stop() timed out — force-cleaning remnants")
            self._cleanup_chromium_remnants()
        except Exception as e:
            logger.warning(f"[selenium] stop() error: {e}")
        finally:
            self.driver = None
            self._initialized = False
