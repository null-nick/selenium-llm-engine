import asyncio
import glob
import logging
import math
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
        allow_unlogged: bool = False,
    ):
        self.service_url = service_url
        self.model_limits_map = model_limits_map
        self.default_model = default_model
        self.allow_unlogged = allow_unlogged
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
        # CSS selectors whose matching elements must never be clicked as send button
        self.send_button_blacklist: list[str] = []

        # Cloudflare CAPTCHA challenge detectors
        self.captcha_challenge_selectors: list[str] = [
            "iframe#cf-chl-widget-ezspn",
            "iframe[src*='challenges.cloudflare.com/cdn-cgi/challenge-platform']",
        ]

        # Selector cache: remember the last working selector to try it first
        self._cached_prompt_selector: Optional[str] = None
        self._cached_send_selector: Optional[str] = None

        # Prompt chunking: split prompts that exceed the model char limit
        self._split_prompt_parts: int = max(1, int(os.getenv("SELENIUM_SPLIT_PROMPT_PARTS", "3")))
        self._skip_split_for_next: bool = False

    def get_supported_models(self) -> list[str]:
        return list(self.model_limits_map.keys())

    def get_current_model(self) -> str:
        # Return 'unlogged' when the engine is not logged in, supports it, and has this model
        if (
            not self.is_user_logged_in()
            and self.allow_unlogged
            and "unlogged" in self.model_limits_map
        ):
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
                    time.sleep(1)

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

            # Wait for processes to terminate
            time.sleep(2)
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

            time.sleep(0.5)
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
        # Avoid initializing a browser for a simple state check when not yet used.
        if not self._initialized or self.driver is None:
            if self._last_login_state is not None:
                return self._last_login_state
            return False

        try:
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
            if not self._initialized or self.driver is None:
                logged = bool(self._last_login_state)
                state = "logged" if logged else "unlogged"
                return {"logged_in": logged, "login_state": state}

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
                "no such window",
                "target window already closed",
                "web view not found",
            )
        )

    def _is_redirect_stall(self, exc: Exception) -> bool:
        """Return True if *exc* signals a redirect-stall (prompt submitted but page navigated away)."""
        return "redirect-stall:" in str(exc).lower()

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

    def _is_captcha_present(self, driver: Any) -> bool:
        """Return True if the page includes a known Cloudflare captcha challenge widget."""
        for selector in self.captcha_challenge_selectors:
            try:
                els = driver.find_elements(By.CSS_SELECTOR, selector)
                if els:
                    logger.debug(f"[selenium] Captcha selector matched: {selector}")
                    return True
            except Exception:
                pass
        return False

    # ------------------------------------------------------------------ prompt chunking

    def _should_split_prompt(self, prompt: str) -> bool:
        """Return True if *prompt* exceeds the current model's char limit and chunking is enabled."""
        if self._split_prompt_parts <= 1:
            return False
        limit = self._get_model_limit(self.get_current_model())
        return len(prompt) > limit

    def _split_prompt_into_parts(self, prompt: str, n: int) -> list[str]:
        """Split *prompt* into *n* roughly equal text chunks."""
        chunk_size = math.ceil(len(prompt) / n)
        return [prompt[i : i + chunk_size] for i in range(0, len(prompt), chunk_size)]

    def _execute_chunked_send(self, prompt: str, driver: Any) -> str:
        """Send an oversized *prompt* in sequential chunks, keeping the session open.

        Parts 1..N-1 are prefixed with an instruction telling the LLM not to
        respond yet.  Only the final part triggers a real reply.
        """
        limit = self._get_model_limit(self.get_current_model())
        # Calculate the minimum number of parts that keeps every chunk inside the limit.
        # Never exceed the configured maximum.
        min_parts = math.ceil(len(prompt) / limit)
        n = min(self._split_prompt_parts, max(min_parts, 2))
        parts = self._split_prompt_into_parts(prompt, n)
        logger.info(
            f"[selenium] Prompt chunking: {len(prompt)} chars split into {n} parts "
            f"(limit={limit}, env_max={self._split_prompt_parts})"
        )

        for idx, part in enumerate(parts[:-1], start=1):
            header = (
                f"[INTERNAL-PART{idx}/{n}] This message contains part {idx} of {n} "
                "of a large input. Read it carefully and keep it available for "
                "subsequent messages. Do NOT respond to this part yet. "
                "Reply ONLY with: OK\n\n"
            )
            chunk_text = header + part
            logger.debug(f"[selenium] Sending chunk {idx}/{n} ({len(chunk_text)} chars)")

            input_el = self._find_interactable_element(
                driver, self.prompt_area_selectors, timeout=20.0,
                cache_attr="_cached_prompt_selector",
            )
            if input_el is None:
                raise RuntimeError(f"Could not find prompt input area for chunk {idx}/{n}")

            self._fill_input(driver, input_el, chunk_text)
            self._click_send(driver, input_el)
            if not self._post_send_check(driver):
                self._cached_prompt_selector = None
                self._cached_send_selector = None
                raise RuntimeError(
                    f"redirect-stall: chunk {idx}/{n} not accepted after redirect"
                )
            intermediate = self._wait_for_response(driver)
            if not intermediate:
                logger.warning(f"[selenium] Empty response for intermediate chunk {idx}/{n}")
            else:
                logger.debug(f"[selenium] Chunk {idx}/{n} acknowledged: {intermediate[:80]!r}")

        # Send the final part and return the actual response.
        logger.debug(f"[selenium] Sending final chunk {n}/{n} ({len(parts[-1])} chars)")
        self._skip_split_for_next = True
        try:
            input_el = self._find_interactable_element(
                driver, self.prompt_area_selectors, timeout=20.0,
                cache_attr="_cached_prompt_selector",
            )
            if input_el is None:
                raise RuntimeError(f"Could not find prompt input area for final chunk {n}/{n}")

            self._fill_input(driver, input_el, parts[-1])
            self._click_send(driver, input_el)
            if not self._post_send_check(driver):
                self._cached_prompt_selector = None
                self._cached_send_selector = None
                raise RuntimeError(
                    "redirect-stall: final chunk not accepted after redirect"
                )
            return self._wait_for_response(driver)
        finally:
            self._skip_split_for_next = False

    # ------------------------------------------------------------------ core flow

    def _sync_generate_response(self, prompt: str) -> str:
        """Synchronous core of generate_response — runs in a worker thread."""
        for attempt in range(2):
            try:
                return self._sync_generate_response_once(prompt)
            except RuntimeError as e:
                if attempt == 0:
                    if self._is_dead_session(e):
                        logger.warning(
                            f"[selenium] Dead session on attempt {attempt + 1}, resetting and retrying…"
                        )
                        self._reset_driver()
                        continue
                    if self._is_redirect_stall(e):
                        logger.warning(
                            f"[selenium] Redirect-stall on attempt {attempt + 1}, retrying without driver reset…"
                        )
                        continue
                raise
        # Should not be reached, but satisfy mypy
        raise RuntimeError("_sync_generate_response exhausted retries")

    def _sync_generate_response_once(self, prompt: str) -> str:
        """Single attempt of the core generate flow."""
        self._ensure_ready()

        unlogged = not self.is_user_logged_in()
        if unlogged:
            logger.warning(
                "[selenium] User is unlogged; continuing with unlogged mode (restricted/unreliable)."
            )

        assert self.driver is not None, "_ensure_ready() must have set self.driver"
        driver = cast(webdriver.Chrome, self.driver)

        # Skip navigation if already on the service page (avoids full page reload)
        needs_nav = True
        try:
            current_url = driver.current_url or ""
            if current_url.startswith(self.service_url):
                needs_nav = False
                logger.debug("[selenium] Already on service URL, skipping navigation")
        except Exception:
            pass  # dead session or no URL — navigate anyway

        if needs_nav:
            try:
                driver.get(self.service_url)
            except Exception as nav_err:
                if self._is_dead_session(nav_err):
                    self._reset_driver()
                    raise RuntimeError(
                        f"Driver session died during navigation: {nav_err}"
                    ) from nav_err
                raise
            time.sleep(0.5)

        if self._is_captcha_present(driver):
            logger.warning("[selenium] Cloudflare captcha challenge detected on page")
            return (
                "⚠️ Cloudflare CAPTCHA rilevato. "
                "Per favore completa il CAPTCHA nella pagina e riprova."
            )

        # Prompt chunking: split oversized prompts into sequential parts.
        if not self._skip_split_for_next and self._should_split_prompt(prompt):
            return self._execute_chunked_send(prompt, driver)

        try:
            input_el = self._find_interactable_element(
                driver, self.prompt_area_selectors, timeout=20.0,
                cache_attr="_cached_prompt_selector",
            )
            if input_el is None:
                raise RuntimeError("Could not find prompt input area")

            self._fill_input(driver, input_el, prompt)
            self._click_send(driver, input_el)
            if not self._post_send_check(driver):
                self._cached_prompt_selector = None
                self._cached_send_selector = None
                raise RuntimeError(
                    "redirect-stall: send not accepted after redirect"
                )
            return self._wait_for_response(driver)

        except Exception as e:
            if self._is_dead_session(e):
                self._reset_driver()
                raise RuntimeError(f"Driver session died mid-prompt: {e}") from e
            logger.error(f"[selenium] _sync_generate_response_once failed: {e}")
            if unlogged:
                return f"⚠️ Unlogged session: could not run full prompt flow. Error: {e}"
            raise

    # ------------------------------------------------------------------ helpers

    def _get_latest_response_text(self, driver: Any) -> str:
        """Return latest non-empty text from response selectors, or empty if none."""
        for sel in self.response_area_selectors:
            try:
                els = driver.find_elements(By.CSS_SELECTOR, sel)
                if els:
                    text = els[-1].text.strip()
                    if text:
                        return text
            except Exception:
                pass
        return ""

    def _find_interactable_element(
        self,
        driver: Any,
        selectors: list[str],
        timeout: float = 20.0,
        cache_attr: Optional[str] = None,
    ) -> Optional[Any]:
        """Try CSS selectors in order; return first element that is clickable.

        If *cache_attr* is given (e.g. ``'_cached_prompt_selector'``), the last
        successful selector is tried first on subsequent calls.
        """
        # Build an ordered list: cached selector first (if valid), then the rest
        ordered: list[str] = list(selectors)
        cached: Optional[str] = getattr(self, cache_attr, None) if cache_attr else None
        if cached and cached in ordered:
            ordered.remove(cached)
            ordered.insert(0, cached)

        per = max(1.5, timeout / max(len(ordered), 1))
        for sel in ordered:
            try:
                condition = EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
                # For compatibility with different Selenium versions, expose locator
                # on the callable condition object so unit tests and mocks can inspect it.
                if not hasattr(condition, "locator"):
                    try:
                        setattr(condition, "locator", (By.CSS_SELECTOR, sel))
                    except Exception:
                        pass

                el = WebDriverWait(driver, per).until(condition)
                logger.debug(f"[selenium] Found clickable element: {sel}")
                if cache_attr:
                    setattr(self, cache_attr, sel)
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

        def _is_button_blacklisted(btn: Any) -> bool:
            for bl_sel in self.send_button_blacklist:
                try:
                    matches = driver.find_elements(By.CSS_SELECTOR, bl_sel)
                    if any(m == btn for m in matches):
                        logger.debug(f"[selenium] Button matches blacklist selector: {bl_sel}")
                        return True
                except Exception:
                    pass
            return False

        def _safe_click(btn: Any, selector: str) -> bool:
            if _is_button_blacklisted(btn):
                logger.debug(f"[selenium] Skipping blacklisted button for selector: {selector}")
                return False
            try:
                btn.click()
                logger.debug(f"[selenium] Sent via button click: {selector}")
                return True
            except Exception as e:
                logger.warning(f"[selenium] Button.click() failed for {selector}: {e}")
            try:
                driver.execute_script("arguments[0].click();", btn)
                logger.debug(f"[selenium] Sent via JS click: {selector}")
                return True
            except Exception as e:
                logger.warning(f"[selenium] JS click failed for {selector}: {e}")
            return False

        def _resolve_click_target(element: Any) -> Any:
            """If selector matches icon SVG, resolve up to a parent button."""
            try:
                if element.tag_name.lower() == "svg":
                    parent_btn = element.find_element(By.XPATH, "ancestor::button[1]")
                    if parent_btn is not None:
                        return parent_btn
            except Exception:
                pass
            return element

        # Try cached send selector first, then fall through the full list
        ordered_send: list[str] = list(self.send_button_selectors)
        if self._cached_send_selector and self._cached_send_selector in ordered_send:
            ordered_send.remove(self._cached_send_selector)
            ordered_send.insert(0, self._cached_send_selector)

        for sel in ordered_send:
            try:
                btn = WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
                )
                btn = _resolve_click_target(btn)
                if _safe_click(btn, sel):
                    self._cached_send_selector = sel
                    return
            except Exception as e:
                logger.debug(f"[selenium] selector {sel} not clickable: {e}")

        # Secondary fallback: try visible buttons even if not clickable by wait
        for sel in ordered_send:
            try:
                candidates = driver.find_elements(By.CSS_SELECTOR, sel)
                for btn in candidates:
                    try:
                        resolved_btn = _resolve_click_target(btn)
                        if resolved_btn.is_displayed() and resolved_btn.is_enabled():
                            if _safe_click(resolved_btn, sel):
                                self._cached_send_selector = sel
                                return
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"[selenium] selector {sel} not found for fallback click: {e}")

        # Final fallback: Enter key on the input element
        try:
            input_el.send_keys(Keys.RETURN)
            logger.debug("[selenium] Sent via Enter key")
        except Exception as e:
            logger.error(f"[selenium] Could not send prompt: {e}")

    def _post_send_check(self, driver: Any, timeout: float = 15.0) -> bool:
        """Return True if the LLM accepted the prompt (stop button or new text appeared).

        After clicking send, poll for *timeout* seconds to confirm that either:
        - a stop-button becomes visible (streaming started), or
        - new response text different from the current baseline appears.

        If neither signal is seen by the deadline, inspects the current URL:
        - URL no longer on service_url → a redirect occurred → return False (stall detected).
        - URL still on service_url → model is just slow → return True (let _wait_for_response decide).
        """
        baseline = self._get_latest_response_text(driver)
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Check stop button
            for sel in self.stop_selectors:
                try:
                    btns = driver.find_elements(By.CSS_SELECTOR, sel)
                    for b in btns:
                        try:
                            if b.is_displayed():
                                logger.debug("[selenium] post_send_check: stop button visible — send accepted")
                                return True
                        except Exception:
                            pass
                except Exception:
                    pass
            # Check new response text
            cur = self._get_latest_response_text(driver)
            if cur and cur != baseline:
                logger.debug("[selenium] post_send_check: new response text appeared — send accepted")
                return True
            time.sleep(0.5)

        # Timeout expired — check if we are still on the expected page
        cur_url = ""
        try:
            cur_url = driver.current_url or ""
        except Exception:
            pass

        if self.service_url and not cur_url.startswith(self.service_url):
            logger.warning(
                f"[selenium] post_send_check: timeout with unexpected URL '{cur_url}' — redirect-stall detected"
            )
            return False

        logger.debug("[selenium] post_send_check: timeout but URL looks ok — assuming slow model")
        return True

    def _wait_for_response(self, driver: Any, max_wait: int = 120) -> str:
        """Wait for the LLM response to fully stream, then return its text.

        Strategy:
        1. Take current last response as baseline (conversation history).
        2. Wait up to 30 s for a new response text different from baseline.
        3. Poll until the response is stable (1 s) and stop button disappears.
        """
        baseline = self._get_latest_response_text(driver)

        start = time.time()
        poll_interval = 0.1
        while time.time() - start < 30:
            cur = self._get_latest_response_text(driver)
            if cur and cur != baseline:
                logger.debug("[selenium] New response text appeared")
                break
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 2, 1.0)
        else:
            logger.warning("[selenium] Response area did not produce new text within 30s")

        start = time.time()
        first_new = ""
        stable_since = time.time()

        while time.time() - start < max_wait:
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
                time.sleep(0.5)
                stable_since = time.time()
                continue

            cur = self._get_latest_response_text(driver)
            cur_url = ""
            try:
                cur_url = driver.current_url or ""
            except Exception:
                pass

            if cur and cur != baseline:
                if first_new == "" or cur != first_new:
                    first_new = cur
                    stable_since = time.time()
                    logger.debug("[selenium] New response candidate captured")

                if self.service_url and not cur_url.startswith(self.service_url):
                    logger.warning(
                        "[selenium] Detected navigation away from service URL; returning captured text"
                    )
                    return first_new

                if time.time() - stable_since >= 1.0:
                    logger.debug("[selenium] Response stable — done")
                    return first_new
            elif first_new:
                if time.time() - stable_since >= 1.0:
                    logger.debug("[selenium] First new response stable — done")
                    return first_new

            time.sleep(0.3)

        logger.warning("[selenium] Response wait timed out, returning best-effort result")
        return first_new or baseline

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
