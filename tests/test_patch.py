import undetected_chromedriver as uc
from selenium.webdriver.chrome.webdriver import WebDriver as ChromeWebDriver

_orig_init = ChromeWebDriver.__init__


def _patched_init(self, *args, **kwargs):
    if "executable_path" in kwargs:
        from selenium.webdriver.chrome.service import Service

        executable_path = kwargs.pop("executable_path")
        if "service" not in kwargs:
            kwargs["service"] = Service(executable_path)
    _orig_init(self, *args, **kwargs)


ChromeWebDriver.__init__ = _patched_init

try:
    options = uc.ChromeOptions()
    options.binary_location = "/usr/bin/chromium"
    # This will simulate an old undetected_chromedriver trying to pass executable_path
    ChromeWebDriver(executable_path="/usr/local/bin/chromedriver", options=options)
except Exception:
    import traceback

    traceback.print_exc()
