import undetected_chromedriver as uc
from selenium.webdriver.chrome.service import Service
import traceback

try:
    options = uc.ChromeOptions()
    options.binary_location = "/usr/bin/chromium"  # dummy
    # Just to trace what error it hits! We pass fake chromedriver path
    uc.Chrome(
        options=options, service=Service("/usr/local/bin/chromedriver"), headless=True
    )
except Exception:
    traceback.print_exc()
