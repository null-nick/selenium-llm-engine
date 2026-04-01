import undetected_chromedriver as uc
from selenium.webdriver.chrome.service import Service
import traceback

try:
    uc.Chrome(service=Service("/usr/local/bin/chromedriver"), headless=True)
except Exception:
    traceback.print_exc()
