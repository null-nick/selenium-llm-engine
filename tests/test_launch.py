from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import undetected_chromedriver as uc
import traceback

print("Testing vanilla webdriver.Chrome...")
try:
    options = webdriver.ChromeOptions()
    options.binary_location = "/usr/bin/chromium"
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service("/usr/bin/chromedriver"), options=options)
    print("Vanilla webdriver SUCCESS!")
    driver.quit()
except Exception:
    print("Vanilla webdriver FAILED:")
    traceback.print_exc()

print("\n------------------------------\nTesting uc.Chrome...")
try:
    uc_options = uc.ChromeOptions()
    uc_options.binary_location = "/usr/bin/chromium"
    uc_options.add_argument("--no-sandbox")
    uc_options.add_argument("--disable-dev-shm-usage")
    driver2 = uc.Chrome(
        options=uc_options,
        driver_executable_path="/usr/bin/chromedriver",
        browser_executable_path="/usr/bin/chromium",
    )
    print("uc.Chrome SUCCESS!")
    driver2.quit()
except Exception:
    print("uc.Chrome FAILED:")
    traceback.print_exc()
