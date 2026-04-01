import undetected_chromedriver as uc
import traceback

try:
    options = uc.ChromeOptions()
    options.binary_location = "/usr/bin/chromium"  # dummy
    uc.Chrome(
        options=options,
        driver_executable_path="/usr/local/bin/chromedriver",
        headless=True,
    )
    print("Success?")
except Exception:
    traceback.print_exc()
