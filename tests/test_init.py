import traceback
from engine.selenium_chatgpt import SeleniumChatGPT


def main():
    engine = SeleniumChatGPT()
    print("Attempting initialization...")
    try:
        engine.is_user_logged_in()
        print("Driver started SUCCESS!")
    except Exception:
        print("Driver started FAILED!")
        traceback.print_exc()


if __name__ == "__main__":
    main()
