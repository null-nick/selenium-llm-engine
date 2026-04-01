import traceback
from pathlib import Path

from core.json_engine import JsonEngine


def main():
    engines_dir = Path(__file__).parent.parent / "engines"
    chatgpt_json = engines_dir / "chatgpt.json"
    engine = JsonEngine(chatgpt_json)
    print("Attempting initialization...")
    try:
        engine.is_user_logged_in()
        print("Driver started SUCCESS!")
    except Exception:
        print("Driver started FAILED!")
        traceback.print_exc()


if __name__ == "__main__":
    main()
