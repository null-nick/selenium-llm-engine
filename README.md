# Selenium LLM Engine (Standalone)

This repository provides a standalone, Docker-friendly Selenium-powered LLM engine proxy. It automates browser access to Web GPTs via Selenium to expose a simple OpenAI-compatible API and web admin UI.

## Features

- Selenium-based support for:
  - ChatGPT (OpenAI ChatGPT frontend via browser automation)
  - Gemini (Google Gemini frontend via browser automation)
- Unified REST API endpoints:
  - `/api/ping`
  - `/models` and `/models/{engine}`
  - `/login/{engine}` and `/login/{engine}/state`
  - `/chatgpt/prompt`, `/gemini/prompt`
  - `/v1/chat/completions` (OpenAI-like compatibility)
  - `/stats`, `/logs`, `/ui`
- SQLite storage for prompt logs and counters
- Web UI for simple login, prompt sending and metrics
- Docker + docker-compose support

## Quickstart

1. Build and run with Docker compose:

```bash
docker compose up --build
```

2. API access:

- `http://localhost:8000/api/ping`
- `http://localhost:8000/models`
- `http://localhost:8000/chatgpt/prompt`
- `http://localhost:8000/gemini/prompt`
- `http://localhost:8000/ui`

3. Prompt example (API call)

Before sending a prompt, make sure you have logged in using `/login/chatgpt` or `/login/gemini`.

```bash
curl -X POST "http://localhost:8000/chatgpt/prompt" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a short poem in English about a robot learning to sing."}'
```

Example response:

```json
{
  "id": "llm_1680390000",
  "object": "chat.completion",
  "created": 1680390000,
  "model": "chatgpt",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "engine": "chatgpt",
  "prompt": "Write a short poem...",
  "elapsed_ms": 850
}
```

4. Login flow (before `/prompt`):

- `POST http://localhost:8000/login/chatgpt`
- `POST http://localhost:8000/login/gemini`

## Notes

- Requires Chromium + chromedriver and undetected-chromedriver.
- Container binds persistent profile path: `/config/.config/chromium-synth`.
- If using Python locally, a compatible FastAPI/Pydantic stack is required.

## Legal / Terms of Service (ToS) Notice

- This software is intended for research and testing purposes only.
- Before using this tool with any online service, verify and comply with that service's Terms of Service (ToS) and usage policies.
- If the target service does not allow automated access or use via browser automation, do not use this software against it.
- Always respect copyright and service contract requirements.
## Development

```bash
python -m pip install -r requirements.txt
pytest -q
```

## Directory structure

- `app.py` - FastAPI entrypoint
- `engine/` - Selenium engine wrappers and manager
- `db/` - SQLite persistence helpers
- `web/` - minimal static UI
- `tests/` - API tests
- `Dockerfile`, `docker-compose.yml` - container setup
