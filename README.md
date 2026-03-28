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

3. Login flow (before `/prompt`):

- `POST http://localhost:8000/login/chatgpt`
- `POST http://localhost:8000/login/gemini`

## Notes

- Requires Chromium + chromedriver and undetected-chromedriver.
- Container binds persistent profile path: `/config/.config/chromium-synth`.
- If using Python locally, a compatible FastAPI/Pydantic stack is required.

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
