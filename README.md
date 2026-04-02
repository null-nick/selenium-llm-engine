# Selenium LLM Engine


> Disclaimer: this software has been vibe-coded.

![Docker Pulls](https://img.shields.io/docker/pulls/xargonwan/selenium-llm-engine)
| Branch    | Build Status                                                                                                                                         | Docs Status                                                                                                                                      |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `main`    | [![CI Status](https://img.shields.io/github/actions/workflow/status/xargonwan/selenium-llm-engine/build-release.yml)](https://github.com/xargonwan/selenium-llm-engine/actions)

This repository provides a standalone, Docker-friendly Selenium-powered LLM engine proxy. It automates browser access to Web GPTs via Selenium to expose a simple OpenAI-compatible API and web admin UI.

Feel free to submit pull requests with improvements, new engines, or engine definitions (`engines/*.json` or `engines/*.py`).

</br>
> [!NOTE]
> The application listens on port `8000` by default. Docker Compose maps this to `14848` in this repository, but you can change this mapping freely in `docker-compose.yml`.


## Features

- Selenium-based support web based LLM.
- Unified REST API endpoints:
  - `/api/ping`
  - `/api/engines`
  - `/api/engines/reload`
  - `/models` and `/models/{engine}`
  - `/login/{engine}` and `/login/{engine}/state`
  - `/engine/{engine}/prompt`
  - `/v1/models`
  - `/v1/models/{model_id:path}`
  - `/v1/chat/completions` (OpenAI-like compatibility)
  - `/chat/completions` (OpenAI-like legacy compatibility)
  - `/stats` (aggregated counters + response time averages)
  - `/api/logs/app` (incremental app log polling)
  - `/api/engines/selector-hints` (runtime selector hints)
  - `/reset` and `/api/reset` (clears engine state and stats counters)
  - `/logs` and `/api/history`
  - `/ui`
- SQLite storage for prompt logs and counters
- Web UI for simple login, prompt sending and metrics
- Docker + docker-compose support

## Quickstart

1. Build and run with Docker compose:

```bash
docker compose up --build
```

2. API access:

- `http://localhost:14848/api/ping`
- `http://localhost:14848/models`

> [!NOTE]
> model discovery is dynamic. Although examples use `chatgpt`, `gemini`, `stepfun`, `calude`, the available engines are those present in `engines/` and reported by `/models`.

- `http://localhost:14848/chatgpt/prompt`
- `http://localhost:14848/gemini/prompt`
- `http://localhost:14848/stepfun/prompt`
- `http://localhost:14848/calude/prompt`
- `http://localhost:14848/v1/chat/completions` (OpenAI-compatible)
- `http://localhost:14848/ui`

2.1 OpenAPI / OpenAI client

This application exposes an OpenAI-compatible endpoint for clients and SDKs:

- Endpoint: `POST http://localhost:14848/v1/chat/completions`
- `model`: `chatgpt` or `gemini`
- `messages`: standard OpenAI array

Example using `openai` (Python):

```python
from openai import OpenAI

client = OpenAI(api_key="dummy")  # the proxy does not require a real key, values can be dummy
client.api_base = "http://localhost:14848"
client.api_type = "openai"
client.api_version = ""

resp = client.chat.completions.create(
    model="chatgpt",
    messages=[{"role": "user", "content": "Write a short poem in English about a robot learning to sing."}]
)
print(resp)
```

If the client does not directly support base URL configuration, use a manual request with `requests`:

```python
import requests

url = "http://localhost:14848/v1/chat/completions"

payload = {
    "model": "chatgpt",
    "messages": [{"role": "user", "content": "Scrivi una breve poesia in italiano."}]
}

r = requests.post(url, json=payload)
print(r.json())
```

3. Prompt example (API call)

Before sending a prompt, make sure you have logged in using `/login/chatgpt` or `/login/gemini`.

```bash
curl -X POST "http://localhost:14848/chatgpt/prompt" \
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

- `POST http://localhost:14848/login/chatgpt`
- `POST http://localhost:14848/login/gemini`

## Notes

- Requires Chromium + chromedriver and undetected-chromedriver.
- Container binds persistent profile path: `/config/.config/chromium-synth`.
- If using Python locally, a compatible FastAPI/Pydantic stack is required, but this approach is not tested nor supported.

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
- `core/` - Selenium engine wrappers and manager
- `db/` - SQLite persistence helpers
- `web/` - minimal static UI
- `tests/` - API tests
- `Dockerfile`, `docker-compose.yml` - container setup
