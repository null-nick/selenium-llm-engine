## Plan: Dockerized Selenium LLM Engine API

TL;DR: estrarre l'engine Selenium (ChatGPT/Gemini) da Synthetic_Heart in un servizio standalone in `selenium-llm-engine`, esponendo Web API OpenAPI + UI basica con login e log/prompt metriche. Include driver Chrome undetected + webtop opzionale come richiesto.

Steps
1. Analisi componenti esistenti (DONE): in `Synthetic_Heart/cortex/selenium_engine`, `selenium_llm_base.py`, `selenium_chatgpt.py`, `selenium_gemini.py`.
2. Definisci struttura del nuovo progetto in `selenium-llm-engine`:
   - `app.py` (FastAPI) con endpoint REST
   - `required/` per `Dockerfile` e `docker-compose.yml`
   - `login/` (Web UI static + endpoint metrics)
   - `db/` per SQLite (prompt logs, contatori, limits)
   - `engine/` copia minimalizzata del Selenium base + plugin
   - `settings.py` config environment
   - `tests/` unit/integration
3. Implementa engine wrapper:
   - `engine/selenium_llm_base.py` (porting core methods, driver management, login state, generate_response)
   - `engine/selenium_chatgpt.py` e `engine/selenium_gemini.py` con `MODEL_LIMITS_MAP`, selectors, `_ensure_logged_in`, `get_current_model`, etc.
   - endpoint inizializzazione `engine_context` per switch engine + model.
   - `engine_manager.py` con factory, singleton, thread-safety
4. Implementa APIs (FastAPI / endpoints):
   a. `POST /v1/chat/completions` (Ollama/OpenAI compatibility) + `POST /chatgpt/prompt`, `/gemini/prompt` (plugin-specific)
   b. `GET /models` + `GET /models/{name}` expose limits per engine
   c. `POST /login/{engine}` per innescare `start_login_flow` e `check_login_state`
   d. `GET /login/{engine}/state` status
   e. `GET /stats` metriche aggregate + logs
   f. `GET /logs` cronologia prompt/response (paginata)
5. Logging e metriche in DB:
   - `prompt_logs` (id, engine, model, request, response, status, elapsed, timestamp)
   - `stats` (requests, responses, errors, last_login)
   - aggiornamento quando /prompt e /login invocate.
6. Implementa UI frontend semplce in `web/`:
   - lista dei modelli + limiti + login status + tasto login per ognuno
   - form prompt e risposta
   - metrica contatori/percentuale
   - tab for limits e login status
7. Docker e permessi:
   - `Dockerfile` con Chromium e undetected_chromedriver, `node`, `python` (o solo Python) per UI + API.
   - `docker-compose.yml` con volume `/config/.config/chromium-synth` per state persistente.
   - documenta variabili (CHROMIUM_HEADLESS, SERVICE_URL, instancem).
8. Test:
   - `tests/test_engine_manager.py` (simulazione engine base non headless, login detection fallback)
   - `tests/test_api_endpoints.py` (FastAPI TestClient prompt count, status)
   - `tests/test_limits.py` (verifica MODEL_LIMITS_MAP in mapping + endpoint /models)
   - `uv run pytest` mandatory.
9. Validazione e CI:
   - `uv run ruff format .`;
   - `uv run ruff check --fix .`;
   - `uv run ty check path/to/edited`; 
   - `uv run pytest`.

Verification
1. Avvia container e verifica a) /api/ping b) /models espone chatgpt/gemini limits c) /chatgpt/prompt produce testo (o error se non loggato)
2. Verifica UI login avvia browser in container e mostra status; verifica counters aggiornano.
3. Esegui test automatici con `pytest`.

Decisions
- Persistenza: SQLite (utente ha richiesto)
- Endpoint: engine-specific (grazie risposta user)
- Auth: no auth (locale)
- Infrastruttura: include webtop + undetected chromedriver (richiesto), lo facciamo via Chromium profile persistente.

Further Considerations
1. Servizio dovrebbe includere un call /health per readiness.
2. Utenti potrebbero volere cronologia piu lunga, obvia paginazione.
3. Fallimenti di login con captcha necessitano messaggio di troubleshooting.
