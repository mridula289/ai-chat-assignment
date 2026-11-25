# Full-stack AI Chat

## 1. Live URLs

- Frontend: Not deployed yet
- Backend: Not deployed yet

## 2. Run locally: with Docker and without

First copy `.env.example` to `.env` and set `LLM_API_KEYS` to two real API keys for the configured LiteLLM model, separated by a comma. Do not commit `.env`.

Docker (requires Docker Compose):

```bash
cp .env.example .env
make up
```

Open `http://localhost:3000`. The backend health endpoint is `http://localhost:8000/health`.

Without Docker:

PowerShell:

```powershell
Copy-Item .env.example .env
# Set the following values in .env before starting the services.
cd backend
python -m pip install -r requirements-dev.txt
$env:DATABASE_URL = "sqlite:///./app.db"
$env:LLM_API_KEYS = "your-provider-key-1,your-provider-key-2"
$env:ALLOWED_ORIGINS = "http://localhost:3000"
$env:ENABLE_FAILURE_SIMULATION = "true"
$env:LLM_CACHE_ENABLED = "true"
uvicorn app.main:app --reload --port 8000

# In a second terminal:
cd frontend
npm ci
$env:VITE_API_URL = "http://localhost:8000"
npm run dev -- --host 0.0.0.0 --port 3000
```

POSIX shell:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt
export DATABASE_URL=sqlite:////tmp/app.db
export LLM_API_KEYS=your-provider-key-1,your-provider-key-2
export ALLOWED_ORIGINS=http://localhost:3000
export ENABLE_FAILURE_SIMULATION=true
export LLM_CACHE_ENABLED=true
uvicorn app.main:app --app-dir backend --reload --port 8000

cd frontend
npm ci
VITE_API_URL=http://localhost:8000 npm run dev -- --host 0.0.0.0 --port 3000
```

## 3. Design decisions: key fallback, caching, SQLite session storage

Each request calls LiteLLM with one key at a time in the configured order. Rate limits, quota errors, authentication errors, and provider 5xx responses move to the next key. The failure-simulation header is enabled by `ENABLE_FAILURE_SIMULATION` and only affects the current request.

LiteLLM's built-in in-memory cache is enabled with the complete model and message list. Cache hits reuse the cached reply and the provider's original usage; they report `key_id: null` and zero fallback attempts. Real provider usage is returned per request; the app does not invent a reply or estimate token counts when no provider is available.

SQLite stores session prompts and append-only user/assistant history. `/sessions/{session_id}/messages` reads oldest-first and excludes the system prompt. The named Docker volume persists `/data/app.db`.

For a frontend hosted separately, set its `VITE_API_URL` to the backend's public base URL and set the backend's `ALLOWED_ORIGINS` to the frontend's exact public origin.

## 4. Verification: how to see failover, cache hits and per-message token usage

Health:

```bash
curl http://localhost:8000/health
```

Chat:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"What is the capital of France?"}'
```

Repeat with another session ID and the same message to observe `LLM_CACHE`. Simulate fallback:

```bash
curl -i -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{"session_id":"failover","message":"Hello"}'
```

Simulate all providers unavailable:

```bash
curl -i -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{"session_id":"no-provider","message":"Hello"}'
```

Run backend tests:

```bash
cd backend
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## 5. Setup Used: your editor and AI tools

VS Code, Docker Compose, FastAPI, React/Vite, LiteLLM, SQLite, and Copilot SDK.

## 6. Bonus features (if any): what you added beyond the requirements and why

- System-prompt override field for each session.
- Responsive chat layout with distinct live/error states.
