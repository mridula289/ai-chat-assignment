# Full-stack AI Chat

## 1. Live URLs

- Frontend: `https://YOUR-FRONTEND.example`
- Backend: `https://YOUR-BACKEND.example`

## 2. Run locally: with Docker and without

Docker:

```bash
cp .env.example .env
make up
```

Open `http://localhost:3000`. The backend health endpoint is `http://localhost:8000/health`.

Without Docker:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt
set DATABASE_URL=sqlite:////tmp/app.db
set LLM_API_KEYS=key-1,key-2
uvicorn app.main:app --app-dir backend --reload --port 8000

cd frontend
npm install
npm run dev -- --port 3000
```

## 3. Design decisions: key fallback, caching, SQLite session storage

LiteLLM receives one ordered key pool and cache configuration. Requests try key_1, key_2, and subsequent keys until a 429, quota, or 5xx response is no longer blocking. The test hook overrides individual key outcomes only for the active request. LiteLLM local disk cache uses the complete system prompt, history, and new message as request context, and cached responses reuse their stored token counts.

SQLite stores session prompts and append-only user/assistant history. `/sessions/{session_id}/messages` reads oldest-first and excludes the system prompt. The named Docker volume persists `/data/app.db`.

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
pytest -q
```

## 5. Setup Used: your editor and AI tools

VS Code, Docker Compose, FastAPI, React/Vite, LiteLLM, SQLite, and Copilot SDK.

## 6. Bonus features (if any): what you added beyond the requirements and why

- System-prompt override field for each session.
- Responsive chat layout with distinct live/error states.
