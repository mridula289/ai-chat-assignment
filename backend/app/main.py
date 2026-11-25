import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import litellm
from litellm import Cache
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////data/app.db")
MODEL = os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash")
CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"
SIMULATION_ENABLED = os.getenv("ENABLE_FAILURE_SIMULATION", "false").lower() == "true"
logger = logging.getLogger(__name__)


def db_path() -> str:
    if not DATABASE_URL.startswith("sqlite:///"):
        raise RuntimeError("DATABASE_URL must be a SQLite URL")
    path = DATABASE_URL[len("sqlite:///"):]
    if path.startswith("//"):
        path = "/" + path.lstrip("/")
    return path


DB_PATH = db_path()


def connect():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_db():
    with connect() as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        db.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)")
        db.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                system_prompt TEXT,
                created_at TEXT NOT NULL
            )"""
        )


if CACHE_ENABLED:
    litellm.cache = Cache(type="local")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


keys = [key.strip() for key in os.getenv("LLM_API_KEYS", "").split(",") if key.strip()]
key_ids = [f"key_{i}" for i in range(1, len(keys) + 1)]


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    system_prompt: Optional[str] = None

    @field_validator("session_id", mode="before")
    @classmethod
    def session_not_blank(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("session_id must not be blank")
        return value.strip()

    @field_validator("message", mode="before")
    @classmethod
    def message_trimmed(cls, value):
        if not isinstance(value, str):
            raise ValueError("message must be a string")
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        if len(value) > 4000:
            raise ValueError("message must be at most 4000 characters")
        return value

    @field_validator("system_prompt")
    @classmethod
    def system_trimmed(cls, value):
        return value.strip() if value and value.strip() else None


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
origins = [item.strip() for item in os.getenv("ALLOWED_ORIGINS", "*").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled request error")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.on_event("startup")
def startup():
    initialize_db()


@app.get("/health")
def health():
    if len(keys) < 2:
        raise HTTPException(status_code=503, detail="at least two LLM API keys must be configured")
    with connect() as db:
        db.execute("SELECT 1").fetchone()
    return {"status": "ok", "keys_configured": len(keys), "cache_enabled": CACHE_ENABLED}

def parse_failures(header):
    if not SIMULATION_ENABLED or not header:
        return {}
    result = {}
    for item in header.split(","):
        item = item.strip()
        if "=" not in item:
            continue
        key, status = item.split("=", 1)
        if status.strip() in {"429", "500", "502", "503"}:
            result[key.strip()] = int(status.strip())
    return result


def should_simulate(key_id, failures):
    return failures.get("*") or failures.get(key_id)


def status_from_exception(exc):
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if value:
            return int(value)
    response = getattr(exc, "response", None)
    if response is not None:
        return getattr(response, "status_code", 0)
    text = str(exc).lower()
    if "429" in text or "rate" in text or "quota" in text:
        return 429
    if any(code in text for code in ("500", "502", "503")):
        for code in ("500", "502", "503"):
            if code in text:
                return int(code)
    return 0


def save_conversation(session_id, system_prompt, user_message, reply, metadata):
    with connect() as db:
        db.execute(
            "INSERT INTO sessions(session_id, system_prompt, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET system_prompt=excluded.system_prompt",
            (session_id, system_prompt, utc_now()),
        )
        db.execute(
            "INSERT INTO messages(session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, "user", user_message, None, utc_now()),
        )
        db.execute(
            "INSERT INTO messages(session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, "assistant", reply, json.dumps(metadata), utc_now()),
        )


@app.post("/chat")
def chat(body: ChatRequest, x_simulate_failure: Optional[str] = Header(default=None)):
    with connect() as db:
        session = db.execute("SELECT system_prompt FROM sessions WHERE session_id=?", (body.session_id,)).fetchone()
    system_prompt = body.system_prompt or (
        session["system_prompt"] if session and session["system_prompt"] else "You are a helpful assistant."
    )

    with connect() as db:
        rows = db.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
            (body.session_id,),
        ).fetchall()
    history = [{"role": row["role"], "content": row["content"]} for row in rows]
    messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": body.message}]

    started = time.perf_counter()
    failures = parse_failures(x_simulate_failure)
    fallback_attempts = 0
    for index, key_id in enumerate(key_ids):
        simulated = should_simulate(key_id, failures)
        if simulated:
            fallback_attempts += 1
            continue
        try:
            response = litellm.completion(
                model=MODEL,
                messages=messages,
                api_key=keys[index],
                caching=CACHE_ENABLED,
            )
            reply = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            output_tokens = getattr(usage, "completion_tokens", None) if usage else None
            if not isinstance(reply, str) or not reply.strip():
                raise ValueError("provider returned an empty reply")
            if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
                raise ValueError("provider returned no token usage")
            cache_hit = bool((getattr(response, "_hidden_params", {}) or {}).get("cache_hit"))
            break
        except Exception as exc:
            status = status_from_exception(exc)
            logger.warning("LLM provider attempt failed: key_id=%s status=%s", key_id, status)
            fallback_attempts += 1
            if status not in {0, 401, 403, 429, 500, 502, 503}:
                raise HTTPException(status_code=502, detail="provider request failed") from exc
    else:
        raise HTTPException(status_code=503, detail="all providers unavailable")

    cache_hit = bool((getattr(response, "_hidden_params", {}) or {}).get("cache_hit"))
    latency = int((time.perf_counter() - started) * 1000)
    metadata = {
        "resolution_source": "LLM_CACHE" if cache_hit else "LIVE_INFERENCE",
        "model": MODEL,
        "latency_ms": latency,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "key_id": None if cache_hit else key_id,
        "fallback_attempts": 0 if cache_hit else fallback_attempts,
    }
    reply = reply.strip()
    save_conversation(body.session_id, system_prompt, body.message, reply, metadata)
    return {"session_id": body.session_id, "reply": reply, "metadata": metadata}


@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str):
    with connect() as db:
        rows = db.execute(
            "SELECT role, content, metadata FROM messages WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="session not found")
    messages = []
    for row in rows:
        item = {"role": row["role"], "content": row["content"]}
        if row["role"] == "assistant" and row["metadata"]:
            full = json.loads(row["metadata"])
            item["metadata"] = {
                "resolution_source": full["resolution_source"],
                "input_tokens": full["input_tokens"],
                "output_tokens": full["output_tokens"],
            }
        messages.append(item)
    return {"session_id": session_id, "messages": messages}
