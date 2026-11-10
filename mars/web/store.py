"""Хранилище веб-сервера на SQLite: пользователи, сессии, задачи."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path

SESSION_TTL = 7 * 24 * 3600
SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, salt BLOB NOT NULL, pw_hash BLOB NOT NULL,
  created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY, username TEXT NOT NULL, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, username TEXT NOT NULL, question TEXT NOT NULL,
  status TEXT NOT NULL, run_id TEXT, created REAL NOT NULL, events TEXT NOT NULL DEFAULT '[]');
CREATE INDEX IF NOT EXISTS jobs_user ON jobs(username, created);
"""


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, **SCRYPT)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(SCHEMA)

    def _run(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock, self._db:
            return self._db.execute(sql, args).fetchall()

    # ---- пользователи
    def has_users(self) -> bool:
        return bool(self._run("SELECT 1 FROM users LIMIT 1"))

    def list_users(self) -> list[str]:
        return [r["username"] for r in self._run("SELECT username FROM users ORDER BY username")]

    def set_user(self, username: str, password: str) -> None:
        if not username or not username.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise ValueError("Имя пользователя: буквы, цифры, точка, дефис, подчёркивание")
        if len(password) < 8:
            raise ValueError("Пароль должен быть не короче 8 символов")
        salt = secrets.token_bytes(16)
        self._run("INSERT INTO users(username, salt, pw_hash, created) VALUES(?,?,?,?) "
                  "ON CONFLICT(username) DO UPDATE SET salt=excluded.salt, pw_hash=excluded.pw_hash",
                  (username, salt, _hash_password(password, salt), time.time()))
        self._run("DELETE FROM sessions WHERE username=?", (username,))  # смена пароля разлогинивает

    def remove_user(self, username: str) -> bool:
        existed = bool(self._run("SELECT 1 FROM users WHERE username=?", (username,)))
        self._run("DELETE FROM users WHERE username=?", (username,))
        self._run("DELETE FROM sessions WHERE username=?", (username,))
        return existed

    def check_password(self, username: str, password: str) -> bool:
        rows = self._run("SELECT salt, pw_hash FROM users WHERE username=?", (username,))
        if not rows:
            _hash_password(password, b"0" * 16)  # одинаковое время ответа для несуществующего имени
            return False
        return hmac.compare_digest(_hash_password(password, rows[0]["salt"]), rows[0]["pw_hash"])

    # ---- сессии
    def create_session(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        self._run("DELETE FROM sessions WHERE expires < ?", (time.time(),))
        self._run("INSERT INTO sessions VALUES(?,?,?)", (_token_hash(token), username, time.time() + SESSION_TTL))
        return token

    def session_user(self, token: str | None) -> str | None:
        if not token:
            return None
        rows = self._run("SELECT username FROM sessions WHERE token_hash=? AND expires > ?",
                         (_token_hash(token), time.time()))
        return rows[0]["username"] if rows else None

    def drop_session(self, token: str | None) -> None:
        if token:
            self._run("DELETE FROM sessions WHERE token_hash=?", (_token_hash(token),))

    # ---- задачи
    def add_job(self, job_id: str, username: str, question: str) -> None:
        self._run("INSERT INTO jobs(id, username, question, status, created) VALUES(?,?,?,?,?)",
                  (job_id, username, question, "queued", time.time()))

    def get_job(self, job_id: str) -> dict | None:
        rows = self._run("SELECT * FROM jobs WHERE id=?", (job_id,))
        return {**dict(rows[0]), "events": json.loads(rows[0]["events"])} if rows else None

    def add_event(self, job_id: str, kind: str, text: str) -> None:
        with self._lock, self._db:
            row = self._db.execute("SELECT events FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row:
                events = json.loads(row["events"]) + [{"type": kind, "text": text}]
                self._db.execute("UPDATE jobs SET events=? WHERE id=?", (json.dumps(events, ensure_ascii=False),
                                                                        job_id))

    def set_status(self, job_id: str, status: str, run_id: str | None = None) -> None:
        self._run("UPDATE jobs SET status=?, run_id=COALESCE(?, run_id) WHERE id=?", (status, run_id, job_id))

    def active_count(self, username: str) -> int:
        return self._run("SELECT COUNT(*) c FROM jobs WHERE username=? AND status IN ('queued','running')",
                         (username,))[0]["c"]

    def queued_ahead(self, job_id: str) -> int:
        return self._run("SELECT COUNT(*) c FROM jobs WHERE status='queued' AND created < "
                         "(SELECT created FROM jobs WHERE id=?)", (job_id,))[0]["c"]

    def history(self, username: str, limit: int = 30) -> list[dict]:
        rows = self._run("SELECT id, question, status, run_id, created FROM jobs WHERE username=? "
                         "ORDER BY created DESC LIMIT ?", (username, limit))
        return [dict(r) for r in rows]

    def owns_run(self, username: str, run_id: str) -> bool:
        return bool(self._run("SELECT 1 FROM jobs WHERE username=? AND run_id=?", (username, run_id)))

    def fail_interrupted(self) -> None:
        """После перезапуска сервера незавершённые задачи уже не выполняются — отмечаем это честно."""
        for r in self._run("SELECT id FROM jobs WHERE status IN ('queued','running')"):
            self.add_event(r["id"], "error", "Сервер был перезапущен, задача прервана. Запустите её заново.")
            self.set_status(r["id"], "error")
