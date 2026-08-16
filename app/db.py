"""Postgres persistence for monitoring.

Two tables: every Q&A round trip lands in `conversations`, and thumbs
up/down from the UI lands in `feedback`. Grafana queries these directly;
the panel SQL lives in monitoring/queries.md.
"""

import psycopg2
from psycopg2.extras import Json

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id              SERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    abstained       BOOLEAN NOT NULL DEFAULT FALSE,
    mode            TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    model           TEXT NOT NULL,
    sources         JSONB,
    response_ms     INTEGER,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER
);

CREATE TABLE IF NOT EXISTS feedback (
    id              SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id),
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    value           INTEGER NOT NULL CHECK (value IN (-1, 1))
);
"""


def connect():
    return psycopg2.connect(
        host=config.PG_HOST,
        port=config.PG_PORT,
        dbname=config.PG_DB,
        user=config.PG_USER,
        password=config.PG_PASSWORD,
    )


def init_db():
    with connect() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)


def log_conversation(result):
    """Persist the dict returned by rag.answer(). Returns the row id so
    the UI can attach feedback to it later."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO conversations
                (question, answer, abstained, mode, prompt_version, model,
                 sources, response_ms, prompt_tokens, completion_tokens)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                result["question"],
                result["answer"],
                result["abstained"],
                result["mode"],
                result["prompt_version"],
                result["model"],
                Json(result["sources"]),
                result["response_ms"],
                result["prompt_tokens"],
                result["completion_tokens"],
            ),
        )
        return cur.fetchone()[0]


def log_feedback(conversation_id, value):
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO feedback (conversation_id, value) VALUES (%s, %s)",
            (conversation_id, value),
        )
