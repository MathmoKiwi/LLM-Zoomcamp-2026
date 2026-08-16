"""Central config. Everything comes from environment variables so the same
code runs locally, in docker-compose, and in CI without edits."""

import os

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "technotes")

# Embedding models (fastembed, CPU-friendly)
DENSE_MODEL = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
DENSE_DIM = int(os.getenv("DENSE_DIM", "384"))
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

# LLM. Any OpenAI-compatible endpoint works: OpenAI, Groq, or a local
# Ollama/TabbyAPI instance. Set OPENAI_BASE_URL accordingly.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)

# Postgres (feedback + monitoring)
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "technote")
PG_USER = os.getenv("PG_USER", "technote")
PG_PASSWORD = os.getenv("PG_PASSWORD", "technote")

# Data paths
DATA_DIR = os.getenv("DATA_DIR", "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
ANSWERABLE_PATH = os.path.join(DATA_DIR, "eval_answerable.jsonl")
IMPOSSIBLE_PATH = os.path.join(DATA_DIR, "eval_impossible.jsonl")

# The exact token the model must emit when the context does not contain
# the answer. eval_llm.py greps for this to measure abstention.
NOT_FOUND = "NOT_FOUND"
