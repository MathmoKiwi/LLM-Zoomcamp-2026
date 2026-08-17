"""Retrieval and answer generation.

Four retrieval modes, all served from one Qdrant collection:
  keyword       : BM25 sparse vectors only
  vector        : dense embeddings only
  hybrid        : both, fused with reciprocal rank fusion (RRF)
  hybrid_rerank : hybrid candidates rescored with a cross-encoder

evals/eval_retrieval.py compares the four and the README records which one
ships as the default.

Extension points, in rough order of payoff (each is a rubric point):
  TODO(query-rewriting): one cheap LLM call that reformulates the user
      question before retrieval. Evaluate against no-rewriting.
"""

import time

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from openai import OpenAI
from qdrant_client import QdrantClient, models

from app import config

MODES = ("keyword", "vector", "hybrid", "hybrid_rerank")

# Loaded once per process. fastembed downloads model weights to local
# cache on first use, so the first call is slow and the rest are not.
_dense = None
_sparse = None
_qdrant = None
_llm = None
_reranker = None


def _models():
    global _dense, _sparse, _qdrant
    if _dense is None:
        _dense = TextEmbedding(config.DENSE_MODEL)
        _sparse = SparseTextEmbedding(config.SPARSE_MODEL)
        _qdrant = QdrantClient(url=config.QDRANT_URL)
    return _dense, _sparse, _qdrant


def _client():
    global _llm
    if _llm is None:
        _llm = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
    return _llm


def _rerank_model():
    # Kept out of _models() so only hybrid_rerank pays the cross-encoder
    # download and load.
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(config.RERANK_MODEL)
    return _reranker


def search(query, mode="hybrid", limit=5):
    """Return a list of payload dicts (filename, title, text, score)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    if mode == "hybrid_rerank":
        return _hybrid_rerank(query, limit)

    dense_model, sparse_model, qdrant = _models()

    dense_vec = None
    sparse_vec = None
    if mode in ("vector", "hybrid"):
        dense_vec = list(dense_model.embed([query]))[0].tolist()
    if mode in ("keyword", "hybrid"):
        sv = list(sparse_model.query_embed(query))[0]
        sparse_vec = models.SparseVector(
            indices=sv.indices.tolist(), values=sv.values.tolist()
        )

    if mode == "vector":
        res = qdrant.query_points(
            collection_name=config.COLLECTION,
            query=dense_vec,
            using="dense",
            limit=limit,
            with_payload=True,
        )
    elif mode == "keyword":
        res = qdrant.query_points(
            collection_name=config.COLLECTION,
            query=sparse_vec,
            using="bm25",
            limit=limit,
            with_payload=True,
        )
    else:
        res = qdrant.query_points(
            collection_name=config.COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=limit * 4),
                models.Prefetch(query=sparse_vec, using="bm25", limit=limit * 4),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

    out = []
    for p in res.points:
        payload = dict(p.payload)
        payload["score"] = p.score
        out.append(payload)
    return out


def _hybrid_rerank(query, limit):
    """Widen the hybrid search to RERANK_CANDIDATES chunks, rescore every
    (query, chunk text) pair with the cross-encoder, then cut back to limit.
    The RRF score is replaced by the cross-encoder logit, which can be
    negative."""
    candidates = max(limit, config.RERANK_CANDIDATES)
    hits = search(query, mode="hybrid", limit=candidates)
    scores = list(_rerank_model().rerank(query, [h["text"] for h in hits]))
    for h, score in zip(hits, scores):
        h["score"] = score
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:limit]


# Two prompt variants so the LLM evaluation compares approaches rather than
# grading a single prompt. v2 is stricter about grounding and citations.
# TODO(prompts): add a v3 of your own design after looking at v1 vs v2
# failure cases. The eval harness picks up any key added to PROMPTS.
PROMPTS = {
    "v1": (
        "You are an IT support assistant. Answer the question using only the "
        "provided context from IBM technotes.\n"
        f"If the context does not contain the answer, reply with exactly {config.NOT_FOUND} "
        "and nothing else.\n\n"
        "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
    ),
    "v2": (
        "You are an IT support assistant answering from IBM technotes.\n"
        "Rules:\n"
        "1. Use only facts present in the context below. Do not use outside knowledge.\n"
        "2. If the context does not fully answer the question, reply with exactly "
        f"{config.NOT_FOUND} and nothing else. A partial or speculative answer is worse "
        "than admitting the information is missing.\n"
        "3. When you do answer, name the technote file(s) you used, quote exact "
        "commands or settings verbatim, and keep the answer under 200 words.\n\n"
        "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
    ),
}


def build_context(hits):
    blocks = []
    for h in hits:
        blocks.append(f"[{h['filename']}] {h['title']}\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def answer(question, mode="hybrid", prompt_version="v2", limit=5):
    """Full RAG round trip. Returns a dict the UI logs to Postgres."""
    t0 = time.time()
    hits = search(question, mode=mode, limit=limit)
    prompt = PROMPTS[prompt_version].format(
        context=build_context(hits), question=question
    )

    resp = _client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    usage = resp.usage

    return {
        "question": question,
        "answer": text,
        "abstained": config.NOT_FOUND in text,
        "mode": mode,
        "prompt_version": prompt_version,
        "model": config.LLM_MODEL,
        "sources": [h["filename"] for h in hits],
        "hits": hits,
        "response_ms": int((time.time() - t0) * 1000),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
    }


def llm_json(prompt, model=None):
    """One-shot completion that must return JSON. Used by the judge in
    evals/eval_llm.py. Strips markdown fences because smaller models add
    them no matter what the prompt says."""
    import json

    resp = _client().chat.completions.create(
        model=model or config.JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)
