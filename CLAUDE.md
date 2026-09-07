# PROJECT CONTEXT

## What this is

Technote Copilot: a RAG application over nvidia/TechQA-RAG-Eval, a 910-question dataset of real IBM support forum questions. Answerable questions carry gold technote documents in a `contexts` field; a third (300 of 910) are labelled `is_impossible` and have no answer in the corpus. The system must answer from retrieved technotes only and reply with exactly NOT_FOUND when the corpus lacks the answer. Abstention on impossible questions is the project's headline metric.

## Layout

* app/config.py: all settings, from env vars
* app/rag.py: retrieval modes (keyword BM25, vector, hybrid RRF, hybrid_rerank cross-encoder) against one Qdrant collection, prompt variants v1/v2, query rewriting, answer() round trip
* app/db.py: Postgres logging (conversations, feedback)
* app/ui.py: Streamlit chat with thumbs feedback
* ingest/ingest.py: downloads dataset, dedupes contexts by filename (496 docs, 2023 chunks), chunks at 1800 chars paragraph-aware, indexes dense (BAAI/bge-small-en-v1.5) + sparse (Qdrant/bm25) via fastembed
* evals/build_eval_set.py: splits answerable vs impossible eval sets
* evals/eval_retrieval.py: document-level hit rate and MRR per mode; iterates rag.MODES, so new modes appear automatically
* evals/eval_llm.py: judge-scored relevance plus abstention rate, per prompt variant; every entry in rag.PROMPTS is evaluated automatically
* docker-compose.yml: qdrant v1.15.1, postgres 16, grafana, app, plus an `ingest` one-shot service behind a compose profile

## Runtime

Everything runs through docker compose. Useful commands: docker compose up -d --build docker compose run --rm ingest make eval-sets | eval-retrieval | eval-llm | psql | logs Dependencies are pinned in requirements.txt (qdrant-client 1.15.1 matching the server image). Do not change pins unless an actual error forces it, and explain any change you make.

## Known risk area

rag.py and ingest/ingest.py were written against the qdrant-client 1.15 query API (query_points, models.Prefetch, models.FusionQuery with Fusion.RRF, named dense vector "dense", named sparse vector "bm25" with IDF modifier) without being executed. If calls fail, verify against the INSTALLED library, not from memory: inspect signatures inside the container, e.g. docker compose run --rm --no-deps ingest python -c "from qdrant_client import QdrantClient; import inspect; print(inspect.signature(QdrantClient.query_points))" Fix the code to match the installed 1.15.1 API rather than upgrading packages to match the code.

## STATUS: both phases complete

Phase 1 and Phase 2 are done, run and pushed to main. Measured results:

* Corpus: 496 unique technotes, 2023 chunks, 610 answerable / 300 impossible.
* Retrieval over all 610 answerable questions: keyword 0.892/0.802, vector
  0.918/0.824, hybrid 0.938/0.858, hybrid_rerank 0.913/0.823 (hit rate/MRR
  at 5). hybrid ships as the default.
* Reranking LOST to plain hybrid. The cross-encoder takes 512 tokens while
  chunks are ~450 tokens plus the question, so candidates are truncated, and
  MS MARCO passages are nothing like technotes. Reported honestly in README.
* LLM output, 100 per subset: v1 relevant 0.54, wrong abstain 0.35,
  hallucination 0.12. v2 relevant 0.25, wrong abstain 0.71, hallucination
  0.13. v1 ships as the default.
* Known weakness: eval_retrieval scores top 5 unique documents drawn from 15
  chunks, but answer() passes only 5 chunks, so the generator sees less than
  the hit rate implies. Prime suspect for the 0.35 wrong-abstain rate.

The phase instructions below are kept as a record of the original brief.

## Hard guardrails

1. NEVER run evals/eval_llm.py. It makes hundreds of paid LLM calls. Only I trigger it.
2. evals/eval_retrieval.py during development only with --sample 10. Full runs are mine.
3. Any single test of rag.answer() costs one LLM call. Keep smoke tests to a handful of calls total.
4. Results tables and the rubric self-assessment in README.md must only ever contain numbers from real runs. Never invent, estimate or placeholder them. Transcribing measured output is fine and expected.
5. Never use em dashes or en dashes anywhere: code, comments, docs, commit messages. Use commas, colons, or parentheses instead. No emoji.
6. Comments explain why, briefly. No decorative or narrating comments.
7. Small atomic commits, imperative mood messages. Never force push, never amend published commits. Ask before any destructive operation (volume deletion, collection drops outside ingest, file deletion).
8. Do not add features beyond the two specified in Phase 2. No v3 prompt, no cloud tooling, no extra abstractions, no speculative refactors.

## PHASE 1: make it run

1. Confirm .env exists and docker is available. If .env is missing, stop and tell me; do not create it.
2. docker compose up -d --build. Verify all services come up and postgres reports healthy.
3. docker compose run --rm ingest. Debug until it completes. Success is the final line reporting the collection point count (measured: 2023 points from 496 unique documents). Iterate on tracebacks yourself using the known-risk guidance above.
4. make eval-sets. Report the exact answerable/impossible counts (measured: 610 answerable, 300 impossible).
5. Smoke test with a throwaway script (do not commit it) that calls rag.answer() three times total: one question copied verbatim from the dataset, one paraphrased version of a dataset question, and one off-domain question like "how do I reset a Fortnite password" which must return NOT_FOUND. Print answer, sources, abstained flag, and response_ms for each.
6. CHECKPOINT: stop and report results. I will verify the Streamlit UI and feedback logging by hand before you continue.

## PHASE 2: two features (only after I confirm Phase 1)

### Feature A: reranking mode

Add a fourth retrieval mode "hybrid_rerank" to app/rag.py:

* Retrieve top 20 with the existing hybrid path, rescore each (query, chunk text) pair with a cross-encoder, return the top `limit`.
* Use fastembed's TextCrossEncoder. Pick a small CPU-friendly model from TextCrossEncoder.list_supported_models(), preferring an ms-marco MiniLM variant or jina-reranker tiny. Lazy-load and cache it exactly like the existing embedding models.
* Add the mode to MODES so eval and UI pick it up automatically.
* Acceptance: python evals/eval_retrieval.py --sample 10 completes and prints four mode rows.

### Feature B: query rewriting

* Add rewrite_query(question) to app/rag.py: one LLM call, temperature 0, prompting the model to rewrite the user question into the technical vocabulary of an enterprise knowledge base, returning only the rewritten question. On any API error, fall back to the original question.
* Add a rewrite=False parameter to search() and answer(). When true, retrieval uses the rewritten question; logging keeps the original.
* Add a --rewrite flag to evals/eval_retrieval.py that appends one extra result row: the best base mode with rewriting on, labelled like "hybrid+rw". Mind guardrail 3: with rewriting, each eval question costs an LLM call, so --sample 10 discipline applies doubly.
* Add a sidebar checkbox in app/ui.py, default off.
* Acceptance: python evals/eval_retrieval.py --sample 10 --rewrite completes and shows the extra row.

## Wrap-up

Update the architecture and evaluation prose in README.md to mention both features (leave all TODO results tables untouched), commit, and report: what changed, files touched, any deviations from this brief with reasons, and the exact commands I should now run for the full evaluations.
