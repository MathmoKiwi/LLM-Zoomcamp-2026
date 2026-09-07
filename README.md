# Technote Copilot

A RAG application that answers IT support questions from a corpus of IBM
technotes, and admits when it can't. Built for the LLM Zoomcamp final
project.

## The problem

Tier 1 support work is mostly recognition: someone has almost certainly hit
this error before, and the fix is written down somewhere. The bottleneck is
finding it. Keyword search over a knowledge base fails when the user
describes symptoms ("scheduled reports fail after changing password") and
the document describes causes ("trusted credential renewal").

This project retrieves relevant technotes for a natural-language question
and generates a grounded answer with sources. The part I care most about:
support corpora never cover everything, and a confident wrong answer costs
an engineer more time than no answer. So the system is built and measured
around abstention. When the corpus doesn't contain the fix, the correct
output is `NOT_FOUND`, and the evaluation quantifies how often the model
holds that line.

## The data

[nvidia/TechQA-RAG-Eval](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval)
(Apache-2.0), a 910-question reduction of IBM's TechQA dataset built for
evaluating RAG systems. Questions are real posts from IBM developer forums,
messy formatting and embedded log output included. Each answerable question
carries its gold technote(s) in a `contexts` field.

Two properties shape the whole design:

1. There is no separate corpus file. The knowledge base is the union of all
   `contexts`, deduplicated by filename (roughly 600-700 technotes). That
   means most documents in the index are the gold answer to something, so
   retrieval scores here will read higher than they would against the
   original TechQA's 800k-technote haystack. I state this openly rather
   than let the numbers flatter the system.
2. About a fifth of the questions are labelled `is_impossible`: real
   questions the corpus cannot answer. They ship with no gold document and
   no reference answer. These become a hallucination benchmark for free,
   with human-made ground truth instead of LLM-generated labels.

## Architecture

```
                        ┌──────────────┐
  question ───────────▶ │  Streamlit    │ ◀── thumbs up/down
                        │  (app/ui.py)  │
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐        ┌────────────┐
                        │  app/rag.py   │───────▶│  LLM (any   │
                        │  rewrite +    │        │  OpenAI-    │
                        │  retrieve +   │        │  compatible │
                        │  rerank +     │        │  endpoint)  │
                        │  prompt       │        └────────────┘
                        └──┬───────┬───┘
                           │       │
                 ┌─────────▼──┐  ┌─▼──────────┐
                 │  Qdrant     │  │  Postgres  │◀── Grafana
                 │  dense+BM25 │  │  logs +    │    dashboard
                 └────────────┘  │  feedback  │
                                 └────────────┘
```

One Qdrant collection holds both a dense vector (`BAAI/bge-small-en-v1.5`
via fastembed, CPU-friendly) and a BM25 sparse vector per chunk, which
gives four retrieval modes from a single index: keyword, vector, hybrid
with reciprocal rank fusion, and hybrid_rerank, which widens the hybrid
search to 20 candidate chunks and rescores each (question, chunk) pair
with a cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`, also fastembed)
before keeping the top few.

Retrieval can optionally be preceded by query rewriting: one cheap LLM
call reformulates the question into the technical vocabulary of a
knowledge base, and only the retrieval step sees the rewritten text. The
answer prompt and the logged conversation keep the user's original
wording. It is a sidebar toggle in the UI, off by default, and falls back
to the original question if the rewrite call fails.

## Running it

Requirements: Docker with compose, and an API key for any OpenAI-compatible
LLM endpoint (OpenAI, Groq, or local Ollama).

```bash
git clone <this repo>
cd technote-copilot
cp .env.example .env        # set OPENAI_API_KEY, optionally OPENAI_BASE_URL

docker compose up -d --build      # qdrant, postgres, grafana, app
docker compose run --rm ingest    # download dataset, chunk, embed, index
```

Then open http://localhost:8501 for the app and http://localhost:3000 for
Grafana. Ingestion downloads the dataset from Hugging Face and embeds
~2-3k chunks on CPU; expect a few minutes on first run.

Without Docker: `pip install -r requirements.txt`, start Qdrant and
Postgres however you like, set the env vars from `.env.example`, then run
the same scripts directly.

## Evaluation

Build the eval sets first (writes `data/eval_answerable.jsonl` and
`data/eval_impossible.jsonl`):

```bash
make eval-sets
```

### Retrieval

```bash
make eval-retrieval
```

Compares keyword, vector, hybrid, and hybrid_rerank on document-level hit
rate and MRR over all answerable questions. Adding `--rewrite` re-runs
whichever mode scored best with query rewriting enabled and appends one
extra row labelled `<mode>+rw`, so the cost of the rewrite call is visible
against the mode it has to beat. Results:

| mode | hit rate@5 | MRR@5 | n |
|---|---|---|---|
| keyword | TODO | TODO | ~700 |
| vector | TODO | TODO | ~700 |
| hybrid | TODO | TODO | ~700 |

<!-- TODO(results): run the eval, paste the table, and state which mode
     ships as the default and why. -->

### LLM output

```bash
make eval-llm
```

Two measurements, run for every prompt variant in `app/rag.py`:

1. Relevance on a sample of answerable questions, scored by an LLM judge
   against the dataset's reference answers. Judged rather than
   string-matched because the reference answers are uneven; some are full
   procedures, some are a product name.
2. Abstention on the impossible questions. Every confident answer to an
   unanswerable question is a hallucination by construction. The
   hallucination rate is the headline number of this project.

| prompt | relevant | partly | non-relevant | wrong abstain | hallucination rate |
|---|---|---|---|---|---|
| v1 | TODO | TODO | TODO | TODO | TODO |
| v2 | TODO | TODO | TODO | TODO | TODO |

<!-- TODO(results): run, paste, and pick the shipped prompt. -->

## Monitoring

Every Q&A round trip is logged to Postgres (mode, prompt version, latency,
token counts, sources, whether the model abstained), and the UI collects
thumbs up/down per answer. Grafana is provisioned against the same
database; panel SQL for six charts lives in
[monitoring/queries.md](monitoring/queries.md).

<!-- TODO(dashboard): build panels, commit the exported dashboard JSON,
     add a screenshot here. -->

## Project layout

```
app/        config, retrieval + generation, Postgres layer, Streamlit UI
ingest/     corpus build + Qdrant indexing (fully scripted, idempotent)
evals/      eval set builder, retrieval eval, LLM eval
monitoring/ grafana provisioning + panel SQL
data/       generated artifacts (gitignored)
```

## Rubric self-assessment

| criterion | status |
|---|---|
| Problem description | described above |
| Retrieval flow | knowledge base (Qdrant) + LLM |
| Retrieval evaluation | 3 approaches compared, best one shipped |
| LLM evaluation | 2+ prompts compared on relevance and hallucination rate |
| Interface | Streamlit UI |
| Ingestion pipeline | automated script (`ingest/ingest.py`) |
| Monitoring | feedback collection + Grafana dashboard |
| Containerization | everything in docker-compose |
| Reproducibility | pinned deps, public dataset, quickstart above |
| Hybrid search | evaluated and available as a mode |
| Document re-ranking | cross-encoder rescoring as the `hybrid_rerank` mode |
| Query rewriting | `--rewrite` flag and UI toggle, falls back on failure |

## Remaining work

Marked as `TODO(...)` in the code so they're greppable:

- [ ] `TODO(results)`: run both evals, paste numbers into this README,
      record which retrieval mode and prompt ship as defaults
- [ ] `TODO(prompts)`: a v3 prompt informed by v1/v2 failure cases
- [ ] `TODO(dashboard)`: build the Grafana panels from monitoring/queries.md,
      commit the dashboard JSON, screenshot for this README
- [ ] Cloud deployment for the bonus points
