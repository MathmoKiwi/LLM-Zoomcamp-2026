"""Ingest the TechQA-RAG-Eval corpus into Qdrant.

The dataset ships gold documents inside each question's `contexts` field
rather than as a separate corpus file. The same technote appears under many
questions, so the corpus is the union of contexts deduplicated by filename.
On the current version that yields roughly 600-700 unique technotes.

Run:  python ingest/ingest.py
Idempotent: re-running recreates the collection from scratch.
"""

import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from app import config

CHUNK_CHARS = 1800  # ~450 tokens, fits several chunks in a prompt
BATCH = 64


def load_unique_documents():
    """Download the dataset and return {filename: text} for every unique
    gold document. Impossible questions have empty contexts and contribute
    nothing here, which is correct: they are unanswerable by design."""
    from datasets import load_dataset

    ds = load_dataset("nvidia/TechQA-RAG-Eval", split="train")
    docs = {}
    for row in ds:
        for ctx in row.get("contexts") or []:
            fname = ctx["filename"]
            if fname not in docs:
                docs[fname] = ctx["text"]
    return docs


def chunk_text(text, max_chars=CHUNK_CHARS):
    """Paragraph-aware chunking. Split on blank lines, then pack paragraphs
    into chunks up to max_chars. A paragraph longer than max_chars gets hard
    split. No overlap: technotes are section-structured, and the retrieval
    eval works at document level anyway, so overlap buys little here."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(p) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(p), max_chars):
                chunks.append(p[i : i + max_chars])
            continue
        if len(current) + len(p) + 2 > max_chars:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)
    return chunks


def build_corpus():
    """Write data/corpus.jsonl with one record per chunk."""
    docs = load_unique_documents()
    print(f"Unique documents: {len(docs)}")

    os.makedirs(config.DATA_DIR, exist_ok=True)
    n_chunks = 0
    with open(config.CORPUS_PATH, "w") as f:
        for fname, text in docs.items():
            # First line of a technote is its title. Keep it on every chunk
            # so retrieval sees the product name even in deep sections.
            title = text.split("\n", 1)[0].strip()
            for i, chunk in enumerate(chunk_text(text)):
                record = {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{fname}:{i}")),
                    "filename": fname,
                    "title": title,
                    "chunk_index": i,
                    "text": chunk,
                }
                f.write(json.dumps(record) + "\n")
                n_chunks += 1
    print(f"Wrote {n_chunks} chunks to {config.CORPUS_PATH}")


def index_corpus():
    """Embed every chunk (dense + BM25 sparse) and upsert into Qdrant."""
    from fastembed import SparseTextEmbedding, TextEmbedding
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=config.QDRANT_URL)

    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(
        collection_name=config.COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=config.DENSE_DIM, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    dense_model = TextEmbedding(config.DENSE_MODEL)
    sparse_model = SparseTextEmbedding(config.SPARSE_MODEL)

    with open(config.CORPUS_PATH) as f:
        records = [json.loads(line) for line in f]

    for start in tqdm(range(0, len(records), BATCH), desc="Indexing"):
        batch = records[start : start + BATCH]
        texts = [r["text"] for r in batch]
        dense_vecs = list(dense_model.embed(texts))
        sparse_vecs = list(sparse_model.embed(texts))

        points = []
        for rec, dv, sv in zip(batch, dense_vecs, sparse_vecs):
            points.append(
                models.PointStruct(
                    id=rec["id"],
                    vector={
                        "dense": dv.tolist(),
                        "bm25": models.SparseVector(
                            indices=sv.indices.tolist(), values=sv.values.tolist()
                        ),
                    },
                    payload={
                        "filename": rec["filename"],
                        "title": rec["title"],
                        "chunk_index": rec["chunk_index"],
                        "text": rec["text"],
                    },
                )
            )
        client.upsert(collection_name=config.COLLECTION, points=points)

    info = client.get_collection(config.COLLECTION)
    print(f"Collection '{config.COLLECTION}' ready: {info.points_count} points")


if __name__ == "__main__":
    build_corpus()
    index_corpus()
