"""Retrieval evaluation: hit rate and MRR at k for every retrieval mode.

Metrics are document-level. The retriever returns chunks; hits collapse to
their parent filename in first-seen order before scoring, because the gold
labels are filenames. A question counts as a hit if any gold filename shows
up in the top k unique documents.

Run:  python evals/eval_retrieval.py [--k 5] [--sample 0] [--rewrite]
      --sample 0 means the full answerable set (~700 questions).
      --rewrite re-runs the best base mode with query rewriting and appends
      one "<mode>+rw" row. That costs one LLM call per question.
Writes evals/retrieval_results.csv and prints a markdown table for the README.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from app import config, rag


def unique_filenames(hits):
    seen = []
    for h in hits:
        if h["filename"] not in seen:
            seen.append(h["filename"])
    return seen


def evaluate_mode(questions, mode, k, rewrite=False):
    label = f"{mode}+rw" if rewrite else mode
    hit, rr_sum = 0, 0.0
    for q in tqdm(questions, desc=label):
        # Retrieve extra chunks since several may come from one document.
        hits = rag.search(q["question"], mode=mode, limit=k * 3, rewrite=rewrite)
        docs = unique_filenames(hits)[:k]
        gold = set(q["gold_filenames"])
        rank = next((i + 1 for i, d in enumerate(docs) if d in gold), None)
        if rank:
            hit += 1
            rr_sum += 1.0 / rank
    n = len(questions)
    return {"mode": label, "hit_rate": hit / n, "mrr": rr_sum / n, "n": n, "k": k}


def best_result(results):
    return max(results, key=lambda r: (r["hit_rate"], r["mrr"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sample", type=int, default=0, help="0 = full set")
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="also evaluate the best base mode with query rewriting",
    )
    args = parser.parse_args()

    with open(config.ANSWERABLE_PATH) as f:
        questions = [json.loads(line) for line in f]
    if args.sample:
        questions = questions[: args.sample]

    results = [evaluate_mode(questions, mode, args.k) for mode in rag.MODES]

    if args.rewrite:
        best = best_result(results)
        results.append(evaluate_mode(questions, best["mode"], args.k, rewrite=True))

    print(f"\n| mode | hit rate@{args.k} | MRR@{args.k} | n |")
    print("|---|---|---|---|")
    for r in results:
        print(f"| {r['mode']} | {r['hit_rate']:.3f} | {r['mrr']:.3f} | {r['n']} |")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieval_results.csv")
    with open(out, "w") as f:
        f.write("mode,hit_rate,mrr,n,k\n")
        for r in results:
            f.write(f"{r['mode']},{r['hit_rate']:.4f},{r['mrr']:.4f},{r['n']},{r['k']}\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
