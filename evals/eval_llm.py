"""LLM output evaluation, two measurements per prompt variant.

1. Relevance (answerable sample): an LLM judge compares the generated
   answer against the dataset's reference answer and scores it RELEVANT,
   PARTLY_RELEVANT, or NON_RELEVANT. Judged rather than string-matched
   because TechQA reference answers are uneven; some are full procedures,
   some are a product name or a bare URL.

2. Abstention (impossible sample): these questions have no answer in the
   corpus, so the only correct behaviour is emitting NOT_FOUND. The
   hallucination rate is the fraction of confident answers to unanswerable
   questions. This is the number that distinguishes a grounded system from
   a fluent one.

Every prompt version in rag.PROMPTS is evaluated, which satisfies the
"multiple approaches evaluated" criterion. Add a v3 to rag.PROMPTS and
re-run; nothing here changes.

Run:  python evals/eval_llm.py [--sample 100] [--mode hybrid]
Writes evals/llm_results.csv and prints a markdown table.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from app import config, rag

JUDGE_PROMPT = """You are grading an IT support assistant.

QUESTION:
{question}

REFERENCE ANSWER (ground truth, may be terse):
{reference}

GENERATED ANSWER:
{generated}

Does the generated answer convey the same resolution as the reference?
Grade RELEVANT if it contains the key fix or information from the reference,
PARTLY_RELEVANT if it is on topic but misses or muddles the key point,
NON_RELEVANT if it answers a different question or contradicts the reference.

Respond with JSON only: {{"verdict": "RELEVANT" | "PARTLY_RELEVANT" | "NON_RELEVANT", "reason": "<one sentence>"}}"""


def eval_relevance(questions, prompt_version, mode):
    counts = {"RELEVANT": 0, "PARTLY_RELEVANT": 0, "NON_RELEVANT": 0, "ABSTAINED": 0}
    for q in tqdm(questions, desc=f"relevance {prompt_version}"):
        result = rag.answer(q["question"], mode=mode, prompt_version=prompt_version)
        if result["abstained"]:
            # Abstaining on an answerable question is a miss, tracked
            # separately from wrong answers because the failure differs.
            counts["ABSTAINED"] += 1
            continue
        verdict = rag.llm_json(
            JUDGE_PROMPT.format(
                question=q["question"],
                reference=q["reference_answer"],
                generated=result["answer"],
            )
        )
        counts[verdict["verdict"]] += 1
    return counts


def eval_abstention(questions, prompt_version, mode):
    abstained = 0
    for q in tqdm(questions, desc=f"abstention {prompt_version}"):
        result = rag.answer(q["question"], mode=mode, prompt_version=prompt_version)
        if result["abstained"]:
            abstained += 1
    return abstained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=100, help="questions per subset")
    parser.add_argument("--mode", default="hybrid", choices=rag.MODES)
    args = parser.parse_args()

    with open(config.ANSWERABLE_PATH) as f:
        answerable = [json.loads(line) for line in f][: args.sample]
    with open(config.IMPOSSIBLE_PATH) as f:
        impossible = [json.loads(line) for line in f][: args.sample]

    rows = []
    for pv in rag.PROMPTS:
        rel = eval_relevance(answerable, pv, args.mode)
        n_ans = len(answerable)
        abstained = eval_abstention(impossible, pv, args.mode)
        n_imp = len(impossible)
        rows.append(
            {
                "prompt": pv,
                "relevant": rel["RELEVANT"] / n_ans,
                "partly": rel["PARTLY_RELEVANT"] / n_ans,
                "non_relevant": rel["NON_RELEVANT"] / n_ans,
                "wrong_abstain": rel["ABSTAINED"] / n_ans,
                "abstention_rate": abstained / n_imp,
                "hallucination_rate": 1 - abstained / n_imp,
            }
        )

    print("\n| prompt | relevant | partly | non-relevant | wrong abstain | hallucination rate |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['prompt']} | {r['relevant']:.2f} | {r['partly']:.2f} "
            f"| {r['non_relevant']:.2f} | {r['wrong_abstain']:.2f} "
            f"| {r['hallucination_rate']:.2f} |"
        )

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_results.csv")
    with open(out, "w") as f:
        f.write("prompt,relevant,partly,non_relevant,wrong_abstain,abstention_rate,hallucination_rate\n")
        for r in rows:
            f.write(
                f"{r['prompt']},{r['relevant']:.4f},{r['partly']:.4f},"
                f"{r['non_relevant']:.4f},{r['wrong_abstain']:.4f},"
                f"{r['abstention_rate']:.4f},{r['hallucination_rate']:.4f}\n"
            )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
