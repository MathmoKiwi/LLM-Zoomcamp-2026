"""Build the two evaluation sets from the dataset itself.

TechQA-RAG-Eval questions come pre-labelled with ground truth, so no
LLM-generated questions are needed:

  answerable  : is_impossible == false. Each has gold document filename(s),
                used for retrieval metrics (hit rate, MRR) and for judging
                answer quality.
  impossible  : is_impossible == true. No document in the corpus answers
                these. A correct system abstains; anything else is a
                hallucination. Roughly a fifth of the dataset.

Run:  python evals/build_eval_set.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from app import config


def main():
    ds = load_dataset("nvidia/TechQA-RAG-Eval", split="train")
    os.makedirs(config.DATA_DIR, exist_ok=True)

    n_ans, n_imp = 0, 0
    with open(config.ANSWERABLE_PATH, "w") as fa, open(
        config.IMPOSSIBLE_PATH, "w"
    ) as fi:
        for row in ds:
            if row["is_impossible"]:
                fi.write(
                    json.dumps({"id": row["id"], "question": row["question"]}) + "\n"
                )
                n_imp += 1
            else:
                gold = sorted({c["filename"] for c in row["contexts"]})
                fa.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "question": row["question"],
                            "gold_filenames": gold,
                            "reference_answer": row["answer"],
                        }
                    )
                    + "\n"
                )
                n_ans += 1

    print(f"answerable: {n_ans} -> {config.ANSWERABLE_PATH}")
    print(f"impossible: {n_imp} -> {config.IMPOSSIBLE_PATH}")


if __name__ == "__main__":
    main()
