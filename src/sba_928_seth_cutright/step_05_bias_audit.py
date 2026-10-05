"""Audit the bias in model conclusions based on target statements."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

EVAL_FILE = Path("artifacts/evaluation/model_comparison.jsonl")
TRAIN_FILE = Path("data/processed/train.jsonl")
OUT_DIR = Path("artifacts/bias_audit")
LOGGER = logging.getLogger(__name__)


def conclusion_type(target):
    """Determine the type of conclusion based on the target text."""
    text = target.lower()
    if "women are more likely" in text:
        return "women"
    if "men are more likely" in text:
        return "men"
    if "similar" in text:
        return "similar"
    return None


def main() -> None:
    """Main function to perform the bias audit."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train = [json.loads(line) for line in f]

    groups = []
    for topic in ["cb2", "cb3", "mt1", "cb3_region", "cb3_age"]:
        topic_rows = [r for r in rows if r["topic"] == topic]
        groups.append(
            {
                "group": f"topic={topic}",
                "records": len(topic_rows),
                "base_correct": sum(r["base_correct"] for r in topic_rows),
                "tuned_correct": sum(r["tuned_correct"] for r in topic_rows),
                "train_examples": len([t for t in train if t["topic"] == topic]),
            }
        )

    for kind in ["women", "men", "similar"]:
        kind_rows = [
            r
            for r in rows
            if (
                r["topic"] == "cb3"
                or r["topic"] == "cb3_region"
                or r["topic"] == "cb3_age"
            )
            and conclusion_type(r["target"]) == kind
        ]
        groups.append(
            {
                "group": f"cb3_family_conclusion={kind}",
                "records": len(kind_rows),
                "base_correct": sum(r["base_correct"] for r in kind_rows),
                "tuned_correct": sum(r["tuned_correct"] for r in kind_rows),
                "train_examples": len(
                    [
                        t
                        for t in train
                        if (
                            t["topic"] == "cb3"
                            or t["topic"] == "cb3_region"
                            or t["topic"] == "cb3_age"
                        )
                        and conclusion_type(t["target"]) == kind
                    ]
                ),
            }
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(groups).to_csv(OUT_DIR / "subgroup_metrics.csv", index=False)
    with open(OUT_DIR / "bias_audit.json", "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2)
    for g in groups:
        LOGGER.info("%s", g)


if __name__ == "__main__":
    main()
