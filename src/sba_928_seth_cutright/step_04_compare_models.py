"""Compare the performance of the original and fine-tuned models."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sba_928_seth_cutright.metrics import conclusion_correct, percent_recall
from sba_928_seth_cutright.modeling import (
    MODEL_NAME,
    generate_reply,
    load_model,
    load_tuned_model,
)

DATA_DIR = Path("data/processed")
ADAPTER_DIR = Path("artifacts/models/qwen-sba-928/final")
OUT_DIR = Path("artifacts/evaluation")
LOGGER = logging.getLogger(__name__)


def run_model(model, tokenizer, records):
    """Run the model on the given records and generate replies."""
    return [generate_reply(model, tokenizer, r["messages"][: 2]) for r in records]


def main() -> None:
    """Compare the performance of the original and fine-tuned models."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(DATA_DIR / "test.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    base_model, tokenizer = load_model(MODEL_NAME)
    base_replies = run_model(base_model, tokenizer, records)
    LOGGER.info("base model done")

    tuned_model, tokenizer = load_tuned_model(model_name=MODEL_NAME, adapter_dir=ADAPTER_DIR)
    tuned_replies = run_model(tuned_model, tokenizer, records)
    LOGGER.info("tuned model done") 


    summary = {
        "test_records": len(records),
        "base_correct": sum(conclusion_correct(r, rec) for r, rec in zip(base_replies, records)),
        "tuned_correct": sum(conclusion_correct(r, rec) for r, rec in zip(tuned_replies, records)),
        "base_avg_percent_recall": sum(percent_recall(r, rec["target"]) for r, rec in zip(base_replies, records)) / len(records),
        "tuned_avg_percent_recall": sum(percent_recall(r, rec["target"]) for r, rec in zip(tuned_replies, records)) / len(records),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "model_comparison.jsonl", "w", encoding="utf-8") as f:
        for record, base_reply, tuned_reply in zip(records, base_replies, tuned_replies):
            row = {
                "record_id": record["record_id"],
                "topic": record["topic"],
                "instruction": record["instruction"],
                "context": record["context"],
                "target": record["target"],
                "base_reply": base_reply,
                "tuned_reply": tuned_reply,
                "base_correct": conclusion_correct(base_reply, record),
                "tuned_correct": conclusion_correct(tuned_reply, record),
                "base_percent_recall": percent_recall(base_reply, record["target"]),
                "tuned_percent_recall": percent_recall(tuned_reply, record["target"]),
            }
            f.write(json.dumps(row) + "\n")

    with open(OUT_DIR / "model_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("summary: %s", summary)

    LOGGER.info("Summary: %s", summary)

if __name__ == "__main__":
    main()