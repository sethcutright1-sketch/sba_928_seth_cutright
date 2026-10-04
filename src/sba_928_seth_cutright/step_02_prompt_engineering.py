"""Run the prompt variants based on model (no training)"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sba_928_seth_cutright.modeling import MODEL_NAME, generate_reply, load_model
from sba_928_seth_cutright.prompts import build_prompt_variants

DATA_DIR = Path("data/processed")
OUT_DIR = Path("artifacts/prompt_baseline")
LIMIT = 12
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Runs each prompt variant on the validation records and saves the results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(DATA_DIR / "validation.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    records = records[:LIMIT]

    model, tokenizer = load_model(MODEL_NAME)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "prompt_predictions.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            versions = build_prompt_variants(record["context"], record["instruction"])
            for variant_name, prompt in versions.items():
                reply = generate_reply(model, tokenizer, [{"role": "user", "content": prompt}])
                row = {
                    "record_id": record["record_id"],
                    "variant": variant_name,
                    "reply": reply,
                    "context": record["context"],
                    "instruction": record["instruction"],
                    "prompt": prompt,
                    "topic": record["topic"],  
                    "target": record["target"],
                }
                f.write(json.dumps(row) + "\n")
            LOGGER.info("done %s", record["record_id"])

if __name__ == "__main__":
    main()