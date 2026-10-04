"""Custom inference script for the SBA 928 Seth Cutright model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sba_928_seth_cutright.modeling import MODEL_NAME, generate_reply, load_tuned_model
from sba_928_seth_cutright.step_01_generate_dataset import SYSTEM_PROMPT

ADAPTER_DIR = Path("artifacts/models/qwen-sba-928/final")
OUT_DIR = Path("artifacts/custom_inference")
DEFAULT_CONTEXT = (
    "Subscribers: 42 of 1,053 bought Boots (4.0%) | "
    "Non-subscribers: 102 of 2,847 bought Boots (3.6%)"
)
DEFAULT_QUESTION = "Are subscribers or non-subscribers more likely to buy Boots?"
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Main function to perform custom inference."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ask the analysis bot one question.")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Which question to ask")
    parser.add_argument("--context", default=DEFAULT_CONTEXT, help="The context data to provide to the model")
    args = parser.parse_args()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Data: {args.context}\n\nQuestion: {args.question}"},
    ]
    model, tokenizer = load_tuned_model(model_name=MODEL_NAME, adapter_dir=ADAPTER_DIR)
    reply = generate_reply(model, tokenizer, messages)

    result = {
        "base_model": MODEL_NAME,
        "adapter": str(ADAPTER_DIR),
        "context": args.context,
        "question": args.question,
        "reply": reply,
        "note": "Model prediction; requires human review.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "custom_inference.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    LOGGER.info("reply: %s", reply)


if __name__ == "__main__":
    main()