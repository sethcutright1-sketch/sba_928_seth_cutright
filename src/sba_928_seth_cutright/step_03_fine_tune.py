"""Fine-tune the model using SFTTrainer with LoRA configuration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from sba_928_seth_cutright.modeling import load_model

CONFIG_PATH = Path("config/default.json")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run the fine-tuning process using the configuration specified in the JSON file."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)

    dataset = load_dataset("json", data_files={"train": config["train_file"], "validation": config["validation_file"]})
    dataset = dataset.select_columns(["messages"])

    model, tokenizer = load_model(config["model_name"])
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        learning_rate=config["learning_rate"],
        max_length=config["max_length"],
        seed=config["seed"],
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        report_to="none",
        bf16=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()


    final_dir = Path(config["output_dir"]) / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(Path(config["output_dir"]) / "training_config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(Path(config["output_dir"]) / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    LOGGER.info("saved adapter to %s", final_dir)

if __name__ == "__main__":
    main()
