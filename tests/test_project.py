"""Test project for the SBA 928 Seth Cutright model."""

from __future__ import annotations

import json
from pathlib import Path

from sba_928_seth_cutright.metrics import conclusion_correct, percent_recall
from sba_928_seth_cutright.prompts import build_prompt_variants


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_split_counts():
    train_path = Path("data/processed/train.jsonl")
    val_path = Path("data/processed/validation.jsonl")
    test_path = Path("data/processed/test.jsonl")

    train_data = read_jsonl(train_path)
    val_data = read_jsonl(val_path)
    test_data = read_jsonl(test_path)

    assert len(train_data) == 180
    assert len(val_data) == 24
    assert len(test_data) == 24

    total_records = len(train_data) + len(val_data) + len(test_data)
    assert total_records == 228


def test_no_id_overlap():
    train_data = read_jsonl("data/processed/train.jsonl")
    val_data = read_jsonl("data/processed/validation.jsonl")
    test_data = read_jsonl("data/processed/test.jsonl")

    train_ids = {r["record_id"] for r in train_data}
    val_ids = {r["record_id"] for r in val_data}
    test_ids = {r["record_id"] for r in test_data}

    assert train_ids & val_ids == set()
    assert train_ids & test_ids == set()
    assert val_ids & test_ids == set()

    all_ids = train_ids | val_ids | test_ids
    assert len(all_ids) == 228


def test_prompt_variants():
    context_text = "Sample background context."
    question_text = "What is the final conclusion?"
    
    variants = build_prompt_variants(context_text, question_text)

    assert len(variants) == 3
    assert len(set(variants.values())) == 3

    for prompt_text in variants.values():
        assert prompt_text.strip().endswith(question_text.strip())


def test_metrics():
    """Test metrics scorers using realistic data records and regression guards."""
    cb3 = {
        "topic": "cb3",
        "target": "Conclusion: Women are more likely to prefer sandals. Men 3.8% vs women 4.7%",
    }
    mt1 = {
        "topic": "mt1",
        "target": "The North region has the highest sales.",
    }

    assert percent_recall("Men 3.8% vs women 4.7%", cb3["target"]) == 1.0

    assert conclusion_correct("Women are more likely to buy Sandals", cb3) is True
    assert conclusion_correct("Therefore, more men than women bought Sandals.", cb3) is False

    assert conclusion_correct("The North region has the highest sales.", mt1) is True
    assert conclusion_correct("The South region has the highest sales and the North region the lowest.", mt1) is False