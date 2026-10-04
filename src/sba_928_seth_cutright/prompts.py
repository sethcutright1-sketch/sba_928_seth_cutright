"""Builds the prompt variants for step 2."""

from __future__ import annotations


def build_prompt_variants(context, question):
    """Build the prompt variants for the given context and question."""
    data_and_question = f"Data: {context}\n\nQuestion: {question}"
    return {
        "minimal": "Just answer the question using the provided data.\n" + data_and_question,
        "structured": "Answer in labeled parts: conclusion, evidence, caveats.\n" + data_and_question,
        "grounded": "Use only the data given. If groups are different sizes, compare percentages, not counts. If the data can't answer the question, say so.\n" + data_and_question,
   
    }