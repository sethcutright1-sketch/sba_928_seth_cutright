"""Metrics for evaluating the model's performance."""

from __future__ import annotations

import re


def percent_recall(reply, target):
    """Calculate the percentage recall of the reply against the target."""
    percents = re.findall(r"\d+\.\d%", target)
    if not percents:
        return 0.0
    found = [p for p in percents if p in reply]
    return len(found) / len(percents)


def conclusion_correct(reply, record):
    """Check if the conclusion in the reply matches the target for the given topic."""
    text = reply.lower()
    target = record["target"].lower()
    topic = record["topic"]

    if topic == "cb3":
        if "women are more likely" in target:
            return "women are more likely" in text
        if "men are more likely" in target:
            return re.search(r"\bmen are more likely", text) is not None
        return "similar" in text

    if topic == "mt1":
        top_region = target.split()[1]
        i = text.find("highest")
        if i == -1:
            return False
        window = text[max(0, i - 25) : i]
        return top_region in window

    if topic == "cb2":
        count = target.split()[0]
        return count in text

    return False
