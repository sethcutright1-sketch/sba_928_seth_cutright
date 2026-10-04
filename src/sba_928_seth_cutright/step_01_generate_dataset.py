"""Generate the training dataset for the SBA 928 Seth Cutright project."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sba_928_seth_cutright.load_data import load_competitors
from sba_928_seth_cutright.prepare_data import prepare_shopping

SYSTEM_PROMPT = (
    "You are a market research analyst. Answer using only the data provided. "
    "If the data does not contain the answer, say so."
)
DATA_DIR = Path("data")
LOGGER = logging.getLogger(__name__)

def answer_sandals(df):
    sandals = (df["Item Purchased"] == "Sandals").sum()
    everyone = len(df)
    share = sandals / everyone * 100
    context = f"Sandal buyers: {sandals:,} | Total customers: {everyone:,} | Share: {share:.1f}%"
    target = f"{sandals:,} sandal buyers out of {everyone:,} customers ({share:.1f}%)."
    return context, target

def answer_cb3(df):
    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]
    men_sandals = (men["Item Purchased"] == "Sandals").sum()
    women_sandals = (women["Item Purchased"] == "Sandals").sum()

    men_total = len(men)
    women_total = len(women)
    men_rate = men_sandals / men_total * 100
    women_rate = women_sandals / women_total * 100
    context = (
        f"Men: {men_sandals:,} of {men_total:,} bought sandals ({men_rate:.1f}%) | "
        f"Women: {women_sandals:,} of {women_total:,} bought sandals ({women_rate:.1f}%)"
    )
    target = f"Women are more likely to buy sandals ({women_rate:.1f}% vs. {men_rate:.1f}%). There are about 2x more men in the data than women so the overall sandal purchase count is skewed towards men."
    return context, target

def answer_mt1(df):
    pieces = [] 
    for region in ["Northeast", "South", "West", "Midwest"]:
        region_data = df[df["Region"] == region]
        region_sandals = (region_data["Item Purchased"] == "Sandals").sum()
        region_total = len(region_data)
        region_rate = region_sandals / region_total * 100
        pieces.append(f"{region}: {region_sandals:,} of {region_total:,} bought sandals ({region_rate:.1f}%)")
    context = " | ".join(pieces)
    target = (
        "The Midwest has the highest rate (5.5%) and the West the lowest (3.4%). "
        "Due to the small number of sandal buyers (about 24-52 per region), the differences are small and could be chance."
    )
    return context, target


def answer_ca3(adidas, nike):
    adidas_discount = (adidas["discount"] > 0).sum()
    nike_discount = (nike["discount"] > 0).sum()
    adidas_total = len(adidas)
    nike_total = len(nike)
    adidas_rate = adidas_discount / adidas_total * 100
    nike_rate = nike_discount / nike_total * 100
    context = (
        f"Adidas: {adidas_discount:,} of {adidas_total:,} products discounted ({adidas_rate:.1f}%) | "
        f"Nike: {nike_discount:,} of {nike_total:,} products discounted ({nike_rate:.1f}%)"
    )
    target = (
        "Adidas has a higher proportion of discounted products compared to Nike. "
        f"Adidas: {adidas_discount:,} of {adidas_total:,} products discounted ({adidas_rate:.1f}%) | "
        f"Nike: {nike_discount:,} of {nike_total:,} products discounted ({nike_rate:.1f}%). "
        "Adidas has more products overall, so the comparison should focus on percentages rather than raw counts. "
        "It is also one day's snapshot (April 2020) during COVID so it may not show normal pricing."
    )
    return context, target


def make_example(question, context, answer):             
    return {
        "messages": [                  # the key the trainer looks for
            {"role": "system", "content": SYSTEM_PROMPT},   # the instructions (your constant)
            {"role": "user", "content": f"Data: {context}\n\nQuestion: {question}"},   # the question (a parameter)
            {"role": "assistant", "content": answer},   # the answer (a parameter)
                ],
        "instruction": question,
        "context": context,
        "target": answer,
    }

def main() -> None:
    """Build train.jsonl from the four prompt groups."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = prepare_shopping()                      #  load once
    adidas, nike = load_competitors()

    cb2_questions = [                        
            "how many sandals were sold?",
            "what is the total number of sandals sold?",
            "how many customers bought sandals?",
        ]

    cb3_questions = [                        
            "What percentage of men, and what percentage of women, buy sandals?",
            "Did more men or women buy sandals?",
            "Of all customers who bought sandals, what proportion were men versus women?",
        ]

    mt1_questions = [
            "What percentage of customers in each region bought sandals?",
            "What is the regional purchase rate for sandals as a proportion of total customers?",
            "In which region are customers most likely to buy sandals?"
    ]


    ca3_questions = [
        "What percentage of each brand's products are discounted?",
        "What proportion of items within each brand catalog are currently on sale?",
        "How does the ratio of discounted merchandise compare across each brand's total inventory?",
    ]

    groups = [
        (cb2_questions, answer_sandals(df)),
        (cb3_questions, answer_cb3(df)),
        (mt1_questions, answer_mt1(df)),
        (ca3_questions, answer_ca3(adidas, nike)),  
    ]
    DATA_DIR.mkdir(exist_ok=True)         # 4. makes the folder (no error if it exists)

    with open(DATA_DIR / "train.jsonl", "w", encoding="utf-8") as f:   # 5. write
        count = 0
        for questions, answers in groups:
            for q in questions:
                row = make_example(q, answers[0], answers[1])
                f.write(json.dumps(row) + "\n")
                count += 1
    LOGGER.info("wrote %d lines", count)


if __name__ == "__main__":
    main()
