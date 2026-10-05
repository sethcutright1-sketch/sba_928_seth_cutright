"""Generate the training dataset for the SBA 928 Seth Cutright project."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from sba_928_seth_cutright.load_data import load_competitors
from sba_928_seth_cutright.prepare_data import prepare_shopping

SYSTEM_PROMPT = (
    "You are a market research analyst. Answer using only the data provided. "
    "If the data does not contain the answer, say so."
)
DATA_DIR = Path("data/processed")
LOGGER = logging.getLogger(__name__)


def answer_cb2(df, item):
    buyer = (df["Item Purchased"] == item).sum()
    everyone = len(df)
    share = buyer / everyone * 100
    context = f"{item} buyers: {buyer:,} | Total customers: {everyone:,} | Share: {share:.1f}%"
    target = f"{buyer:,} {item} buyers out of {everyone:,} customers ({share:.1f}%)."
    return context, target


def answer_cb3(df, item):
    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]
    men_buyers = (men["Item Purchased"] == item).sum()
    women_buyers = (women["Item Purchased"] == item).sum()

    men_total = len(men)
    women_total = len(women)
    men_rate = men_buyers / men_total * 100
    women_rate = women_buyers / women_total * 100
    gap = women_rate - men_rate
    if abs(gap) < 0.5:
        conclusion = f"{item} is purchased at a similar rate by men and women."
    elif women_rate > men_rate:
        conclusion = f"Women are more likely to buy {item}."
    else:
        conclusion = f"Men are more likely to buy {item}."
    context = (
        f"Men: {men_buyers:,} of {men_total:,} bought {item} ({men_rate:.1f}%) | "
        f"Women: {women_buyers:,} of {women_total:,} bought {item} ({women_rate:.1f}%)"
    )
    target = (
        f"Conclusion: {conclusion} "
        f"Men: {men_buyers:,} of {men_total:,} bought {item} ({men_rate:.1f}%) | "
        f"Women: {women_buyers:,} of {women_total:,} bought {item} ({women_rate:.1f}%). "
        f"There are about 2x more men in the data than women so the overall {item} purchase count is skewed towards men."
    )
    return context, target


def answer_mt1(df, item):
    pieces = []
    results = []
    counts = []
    for region in ["Northeast", "South", "West", "Midwest"]:
        region_data = df[df["Region"] == region]
        region_buyers = (region_data["Item Purchased"] == item).sum()
        region_total = len(region_data)
        region_rate = region_buyers / region_total * 100
        pieces.append(
            f"{region}: {region_buyers:,} of {region_total:,} bought {item} ({region_rate:.1f}%)"
        )
        results.append(region_rate)
        counts.append(region_buyers)
    context = " | ".join(pieces)
    top_rate = max(results)
    low_rate = min(results)
    top_region = ["Northeast", "South", "West", "Midwest"][results.index(top_rate)]
    low_region = ["Northeast", "South", "West", "Midwest"][results.index(low_rate)]
    target = (
        f"The {top_region} has the highest rate ({top_rate:.1f}%) and the {low_region} the lowest ({low_rate:.1f}%). "
        f"Due to the small number of {item} buyers (about {min(counts):,}-{max(counts):,} per region), the differences are small and could be chance."
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


def make_example(question, context, answer, topic):
    return {
        "messages": [  # the key the trainer looks for
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },  # the instructions (your constant)
            {
                "role": "user",
                "content": f"Data: {context}\n\nQuestion: {question}",
            },  # the question (a parameter)
            {"role": "assistant", "content": answer},  # the answer (a parameter)
        ],
        "instruction": question,
        "context": context,
        "target": answer,
        "topic": topic,
    }


def main() -> None:
    """Build train.jsonl from the four prompt groups."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = prepare_shopping()  #  load once
    adidas, nike = load_competitors()

    cb2_questions = [
        "how many {item} were sold?",
        "what is the total number of {item} sold?",
        "how many customers bought {item}?",
    ]

    cb3_questions = [
        "What percentage of men, and what percentage of women, buy {item}?",
        "Did more men or women buy {item}?",
        "Of all customers who bought {item}, what proportion were men versus women?",
    ]

    mt1_questions = [
        "What percentage of customers in each region bought {item}?",
        "What is the regional purchase rate for {item} as a proportion of total customers?",
        "In which region are customers most likely to buy {item}?",
    ]

    ca3_questions = [
        "What percentage of each brand's products are discounted?",
        "What proportion of items within each brand catalog are currently on sale?",
        "How does the ratio of discounted merchandise compare across each brand's total inventory?",
    ]

    cb3_region_questions = [
        "What percentage of men, and what percentage of women, bought {item} in the {region} region?",
        "Did more men or women buy {item} in the {region} region?",
        "Comparing men versus women, who was more likely to buy {item} in the {region} region?"
    ]
    
    cb3_age_questions = [
        "What percentage of men, and what percentage of women, who are {age} bought {item}?",
        "Did more men or women who are {age} buy {item}?",
        "Comparing men versus women, who was more likely to buy {item} when aged {age}?"
    ]

    groups = [
        ("ca3", ca3_questions, answer_ca3(adidas, nike)),
    ]
    for item in df["Item Purchased"].unique():
        groups.append(
            ("cb2", [t.format(item=item) for t in cb2_questions], answer_cb2(df, item))
        )
        groups.append(
            ("cb3", [t.format(item=item) for t in cb3_questions], answer_cb3(df, item))
        )
        groups.append(
            ("mt1", [t.format(item=item) for t in mt1_questions], answer_mt1(df, item))
        )

        for region in ["Northeast", "South", "West", "Midwest"]:
            region_df = df[df["Region"] == region]
            groups.append(
                ("cb3_region", [t.format(item=item, region=region) for t in cb3_region_questions], answer_cb3(region_df, item))
            )
        for age in ["18-29", "30-39", "40-49", "50-59", "60-70"]:
            age_df = df[df["Age Group"] == age]
            groups.append(
                ("cb3_age", [t.format(item=item, age=age) for t in cb3_age_questions], answer_cb3(age_df, item))
            )

    rng = random.Random(42)
    rng.shuffle(groups)

    train_end = int(len(groups) * 0.8)
    val_end = int(len(groups) * 0.9)

    train_groups = groups[:train_end]
    val_groups = groups[train_end:val_end]
    test_groups = groups[val_end:]

    LOGGER.info(
        "groups: train %d, validation %d, test %d",
        len(train_groups),
        len(val_groups),
        len(test_groups),
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    split_counts = []
    for split_name, split_groups in [
        ("train", train_groups),
        ("validation", val_groups),
        ("test", test_groups),
    ]:
        start = count
        with open(DATA_DIR / f"{split_name}.jsonl", "w", encoding="utf-8") as f:
            for group_name, questions, answers in split_groups:
                for q in questions:
                    count += 1
                    row = make_example(q, answers[0], answers[1], group_name)
                    row["record_id"] = f"MR-{count:04d}"
                    row["split"] = split_name
                    f.write(json.dumps(row) + "\n")
        split_counts.append({split_name: count - start})
    LOGGER.info("wrote %d lines", count)
    manifest = {
        "seed": 42,
        "groups": {
            "train": len(train_groups),
            "validation": len(val_groups),
            "test": len(test_groups),
        },
        "records": split_counts,
        "total_records": count,
    }
    with open(DATA_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
