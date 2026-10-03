

import json                               
from pathlib import Path                    
from sba_928_seth_cutright.prepare_data import prepare_shopping   

SYSTEM_PROMPT = (
    "You are a market research analyst. Answer using only the shopping data. "
    "If the shopping data does not contain the answer, say so."
)
DATA_DIR = Path("data")


def answer_sandals(df):
    sandals = (df["Item Purchased"] == "Sandals").sum()
    everyone = len(df)
    share = sandals / everyone * 100
    return f"{sandals:,} sandal buyers out of {everyone:,} customers ({share:.1f}%)."

def answer_cb3(df):
    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]
    men_sandals = (men["Item Purchased"] == "Sandals").sum()
    women_sandals = (women["Item Purchased"] == "Sandals").sum()

    men_total = len(men)
    women_total = len(women)
    men_rate = men_sandals / men_total * 100
    women_rate = women_sandals / women_total * 100
    return (
        f"{men_sandals:,} men bought sandals out of {men_total:,} men ({men_rate:.1f}%), "
        f"{women_sandals:,} women bought sandals out of {women_total:,} women ({women_rate:.1f}%). "
        "There are about 2x more men in the data than women so the overall sandal purchase count is skewed towards men."
        
    )


def make_example(question, answer):             
    return {
        "messages": [                  # the key the trainer looks for
            {"role": "system", "content": SYSTEM_PROMPT},   # the instructions (your constant)
            {"role": "user", "content": question},   # the question (a parameter)
            {"role": "assistant", "content": answer},   # the answer (a parameter)
        ]
    }
if __name__ == "__main__":
    df = prepare_shopping()                      #  load once
    my_answer = answer_sandals(df)             #  compute once

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

    
    DATA_DIR.mkdir(exist_ok=True)         # 4. makes the folder (no error if it exists)

    with open(DATA_DIR / "train.jsonl", "w", encoding="utf-8") as f:   # 5. write
        for q in cb2_questions:
            row = make_example(q, my_answer)
            f.write(json.dumps(row) + "\n")

        for q in cb3_questions:
            row = make_example(q, answer_cb3(df))
            f.write(json.dumps(row) + "\n")

    print("wrote", len(cb2_questions) + len(cb3_questions), "lines")

