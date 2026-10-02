

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

    my_questions = [                        
        "how many sandals were sold?",
        "what is the total number of sandals sold?",
        "how many customers bought sandals?",
    ]

    DATA_DIR.mkdir(exist_ok=True)         # 4. makes the folder (no error if it exists)

    with open(DATA_DIR / "train.jsonl", "w", encoding="utf-8") as f:   # 5. write
        for q in my_questions:
            row = make_example(q, my_answer)
            f.write(json.dumps(row) + "\n")

    print("wrote", len(my_questions), "lines")