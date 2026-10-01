#import the load_shopping function from the load_data.py file in the sba_928_seth_cutright package
from sba_928_seth_cutright.load_data import load_shopping
import pandas as pd
AGE_BINS = [17, 29, 39, 49, 59, 70]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60-70"]

def prepare_shopping():
    df = load_shopping()
    df["Age Group"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS)
    return df

if __name__ == "__main__":
    prepared_df = prepare_shopping()
    print(prepared_df.head())
    print(prepared_df["Age Group"].isna().sum())
         