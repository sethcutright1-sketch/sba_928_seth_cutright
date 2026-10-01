#import the load_shopping function from the load_data.py file in the sba_928_seth_cutright package
from sba_928_seth_cutright.load_data import load_shopping
import pandas as pd
AGE_BINS = [17, 29, 39, 49, 59, 70]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60-70"]
STATE_TO_REGION = {
    # US Census Bureau's 4 regions (state -> region lookup)
    # Northeast (9)
    "Connecticut": "Northeast",
    "Maine": "Northeast",
    "Massachusetts": "Northeast",
    "New Hampshire": "Northeast",
    "Rhode Island": "Northeast",
    "Vermont": "Northeast",
    "New Jersey": "Northeast",
    "New York": "Northeast",
    "Pennsylvania": "Northeast",
    # Midwest (12)
    "Illinois": "Midwest",
    "Indiana": "Midwest",
    "Michigan": "Midwest",
    "Ohio": "Midwest",
    "Wisconsin": "Midwest",
    "Iowa": "Midwest",
    "Kansas": "Midwest",
    "Minnesota": "Midwest",
    "Missouri": "Midwest",
    "Nebraska": "Midwest",
    "North Dakota": "Midwest",
    "South Dakota": "Midwest",
    # South (16)
    "Delaware": "South",
    "Florida": "South",
    "Georgia": "South",
    "Maryland": "South",
    "North Carolina": "South",
    "South Carolina": "South",
    "Virginia": "South",
    "West Virginia": "South",
    "Alabama": "South",
    "Kentucky": "South",
    "Mississippi": "South",
    "Tennessee": "South",
    "Arkansas": "South",
    "Louisiana": "South",
    "Oklahoma": "South",
    "Texas": "South",
    # West (13)
    "Arizona": "West",
    "Colorado": "West",
    "Idaho": "West",
    "Montana": "West",
    "Nevada": "West",
    "New Mexico": "West",
    "Utah": "West",
    "Wyoming": "West",
    "Alaska": "West",
    "California": "West",
    "Hawaii": "West",
    "Oregon": "West",
    "Washington": "West",
}
def prepare_shopping():
    df = load_shopping()
    df["Age Group"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS)
    df["Region"] = df["Location"].map(STATE_TO_REGION)

    return df


if __name__ == "__main__":
    prepared_df = prepare_shopping()
    print(prepared_df.head())
    print(prepared_df["Age Group"].isna().sum())
    print(prepared_df["Region"].isna().sum())
    print(prepared_df["Region"].value_counts())
         