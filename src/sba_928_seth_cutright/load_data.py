"""Download the SBA 928 datasets from Kaggle and print an overview of each.

1. Shopping behavior (consumer behavior + market trends)
   Public copy at lakshitmalhotra28/consumer-behavior-and-shopping-habits
   (the original zeesolver dataset requires a Kaggle login). Ships as an
   Excel workbook with one sheet, "shopping_behavior_updated".
2. Adidas vs Nike products (competitor analysis)
   mexwell/adidas-x-nike: two CSVs scraped from the Adidas and Nike India
   online stores in April 2020. Prices are in Indian rupees (INR).
"""
from pathlib import Path

import kagglehub
import pandas as pd

SHOPPING_DATASET = "lakshitmalhotra28/consumer-behavior-and-shopping-habits"
SHOPPING_FILE = "customer behaviour.xlsx"

COMPETITOR_DATASET = "mexwell/adidas-x-nike"
ADIDAS_FILE = "adidas_final.csv"
NIKE_FILE = "nike_2020_04_13.csv"

# Long text/URL columns that clutter the preview
HIDE_IN_PREVIEW = ["url", "description", "images"]


def load_shopping() -> pd.DataFrame:
    path = Path(kagglehub.dataset_download(SHOPPING_DATASET))
    return pd.read_excel(path / SHOPPING_FILE)


def load_competitors() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(kagglehub.dataset_download(COMPETITOR_DATASET))
    return pd.read_csv(path / ADIDAS_FILE), pd.read_csv(path / NIKE_FILE)


def overview(name: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(name)
    print("Shape (rows, columns):", df.shape)
    print("\nColumns and data types:")
    print(df.dtypes.to_string())
    print("\nFirst 3 rows:")
    print(df.drop(columns=HIDE_IN_PREVIEW, errors="ignore").head(3).to_string())


if __name__ == "__main__":
    shopping = load_shopping()
    adidas, nike = load_competitors()
    overview("Shopping behavior", shopping)
    overview("Adidas products", adidas)
    overview("Nike products", nike)
