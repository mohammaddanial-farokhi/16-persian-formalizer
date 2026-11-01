import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset, DatasetDict
import os
from pathlib import Path

# =====================================================
# 1. DATA LOADING & ANALYSIS
# =====================================================
def convert_dataset_to_json():
    BASE_DIR = Path(__file__).resolve().parent
    excel_path = BASE_DIR / "data" / "dataset" / "trainset_40k_onlySent.xlsx"
    
    print("Loading ....")
    df = pd.read_excel(excel_path)
    
    jsonl_path = BASE_DIR / "data" / "dataset" / "parsmap_dataset.jsonl"
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    print(f"\njsonlfile create in: {jsonl_path}")


def loading_dataset(dataset_path):
    """
    This function loads a JSONL file using the Hugging Face library.
    """

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"File path not found: {dataset_path}")
    
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Dataset loaded successfully. Number of rows: {len(dataset)}")

    return dataset


if __name__ == "__main__":
    MODEL_NAME="google/gemma-3-4b-it"
    DATASET_PATH = "data/dataset/parsmap_dataset.jsonl"

    # convert_dataset_to_json()
    raw_dataset=loading_dataset(DATASET_PATH)
