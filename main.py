import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset, DatasetDict
from collections import Counter
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


def dataset_first_look(dataset, sample_size: int = 2, text_columns=None):
    """
    Perform a quick initial analysis of a parallel text dataset (JSON format).
    Optimized for 'formalForm' and 'inFormalForm' columns.
    """
    line = "=" * 70
    print(f"\n{line}\nDATASET FIRST LOOK\n{line}")

    # ---------- 1) Splits and Shape ----------
    print("\n[1] SPLITS AND SHAPE")
    if isinstance(dataset, DatasetDict):
        splits = list(dataset.keys())
        for split in splits:
            print(f"    - split '{split}': {len(dataset[split]):,} rows")
        main_split = "train" if "train" in splits else splits[0]
        data = dataset[main_split]
        print(f"    * Analyzing main split: '{main_split}'")
    else:
        data = dataset
        print(f"    - single dataset: {len(data):,} rows")

    # ---------- 2) Columns ----------
    print("\n[2] COLUMNS")
    for i, col in enumerate(data.column_names):
        print(f"    [{i}] {col!r}  type={data.features[col]}")

    # Set default text columns specific to this dataset
    if text_columns is None:
        text_columns = ["formalForm", "inFormalForm"]

    # Extract data directly as lists for faster processing
    col_data = {col: data[col] for col in data.column_names}

    # ---------- 3) Null / empty ----------
    print("\n[3] NULL / EMPTY VALUES")
    for col in data.column_names:
        # Check for None, empty strings, or strings with just spaces
        n_empty = sum(1 for x in col_data[col] if x is None or len(str(x).strip()) == 0)
        pct = 100 * n_empty / len(data) if len(data) else 0
        status = "OK  " if n_empty == 0 else "WARN"
        print(f"    [{status}] {col}: {n_empty:,} empty ({pct:.2f}%)")

    # ---------- 4) Unique values ----------
    print("\n[4] UNIQUE VALUES PER COLUMN")
    for col in data.column_names:
        # Convert to string just in case to avoid unhashable types
        extracted_list = [str(x) for x in col_data[col]]
        n_unique = len(set(extracted_list))
        if n_unique <= 20:
            print(f"    - {col}: {n_unique} unique values")
            for val, count in Counter(extracted_list).most_common():
                pct = 100 * count / len(data)
                print(f"        * {val!r}: {count:,} ({pct:.1f}%)")
        else:
            print(f"    - {col}: {n_unique:,} unique values")
            # Show the duplicate count for context
            n_dupes = len(extracted_list) - n_unique
            pct_dupes = 100 * n_dupes / len(data)
            print(f"        * (Note: {n_dupes:,} duplicated values [{pct_dupes:.2f}%])")

    # ---------- 5) Length Comparison (Formal vs Informal) ----------
    if "formalForm" in text_columns and "inFormalForm" in text_columns:
        print("\n[5] LENGTH COMPARISON: FORMAL vs INFORMAL")
        len_formal = np.array([len(str(x)) for x in col_data["formalForm"]])
        len_informal = np.array([len(str(x)) for x in col_data["inFormalForm"]])

        diff = len_formal - len_informal
        print(f"    - Mean Formal length:   {np.mean(len_formal):.1f} chars")
        print(f"    - Mean Informal length: {np.mean(len_informal):.1f} chars")
        print(
            f"    - Formal is longer in:  {(diff > 0).sum():,} rows ({(diff > 0).mean()*100:.1f}%)"
        )
        print(
            f"    - Informal is longer in:{(diff < 0).sum():,} rows ({(diff < 0).mean()*100:.1f}%)"
        )
        print(
            f"    - Equal length:         {(diff == 0).sum():,} rows ({(diff == 0).mean()*100:.1f}%)"
        )

    # ---------- 6) Text length distribution ----------
    print("\n[6] TEXT LENGTH DISTRIBUTION")
    print(f"    (columns: {text_columns})")

    for col in text_columns:
        lengths = np.array([len(str(x)) for x in col_data[col]])
        if lengths.size == 0:
            continue
        print(f"    - {col}:")
        print(f"        * min:    {lengths.min()}")
        print(f"        * max:    {lengths.max()}")
        print(f"        * mean:   {lengths.mean():.1f}")
        print(f"        * median: {np.median(lengths):.1f}")
        print(f"        * std:    {lengths.std():.1f}")

        buckets = [0, 20, 50, 100, 200, 500, 1000, 2000, 5000, np.inf]
        labels = ["0-20", "20-50", "50-100", "100-200", "200-500", "500-1k", "1k-2k", "2k-5k", "5k+"]
        hist = np.histogram(lengths, bins=buckets)[0]
        total = hist.sum() or 1
        for label, count in zip(labels, hist):
            pct = 100 * count / total
            bar = "#" * int(pct / 2)
            print(f"          {label:>7}: {count:>7,} ({pct:5.1f}%) {bar}")

    # ---------- 7) Duplicates ----------
    print("\n[7] DUPLICATES")
    # Check exact duplicate pairs
    pairs = list(zip(col_data["formalForm"], col_data["inFormalForm"]))
    n_dupe_pairs = len(pairs) - len(set(pairs))
    print(f"    - exact duplicate row pairs: {n_dupe_pairs:,}")

    # ---------- 8) Samples ----------
    print("\n[8] SAMPLES")
    for i in range(min(sample_size, len(data))):
        print(f"\n    --- Sample {i + 1} ---")
        for col in data.column_names:
            val_str = str(col_data[col][i]).replace("\n", " ")
            if len(val_str) > 250:
                val_str = val_str[:250] + "..."
            print(f"        [{col}] {val_str!r}")

    print(f"\n{line}\nAnalysis finished.\n{line}")

# =====================================================
# 2. DATA PREPROCESS
# =====================================================
def clean_jsonl_dataset(input_path):
    """
    Reads a JSONL file, removes exact duplicate rows, and saves it to a new JSONL file.
    """
    output_path = "data/dataset/parsmap_clean_dataset.jsonl"
    print(f"Loading dataset from: {input_path}")
    
    df = pd.read_json(input_path, lines=True)
    initial_len = len(df)
    
    df_cleaned = df.drop_duplicates()
    final_len = len(df_cleaned)
    
    removed_count = initial_len - final_len
    print(f"Removed {removed_count} exact duplicate rows.")
    print(f"Dataset size changed from {initial_len:,} to {final_len:,} rows.")
    
    df_cleaned.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Cleaned dataset saved successfully to: {output_path}")

    return df_cleaned

if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    MODEL_NAME="google/gemma-3-4b-it"
    DATASET_PATH = "data/dataset/parsmap_dataset.jsonl"

    # ==================== STEP 1: LOAD RAW DATA ====================
    # convert_dataset_to_json()
    raw_dataset=loading_dataset(DATASET_PATH)
    # dataset_first_look(raw_dataset)

    # ==================== STEP 2: PREPARE DATA ====================
    clean_dataset=clean_jsonl_dataset(DATASET_PATH)

