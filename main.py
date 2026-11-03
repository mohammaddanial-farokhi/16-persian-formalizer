import unsloth
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset, DatasetDict
from collections import Counter
import os
from pathlib import Path
from transformers import  AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template,train_on_responses_only
import torch
from trl import SFTConfig,SFTTrainer
from transformers.trainer_utils import get_last_checkpoint

# =====================================================
# 1. DATA LOADING & ANALYSIS
# =====================================================
def loading_dataset(dataset_path):
    """
    Reads an Excel file and converts it directly to a Hugging Face DatasetDict.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"File path not found: {dataset_path}")

    print(f"Loading Excel file from: {dataset_path}")
    
    df = pd.read_excel(dataset_path)
    hf_dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset_dict = DatasetDict({"train": hf_dataset})

    print(f"Dataset loaded successfully as DatasetDict. Number of rows: {len(dataset_dict['train']):,}")
    
    return dataset_dict


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
def clean_dataset(dataset_dict):
    """
    Takes a DatasetDict, removes exact duplicate rows from all splits,
    """
    print("\n--- Removing Duplicates ---")
    
    if not isinstance(dataset_dict, DatasetDict):
        raise TypeError("Input must be a Hugging Face DatasetDict.")

    cleaned_dict = {}

    for split_name, split_data in dataset_dict.items():
        print(f"Processing split: '{split_name}'")
        
        df = split_data.to_pandas()
        initial_len = len(df)
        
        df_cleaned = df.drop_duplicates()
        final_len = len(df_cleaned)
        
        removed_count = initial_len - final_len
        print(f"    - Removed {removed_count} exact duplicate rows. (Size: {initial_len:,} -> {final_len:,})")
        
        clean_split = Dataset.from_pandas(df_cleaned, preserve_index=False)
        cleaned_dict[split_name] = clean_split

    return DatasetDict(cleaned_dict)


def split_dataset(dataset_dict, val_size=0.1, seed=42):
    """
    Takes a DatasetDict, extracts the 'train' split, splits it into 
    train and validation sets, and returns the two Dataset objects.
    """
    print("\n--- Splitting Dataset ---")
    
    if "train" not in dataset_dict:
        raise ValueError("DatasetDict must contain a 'train' split.")
    
    raw_train = dataset_dict["train"]
    total_rows = len(raw_train)
    
    split_data = raw_train.train_test_split(test_size=val_size, seed=seed)
    
    train_data = split_data["train"]
    val_data = split_data["test"]
    
    print(f"Split complete:")
    print(f"    - Train rows: {len(train_data):,} ({100 * len(train_data)/total_rows:.1f}%)")
    print(f"    - Validation rows: {len(val_data):,} ({100 * len(val_data)/total_rows:.1f}%)")
    
    return train_data, val_data

# =====================================================
# 3. MODEL LOADING & QLORA SETUP
# =====================================================
def load_model_and_tokenizer(model_name, max_seq_length=512):
    """
    Loads a model using Unsloth for fast QLoRA fine-tuning.
    """
    print(f"Loading model with Unsloth: {model_name}")


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length, 
        dtype=torch.bfloat16,           
        load_in_4bit=True, 
        local_files_only=True,            
    )


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    model = FastLanguageModel.get_peft_model(
        model,
        r=16, # رنک LoRA
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.0, 
        bias="none",
        use_gradient_checkpointing="unsloth", 
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    print("✅ Model loaded with Unsloth and ready for QLoRA!")
    model.print_trainable_parameters() 

    return model, tokenizer


# =====================================================
# 4. MODEL PRE-TRAIN
# =====================================================
def prepare_dataset_for_training(dataset, tokenizer):
    """
    Formats the dataset into a conversation structure and applies the Gemma 3 chat template.
    """
    print("\n--- Preparing Dataset for Training ---")
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )

    def format_prompts(examples):
        texts = []
        for informal, formal in zip(examples['inFormalForm'], examples['formalForm']):
            messages = [
                {"role": "user", "content": f"متن محاوره‌ای زیر را به متن رسمی تبدیل کن:\n{informal}"},
                {"role": "assistant", "content": formal}
            ]
            
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            texts.append(text)
            
        return {"text": texts}
    

    formatted_dataset = dataset.map(format_prompts, batched=True)
    print("Dataset formatted successfully!")
    
    return formatted_dataset, tokenizer


# =====================================================
# 5. TRAIN
# =====================================================
def start_training(model, tokenizer, train_dataset, valid_dataset):
    print("\n--- Starting Fine-Tuning (Optimized with Unsloth) ---")
    
    # 1. Define paths
    checkpoint_dir = "data/model/checkpoints"
    final_model_dir = "data/model/final_model"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)

    # 2. Check for existing checkpoints
    last_checkpoint = None
    if os.path.exists(checkpoint_dir):
        last_checkpoint = get_last_checkpoint(checkpoint_dir)
        
    if last_checkpoint is not None:
        print(f"Found previous checkpoint! Resuming from: {last_checkpoint}")
    else:
        print("No checkpoint found. Starting from scratch.")

    # 3. Optimized configuration for RTX 4060 8GB + Unsloth
    training_args = SFTConfig(
        output_dir=checkpoint_dir,
        dataset_text_field="text",
        max_seq_length=512,                 # Sentences are short, no need for a larger number
        per_device_train_batch_size=2,      # Since we use train_on_responses_only, we can set batch size to 2 (change to 1 if OOM occurs)
        gradient_accumulation_steps=4,      # Equivalent to batch size 8
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,                          # RTX 4060 supports this
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=1500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,                 # Exactly as requested: keep only 3 checkpoints
        optim="adamw_8bit",                 # 8-bit optimizer to save memory
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        packing=False,                      # For this dataset, it's better to disable packing
    )

    # 4. Create SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
    )

    # 5. Unsloth magic: train only on responses
    # This tells the model to only learn the text after <start_of_turn>model
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    # 6. Start training (with Resume capability)
    print("\nTraining started... Check your terminal for loss updates.")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # 7. Save the final model
    print("\n--- Saving Final Model ---")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"✅ Final model saved successfully to: {final_model_dir}")

    return model, tokenizer

if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    MODEL_NAME="unsloth/gemma-3-4b-it-unsloth-bnb-4bit" 
    DATASET_PATH = "data/dataset/trainset_40k_onlySent.xlsx"

    # ==================== STEP 1: LOAD RAW DATA ====================
    # convert_dataset_to_json()
    raw_dataset=loading_dataset(DATASET_PATH)
    # dataset_first_look(raw_dataset)

    # ==================== STEP 2: PREPARE DATA ====================
    cleaned_dataset =clean_dataset(raw_dataset)
    train_dataset,valid_dataset=split_dataset(cleaned_dataset)

    # ==================== STEP 3: LOAD MODEL ====================
    base_model,base_tokenizer=load_model_and_tokenizer(MODEL_NAME)

    # ==================== STEP 4: PRE-TRAIN ====================
    train_dataset_formatted, formatted_tokenizer=prepare_dataset_for_training(train_dataset, base_tokenizer)
    valid_dataset_formatted, _ = prepare_dataset_for_training(valid_dataset, base_tokenizer)

    # ==================== STEP 5: TRAIN ====================
    SFT_model,SFT_tokenizer=start_training(base_model, base_tokenizer, train_dataset_formatted, valid_dataset_formatted,)