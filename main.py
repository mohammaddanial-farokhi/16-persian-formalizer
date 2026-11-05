# =====================================================================
# TABLE OF CONTENTS
# =====================================================================
# 1. IMPORTS & ENVIRONMENT SETUP
# 2. DATA LOADING & ANALYSIS
# 3. DATA PREPROCESSING (Clean & Split)
# 4. MODEL LOADING & QLORA SETUP
# 5. DATASET FORMATTING (Prompt Engineering)
# 6. MODEL TRAINING (SFT)
# 7. MODEL EVALUATION (Generation & Metrics)
# 8. MAIN EXECUTION PIPELINE
# =====================================================================

# =====================================================================
# 1. IMPORTS & ENVIRONMENT SETUP
# =====================================================================
import os
# Set Unsloth env var before importing it
os.environ["UNSLOTH_MOE_DISABLE_AUTOTUNE"] = "1" 

import gc
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# Hugging Face & Evaluation
from datasets import Dataset, load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
import evaluate

# Unsloth & TRL
import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTConfig, SFTTrainer


# =====================================================================
# 2. DATA LOADING & ANALYSIS
# =====================================================================
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
        n_empty = sum(1 for x in col_data[col] if x is None or len(str(x).strip()) == 0)
        pct = 100 * n_empty / len(data) if len(data) else 0
        status = "OK  " if n_empty == 0 else "WARN"
        print(f"    [{status}] {col}: {n_empty:,} empty ({pct:.2f}%)")

    # ---------- 4) Unique values ----------
    print("\n[4] UNIQUE VALUES PER COLUMN")
    for col in data.column_names:
        extracted_list = [str(x) for x in col_data[col]]
        n_unique = len(set(extracted_list))
        if n_unique <= 20:
            print(f"    - {col}: {n_unique} unique values")
            for val, count in Counter(extracted_list).most_common():
                pct = 100 * count / len(data)
                print(f"        * {val!r}: {count:,} ({pct:.1f}%)")
        else:
            print(f"    - {col}: {n_unique:,} unique values")
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
        print(f"    - Formal is longer in:  {(diff > 0).sum():,} rows ({(diff > 0).mean()*100:.1f}%)")
        print(f"    - Informal is longer in:{(diff < 0).sum():,} rows ({(diff < 0).mean()*100:.1f}%)")
        print(f"    - Equal length:         {(diff == 0).sum():,} rows ({(diff == 0).mean()*100:.1f}%)")

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


# =====================================================================
# 3. DATA PREPROCESSING (Clean & Split)
# =====================================================================
def clean_dataset(dataset_dict):
    """
    Takes a DatasetDict, removes exact duplicate rows from all splits.
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


# =====================================================================
# 4. MODEL LOADING & QLORA SETUP
# =====================================================================
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
        r=16, # LoRA rank
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


# =====================================================================
# 5. DATASET FORMATTING (Prompt Engineering)
# =====================================================================
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


# =====================================================================
# 6. MODEL TRAINING (SFT)
# =====================================================================
def start_training(model, tokenizer, train_dataset, valid_dataset):
    print("\n--- Starting Fine-Tuning (Optimized with Unsloth) ---")
    
    checkpoint_dir = "data/model/checkpoints"
    final_model_dir = "data/model/final_model"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)

    last_checkpoint = None
    if os.path.exists(checkpoint_dir):
        last_checkpoint = get_last_checkpoint(checkpoint_dir)
        
    if last_checkpoint is not None:
        print(f"Found previous checkpoint! Resuming from: {last_checkpoint}")
    else:
        print("No checkpoint found. Starting from scratch.")

    training_args = SFTConfig(
        output_dir=checkpoint_dir,
        dataset_text_field="text",
        max_seq_length=256,                 
        per_device_train_batch_size=2,      
        gradient_accumulation_steps=5,      
        per_device_eval_batch_size=1,       
        eval_accumulation_steps=4,          
        warmup_steps=10,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,                          
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=1500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,                 
        optim="adamw_8bit",                 
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        packing=False,                      
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    print("\nTraining started... Check your terminal for loss updates.")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    print("\n--- Saving Final Model ---")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"✅ Final model saved successfully to: {final_model_dir}")

    return model, tokenizer


# =====================================================================
# 7. MODEL EVALUATION (Generation & Metrics)
# =====================================================================
def create_evaluation_dataset(model, tokenizer, valid_dataset, num_samples=2000, batch_size=16):
    """
    Optimized function to generate predictions in batches for much faster inference.
    """
    csv_path = "data/dataset/evaluation_dataset.csv"

    if os.path.exists(csv_path):
        print(f"\n✅ Found existing dataset at '{csv_path}'. Loading it...")
        return pd.read_csv(csv_path)

    print(f"\n--- Generating Predictions for {num_samples} samples (Batch Size: {batch_size}) ---")
    
    FastLanguageModel.for_inference(model)
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_data = {
        "informal_text": [],
        "target_formal_text": [],
        "predicted_formal_text": []
    }
    
    subset = valid_dataset.select(range(min(num_samples, len(valid_dataset))))
    
    for i in tqdm(range(0, len(subset), batch_size), desc="Processing batches"):
        batch_samples = subset.select(range(i, min(i + batch_size, len(subset))))
        
        batch_informals = batch_samples['inFormalForm']
        batch_targets = batch_samples['formalForm']
        
        prompts = [
            f"<start_of_turn>user\nمتن محاوره‌ای زیر را به متن رسمی تبدیل کن:\n{inf}<end_of_turn>\n<start_of_turn>model\n" 
            for inf in batch_informals
        ]
        
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=128,       
                use_cache=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_length:]
        
        predicted_formals = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for inf, tgt, pred in zip(batch_informals, batch_targets, predicted_formals):
            eval_data["informal_text"].append(inf)
            eval_data["target_formal_text"].append(tgt)
            eval_data["predicted_formal_text"].append(pred.strip())
            
    df = pd.DataFrame(eval_data)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Evaluation dataset created and saved to '{csv_path}'!")
    return df


def calculate_metrics(evaluation_df):
    """
    This function takes the evaluation DataFrame and calculates 
    BLEU, METEOR, chrF, and BERTScore metrics.
    """
    print("\n--- Calculating Metrics ---")
    
    predictions = evaluation_df["predicted_formal_text"].tolist()
    references = evaluation_df["target_formal_text"].tolist()
    
    results = {}
    
    bleu_metric = evaluate.load("bleu")
    bleu_results = bleu_metric.compute(predictions=predictions, references=references)
    results['BLEU'] = bleu_results['bleu'] * 100 
    
    meteor_metric = evaluate.load("meteor")
    meteor_results = meteor_metric.compute(predictions=predictions, references=references)
    results['METEOR'] = meteor_results['meteor'] * 100
    
    chrf_metric = evaluate.load("chrf")
    chrf_results = chrf_metric.compute(predictions=predictions, references=references)
    results['chrF'] = chrf_results['score']
    
    print("Calculating BERTScore (this might take a minute)...")
    bertscore_metric = evaluate.load("bertscore")
    bertscore_results = bertscore_metric.compute(
        predictions=predictions, 
        references=references, 
        model_type="bert-base-multilingual-cased", 
        lang="fa" 
    )
    results['BERTScore_F1'] = sum(bertscore_results['f1']) / len(bertscore_results['f1'])
    
    print("\n" + "="*40)
    print("📊 FINAL EVALUATION RESULTS 📊")
    print("="*40)
    for metric, score in results.items():
        print(f"{metric:15}: {score:.2f}")
    print("="*40)
    
    return results


# =====================================================================
# 8. MAIN EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    
    # ==================== CONFIGURATION ====================
    MODEL_NAME = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit" 
    DATASET_PATH = "data/dataset/trainset_40k_onlySent.xlsx"
    FINAL_MODEL_PATH = "data/model/final_model" 

    # ==================== PHASE 1: DATA PREPARATION ====================
    print("\n>>> PHASE 1: DATA PREPARATION")
    raw_dataset = loading_dataset(DATASET_PATH)
    # dataset_first_look(raw_dataset) # Uncomment for initial analysis
    
    cleaned_dataset = clean_dataset(raw_dataset)
    train_dataset, valid_dataset = split_dataset(cleaned_dataset)


    # ==================== PHASE 2: MODEL TRAINING ====================
    # (Currently commented out to run Evaluation only)
    
    # print("\n>>> PHASE 2: MODEL TRAINING")
    # base_model, base_tokenizer = load_model_and_tokenizer(MODEL_NAME)
    # train_dataset_formatted, formatted_tokenizer = prepare_dataset_for_training(train_dataset, base_tokenizer)
    # valid_dataset_formatted, _ = prepare_dataset_for_training(valid_dataset, base_tokenizer)
    # valid_dataset_formatted = valid_dataset_formatted.shuffle(seed=42).select(range(2000))
    # SFT_model, SFT_tokenizer = start_training(base_model, base_tokenizer, train_dataset_formatted, valid_dataset_formatted)


    # ==================== PHASE 3: MODEL EVALUATION ====================
    print("\n>>> PHASE 3: MODEL EVALUATION")
    
    # 1. Load the fine-tuned model
    print("\nLoading Fine-Tuned Model...")
    SFT_model, SFT_tokenizer = FastLanguageModel.from_pretrained(
        model_name=FINAL_MODEL_PATH,
        max_seq_length=512,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    SFT_tokenizer = get_chat_template(SFT_tokenizer, chat_template="gemma-3")
    
    # Clear memory to avoid OOM
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Create/load evaluation dataset
    eval_df = create_evaluation_dataset(
        model=SFT_model,            
        tokenizer=SFT_tokenizer, 
        valid_dataset=valid_dataset, # Using the raw validation dataset
        num_samples=2000
    )
    
    # 3. Calculate metrics
    final_scores = calculate_metrics(eval_df)
    
    print("\n✅ Evaluation process completed successfully!")