# Persian Formalizer (Gemma-3-4B + QLoRA)

![Python](https://img.shields.io/badge/Python-3.12.16-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.x-yellow.svg)
![Unsloth](https://img.shields.io/badge/Unsloth-QLoRA-orange.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio-ff69b4.svg)
![Task](https://img.shields.io/badge/Task-Text%20Style%20Transfer-success.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 🧠 About The Project

A complete end-to-end pipeline to fine-tune the **Gemma-3-4B-IT** language model for converting informal Persian (Farsi) conversational text into formal, written Persian. 

This project utilizes **Unsloth** and **TRL** for highly efficient Supervised Fine-Tuning (SFT) using **QLoRA** (4-bit quantization). By leveraging Unsloth's optimized kernels, the training is extremely fast and memory-efficient, making it accessible on consumer hardware. The final model is served via a beautifully designed, responsive **Gradio** web interface with full RTL (Right-to-Left) support.

**Result:** The fine-tuned model demonstrates exceptional performance in preserving semantic meaning while transforming the tone, achieving a **BERTScore of 0.96** and a **BLEU score of 72.99**.

---

## 🎯 Key Features

1. **Efficient Fine-Tuning:** Uses Unsloth's `FastLanguageModel` and QLoRA to drastically reduce VRAM usage and speed up training.
2. **Prompt Engineering & Chat Templates:** Formats data using the official Gemma-3 chat template for optimal instruction-following.
3. **Response-Only Training:** Utilizes `train_on_responses_only` to ensure the model only learns from the assistant's formal outputs, masking the user prompts and saving compute.
4. **Comprehensive Evaluation:** Computes BLEU, METEOR, chrF, and BERTScore (using `bert-base-multilingual-cased`) for robust evaluation.
5. **Premium Web UI:** A custom Gradio interface with modern glassmorphism CSS, Vazirmatn font, fully responsive RTL layout, and side-by-side input/output panels.
6. **Checkpoint Resume:** Automatically detects and resumes from the latest checkpoint if the training is interrupted.

---

## 📂 Project Structure

```text
.
├── data/
│   ├── dataset/
│   │   ├── trainset_40k_onlySent.xlsx   # Raw dataset (Excel format)
│   │   └── evaluation_dataset.csv       # Generated predictions & targets
│   │
│   └── model/
│       ├── checkpoints/                 # SFT training checkpoints
│       └── final_model/                 # Final fine-tuned LoRA adapter
│
├── main.py                              # Main training & evaluation pipeline
├── app.py                               # Gradio web UI application
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## 📊 Dataset

**Format:** Parallel corpus in an Excel file (`trainset_40k_onlySent.xlsx`).
*   **Columns:** 
    *   `inFormalForm`: The conversational/colloquial Persian text.
    *   `formalForm`: The formal, written Persian equivalent.
*   The script automatically cleans duplicates, handles null values, and splits the dataset into Train (90%) and Validation (10%) sets.

---

## ⚙️ Training Configuration

| Parameter | Value | Note |
| :--- | :--- | :--- |
| **Base Model** | `unsloth/gemma-3-4b-it-unsloth-bnb-4bit` | Pre-quantized 4-bit model |
| **Quantization** | 4-bit (QLoRA) | To fit into limited VRAM |
| **LoRA `r`** | 16 | Rank of LoRA matrices |
| **LoRA `alpha`** | 32 | Scaling factor |
| **Max Seq Length** | 512 (Load) / 256 (Train) | Optimized for sentence-level conversion |
| **Batch Size** | 2 | With gradient accumulation steps = 5 |
| **Learning Rate** | 2e-4 | Linear scheduler |
| **Epochs** | 2 | Validated to prevent overfitting |
| **Optimizer** | `adamw_8bit` | 8-bit AdamW for memory efficiency |

---

## 📈 Evaluation Results

The model was evaluated on a hold-out validation set of conversational data.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **BERTScore F1** | **0.96** | Semantic similarity (Multilingual BERT) |
| **chrF** | **88.79** | Character n-gram F-score |
| **METEOR** | **87.92** | Vocabulary & synonym overlap |
| **BLEU** | **72.99** | Exact n-gram overlap |
| **Final Eval Loss** | **0.248** | Evaluated at Epoch 2 |
| **Final Train Loss** | **0.02231** | Training completed in 2375 seconds |

---

## 🚀 Usage

### 1. Install Dependencies
Ensure you have Python 3.12.16 and CUDA installed. Then install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Training & Evaluation Pipeline
To start fine-tuning the model from scratch or resume from the last checkpoint, and subsequently run the evaluation metrics:
```bash
python train.py
```
*(Note: The evaluation script generates a `evaluation_dataset.csv` with predictions, then runs BLEU, METEOR, chrF, and BERTScore.)*

### 3. Web Application (Local)
To launch the Gradio UI and interact with the fine-tuned model:
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser. The interface allows you to input informal text, see the formal conversion, and view pre-calculated metric cards.

---

## ⚠️ Known Limitations

*   **Domain Specificity:** While the model handles general conversations well, performance may degrade on highly technical, scientific, or archaic Persian texts if such examples are not present in the training data.
*   **Output Length:** The model is constrained by `max_new_tokens=128` during inference. Extremely long input paragraphs should be chunked before feeding to the model.
*   **Hardware:** The Gradio app expects a CUDA-enabled GPU for fast inference (`load_in_4bit=True`). Running purely on CPU may result in slow generation times.

---

## 📜 License
This project is licensed under the MIT License - free to use, modify, and distribute.