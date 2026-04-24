# 🗣️ Can a Machine Know What You're Talking About?
## Topic Classification of the Yahoo! Answers Dataset

**Course:** CSCE 676 — Data Mining  
**Dataset:** Yahoo! Answers Topic Classification — Zhang et al., 2015  
*(1.4 million training posts · 10 topic categories · real user-generated text)*

> *"Yahoo Answers was the internet's most chaotic public square — grammar optional, sincerity maximum. Can a data mining pipeline make sense of it?"*

This project builds a full NLP classification pipeline on the Yahoo Answers dataset — from exploratory data analysis through classical baselines (TF-IDF + Logistic Regression, LSA) to transformer fine-tuning (BERT, DistilBERT, ULMFiT) — answering the question: **does reading words in context matter, or is knowing which words appear enough?**

---

## 👉 Start Here: [`main_notebook.ipynb`](main_notebook.ipynb)

The main deliverable is **`main_notebook.ipynb`** — a curated, narrative-driven notebook that walks through all nine phases of the project, from environment setup and EDA through every model, cross-validation, and the final conclusions.

---

## 🎥 Project Video

**[▶ Watch the Project Walkthrough on YouTube](https://www.youtube.com/watch?v=b1BY9Ax18tY)**

---

## 🔬 Research Questions

This project is organized around three research questions, each answered by a dedicated modeling stage:

| RQ | Question | Technique | Macro F1 |
|----|----------|-----------|----------|
| **RQ1** | How far can word frequencies alone take us? | TF-IDF + Logistic Regression | **0.6653** |
| **RQ2** | Can compressing features into topics help? | Latent Semantic Analysis (SVD) | 0.6350 |
| **RQ3** | Does reading words in context close the gap? | BERT Fine-Tuning | **0.7281** |

---

## 📊 Results Summary

The best model — BERT fine-tuned on the full dataset — achieves **75.7% Macro F1**, a 9-point improvement over the TF-IDF baseline. The key finding: contextual representations (BERT) outperform word-frequency representations (TF-IDF) by 6+ points **even with 12× less training data**, and the gap widens further to 9 points when BERT is given the full dataset.

| Model | Macro F1 | Accuracy | Notes |
|-------|----------|----------|-------|
| Random Baseline | 0.100 | — | Floor |
| LSA (k=300) | 0.635 | 0.641 | Worse than TF-IDF |
| TF-IDF + LR | 0.665 | 0.671 | Strong classical baseline |
| Zero-Shot BART-MNLI | 0.610 | — | 0 task-specific examples |
| ULMFiT (AWD-LSTM) | 0.680 | 0.687 | Sequential context |
| DistilBERT | 0.720 | 0.725 | 40% smaller, near-BERT quality |
| BERT (50K examples) | 0.728 | 0.732 | 12× less data than TF-IDF |
| **BERT (Full Dataset)** | **0.757** | **0.763** | **Best overall** |

**Central conclusion:** Context matters — significantly and measurably. The 6-point Macro F1 gain from Stage 1 to Stage 3 represents thousands of correctly routed posts per day in a production system. See `main_notebook.ipynb` Phase 9 for the full analysis.

---

## 📁 Repo Structure

```
yahoo-answers-nlp-project/
│
├── main_notebook.ipynb          # 👈 Main deliverable — start here
├── requirements.txt             # Full dependency list (exported from Colab)
├── README.md                    # This file
│
├── checkpoints/
│   ├── checkpoint_1.ipynb       # Checkpoint 1: Dataset selection & initial EDA
│   └── checkpoint_2.ipynb       # Checkpoint 2: Research questions & experimental design
│
├── data/
│   └── README_data.md           # Instructions for downloading the dataset
│
└── assets/
    └── figures/                 # Output figures referenced in the notebook
```

---

## 📂 Dataset

**Name:** Yahoo Answers Topic Classification Dataset  
**Source:** [LC-John/Yahoo-Answers-Topic-Classification-Dataset](https://github.com/LC-John/Yahoo-Answers-Topic-Classification-Dataset)  
**Original paper:** Zhang et al., 2015 — *Character-level Convolutional Networks for Text Classification*

### Structure

Each record contains four fields (no header row in the raw CSV):

| Column | Content | Example |
|--------|---------|---------|
| 1 | Topic label (integer 1–10) | `5` |
| 2 | Question title | *"why doesn't an optical mouse work on glass?"* |
| 3 | Question body | *"or even on some surfaces?"* |
| 4 | Best answer | *"optical sensors need contrast..."* |

### Size

- **Training set:** ~1.4 million samples  
- **Test set:** ~60,000 samples  
- **Classes:** 10 topic categories (Society & Culture, Science & Math, Health, Education & Reference, Computers & Internet, Sports, Business & Finance, Entertainment & Music, Family & Relationships, Politics & Government)

### Downloading the Data

The dataset is too large to commit to this repo. Download it from the source above and place the files at:

```
data/
├── train.csv    # ~1.4M rows, no header
└── test.csv     # ~60K rows, no header
```

The notebook's Phase 2 (Data Loading) will auto-resolve the path when running in Google Colab with Google Drive mounted.

### Preprocessing

All preprocessing is done inside `main_notebook.ipynb` Phase 2. Key steps:
1. Assign column names (`label`, `title`, `body`, `answer`)
2. Handle embedded newlines in user posts using `quoting=csv.QUOTE_ALL`
3. Combine `title + " " + body + " " + answer` into a single `text` field
4. Strip `NaN` entries and normalize whitespace
5. Map integer labels (1–10) to human-readable class names

---

## ▶️ How to Reproduce

This project was built and run entirely in **Google Colab** with a T4 GPU.

### Quick Start

1. Open [Google Colab](https://colab.research.google.com/) and upload (or open from GitHub) `main_notebook.ipynb`
2. Go to **Runtime → Change runtime type → T4 GPU** (required for Phase 7 BERT training)
3. Mount Google Drive and place the dataset files at the path specified in Phase 1
4. Run all cells top to bottom

### Install Dependencies

Run this in the first code cell (already included in the notebook):

```python
!pip install transformers datasets accelerate scikit-learn pandas matplotlib seaborn fastai -q
```

Or install from the full requirements file:

```bash
pip install -r requirements.txt
```

### Run Order

| Step | File | Description |
|------|------|-------------|
| 1 | `checkpoints/checkpoint_1.ipynb` | Dataset exploration and selection |
| 2 | `checkpoints/checkpoint_2.ipynb` | Research question formalization |
| 3 | `main_notebook.ipynb` | Full pipeline — EDA through conclusions |

> ⚠️ **Note on Phase 8.3 (Full-Dataset BERT):** This experiment required ~3.6 hours on a T4 GPU. The results are stored in the notebook and the training code is preserved (commented out) for transparency. Re-running it interactively in a standard Colab session is not feasible without extended runtime access.

---

## 🔑 Key Dependencies

| Package | Version | Used For |
|---------|---------|---------|
| Python | 3.11 | Runtime |
| pandas | 2.2.0 | Data loading and manipulation |
| numpy | 1.26.4 | Numerical operations |
| scikit-learn | 1.4.1 | TF-IDF, Logistic Regression, LSA, CV |
| matplotlib | 3.8.0 | Visualizations |
| seaborn | 0.13.2 | Heatmaps and EDA plots |
| transformers | 4.40.0 | BERT, DistilBERT, BART (HuggingFace) |
| datasets | 2.19.0 | HuggingFace dataset utilities |
| accelerate | 0.30.0 | HuggingFace Trainer GPU support |
| torch | 2.2.0 | PyTorch backend for all neural models |
| fastai | 2.7.14 | ULMFiT / AWD-LSTM |

The complete list of every package and version from the Colab session lives in [`requirements.txt`](requirements.txt).

---

## 🗂️ Checkpoint Notebooks

| Notebook | Contents |
|----------|----------|
| [`checkpoints/checkpoint_1.ipynb`](checkpoints/checkpoint_1.ipynb) | Three candidate datasets evaluated; Yahoo Answers selected; initial EDA and data quality assessment |
| [`checkpoints/checkpoint_2.ipynb`](checkpoints/checkpoint_2.ipynb) | Research questions formalized; experimental framework designed; hypotheses stated with EDA support |

---

## 🚀 Project Overview

Online question-and-answer platforms process millions of posts every day. Routing each post to the correct topic category enables better search, smarter recommendations, and effective content moderation. Done manually, this is impossibly slow at scale; done naively with keyword matching, it fails on informal language.

This project builds and compares a full suite of NLP approaches on the Yahoo Answers Topic Classification Dataset — one of the largest public benchmarks for multi-class text classification — testing methods from classical data mining (TF-IDF, SVD) through modern deep learning (BERT, DistilBERT, ULMFiT, zero-shot BART).

The central experimental design uses **one stratified 80/20 train/validation split for all models** to ensure fair comparison, with 5-fold cross-validation to validate single-split estimates.

---

## 🔮 Future Scope

- **Topic Modeling:** LDA, BERTopic for contextual topic discovery
- **Semantic Representation Learning:** Sentence embeddings (SBERT), contrastive learning
- **Graph-Based Learning:** Knowledge graph construction, graph neural networks
- **Scalability:** Distributed training, large-scale embedding indexing, efficient inference pipelines
