# 🏀 NBA/WNBA Interview Classification — NLP Project

This project builds a **multi-label text classification pipeline** that analyzes post-game and draft-event interviews from NBA and WNBA players and coaches.
Each interview quote is classified along two binary axes:

* **Sentiment:** Positive 🟢 or Negative 🔴
* **Focus:** Team-oriented 🤝 or Individual-focused 👤

The dataset is sourced from [asapsports.com](https://asapsports.com), structured in CSV format with labeled training data and unlabeled test data.

---

## 📊 Project Overview

This repository contains multiple modeling approaches — from baselines to transformer-based models — to compare performance and evaluate trade-offs between simplicity and accuracy.

| Model                           | Technique                    | Description                                                                              |
| ------------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------- |
| `baseline_model.py`             | Hardcoded baseline           | Always predicts “Positive” and “Team” for benchmarking.                                  |
| `lr_model.py`                   | Logistic Regression (TF-IDF) | Trains two independent models (Sentiment, Focus) using scikit-learn.                     |
| `zero_shot_model.py`            | Zero-Shot Classification     | Uses `facebook/bart-large-mnli` to classify without training via Hugging Face pipelines. |
| `distilBERT_finetuned_model.py` | Transformer Fine-Tuning      | Fine-tunes **DistilBERT** on custom labels using the Hugging Face Trainer API.           |

---

## ⚙️ How to Run

Clone the repo and navigate to the project directory:

```bash
git clone https://github.com/Rehmanmohammed03/NBA-Interview-Sentiment-Classifier.git
cd NBA-Interview-Sentiment-Classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

*(or manually install the listed libraries below)*

Run a specific model:

```bash
# Baseline
python baseline_model.py

# Logistic Regression
python lr_model.py

# Zero-Shot BART
python zero_shot_model.py

# Fine-tuned DistilBERT
python distilBERT_finetuned_model.py
```

Each script outputs:

* `submission.csv` → stored in the `sample_outputs/` folder
* Accuracy metrics printed to the console

---

## 🧰 Requirements

```
pandas  
numpy  
scikit-learn  
torch  
transformers  
datasets
```

Install all at once:

```bash
pip install pandas numpy scikit-learn torch transformers datasets
```

---

## 📁 Submission Format

All models produce predictions in a CSV with the following columns:

| Positive | Negative | Team | Individual |
| -------- | -------- | ---- | ---------- |
| 1        | 0        | 1    | 0          |
| 0        | 1        | 0    | 1          |
| 1        | 0        | 0    | 1          |

Each row corresponds to one interview quote.

---

## 📈 Future Improvements

* Integrate **Airflow or Prefect** to automate training/evaluation pipelines
* Expand dataset with more sports domains for cross-league generalization
* Log results to a central database for model comparison and dashboarding


Would you like me to create a small **`requirements.txt`** file next (so you can just push it with your code)? It’ll make the `pip install -r requirements.txt` step work immediately.
