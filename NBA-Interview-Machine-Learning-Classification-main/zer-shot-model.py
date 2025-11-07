import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv("test_data.csv")

for col in ["Positive", "Negative", "Team", "Individual"]:
    df[col] = df[col].fillna(0).astype(int)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sentiment_labels = ["Positive", "Negative"]
focus_labels = ["Team", "Individual"]

sentiment_preds = []
focus_preds = []

print("Running zero-shot classification...")

for quote in df["quote"]:
    # sentiment prediction
    result_sentiment = classifier(quote, sentiment_labels, multi_label=False)
    sentiment_label = result_sentiment["labels"][0]
    sentiment_preds.append(1 if sentiment_label == "Positive" else 0)

    # Focus prediction
    result_focus = classifier(quote, focus_labels, multi_label=False)
    focus_label = result_focus["labels"][0]
    focus_preds.append(1 if focus_label == "Team" else 0)

# accuracy scores
true_sentiment = df["Positive"]
true_focus = df["Team"]

sentiment_acc = accuracy_score(true_sentiment, sentiment_preds)
focus_acc = accuracy_score(true_focus, focus_preds)



# === Optional: Save predictions in CSV format ===
output_df = pd.DataFrame({
    "Positive": sentiment_preds,
    "Negative": [1 - p for p in sentiment_preds],
    "Team": focus_preds,
    "Individual": [1 - p for p in focus_preds]
})
output_df.to_csv("sample_outputs/submission.csv", index=False)

