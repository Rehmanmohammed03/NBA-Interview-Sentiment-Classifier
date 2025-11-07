import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os

class DistilBERTClassifier:
    def __init__(self, model_name, label_list, output_path):
        self.label_list = label_list
        self.model_path = output_path
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.label_encoder = LabelEncoder().fit(label_list)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label_list),
            id2label={i: l for i, l in enumerate(label_list)},
            label2id={l: i for i, l in enumerate(label_list)},
        )

    def tokenize(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def preprocess(self, X, y):
        df = pd.DataFrame({"text": X, "label": y})
        df["label"] = self.label_encoder.transform(df["label"])
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(self.tokenize, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        return dataset

    def train(self, X_train, y_train, X_val, y_val):
        train_dataset = self.preprocess(X_train, y_train)
        val_dataset = self.preprocess(X_val, y_val)

        args = TrainingArguments(
            output_dir=self.model_path,
            evaluation_strategy="epoch",
            save_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            report_to="none",
            fp16=torch.cuda.is_available()
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=lambda p: {
                "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))
            }
        )
        trainer.train()
        self.trainer = trainer

    def predict(self, X_test):
        df = pd.DataFrame({"text": X_test})
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(self.tokenize, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        preds = self.trainer.predict(dataset).predictions
        pred_ids = np.argmax(preds, axis=1)
        return self.label_encoder.inverse_transform(pred_ids)


os.makedirs("sample_outputs", exist_ok=True)

train_df = pd.read_csv("training_data.csv")
test_df = pd.read_csv("test_data.csv")

X_test = test_df["quote"]

train_subset = train_df.iloc[:20]
val_subset = train_df.iloc[21:31]

X_train = train_subset["quote"]
X_val = val_subset["quote"]

y_train_sentiment = train_subset["Positive"].apply(lambda x: "Positive" if x == 1 else "Negative")
y_val_sentiment = val_subset["Positive"].apply(lambda x: "Positive" if x == 1 else "Negative")

sentiment_model = DistilBERTClassifier("distilbert-base-uncased", ["Positive", "Negative"], "./sentiment_model")
sentiment_model.train(X_train, y_train_sentiment, X_val, y_val_sentiment)
y_pred_sentiment = sentiment_model.predict(X_test)

y_train_focus = train_subset["Team"].apply(lambda x: "Team" if x == 1 else "Individual")
y_val_focus = val_subset["Team"].apply(lambda x: "Team" if x == 1 else "Individual")

focus_model = DistilBERTClassifier("distilbert-base-uncased", ["Team", "Individual"], "./focus_model")
focus_model.train(X_train, y_train_focus, X_val, y_val_focus)
y_pred_focus = focus_model.predict(X_test)

submission_df = pd.DataFrame({
    "Positive": (y_pred_sentiment == "Positive").astype(int),
    "Negative": (y_pred_sentiment == "Negative").astype(int),
    "Team": (y_pred_focus == "Team").astype(int),
    "Individual": (y_pred_focus == "Individual").astype(int)
})
submission_df.to_csv("sample_outputs/submission.csv", index=False)
print("Predictions saved to sample_outputs/submission.csv")
