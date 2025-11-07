import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


train_df = pd.read_csv("training_data.csv")
test_df = pd.read_csv("test_data.csv")


for col in ["Positive", "Negative", "Team", "Individual"]:
    train_df[col] = train_df[col].fillna(0).astype(int)
    test_df[col] = test_df[col].fillna(0).astype(int)

X_train = train_df["quote"]
X_test = test_df["quote"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

y_train_sentiment = train_df["Positive"]  # 1 = Positive, 0 = Negative
sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X_train_vec, y_train_sentiment)
y_pred_sentiment = sentiment_model.predict(X_test_vec)

y_train_focus = train_df["Team"]  # 1 = Team, 0 = Individual
focus_model = LogisticRegression(max_iter=1000)
focus_model.fit(X_train_vec, y_train_focus)
y_pred_focus = focus_model.predict(X_test_vec)

#accuracy 
y_test_sentiment = test_df["Positive"]
y_test_focus = test_df["Team"]

sentiment_acc = accuracy_score(y_test_sentiment, y_pred_sentiment)
focus_acc = accuracy_score(y_test_focus, y_pred_focus)


submission_df = pd.DataFrame({
    "Positive": y_pred_sentiment,
    "Negative": 1 - y_pred_sentiment,
    "Team": y_pred_focus,
    "Individual": 1 - y_pred_focus
})
submission_df.to_csv("sample_outputs/submission.csv", index=False)
print("Predictions saved to sample_outputs/submission.csv")
