import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

train_df = pd.read_csv("training_data.csv")
test_df = pd.read_csv("test_data.csv")


test_df["pred_Positive"] = 1
test_df["pred_Negative"] = 0
test_df["pred_Team"] = 1
test_df["pred_Individual"] = 0


for col in ["Positive", "Negative", "Team", "Individual"]:
    test_df[col] = test_df[col].fillna(0).astype(int)


sentiment_true = test_df[["Positive", "Negative"]].values
sentiment_pred = test_df[["pred_Positive", "pred_Negative"]].values
focus_true = test_df[["Team", "Individual"]].values
focus_pred = test_df[["pred_Team", "pred_Individual"]].values

sentiment_accuracy = accuracy_score(sentiment_true[:, 0], sentiment_pred[:, 0])
focus_accuracy = accuracy_score(focus_true[:, 0], focus_pred[:, 0])


