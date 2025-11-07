NBA/WNBA INTERVIEW CLASSIFICATION - NLP PROJECT

This project performs multi-label classification of player and coach interview quotes from NBA and WNBA post-game and draft events. 
Each quote is classified on two binary axes:

1. Sentiment: Positive or Negative
2. Focus: Team or Individual

Quotes are sourced from asapsports.com and provided in a structured CSV format with labeled training data and unlabeled test data.

--------------------------------------------------------------------------------
FILE DESCRIPTIONS AND HOW TO RUN EACH MODEL

1. baseline_model.py
- Predicts "Positive" and "Team" for every quote (hardcoded baseline)
- Run using:
    python baseline_model.py
- Output: submission.csv file in sample_outputs/ with accuracy printed in console

2. Lr_model.py
- Trains logistic regression models using TF-IDF features (one for Sentiment, one for Focus)
- Run using:
    python Lr_model.py
- Output: submission.csv file in sample_outputs/ with accuracy printed in console

3. zer-shot-model.py
- Uses zero-shot classification with facebook/bart-large-mnli (no training)
- Classifies based on prompted labels using Hugging Face pipeline
- Run using:
    python zer-shot-model.py
- Output: submission.csv file in sample_outputs/ with accuracy printed in console

4. distilBERT_finetuned_model.py
- Fine-tunes DistilBERT for both Sentiment and Focus classification using Hugging Face Trainer API
- Trains on 100 rows and validates on 20 rows
- Run using:
    python distilBERT_finetuned_model.py
- Output: 
    - submission.csv file in sample_outputs/
    - Accuracy for Sentiment and Focus printed in console

--------------------------------------------------------------------------------
REQUIREMENTS

Install the following Python libraries using pip:

- pandas
- numpy
- scikit-learn
- torch
- transformers
- datasets

Command to install all:
pip install pandas numpy scikit-learn torch transformers datasets

--------------------------------------------------------------------------------
SUBMISSION FORMAT

All models output predictions in the following CSV format:

Positive,Negative,Team,Individual
1,0,1,0
0,1,0,1
1,0,0,1
...


