# src/model_training.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib


def train_model():
  # Load preprocessed data
  X_train = pd.read_csv('./data/X_train_preprocessed.csv')
  y_train = pd.read_csv('./data/y_train.csv').squeeze()

  # Initialize and train the Logistic Regression model with class balancing
  model = LogisticRegression(random_state=42, class_weight='balanced')
  model.fit(X_train, y_train)

  # Save the trained model using joblib
  joblib.dump(model, './data/logistic_model.pkl')

  print("Model training complete and saved.")
  return model
