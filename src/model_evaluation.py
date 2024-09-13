# src/model_evaluation.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Adding zero_division=0 to suppress warnings
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix visualization
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC-AUC score
    roc_auc = roc_auc_score(
        pd.get_dummies(y_test)['Yes'],
        model.predict_proba(X_test)[:, 1])
    print(f"ROC-AUC Score: {roc_auc:.2f}")
