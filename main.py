# main.py

from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from sklearn.model_selection import train_test_split

# Filepath to dataset
filepath = './data/hospital_data.csv'

# 1. Load and preprocess the data
X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)

# 2. Train the model
model = train_model()

# 3. Evaluate the model
evaluate_model(model, X_test, y_test)

print("Pipeline executed successfully!")
