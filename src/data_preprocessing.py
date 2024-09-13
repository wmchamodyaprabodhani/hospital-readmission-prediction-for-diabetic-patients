import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_and_preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)

    # Features and target
    X = data[['Age', 'Gender', 'HbA1c_Level']]
    y = data['Readmitted']

    # Preprocessing: One-hot encode 'Gender', scale 'Age' and 'HbA1c_Level'
    preprocessor = ColumnTransformer(transformers=[(
        'num', StandardScaler(),
        ['Age', 'HbA1c_Level']), ('cat', OneHotEncoder(), ['Gender'])])

    # Split into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Apply transformations
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Get the feature names after one-hot encoding and scaling
    feature_names_num = ['Age', 'HbA1c_Level']
    feature_names_cat = preprocessor.named_transformers_[
        'cat'].get_feature_names_out(['Gender'])
    feature_names = list(feature_names_num) + list(feature_names_cat)

    # Convert the transformed arrays back to DataFrames with proper column names
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    # Save preprocessed data
    X_train_df.to_csv('./data/X_train_preprocessed.csv', index=False)
    X_test_df.to_csv('./data/X_test_preprocessed.csv', index=False)
    pd.DataFrame(y_train).to_csv('./data/y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv('./data/y_test.csv', index=False)

    print("Data preprocessing complete and saved.")
    return X_train_df, X_test_df, y_train, y_test
