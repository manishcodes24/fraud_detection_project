import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the data into train and test sets.
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]

    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """
    Train a random forest classifier.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    """
    Save trained model using joblib.
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")
