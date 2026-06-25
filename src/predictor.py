import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_disease_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "heart.csv")


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    data = pd.read_csv(DATA_PATH)
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def predict_heart_disease(input_data):
    model = load_model()
    df = pd.DataFrame([input_data], columns=[
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return int(prediction), float(probability)
