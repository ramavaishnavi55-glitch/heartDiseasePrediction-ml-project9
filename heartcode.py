
# Heart Disease Prediction using Decision Tree

# ==============================
# Import Libraries
# ==============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("heart.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ==============================
# Check Missing Values
# ==============================

print("\nMissing Values:")
print(df.isnull().sum())

# ==============================
# Features and Target
# ==============================

X = df.drop("target", axis=1)

y = df["target"]

# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==============================
# Build Decision Tree Model
# ==============================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

# ==============================
# Train Model
# ==============================

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ==============================
# Predictions
# ==============================

y_pred = model.predict(X_test)

# ==============================
# Evaluation
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# Sample Prediction
# ==============================

sample_patient = [[
    52,  # age
    1,   # sex
    0,   # cp
    125, # trestbps
    212, # chol
    0,   # fbs
    1,   # restecg
    168, # thalach
    0,   # exang
    1.0, # oldpeak
    2,   # slope
    2,   # ca
    3    # thal
]]

prediction = model.predict(sample_patient)

if prediction[0] == 1:
    print("\nPrediction: Heart Disease Detected")
else:
    print("\nPrediction: No Heart Disease")

# ==============================
# Feature Importance
# ==============================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# ==============================
# Save Model
# ==============================

import joblib

joblib.dump(
    model,
    "heart_disease_model.pkl"
)

print("\nModel Saved Successfully!")
