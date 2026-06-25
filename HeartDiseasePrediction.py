import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
try:
    data = pd.read_csv("heart.csv")
except FileNotFoundError:
    raise FileNotFoundError("Dataset file 'heart.csv' not found. Place the CSV file in the same folder as this script.")

print("Dataset Preview")
print("=" * 50)
print(data.head())
print("\nDataset Shape:", data.shape)
print("\nDataset Information:")
print(data.info())
print("\nDescriptive Statistics:")
print(data.describe())
print("\nMissing Values:")
print(data.isnull().sum())
print("\nTarget Distribution:")
print(data["target"].value_counts())

# Separate features and target
X = data.drop(columns=["target"])
y = data["target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# Create and train the Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))

# Save the trained model
joblib.dump(model, "heart_disease_model.joblib")
print("\nModel saved as heart_disease_model.joblib")
