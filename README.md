# Heart Disease Prediction Web App

A Streamlit-based web application that predicts whether a patient is at risk of heart disease using a machine learning model trained on the Heart Disease UCI dataset.

## Project Overview

This project demonstrates a complete machine learning workflow:
- Data loading and preprocessing
- Model training
- Model evaluation
- Prediction through a user-friendly web interface

The app allows users to enter patient health details and receive a prediction indicating whether the patient is at low or high risk of heart disease.

## Features

- Interactive web interface built with Streamlit
- Input form for medical parameters like age, blood pressure, cholesterol, and heart rate
- Prediction output with probability score
- Model training and persistence support
- Easy setup and local deployment

## Project Structure

```text
heart-disease-prediction/
├── app.py                 # Streamlit frontend application
├── heart.csv              # Dataset used for training
├── src/
│   ├── __init__.py
│   └── predictor.py       # Backend prediction logic and model loading
├── models/
│   └── heart_disease_model.pkl  # Trained model file (generated after first run)
└── README.md
```

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib

## Installation

1. Clone the project folder.
2. Navigate to the project directory.
3. Install the required dependencies:

```bash
pip install streamlit pandas scikit-learn joblib
```

## Running the Application

Start the Streamlit app with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How It Works

1. The app loads the dataset from heart.csv.
2. If a trained model file is missing, it trains a Decision Tree classifier.
3. The user enters medical values through the Streamlit interface.
4. The app converts the values into a DataFrame and sends them to the model.
5. The model returns a prediction and probability score.

## Model Details

The current implementation uses a Decision Tree classifier with:
- Criterion: gini
- Maximum depth: 5
- Random state: 42

## Input Features

The app expects the following features:
- age
- sex
- cp (chest pain type)
- trestbps (resting blood pressure)
- chol (cholesterol)
- fbs (fasting blood sugar)
- restecg (resting electrocardiogram results)
- thalach (maximum heart rate)
- exang (exercise-induced angina)
- oldpeak
- slope
- ca (number of major vessels)
- thal

## Output

The prediction result shows:
- Low Risk of Heart Disease
- High Risk of Heart Disease
- Probability score for the positive class

## Notes

- The trained model file is created automatically when it is not found.
- The current version uses a simple Decision Tree model for demonstration purposes.
- For production use, you may want to improve the model with more data and better evaluation.

## Future Improvements

Possible enhancements include:
- Using a Random Forest or XGBoost model
- Adding better data preprocessing and feature scaling
- Creating a more advanced UI
- Deploying the app to Streamlit Cloud, Heroku, or Azure

## License

This project is for educational and demonstration purposes.
