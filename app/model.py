# model.py
import joblib
import os
import numpy as np

# Load model and vectorizer (adjust path as needed)
MODEL_PATH = os.path.join("model", "classifier.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_response_type(context_text: str) -> str:
    """
    Takes in patient context and returns predicted response type.
    """
    vect_input = vectorizer.transform([context_text])
    prediction = model.predict(vect_input)
    return prediction[0]
