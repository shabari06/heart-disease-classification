from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from pathlib import Path

# Create FastAPI application
app = FastAPI(title="Heart Disease Classification API")

# Locate the model relative to this file
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "heart_model.pkl"


# Load the trained model
with open(model_path, "rb") as file:
    model = pickle.load(file)


# Define input features
class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Heart Disease Classification API is running!"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: HeartData):

    # Convert input into a DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    # Match the feature order used during training
    if hasattr(model, "feature_names_in_"):
        input_data = input_data[model.feature_names_in_]

    # Predict using the saved model
    prediction = int(model.predict(input_data)[0])

    # Prepare the response
    if prediction == 1:
        result = "Heart disease class (1)"
    else:
        result = "No heart disease class (0)"

    return {
        "prediction": prediction,
        "result": result,
        "message": "Educational prediction only, not a medical diagnosis."
    }