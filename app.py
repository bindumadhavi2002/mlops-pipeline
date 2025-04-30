from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

# Create FastAPI app
app = FastAPI()

# Define request body schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define prediction endpoint
@app.post("/predict")
def predict_species(input_data: IrisInput):
    features = np.array([[input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]])
    prediction = model.predict(features)
    return {"predicted_class": int(prediction[0])}
