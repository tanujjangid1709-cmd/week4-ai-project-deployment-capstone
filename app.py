from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="Iris Prediction API",
    description="Week 4 AI Project Deployment and Capstone",
    version="1.0.0"
)

model = joblib.load("iris_model.joblib")
class_names = ["setosa", "versicolor", "virginica"]

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Iris Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: IrisInput):
    values = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])
    prediction = int(model.predict(values)[0])
    probability = model.predict_proba(values)[0].tolist()

    return {
        "prediction": prediction,
        "class_name": class_names[prediction],
        "probabilities": probability
    }
