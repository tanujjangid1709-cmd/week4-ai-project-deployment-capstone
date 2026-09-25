# Week 4 - AI Project Deployment & Capstone

This project trains a Logistic Regression model on the Iris dataset, serializes it with Joblib,
and exposes a prediction API using FastAPI.

## Files
- app.py - FastAPI prediction service
- train_model.py - trains and saves the model
- iris_model.joblib - serialized trained model
- iris_dataset_week4.csv - dataset
- requirements.txt - Python dependencies

## Run locally
1. Install dependencies:
   pip install -r requirements.txt
2. Train the model if needed:
   python train_model.py
3. Start the API:
   uvicorn app:app --reload
4. Open:
   http://127.0.0.1:8000/docs

## API
POST /predict

Example JSON:
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

The API returns the predicted class and prediction probabilities.
