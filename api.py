from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('models/credit_risk_model.pkl')

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"result": int(prediction[0])}