from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Telco Churn MLOps API")

# Load the best model we just trained
model = joblib.load('models/best_model.pkl')

# Define the data structure the API expects to receive
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {"message": "Welcome to the Telco Churn Prediction API!"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert the incoming JSON data into a Pandas DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # Return the result
    result = "Churn" if prediction == 1 else "No Churn"
    return {
        "prediction": result,
        "churn_probability": round(float(probability), 4)
    }