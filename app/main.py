from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib

from app.schemas import (
    CustomerData,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using machine learning.",
    version="1.2.0"
)

# Load trained model and feature names
try:
    model = joblib.load("artifacts/churn_model.pkl")
    feature_names = joblib.load("artifacts/feature_names.pkl")
except Exception as e:
    raise RuntimeError(f"Could not load model artifacts: {e}")


# Allowed categorical values
ALLOWED_VALUES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}


# Numeric fields and their accepted ranges
NUMERIC_FIELDS = {
    "tenure": (0, 100),
    "MonthlyCharges": (0, 1000),
    "TotalCharges": (0, 100000)
}


def validate_customer(customer: dict):
    # Check for missing features
    missing = [
        col for col in feature_names
        if col not in customer
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    errors = []

    # Validate categorical values
    for field, allowed in ALLOWED_VALUES.items():
        if customer[field] not in allowed:
            errors.append(
                f"{field} must be one of {allowed}"
            )

    # Validate numeric values and ranges
    for field, (minimum, maximum) in NUMERIC_FIELDS.items():
        try:
            value = float(customer[field])

            if not pd.notna(value):
                raise ValueError()

            if value < minimum or value > maximum:
                errors.append(
                    f"{field} must be between {minimum} and {maximum}"
                )

        except (TypeError, ValueError):
            errors.append(
                f"{field} must be a valid number"
            )

    # Validate customer ID
    if not isinstance(customer["customerID"], str) or not customer["customerID"].strip():
        errors.append("customerID must be a non-empty string")

    if errors:
        raise ValueError(errors)

    return customer


def predict_customer(customer: dict):
    validate_customer(customer)

    # Create input DataFrame in the expected feature order
    input_df = pd.DataFrame(
        [[customer[col] for col in feature_names]],
        columns=feature_names
    )

    # Generate prediction
    prediction = int(model.predict(input_df)[0])

    # Get churn probability
    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    # Assign risk level
    if probability >= 0.70:
        risk = "High"
    elif probability >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(probability, 4),
        "risk_level": risk
    }


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: CustomerData):
    try:
        return predict_customer(request.data)

    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse
)
def predict_batch(request: BatchPredictionRequest):
    try:
        results = [
            predict_customer(customer)
            for customer in request.customers
        ]

        return {
            "predictions": results
        }

    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

