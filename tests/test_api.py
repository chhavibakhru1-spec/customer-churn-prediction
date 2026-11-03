
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# Sample customer data
sample_customer = {
    "customerID": "TEST001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 845.5
}


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Customer Churn Prediction API is running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_single_prediction():
    response = client.post(
        "/predict",
        json={"data": sample_customer}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in ["Churn", "No Churn"]
    assert 0 <= result["churn_probability"] <= 1
    assert result["risk_level"] in ["Low", "Medium", "High"]


def test_batch_prediction():
    response = client.post(
        "/predict/batch",
        json={
            "customers": [
                sample_customer,
                sample_customer
            ]
        }
    )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_invalid_customer():
    invalid_customer = sample_customer.copy()
    invalid_customer["gender"] = "Unknown"

    response = client.post(
        "/predict",
        json={"data": invalid_customer}
    )

    assert response.status_code == 422


def test_missing_features():
    incomplete_customer = sample_customer.copy()
    del incomplete_customer["tenure"]

    response = client.post(
        "/predict",
        json={"data": incomplete_customer}
    )

    assert response.status_code == 422
