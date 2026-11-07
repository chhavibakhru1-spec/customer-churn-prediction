# Customer Churn Prediction System

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to leave a company.

This project combines machine learning, model explainability, REST API development, input validation, automated testing, and Docker containerization.

## Project Objectives

- Predict customer churn using machine learning.
- Estimate the probability of customer churn.
- Categorize customers into Low, Medium, and High risk levels.
- Provide individual and batch predictions through a REST API.
- Validate customer input before generating predictions.
- Containerize the application using Docker.

## Dataset

**IBM Telco Customer Churn Dataset (Kaggle)**

The dataset contains customer information, including:
- Demographics
- Account information
- Services subscribed
- Contract details
- Monthly and total charges
- Customer churn status

## Machine Learning

**Models explored:**
- Logistic Regression
- Random Forest
- XGBoost

**Model development process:**
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Stratified train, validation, and test split
- Cross-validation
- Hyperparameter tuning
- Model evaluation
- SHAP-based model explainability

**Final selected model:** To be specified after confirming the final model-selection results.

The trained model is saved as `artifacts/churn_model.pkl`.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- FastAPI
- Pydantic
- Uvicorn
- Pytest
- Docker
- Git and GitHub

## API Features

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API home |
| `/health` | GET | Health check |
| `/predict` | POST | Single customer prediction |
| `/predict/batch` | POST | Batch customer predictions |
| `/docs` | GET | Interactive API documentation |

## Example Prediction

**Sample API response:**

```json
{
  "prediction": "No Churn",
  "churn_probability": 0.4765,
  "risk_level": "Medium"
}
```

The probability represents the model's estimated likelihood of churn. The prediction and risk category are separate outputs.

## Risk Classification

The API uses the following probability thresholds:

- Low: probability below 0.40
- Medium: probability from 0.40 to below 0.70
- High: probability of 0.70 or above

These are illustrative project thresholds and are not calibrated business risk categories.

## Project Structure

```text
customer-churn-prediction/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── artifacts/
│   ├── churn_model.pkl
│   └── feature_names.pkl
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/chhavibakhru1-spec/customer-churn-prediction.git

cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python3 -m venv venv

source venv/bin/activate
```

Python 3.12 or later is recommended.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
python3 -m uvicorn app.main:app --reload
```

### 5. Open API documentation

Visit:

http://127.0.0.1:8000/docs

## Running Automated Tests

Run:

```bash
python3 -m pytest tests/ -v
```

**Test results:** 6 tests passed.

Tests cover:
- Home endpoint
- Health endpoint
- Single prediction
- Batch prediction
- Invalid customer input
- Missing features

## Docker Deployment

The application is containerized using Docker.

### 1. Build the Docker image

```bash
docker build -t customer-churn-api .
```

### 2. Run the Docker container

```bash
docker run -d --name customer-churn-container -p 8000:8000 customer-churn-api
```

### 3. Verify the container

```bash
docker ps
```

### 4. Test the API

Health check:

http://127.0.0.1:8000/health

Swagger documentation:

http://127.0.0.1:8000/docs

### 5. Stop the container

```bash
docker stop customer-churn-container
```

## Future Improvements

- Deploy the API to a cloud platform.
- Build an interactive Streamlit dashboard.
- Add monitoring and logging.
- Add model performance monitoring.
- Improve model explainability through API responses.

## Author

**Chhavi Bakhru**

Aspiring AI/ML Engineer

GitHub: https://github.com/chhavibakhru1-spec
