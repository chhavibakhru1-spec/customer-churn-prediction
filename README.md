
# Customer Churn Prediction System

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to leave a company.

The project combines machine learning, model explainability, REST API development, input validation, and automated testing.

## Project Objectives

- Predict customer churn using machine learning.
- Estimate the probability of customer churn.
- Categorize customers into Low, Medium, and High risk levels.
- Provide individual and batch predictions through a REST API.
- Validate customer input before generating predictions.

## Machine Learning

**Dataset:** IBM Telco Customer Churn Dataset (Kaggle)

**Models explored:**
- Logistic Regression
- Random Forest
- XGBoost

**Model development:**
- Exploratory Data Analysis
- Data preprocessing
- Feature engineering
- Train, validation, and test split
- Cross-validation
- Hyperparameter tuning
- Model evaluation
- SHAP-based explainability

**Selected model:** Add your actual final selected model here.

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
- Git and GitHub

## API Features

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API home |
| `/health` | GET | Health check |
| `/predict` | POST | Single customer prediction |
| `/predict/batch` | POST | Batch customer predictions |
| `/docs` | GET | Interactive API documentation |

## Prediction Output

Example:

```json
{
  "prediction": "No Churn",
  "churn_probability": 0.2797,
  "risk_level": "Low"
}
```

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
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

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

```bash
python3 -m pytest tests/ -v
```

**Current test result:** 6 tests passed.

## Risk Classification

The API currently uses these probability thresholds:

- Low: probability below 0.40
- Medium: probability from 0.40 to below 0.70
- High: probability of 0.70 or above

These are illustrative project thresholds, not calibrated business risk categories.

## Future Improvements

- Deploy the API to a cloud platform.
- Build an interactive Streamlit dashboard.
- Add monitoring and logging.
- Improve API error handling.
- Add model performance monitoring.

## Author

Chhavi Bakhru

Aspiring AI/ML Engineer
