
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------
# Configuration
# --------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

FEATURE_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner",
    "Dependents", "tenure", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

# --------------------------------
# Load trained model
# --------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "artifacts/churn_model.pkl")
    features = joblib.load(BASE_DIR / "artifacts/feature_names.pkl")
    return model, features

try:
    model, feature_names = load_model()
    st.sidebar.success("ML Model Loaded Successfully")
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

# --------------------------------
# Prediction function
# --------------------------------

classes = list(model.classes_)

positive_class = next(
    (label for label in [1, "Yes", "yes", "Churn", True]
     if label in classes),
    classes[-1]
)

positive_index = classes.index(positive_class)


def predict_customers(data):

    X = pd.DataFrame(data, columns=feature_names)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, positive_index]

    results = []

    for prediction, probability in zip(predictions, probabilities):

        is_churn = prediction == positive_class

        risk = (
            "High" if probability >= 0.70
            else "Medium" if probability >= 0.40
            else "Low"
        )

        results.append({
            "prediction": "Churn" if is_churn else "No Churn",
            "churn_probability": round(float(probability), 4),
            "risk_level": risk
        })

    return pd.DataFrame(results)


# --------------------------------
# Dashboard
# --------------------------------

st.title("Customer Churn Prediction Dashboard")

st.write(
    "Predict customer churn using a trained "
    "Machine Learning model."
)

st.divider()

single_tab, batch_tab = st.tabs([
    "Single Customer Prediction",
    "Batch Prediction"
])

# =================================
# SINGLE CUSTOMER
# =================================

with single_tab:

    st.subheader("Enter Customer Details")

    with st.form("customer_form"):

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Personal Information")

            customer_id = st.text_input(
                "Customer ID", "DASH001"
            )

            gender = st.selectbox(
                "Gender", ["Female", "Male"]
            )

            senior_citizen = st.selectbox(
                "Senior Citizen", [0, 1]
            )

            partner = st.selectbox(
                "Partner", ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents", ["Yes", "No"]
            )

            tenure = st.number_input(
                "Tenure (months)", 0, 72, 12
            )

        with col2:

            st.markdown("### Services")

            phone_service = st.selectbox(
                "Phone Service", ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No phone service", "No", "Yes"]
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["No internet service", "No", "Yes"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["No internet service", "No", "Yes"]
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["No internet service", "No", "Yes"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["No internet service", "No", "Yes"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No internet service", "No", "Yes"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No internet service", "No", "Yes"]
            )

        st.markdown("### Billing Information")

        col3, col4 = st.columns(2)

        with col3:

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with col4:

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.35
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=845.50
            )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            use_container_width=True
        )

    if submitted:

        if not customer_id.strip():
            st.error("Enter a Customer ID.")

        else:

            customer = {
                "customerID": customer_id.strip(),
                "gender": gender,
                "SeniorCitizen": int(senior_citizen),
                "Partner": partner,
                "Dependents": dependents,
                "tenure": int(tenure),
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": float(monthly_charges),
                "TotalCharges": float(total_charges)
            }

            try:
                result = predict_customers(
                    [customer]
                ).iloc[0]

                st.divider()
                st.subheader("Prediction Results")

                c1, c2, c3 = st.columns(3)

                c1.metric("Prediction", result["prediction"])

                c2.metric(
                    "Churn Probability",
                    f'{result["churn_probability"] * 100:.2f}%'
                )

                c3.metric("Risk Level", result["risk_level"])

                st.progress(result["churn_probability"])

                if result["risk_level"] == "High":
                    st.error("High churn risk")
                elif result["risk_level"] == "Medium":
                    st.warning("Medium churn risk")
                else:
                    st.success("Low churn risk")

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# =================================
# BATCH PREDICTION
# =================================

with batch_tab:

    st.subheader("Upload Customer CSV")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file:

        try:

            df = pd.read_csv(uploaded_file)

            st.write("Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            missing = [
                col for col in feature_names
                if col not in df.columns
            ]

            if missing:
                st.error(f"Missing columns: {missing}")

            elif len(df) == 0:
                st.error("CSV is empty.")

            elif len(df) > 1000:
                st.error("Please upload at most 1000 customers at a time.")

            else:

                if st.button("Run Batch Predictions"):

                    numeric_columns = [
                        "SeniorCitizen", "tenure",
                        "MonthlyCharges", "TotalCharges"
                    ]

                    input_df = df[feature_names].copy()

                    for col in numeric_columns:
                        input_df[col] = pd.to_numeric(
                            input_df[col], errors="coerce"
                        )

                    if input_df[numeric_columns].isnull().any().any():
                        st.error("Some numeric values are missing or invalid.")

                    elif not input_df["SeniorCitizen"].isin([0, 1]).all():
                        st.error("SeniorCitizen must contain only 0 or 1.")

                    else:

                        try:

                            with st.spinner("Predicting customers..."):
                                predictions = predict_customers(
                                    input_df.to_dict(orient="records")
                                )

                            final_df = pd.concat(
                                [
                                    df.reset_index(drop=True),
                                    predictions.reset_index(drop=True)
                                ],
                                axis=1
                            )

                            st.success("Predictions completed!")

                            c1, c2, c3 = st.columns(3)

                            c1.metric("Total Customers", len(final_df))

                            c2.metric(
                                "Predicted Churn",
                                int((final_df["prediction"] == "Churn").sum())
                            )

                            c3.metric(
                                "Predicted No Churn",
                                int((final_df["prediction"] == "No Churn").sum())
                            )

                            st.subheader("Risk Distribution")

                            st.bar_chart(
                                final_df["risk_level"].value_counts()
                            )

                            st.subheader("Prediction Results")

                            st.dataframe(
                                final_df,
                                use_container_width=True
                            )

                            st.download_button(
                                "Download Results CSV",
                                final_df.to_csv(index=False).encode("utf-8"),
                                "churn_predictions.csv",
                                "text/csv"
                            )

                        except Exception as e:
                            st.error(f"Batch prediction failed: {e}")

        except Exception as e:
            st.error(f"Could not read CSV: {e}")


st.divider()

st.caption(
    "Customer Churn Prediction | "
    "Machine Learning | Streamlit | FastAPI | Docker"
)

