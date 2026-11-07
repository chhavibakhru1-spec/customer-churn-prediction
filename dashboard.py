import streamlit as st
import requests
import pandas as pd
import json

# -------------------------------
# Configuration
# -------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

FEATURE_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner",
    "Dependents", "tenure", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

# -------------------------------
# Dashboard styling
# -------------------------------

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown(
    "Predict customer churn using our trained "
    "Machine Learning model."
)

st.divider()

# -------------------------------
# API connection
# -------------------------------

st.sidebar.header("API Configuration")

api_url = st.sidebar.text_input(
    "FastAPI URL",
    value="http://127.0.0.1:8000"
).rstrip("/")

try:
    health = requests.get(
        f"{api_url}/health",
        timeout=5
    )

    if health.status_code == 200:
        st.sidebar.success("API Connected")
    else:
        st.sidebar.error("API is not healthy")

except requests.RequestException:
    st.sidebar.error("API is not reachable")

st.sidebar.info(
    "Start your FastAPI Docker container "
    "before making predictions."
)

# -------------------------------
# Prediction display function
# -------------------------------

def display_prediction(result):

    prediction = result["prediction"]
    probability = result["churn_probability"]
    risk = result["risk_level"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction", prediction)

    with col2:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    with col3:
        st.metric("Risk Level", risk)

    st.progress(float(probability))

    if risk == "High":
        st.error("High churn risk")

    elif risk == "Medium":
        st.warning("Medium churn risk")

    else:
        st.success("Low churn risk")

    st.caption(
        "Risk categories use the thresholds configured "
        "in the FastAPI application."
    )


# -------------------------------
# Dashboard tabs
# -------------------------------

single_tab, batch_tab = st.tabs(
    ["👤 Single Customer Prediction", "📁 Batch Prediction"]
)

# =====================================================
# SINGLE CUSTOMER PREDICTION
# =====================================================

with single_tab:

    st.subheader("Enter Customer Details")

    with st.form("customer_form"):

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Personal Information")

            customer_id = st.text_input(
                "Customer ID",
                value="DASH001"
            )

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12
            )

        with col2:

            st.markdown("### Services")

            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
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
                "Paperless Billing",
                ["Yes", "No"]
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
                max_value=200.0,
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
            st.error("Please enter a Customer ID.")

        else:

            customer_data = {
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

                with st.spinner("Predicting customer churn..."):

                    response = requests.post(
                        f"{api_url}/predict",
                        json={"data": customer_data},
                        timeout=30
                    )

                response.raise_for_status()

                result = response.json()

                st.divider()
                st.subheader("Prediction Results")

                display_prediction(result)

            except requests.RequestException as e:
                st.error(f"Prediction request failed: {e}")

# =====================================================
# BATCH PREDICTION
# =====================================================

with batch_tab:

    st.subheader("Upload Customer CSV File")

    st.write(
        "Upload a CSV containing customer records. "
        "The dashboard will predict churn for each customer."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.write("Preview of uploaded data")
            st.dataframe(df.head(), use_container_width=True)

            missing_columns = [
                col for col in FEATURE_COLUMNS
                if col not in df.columns
            ]

            if missing_columns:

                st.error(
                    f"Missing required columns: {missing_columns}"
                )

            elif len(df) == 0:

                st.error("The uploaded CSV is empty.")

            else:

                st.info(
                    f"Total customers uploaded: {len(df)}"
                )

                if st.button(
                    "Run Batch Predictions",
                    use_container_width=True
                ):

                    try:

                        batch_df = df.copy()

                        numeric_columns = [
                            "SeniorCitizen",
                            "tenure",
                            "MonthlyCharges",
                            "TotalCharges"
                        ]

                        for col in numeric_columns:
                            batch_df[col] = pd.to_numeric(
                                batch_df[col],
                                errors="coerce"
                            )

                        if batch_df[numeric_columns].isnull().any().any():
                            st.error(
                                "Some numeric values are missing or invalid. "
                                "Please correct the CSV."
                            )

                        elif not batch_df["SeniorCitizen"].isin([0, 1]).all():
                            st.error(
                                "SeniorCitizen must contain only 0 or 1."
                            )

                        else:

                            batch_df["SeniorCitizen"] = (
                                batch_df["SeniorCitizen"].astype(int)
                            )

                            batch_df["tenure"] = (
                                batch_df["tenure"].astype(int)
                            )

                            records = json.loads(
                                batch_df[FEATURE_COLUMNS].to_json(
                                    orient="records"
                                )
                            )

                            all_predictions = []

                            progress = st.progress(0)

                            total_batches = (
                                len(records) + 99
                            ) // 100

                            for i in range(0, len(records), 100):

                                customer_batch = records[i:i + 100]

                                response = requests.post(
                                    f"{api_url}/predict/batch",
                                    json={
                                        "customers": customer_batch
                                    },
                                    timeout=120
                                )

                                response.raise_for_status()

                                result = response.json()

                                all_predictions.extend(
                                    result["predictions"]
                                )

                                progress.progress(
                                    min(
                                        (i // 100 + 1) / total_batches,
                                        1.0
                                    )
                                )

                            result_df = pd.DataFrame(
                                all_predictions
                            )

                            final_df = pd.concat(
                                [
                                    df.reset_index(drop=True),
                                    result_df.reset_index(drop=True)
                                ],
                                axis=1
                            )

                            st.success(
                                "Batch predictions completed successfully!"
                            )

                            st.subheader("Prediction Summary")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    "Total Customers",
                                    len(final_df)
                                )

                            with col2:
                                churn_count = (
                                    final_df["prediction"] == "Churn"
                                ).sum()

                                st.metric(
                                    "Predicted Churn",
                                    int(churn_count)
                                )

                            with col3:
                                no_churn_count = (
                                    final_df["prediction"] == "No Churn"
                                ).sum()

                                st.metric(
                                    "Predicted No Churn",
                                    int(no_churn_count)
                                )

                            st.subheader("Risk Distribution")

                            risk_counts = (
                                final_df["risk_level"].value_counts()
                            )

                            st.bar_chart(risk_counts)

                            st.subheader("Customer Predictions")

                            st.dataframe(
                                final_df,
                                use_container_width=True
                            )

                            csv = final_df.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                "Download Prediction Results",
                                data=csv,
                                file_name="churn_predictions.csv",
                                mime="text/csv"
                            )

                    except requests.RequestException as e:
                        st.error(f"Batch API request failed: {e}")

                    except (KeyError, ValueError) as e:
                        st.error(f"Could not process predictions: {e}")

        except Exception as e:
            st.error(f"Could not read CSV file: {e}")


st.divider()

st.caption(
    "Customer Churn Prediction System | "
    "Machine Learning + FastAPI + Streamlit + Docker"
)
