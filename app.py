import streamlit as st
import pandas as pd
import pickle

# Load model, scaler, and columns
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/columns.pkl", "rb") as f:
    columns = pickle.load(f)


def main():
    st.title("Customer Churn Predictor")
    st.write("Enter customer details to predict customer churn.")

    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

    contract_type = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
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

    # Create empty dataframe with same columns used during training
    input_data = pd.DataFrame(0, index=[0], columns=columns)

    # Fill numerical columns
    if "tenure" in input_data.columns:
        input_data["tenure"] = tenure

    if "MonthlyCharges" in input_data.columns:
        input_data["MonthlyCharges"] = monthly_charges

    if "TotalCharges" in input_data.columns:
        input_data["TotalCharges"] = total_charges

    # Fill encoded categorical columns
    contract_col = "Contract_" + contract_type
    internet_col = "InternetService_" + internet_service
    payment_col = "PaymentMethod_" + payment_method

    if contract_col in input_data.columns:
        input_data[contract_col] = 1

    if internet_col in input_data.columns:
        input_data[internet_col] = 1

    if payment_col in input_data.columns:
        input_data[payment_col] = 1

    if st.button("Predict Churn"):
        input_scaled = scaler.transform(input_data)

        churn_prediction = model.predict(input_scaled)[0]
        churn_probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Prediction Results")

        if churn_prediction == 1:
            st.error("Churn Prediction: Yes")
            st.write(f"Churn Probability: **{churn_probability * 100:.2f}%**")
            st.write("Explanation: This customer is likely to leave based on the given details.")
        else:
            st.success("Churn Prediction: No")
            st.write(f"Churn Probability: **{churn_probability * 100:.2f}%**")
            st.write("Explanation: This customer is less likely to leave based on the given details.")


if __name__ == "__main__":
    main()