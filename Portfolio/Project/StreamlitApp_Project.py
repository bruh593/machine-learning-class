"""
Loan Default Risk Predictor
Streamlit app for the Spring 2026 ML Project — Austin Smith
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="💰",
    layout="wide",
)

st.title("💰 Loan Default Risk Predictor")
st.markdown(
    "**Author:** Austin Smith &nbsp;|&nbsp; "
    "**Course:** Intro to Machine Learning &nbsp;|&nbsp; "
    "**Dataset:** LendingClub 2007–2018"
)
st.markdown("---")

# ============================================================
# Load model + SHAP explainer (cached so they only load once)
# ============================================================
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline = joblib.load(os.path.join(base_dir, "best_xgb_pipeline.pkl"))
    explainer = joblib.load(os.path.join(base_dir, "shap_explainer.pkl"))
    return pipeline, explainer

with st.spinner("Loading model..."):
    pipeline, explainer = load_artifacts()

st.success("Model loaded successfully.")

# ============================================================
# Sidebar — Loan application input form
# ============================================================
st.sidebar.header("📝 Loan Application")
st.sidebar.markdown("Enter the applicant's information below.")

# Loan characteristics
st.sidebar.subheader("Loan Details")
loan_amnt = st.sidebar.number_input("Loan Amount ($)", 1000, 40000, 15000, step=500)
term = st.sidebar.selectbox("Term", [36, 60])
int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 31.0, 12.0, step=0.1)
installment = st.sidebar.number_input("Monthly Installment ($)", 30.0, 1500.0, 450.0, step=10.0)
grade = st.sidebar.selectbox("LendingClub Grade", ["A", "B", "C", "D", "E", "F", "G"])
purpose = st.sidebar.selectbox("Purpose", [
    "debt_consolidation", "credit_card", "home_improvement", "other",
    "major_purchase", "small_business", "car", "medical", "moving",
    "vacation", "house", "wedding", "renewable_energy", "educational"
])
application_type = st.sidebar.selectbox("Application Type", ["Individual", "Joint App"])
initial_list_status = st.sidebar.selectbox("Initial List Status", ["w", "f"])

# Borrower details
st.sidebar.subheader("Borrower Details")
emp_length = st.sidebar.slider("Employment Length (years)", 0, 10, 5)
home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
annual_inc = st.sidebar.number_input("Annual Income ($)", 0, 500000, 65000, step=5000)
verification_status = st.sidebar.selectbox(
    "Income Verification", ["Verified", "Source Verified", "Not Verified"]
)

# Credit history
st.sidebar.subheader("Credit History")
dti = st.sidebar.slider("Debt-to-Income Ratio (%)", 0.0, 50.0, 18.0, step=0.5)
fico_range_low = st.sidebar.slider("FICO Score (low)", 600, 850, 690, step=5)
fico_range_high = fico_range_low + 4
credit_history_months = st.sidebar.slider("Credit History (months)", 12, 600, 200, step=12)
delinq_2yrs = st.sidebar.number_input("Delinquencies (last 2 yrs)", 0, 30, 0)
inq_last_6mths = st.sidebar.number_input("Recent Credit Inquiries (6 mo)", 0, 30, 0)
open_acc = st.sidebar.number_input("Open Credit Accounts", 0, 80, 11)
total_acc = st.sidebar.number_input("Total Credit Accounts", 0, 150, 25)
revol_bal = st.sidebar.number_input("Revolving Balance ($)", 0, 200000, 15000, step=1000)
revol_util = st.sidebar.slider("Revolving Utilization (%)", 0.0, 150.0, 50.0, step=1.0)
mort_acc = st.sidebar.number_input("Mortgage Accounts", 0, 30, 1)
pub_rec = st.sidebar.number_input("Public Records", 0, 80, 0)
pub_rec_bankruptcies = st.sidebar.number_input("Bankruptcies", 0, 10, 0)

# ============================================================
# Build single-row DataFrame to feed into the pipeline
# ============================================================
input_dict = {
    "loan_amnt": loan_amnt,
    "term": term,
    "int_rate": int_rate,
    "installment": installment,
    "grade": grade,
    "emp_length": emp_length,
    "home_ownership": home_ownership,
    "annual_inc": annual_inc,
    "verification_status": verification_status,
    "purpose": purpose,
    "dti": dti,
    "delinq_2yrs": delinq_2yrs,
    "fico_range_low": fico_range_low,
    "fico_range_high": fico_range_high,
    "inq_last_6mths": inq_last_6mths,
    "open_acc": open_acc,
    "pub_rec": pub_rec,
    "revol_bal": revol_bal,
    "revol_util": revol_util,
    "total_acc": total_acc,
    "initial_list_status": initial_list_status,
    "application_type": application_type,
    "mort_acc": mort_acc,
    "pub_rec_bankruptcies": pub_rec_bankruptcies,
    "credit_history_months": credit_history_months,
}

input_df = pd.DataFrame([input_dict])

# ============================================================
# Main panel — Prediction
# ============================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Application Summary")
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

with col2:
    st.subheader("🎯 Risk Prediction")

    if st.button("Run Prediction", type="primary", use_container_width=True):
        proba = pipeline.predict_proba(input_df)[0, 1]
        pred = int(proba >= 0.5)

        st.metric("Default Probability", f"{proba:.1%}")

        if proba < 0.20:
            st.success(f"✅ **LOW RISK** — Recommend approval")
        elif proba < 0.40:
            st.warning(f"⚠️ **MEDIUM RISK** — Recommend manual review")
        else:
            st.error(f"🚨 **HIGH RISK** — Recommend decline or adjusted terms")

        st.markdown(f"**Model output:** {'Default' if pred else 'No Default'}")

        # ============================================================
        # SHAP local explanation
        # ============================================================
        st.markdown("---")
        st.subheader("🔍 Why this prediction?")
        st.markdown(
            "The chart below shows which features pushed this risk score up (red) "
            "or down (blue) compared to the model's average prediction."
        )

        # Run input through the preprocessor only (skip SMOTE — only for training)
        preprocessor_step = pipeline.named_steps["preprocess"]
        input_processed = preprocessor_step.transform(input_df)

        # Get expanded feature names
        numeric_cols = preprocessor_step.transformers_[0][2]
        categorical_cols = preprocessor_step.transformers_[1][2]
        cat_encoder = preprocessor_step.named_transformers_["cat"].named_steps["encode"]
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        all_feature_names = list(numeric_cols) + cat_feature_names

        # Compute SHAP values for this single applicant
        shap_value = explainer.shap_values(input_processed)
        expected_value = explainer.expected_value

        # Render waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_value[0],
                base_values=expected_value,
                data=input_processed[0],
                feature_names=all_feature_names,
            ),
            max_display=12,
            show=False,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "Model: XGBoost (fine-tuned via Grid Search) &nbsp;|&nbsp; "
    "Trained on LendingClub 2007–2018 accepted loans &nbsp;|&nbsp; "
    "CV AUC: 0.7325"
)
