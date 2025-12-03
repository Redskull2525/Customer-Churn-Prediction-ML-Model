import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Load saved pipeline + metadata
# -----------------------------

models_dir = Path("models")

pipeline_path = models_dir / "churn_pipeline.pkl"
feature_cols_path = models_dir / "feature_cols.pkl"

# Load pipeline (preprocessing + model)
pipeline = joblib.load(pipeline_path)

# Load feature column order (raw input columns)
feature_cols = joblib.load(feature_cols_path)

st.set_page_config(page_title="Customer Churn Prediction App")

# -----------------------------
# App UI
# -----------------------------
st.title("📞 Customer Churn Prediction")
st.write("Enter customer details to predict the probability of churn using the trained ML model.")

# Input form
user_input = {}

st.subheader("Enter Customer Details")

for col in feature_cols:
    st.write(f"### {col}")

    # Numeric inputs
    if col.lower() not in ["state", "international plan", "voice mail plan"]:
        try:
            default_value = 0.0
            user_input[col] = st.number_input(col, value=default_value)
        except:
            user_input[col] = st.text_input(col)

    # Categorical inputs
    else:
        if col.lower() == "state":
            user_input[col] = st.text_input(col, value="OH")   # enter any default state
        elif col.lower() == "international plan":
            user_input[col] = st.selectbox(col, ["Yes", "No"])
        elif col.lower() == "voice mail plan":
            user_input[col] = st.selectbox(col, ["Yes", "No"])
        else:
            user_input[col] = st.text_input(col)

# Predict button
if st.button("Predict Churn"):
    try:
        # Create a DataFrame with correct order
        X_input = pd.DataFrame([user_input], columns=feature_cols)

        # Pipeline handles encoding + scaling + prediction
        prediction = pipeline.predict(X_input)[0]
        proba = pipeline.predict_proba(X_input)[0][1]

        st.success(f"Prediction: **{prediction}**")
        st.info(f"Churn Probability: **{proba:.2%}**")

        st.progress(int(proba * 100))

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.write("---")
st.caption("Model: Random Forest + ColumnTransformer Pipeline")
