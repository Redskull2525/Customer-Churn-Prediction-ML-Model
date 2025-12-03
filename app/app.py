

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Paths (adjust if needed)
# -------------------------
ROOT = Path(__file__).resolve().parents[1]  # project root
DATA_PATH = ROOT / "data" / "churn-bigml-80.csv"
MODEL_PATH = ROOT / "models" / "best_churn_model.pkl"
SCALER_PATH = ROOT / "models" / "churn_scaler.pkl"

# -------------------------
# Utility functions
# -------------------------
@st.cache_data
def load_train_df(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Training CSV not found at: {path}")
    return pd.read_csv(path)

@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found at: {path}")
    return joblib.load(path)

@st.cache_resource
def load_scaler(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found at: {path}")
    return joblib.load(path)

def build_preprocessors(train_df: pd.DataFrame):
    """
    Build encoders, numeric medians, categorical modes based on train_df.
    Returns:
      - feature_cols: list of feature column names used by the model (in order)
      - cat_cols: list of categorical columns
      - num_cols: list of numeric columns
      - encoders: dict of LabelEncoder objects for categorical columns
      - num_impute: dict of medians for numeric columns
      - cat_impute: dict of modes for categorical columns
      - target_encoder: LabelEncoder for the target column 'Churn'
    """
    df = train_df.copy()
    # Drop identifier cols not used as features
    drop_cols = ['phone', 'Phone', 'PhoneNumber']  # handle case variations
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Make sure 'Churn' exists
    if 'Churn' not in df.columns:
        raise ValueError("Target column 'Churn' not present in train CSV")

    # Build target encoder
    target_encoder = LabelEncoder()
    target_encoder.fit(df['Churn'].astype(str))

    # Feature columns: everything except 'Churn'
    feature_cols = [c for c in df.columns if c != 'Churn']

    # Detect categorical columns (object / string dtype) among features
    cat_cols = [c for c in feature_cols if df[c].dtype == 'object' or df[c].dtype.name == 'category']
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Build encoders for categorical columns
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        encoders[c] = le.fit(df[c].astype(str))

    # Build numeric impute values (median)
    num_impute = {c: float(df[c].median()) for c in num_cols}

    # Build categorical impute values (mode)
    cat_impute = {c: str(df[c].mode()[0]) if not df[c].mode().empty else "" for c in cat_cols}

    return feature_cols, cat_cols, num_cols, encoders, num_impute, cat_impute, target_encoder

def preprocess_input(user_input: dict,
                     feature_cols, cat_cols, num_cols,
                     encoders, num_impute, cat_impute):
    """
    Turn single-record user_input (dict) into numeric array ready for scaler & model.
    """
    # Create DF with one row
    X = pd.DataFrame([user_input], columns=feature_cols)

    # Fill missing values
    for c in num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce')
            X[c] = X[c].fillna(num_impute[c])

    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].fillna(cat_impute[c]).astype(str)

    # Encode categorical columns using fitted label encoders
    for c in cat_cols:
        if c in X.columns:
            le = encoders[c]
            # Map unseen categories to a new label: append if necessary
            vals = X[c].astype(str).tolist()
            mapped = []
            # If user supplies a category unseen in train, we add it to encoder classes_ temporarily
            for val in vals:
                if val not in le.classes_:
                    # extend classes_ with new value (this keeps encoding deterministic)
                    le.classes_ = np.append(le.classes_, val)
                mapped.append(val)
            X[c] = le.transform(mapped)

    # Return numpy array in the same column order
    return X[feature_cols].to_numpy()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("Customer Churn Prediction")
st.markdown("Enter customer details below and get the predicted probability of churn.")

# Load data / model / scaler and build preprocessors
try:
    train_df = load_train_df(DATA_PATH)
    model = load_model(MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# Prepare preprocessors
feature_cols, cat_cols, num_cols, encoders, num_impute, cat_impute, target_encoder = build_preprocessors(train_df)

st.sidebar.header("Input options")
st.sidebar.markdown("You can load example customers from the training set or enter custom values.")

# Example selector: show few rows for quick test
if st.sidebar.checkbox("Show sample rows from train data", value=False):
    st.subheader("Sample of training data")
    st.dataframe(train_df.sample(5, random_state=42).reset_index(drop=True))

# Build input form using feature_cols
with st.form("input_form"):
    inputs = {}
    st.subheader("Customer Features")

    for col in feature_cols:
        # Skip any column we cannot or should not request manually (e.g., area code might be numeric)
        if col.lower() in ['state']:
            # give text input for state
            inputs[col] = st.text_input(f"{col}", value=str(train_df[col].mode()[0] if col in train_df.columns else ""))
        elif col.lower() in ['area code', 'area_code', 'areacode', 'area']:
            # treat as categorical/short numeric input
            default_area = train_df[col].mode()[0] if col in train_df.columns else ""
            inputs[col] = st.text_input(f"{col}", value=str(default_area))
        elif col.lower() in ['international plan', 'international_plan']:
            opts = sorted(train_df[col].astype(str).unique()) if col in train_df.columns else ["Yes","No"]
            inputs[col] = st.selectbox(f"{col}", options=opts, index=opts.index(str(train_df[col].mode()[0])) if col in train_df.columns else 0)
        elif col.lower() in ['voice mail plan', 'voice_mail_plan', 'voicemailplan']:
            opts = sorted(train_df[col].astype(str).unique()) if col in train_df.columns else ["Yes","No"]
            inputs[col] = st.selectbox(f"{col}", options=opts, index=opts.index(str(train_df[col].mode()[0])) if col in train_df.columns else 0)
        else:
            # numeric fields
            if col in train_df.columns and pd.api.types.is_numeric_dtype(train_df[col]):
                default = float(train_df[col].median())
                inputs[col] = st.number_input(f"{col}", value=float(default))
            else:
                # fallback: text input
                inputs[col] = st.text_input(f"{col}", value=str(train_df[col].mode()[0] if col in train_df.columns else ""))

    submit = st.form_submit_button("Predict")

if submit:
    try:
        X_pre = preprocess_input(inputs, feature_cols, cat_cols, num_cols, encoders, num_impute, cat_impute)
        X_scaled = scaler.transform(X_pre)  # scaler from training
        proba = model.predict_proba(X_scaled)[0]  # array of probabilities

        # find index that corresponds to 'Yes' in target encoder
        try:
            yes_index = int(target_encoder.transform(["Yes"])[0])
            churn_prob = float(proba[yes_index])
        except Exception:
            # if transform fails (maybe 'Yes' not in classes), assume second column is positive class
            churn_prob = float(proba[1] if proba.shape[0] > 1 else proba[0])

        pred_label_num = model.predict(X_scaled)[0]
        # map numeric label back to 'Yes'/'No' using target_encoder
        try:
            pred_label = target_encoder.inverse_transform([pred_label_num])[0]
        except Exception:
            # fallback
            pred_label = str(pred_label_num)

        st.success(f"Predicted label: **{pred_label}**")
        st.metric(label="Churn probability", value=f"{churn_prob:.2%}")

        # Show probability bar
        st.progress(min(max(churn_prob, 0.0), 1.0))

        st.markdown("### Model details")
        st.write(f"Model type: {type(model).__name__}")
        st.write(f"Classes (target encoder): {list(target_encoder.classes_)}")

    except Exception as err:
        st.error(f"Prediction failed: {err}")

st.markdown("---")
st.caption("This app rebuilds encoders from `data/churn-bigml-80.csv`. Make sure your `models/` and `data/` files are present.")
