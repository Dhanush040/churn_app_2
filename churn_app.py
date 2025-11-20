import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telco Churn Predictor (SVM) + Plots", layout="centered")
st.title(" Telco Churn Predictor ")

# ------------------ Configuration ------------------
MODEL_FILE = "svm_churn_model.joblib"
TRAIN_CSV = "Telco_Customer_Churn.csv"

FEATURES = [
    'tenure',
    'InternetService',
    'Contract',
    'MonthlyCharges',
    'TotalCharges',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling'
]

CATEGORICAL_COLS = [
    'InternetService', 'Contract', 'Partner',
    'Dependents', 'PhoneService', 'PaperlessBilling'
]

# ------------------ Helpers ------------------
@st.cache_resource
def load_model_and_encoders(model_path=MODEL_FILE, csv_path=TRAIN_CSV):
    # Load model
    try:
        model = load(model_path)
    except Exception as e:
        st.error(f"Could not load model from '{model_path}': {e}")
        return None, None

    # Load training CSV to re-fit LabelEncoders for categorical columns
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.warning(f"Could not load training CSV '{csv_path}': {e}. The app will still try to predict, but categorical encoding may fail.")
        return model, {}

    # Ensure TotalCharges numeric and fill missing same as training
    df['TotalCharges'] = pd.to_numeric(df.get('TotalCharges', pd.Series()), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Fit label encoders for each categorical column using training data,
    # to replicate how training code encoded them.
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        col_vals = df[col].astype(str).fillna("missing")
        le.fit(col_vals)
        encoders[col] = le

    # Also fit a label encoder for 'Churn' if present (not required for prediction but kept for completeness)
    if 'Churn' in df.columns:
        le_churn = LabelEncoder()
        le_churn.fit(df['Churn'].astype(str).fillna("missing"))
        encoders['Churn'] = le_churn

    return model, encoders

model, encoders = load_model_and_encoders()

if model is None:
    st.stop()

st.sidebar.header("Single customer input")

# Default example values
defaults = {
    "tenure": 12,
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "PaperlessBilling": "Yes"
}

# Sidebar inputs
tenure = st.sidebar.number_input("tenure (months)", min_value=0, max_value=200, value=defaults["tenure"])
internet = st.sidebar.selectbox("InternetService", options=["DSL", "Fiber optic", "No"], index=1)
contract = st.sidebar.selectbox("Contract", options=["Month-to-month", "One year", "Two year"], index=0)
monthly_charges = st.sidebar.number_input("MonthlyCharges", min_value=0.0, value=float(defaults["MonthlyCharges"]))
total_charges = st.sidebar.number_input("TotalCharges", min_value=0.0, value=float(defaults["TotalCharges"]))
senior = st.sidebar.selectbox("SeniorCitizen", options=[0, 1], index=0)
partner = st.sidebar.selectbox("Partner", options=["Yes", "No"], index=1)
dependents = st.sidebar.selectbox("Dependents", options=["Yes", "No"], index=1)
phone = st.sidebar.selectbox("PhoneService", options=["Yes", "No"], index=0)
paperless = st.sidebar.selectbox("PaperlessBilling", options=["Yes", "No"], index=0)

single_input = pd.DataFrame([{
    'tenure': tenure,
    'InternetService': internet,
    'Contract': contract,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'PhoneService': phone,
    'PaperlessBilling': paperless
}])

st.subheader("Input preview")
st.dataframe(single_input)

# Function to encode a DataFrame exactly as training code did (LabelEncode categorical cols, fill TotalCharges)
def encode_input(df_input, encoders_dict):
    df = df_input.copy()
    # TotalCharges behavior in your training code: convert to numeric and fillna(0)
    df['TotalCharges'] = pd.to_numeric(df.get('TotalCharges', pd.Series()), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Apply label encoders per column if available; otherwise fit on-the-fly (best-effort)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            if encoders_dict and col in encoders_dict:
                le = encoders_dict[col]
                vals = df[col].astype(str).fillna("missing")
                try:
                    df[col] = le.transform(vals)
                except Exception:
                    mapping = {c: i for i, c in enumerate(le.classes_)}
                    df[col] = vals.map(mapping).fillna(-1).astype(int)
            else:
                le = LabelEncoder()
                vals = df[col].astype(str).fillna("missing")
                df[col] = le.fit_transform(vals)
    # Ensure column order matches FEATURES
    df = df.reindex(columns=FEATURES)
    return df

col1, col2 = st.columns([1,1])

with col1:
    if st.button("Predict single"):
        try:
            X_single = encode_input(single_input, encoders)
            pred = model.predict(X_single)[0]
            prob = None
            try:
                proba = model.predict_proba(X_single)[0]
                # assume class 1 is churn
                prob = proba[1]
            except Exception:
                prob = None

            st.write("### Prediction")
            st.write(f"Predicted label (encoded): **{pred}**")
            if prob is not None:
                st.write(f"Churn probability (class=1): **{prob*100:.2f}%**")
            else:
                st.write("Model does not support `predict_proba` (probability unavailable).")

            # --- Matplotlib visualization for single prediction ---
            fig, ax = plt.subplots(figsize=(4,4))
            if prob is not None:
                sizes = [prob, 1 - prob]
                labels = [f"Churn ({prob*100:.1f}%)", f"No churn ({(1-prob)*100:.1f}%)"]
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.set_title("Predicted churn probability")
            else:
                # simple bar showing encoded label
                vals = [0, 0]
                idx = int(pred)
                if idx >= 0 and idx < len(vals):
                    vals[idx] = 1
                ax.bar(["Label 0", "Label 1"], vals)
                ax.set_ylabel("Predicted (encoded)")
                ax.set_title("Predicted encoded label")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

with col2:
    st.info("Notes:\n• This app re-fits label encoders using the training CSV to reproduce the encoding used when training the SVM.\n• If you trained with different preprocessing, the app must use the same transformations to get correct results.")

# ---------------- Batch upload ----------------
st.markdown("---")
st.subheader("Telco_Customer_Churn")
st.write("Upload a CSV containing the same 10 feature columns (header required).")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    try:
        df_upload = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df_upload = None

    if df_upload is not None:
        missing = set(FEATURES) - set(df_upload.columns)
        if missing:
            st.error(f"CSV missing columns: {missing}")
        else:
            to_predict = df_upload[FEATURES].copy()
            try:
                X_batch = encode_input(to_predict, encoders)
                preds = model.predict(X_batch)
                probs = None
                try:
                    probs = model.predict_proba(X_batch)[:, 1]
                except Exception:
                    probs = None

                out = to_predict.copy()
                out['pred_churn_label'] = preds
                if probs is not None:
                    out['pred_churn_probability'] = probs

                st.success(f"Predicted {len(out)} rows")
                st.dataframe(out.head(100))

                # --- Matplotlib visualizations for batch predictions ---
                # 1) Bar chart of counts by predicted label
                fig1, ax1 = plt.subplots(figsize=(5,3))
                unique, counts = np.unique(preds, return_counts=True)
                labels = [str(u) for u in unique]
                ax1.bar(labels, counts)
                ax1.set_title("Predicted label counts")
                ax1.set_xlabel("Encoded label")
                ax1.set_ylabel("Count")
                st.pyplot(fig1)

                # 2) Histogram of predicted probabilities (if available)
                if probs is not None:
                    fig2, ax2 = plt.subplots(figsize=(5,3))
                    ax2.hist(probs, bins=20)
                    ax2.set_title("Predicted churn probability distribution")
                    ax2.set_xlabel("Probability (class=1)")
                    ax2.set_ylabel("Frequency")
                    st.pyplot(fig2)
                else:
                    st.info("Model does not support predict_proba — probability plots unavailable.")

                # 3) Scatter: tenure vs MonthlyCharges colored by predicted label
                fig3, ax3 = plt.subplots(figsize=(5,4))
                # jitter small to avoid overplotting if needed
                x = out['tenure'].astype(float)
                y = out['MonthlyCharges'].astype(float)
                # use predicted labels as grouping index; matplotlib will choose default colors
                for lbl in np.unique(preds):
                    mask = preds == lbl
                    ax3.scatter(x[mask], y[mask], label=f"Label {lbl}", alpha=0.7)
                ax3.set_title("Tenure vs MonthlyCharges (grouped by predicted label)")
                ax3.set_xlabel("tenure (months)")
                ax3.set_ylabel("MonthlyCharges")
                ax3.legend()
                st.pyplot(fig3)

                # Download predictions button
                csv_bytes = out.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions", data=csv_bytes, file_name="churn_predictions.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Batch prediction failed: {e}")
