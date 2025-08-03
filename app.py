import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# ----------------------------
# Cache model and scaler loading
# ----------------------------
@st.cache_resource
def load_models():
    rf = joblib.load("Models/random_forest_model.pkl")
    iso = joblib.load("Models/isolation_forest_model.pkl")
    ae = load_model("Models/autoencoder_model.h5", compile=False)
    scaler = joblib.load("Models/scaler.pkl")
    feature_cols = joblib.load("Models/feature_columns.pkl")
    return rf, iso, ae, scaler, feature_cols

rf_model, iso_model, autoencoder_model, scaler, feature_columns = load_models()

# ----------------------------
# Initialize Streamlit App
# ----------------------------
st.title("💳 Real-Time Fraud Detection App")
st.markdown("Upload transaction data and choose a model to predict fraud risk.")

if "drift_reference" not in st.session_state:
    st.session_state.drift_reference = None

if "feedback_db" not in st.session_state:
    st.session_state.feedback_db = pd.DataFrame()

# ----------------------------
# Upload CSV File
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# ----------------------------
# Helper: Preprocess input
# ----------------------------
def preprocess_input_data(df):
    drop_cols = ['cc_num', 'first', 'last', 'street', 'trans_date_trans_time', 'unix_time']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
        df['month'] = df['trans_date_trans_time'].dt.month

    from sklearn.preprocessing import LabelEncoder
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Align features
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    return df

# ----------------------------
# Helper: Real-time scoring
# ----------------------------
def real_time_score(model_name, scaled_data):
    if model_name == "Random Forest":
        proba = rf_model.predict_proba(scaled_data)[:, 1]
        labels = (proba >= 0.5).astype(int)
        return pd.DataFrame({"fraud_flag": labels, "risk_score": (proba * 100).round(2), "fraud_probability": proba})

    elif model_name == "Isolation Forest":
        preds = iso_model.predict(scaled_data)
        labels = np.where(preds == -1, 1, 0)
        return pd.DataFrame({"fraud_flag": labels, "risk_score": np.where(labels == 1, 90 + np.random.rand(len(labels)) * 10, np.random.rand(len(labels)) * 40)})

    elif model_name == "Autoencoder":
        reconstructed = autoencoder_model.predict(scaled_data)
        mse = np.mean(np.power(scaled_data - reconstructed, 2), axis=1)
        threshold = np.percentile(mse, 95)
        labels = (mse > threshold).astype(int)
        return pd.DataFrame({"fraud_flag": labels, "risk_score": (mse * 1000).round(2)})

# ----------------------------
# Helper: Concept Drift Detection
# ----------------------------
def detect_drift(current_sample):
    ref = st.session_state.drift_reference
    if ref is None:
        st.session_state.drift_reference = current_sample.copy()
        return "No reference yet. Drift monitoring started."
    diff = np.mean(np.abs(current_sample.mean() - ref.mean()))
    if diff > 0.1:
        return f"⚠️ Concept drift detected! Mean shift: {diff:.4f}"
    else:
        return f"✅ No significant concept drift. Mean shift: {diff:.4f}"

# ----------------------------
# Helper: Feedback Integration
# ----------------------------
import os

def store_feedback(index, is_fraud):
    if "last_predictions" in st.session_state:
        row = st.session_state.last_predictions.iloc[[index]].copy()
        row["investigator_feedback"] = is_fraud
        st.session_state.feedback_db = pd.concat([st.session_state.feedback_db, row], ignore_index=True)

        # Save feedback to CSV file
        feedback_file = "investigator_feedback.csv"
        if os.path.exists(feedback_file):
            existing_df = pd.read_csv(feedback_file)
            updated_df = pd.concat([existing_df, row], ignore_index=True)
        else:
            updated_df = row
        updated_df.to_csv(feedback_file, index=False)

        st.success("✔️ Feedback saved and written to investigator_feedback.csv.")
    else:
        st.error("⚠️ Please run prediction first before submitting feedback.")

# ----------------------------
# Main App Logic
# ----------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    if "is_fraud" in df.columns:
        df.drop(columns=["is_fraud"], inplace=True)

    df_processed = preprocess_input_data(df)
    scaled_data = scaler.transform(df_processed)

    st.subheader("🔢 Model Selection")
    model_choice = st.selectbox("Choose a Model", ["Random Forest", "Isolation Forest", "Autoencoder"])

    # Predict if button clicked or no previous prediction
    if st.button("📊 Predict Fraud Risk") or "last_predictions" not in st.session_state:
        result_df = real_time_score(model_choice, scaled_data)
        combined = pd.concat([df.reset_index(drop=True), result_df], axis=1)
        combined = combined.sort_values(by="risk_score", ascending=False)

        # ⬇️ Label if the probability of fraud>0.5
        if "fraud_probability" in combined.columns:
            combined["true_fraud"] = (combined["fraud_probability"] >= 0.5).astype(int)

    st.session_state.last_predictions = combined.copy()

    # Show predictions if available
    if "last_predictions" in st.session_state:
        combined = st.session_state.last_predictions.copy()
        st.dataframe(combined.head(20))

        st.subheader("🐞 Concept Drift Monitor")
        drift_msg = detect_drift(pd.DataFrame(scaled_data))
        st.info(drift_msg)

        csv = combined.to_csv(index=False).encode("utf-8")
        st.download_button("📂 Download Results", data=csv, file_name="fraud_predictions.csv")

        st.subheader("📝 Investigator Feedback")
        with st.form("feedback_form"):
            selected_index = st.number_input(
                "Select Row Index for Feedback",
                min_value=0,
                max_value=len(combined) - 1,
                step=1,
                key="feedback_index"
            )

            feedback_label = st.selectbox(
                "Is this transaction fraud?",
                ["Yes", "No"],
                key="feedback_label"
            )

            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                is_fraud = 1 if feedback_label == "Yes" else 0
                store_feedback(selected_index, is_fraud)

        if not st.session_state.feedback_db.empty:
            st.write("Stored Feedback (for retraining later):")
            st.dataframe(st.session_state.feedback_db.tail(5))
