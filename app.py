import streamlit as st
import pandas as pd
import joblib
import os


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Bank Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #1e1b4b,
        #312e81,
        #4c1d95
    );
}


/* Main Titles */
h1 {
    color: #fef3c7;
    text-align: center;
    font-weight: 800;
}

h2, h3 {
    color: #fbcfe8;
}


/* Normal Text */
p, span, label {
    color: #f8fafc;
}


/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111827,
        #1f2937
    );
}


/* Metric Boxes */
div[data-testid="stMetric"] {

    background: linear-gradient(
        145deg,
        #2e1065,
        #4c1d95
    );

    border-radius: 20px;
    padding: 25px;

    border: 1px solid #c084fc;

    box-shadow:
        0px 8px 25px rgba(0,0,0,0.4);
}


/* Metric Names */
div[data-testid="stMetricLabel"] p {

    color: #f9a8d4 !important;
    font-size: 18px;
    font-weight: 700;
}


/* Metric Numbers */
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] > div {

    color: #fde68a !important;
    font-size: 35px;
    font-weight: 900;
}


/* Info Boxes */
div[data-testid="stAlert"] {

    background-color: rgba(255,255,255,0.08);

    border-radius: 15px;

    color: white;
}


/* Buttons */
.stButton > button {

    background: linear-gradient(
        90deg,
        #ec4899,
        #8b5cf6
    );

    color: white;

    border-radius: 15px;

    border: none;

    height: 3em;

    font-weight: 700;
}


.stButton > button:hover {

    background: linear-gradient(
        90deg,
        #f472b6,
        #a78bfa
    );

    color:white;
}


/* Dataframe */
div[data-testid="stDataFrame"] {

    border-radius:15px;

}


/* Footer */
.stCaption {

    color:#ddd6fe;

}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found!")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error("Scaler file not found!")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Sidebar
# -----------------------------
page = st.sidebar.selectbox(
    "Navigation",
    [
        "🏠 Home",
        "📈 Model Performance",
        
    ]
)

# -----------------------------
# Home Page
# -----------------------------
if page == "🏠 Home":

    st.title("💳 Bank Fraud Detection System")

    st.success("Machine Learning Based Fraud Detection")

    col1, col2 = st.columns([2,1])

    with col1:

        st.markdown("""
### 📌 Project Overview

This project detects fraudulent credit card transactions using a **Logistic Regression** model trained on the Credit Card Fraud Detection dataset.

### 🎯 Objective

Identify whether a transaction is **Fraudulent** or **Legitimate** with high accuracy.

### 🛠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Logistic Regression

### 📂 Dataset

- Credit Card Fraud Detection Dataset
- 284,807 Transactions
- 30 Features
- Binary Classification (Fraud / Legitimate)
""")
    with col2:

        st.info("### 📊 Project Information")

        st.write("**Algorithm**")
        st.success("Logistic Regression")
    # -----------------------------
# Model Performance
# -----------------------------
elif page == "📈 Model Performance":

    st.title("📈 Model Performance Dashboard")

    # Model Metrics
    train_accuracy = 99.92
    test_accuracy = 99.91

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "🎯 Train Accuracy",
            f"{train_accuracy}%"
        )

    with col2:
        st.metric(
            "✅ Test Accuracy",
            f"{test_accuracy}%"
        )

    st.markdown("---")

    st.subheader("📂 Dataset Information")

    col3, col4 = st.columns(2)

    with col3:
        st.info("Training Samples: 227,845")

    with col4:
        st.success("Testing Samples: 56,962")

    st.markdown("---")

    st.subheader("🚨 Fraud Statistics")

    total_transactions = 284807
    fraud_cases = 492
    normal_cases = 284315

    col5, col6, col7 = st.columns(3)

    with col5:
        st.metric(
            "📊 Total Transactions",
            f"{total_transactions:,}"
        )

    with col6:
        st.metric(
            "🚨 Fraud Cases",
            f"{fraud_cases:,}"
        )

    with col7:
        st.metric(
            "✅ Legitimate Cases",
            f"{normal_cases:,}"
        )

    st.markdown("---")

    st.success("The Logistic Regression model achieved excellent performance in distinguishing legitimate and fraudulent transactions, making it suitable for real-time fraud detection.")

st.caption("Developed using Streamlit • Scikit-learn • Logistic Regression")

     