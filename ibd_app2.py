import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Load trained models safely
# -----------------------------
try:
    log_model = joblib.load("logistic_final.pkl")
    svc_model = joblib.load("svc_final.pkl")
    ann_model = load_model("ann_final.h5", compile=False)
    scaler = joblib.load("scaler_final.pkl")
except FileNotFoundError as e:
    st.error(f"Error: Missing model file - {e}")
    st.stop()

# -----------------------------
# Get exact feature names
# -----------------------------
try:
    feature_names = list(log_model.feature_names_in_)
except AttributeError:
    try:
        feature_names = scaler.feature_names_in_.tolist()
    except AttributeError:
        feature_names = [
            "WHEAT(CHAPATI,ROTI,NAAN,DALIA,RAWA/SOOJI,SEVIYAAN)",
            "WHEAT FREE CEREALS",
            "FRUITS",
            "OTHER VEGETABLES",
            "STARCHY(POTATO,SWEET PATATO,ARBI ETC)",
            "PULSES AND LEGUMES",
            "PREDOMINANT SATURATED FATS",
            "PREDOMINANT UNSATURATED FATS",
            "TRANS FATS",
            "NUTS AND OILSEEDS",
            "EGGS,FISH AND POULTRY",
            "RED MEAT",
            "MILK",
            "LOW LACTOSE DAIRY",
            "SWEETEND BEVERAGES",
            "ULTRA PROCESSED FOODS",
            "READT TO EAT PACKAGED SNACKS",
            "SAVORY SNACKS",
            "PROCESSED FOODS",
            "INDIAN SWEET MEATS",
            "FOOD SUPPLEMENTS",
            "ERGOGENIC SUPPLEMENTS"
        ]

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="IBD Risk Prediction", layout="wide")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
    <style>
    .stApp { background-color: #FFE4E1; }
    .logo-left, .logo-right { width: 120px; display:block; margin:auto; }
    .institute-name { text-align:center; font-weight:bold; font-size:16px; margin-top:5px; }
    .large-score {
        font-size: 38px !important;
        font-weight: bold;
        color: #8B0000;
        text-align: center;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    .pred-label {
        font-size: 18px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 0;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Logos and Title
# -----------------------------
col_logo_left, col_title, col_logo_right = st.columns([1, 5, 1])

with col_logo_left:
    st.markdown('<img src="https://brandlogovector.com/wp-content/uploads/2022/04/IIT-Delhi-Icon-Logo.png" class="logo-left">', unsafe_allow_html=True)
    st.markdown('<div class="institute-name">Indian Institute of Technology Delhi</div>', unsafe_allow_html=True)

with col_title:
    st.markdown(
        "<h1 style='text-align:center; font-size:36px; color:black;'>DMCH-IITD Machine Learning Tool for Estimating the Diet Percentage Similarity with Respect to Diets Consumed by Inflammatory Bowel Disease Patients Prior to Diagnosis</h1>",
        unsafe_allow_html=True
    )

with col_logo_right:
    st.markdown('<img src="https://tse2.mm.bing.net/th/id/OIP.fNb1hJAUj-8vwANfP3SDJgAAAA?pid=Api&P=0&h=180" class="logo-right">', unsafe_allow_html=True)
    st.markdown('<div class="institute-name">Dayanand Medical College and Hospital Ludhiana</div>', unsafe_allow_html=True)

# -----------------------------
# Intro Paragraph
# -----------------------------
st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:left; font-size:20px; color:black; line-height:1.5;'>
This tool uses a machine learning model to estimate the similarity of your diet with those consumed by patients prior to an Inflammatory Bowel Disease (IBD) diagnosis. It Uses a Logistic Regression, Support Vector Classifier and Artificial Neural Network models to estimate prediction. The ML models were trained based on data from a dietary survey conducted by DMCH Ludhiana among IBD patients and controls without IBD. IBD patients were asked to report their dietary habits prior to diagnosis, and controls were asked to report current food habits.</p>
""", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)

# -----------------------------
# Helper
# -----------------------------
def clean_feature_name(name):
    return name.replace("_", " ").title()

# -----------------------------
# Feature value limits
# -----------------------------
feature_value_limits = {
    "WHEAT(CHAPATI,ROTI,NAAN,DALIA,RAWA/SOOJI,SEVIYAAN": list(range(0, 6)),
    "WHEAT FREE CEREALS": list(range(0, 36)),
    "FRUITS": list(range(0, 21)),
    "OTHER VEGETABLES": list(range(0, 26)),
    "STARCHY(POTATO,SWEET PATATO,ARBI ETC)": list(range(0, 6)),
    "PULSES AND LEGUMES": list(range(0, 16)),
    "PREDOMINANT SATURATED FATS": list(range(0, 11)),
    "PREDOMINANT UNSATURATED FATS": list(range(0, 11)),
    "TRANS FATS": list(range(0, 6)),
    "NUTS AND OILSEEDS": list(range(0, 6)),
    "EGGS,FISH AND POULTRY": list(range(0, 16)),
    "RED MEAT": list(range(0, 6)),
    "MILK ": list(range(0, 6)),
    "LOW LACTOSE DAIRY": list(range(0, 16)),
    "SWEETEND BEVERAGES": list(range(0, 21)),
    "ULTRA PROCESSED FOODS": list(range(0, 76)),
    "READT TO EAT PACKAGED SNACKS": list(range(0, 11)),
    "SAVORY SNACKS": list(range(0, 21)),
    "PROCESSED FOODS": list(range(0, 46)),
    "INDIAN SWEET MEATS": list(range(0, 11)),
    "FOOD SUPPLEMENTS": list(range(0, 26)),
    "ERGOGENIC SUPPLEMENTS": list(range(0, 6))
}

# -----------------------------
# Layout input and output
# -----------------------------
col_input, col_output = st.columns([3, 1])

features = {}
with col_input:
    st.header("In the below fields, provide information about your dietary habits. Select the level of consumption for each food item.")
    n = len(feature_names)
    half = n // 2

    for i in range(half):
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            fname1 = feature_names[i]
            options1 = feature_value_limits.get(fname1.upper(), list(range(38)))  # ✅ use correct limit
            features[fname1] = st.selectbox(
                label=clean_feature_name(fname1),
                options=options1,
                index=0,
                key=fname1
            )
        with c2:
            fname2 = feature_names[i + half]
            options2 = feature_value_limits.get(fname2.upper(), list(range(38)))  # ✅ use correct limit
            features[fname2] = st.selectbox(
                label=clean_feature_name(fname2),
                options=options2,
                index=0,
                key=fname2
            )

# Create DataFrame
input_df = pd.DataFrame([features], columns=feature_names)

# -----------------------------
# Prediction Section
# -----------------------------
with col_output:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Results")
    predict_clicked = st.button("Predict")

    if predict_clicked:
        try:
            scaled_input = scaler.transform(input_df)
            logistic_score = log_model.predict_proba(scaled_input)[0][1] * 100
            svc_score = svc_model.predict_proba(scaled_input)[0][1] * 100
            ann_score = float(ann_model.predict(scaled_input)[0][0]) * 100

            st.markdown("<p style='font-size:18px; font-weight:bold; margin-bottom:5px;'>Similarity Scores (0–100):</p>", unsafe_allow_html=True)

            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                st.markdown("<p class='pred-label'>Logistic</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='large-score'>{logistic_score:.0f}</div>", unsafe_allow_html=True)
            with pcol2:
                st.markdown("<p class='pred-label'>SVC</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='large-score'>{svc_score:.0f}</div>", unsafe_allow_html=True)
            with pcol3:
                st.markdown("<p class='pred-label'>ANN</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='large-score'>{ann_score:.0f}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
