import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("laptop_price_model.pkl")

model = load_model()

# ----------------------------
# UI Header
# ----------------------------
st.markdown(
    """
    <h1 style="text-align:center;">💻 Laptop Price Prediction</h1>
    <p style="text-align:center; color:gray;">
    Powered by Gradient Boosting Regression
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("🛠 Laptop Configuration")

with st.sidebar.expander("📌 Basic Details", expanded=True):
    company = st.selectbox("Brand", [
        'Apple', 'HP', 'Dell', 'Lenovo', 'Asus', 'Acer', 'MSI',
        'Toshiba', 'Samsung', 'Razer', 'Microsoft', 'Xiaomi', 'Others'
    ])

    typename = st.selectbox("Laptop Type", [
        'Ultrabook', 'Notebook', 'Gaming',
        '2 in 1 Convertible', 'Workstation', 'Netbook'
    ])

    os = st.selectbox("Operating System", [
        'Windows', 'Mac', 'Others/No OS/Linux'
    ])

with st.sidebar.expander("⚙ Hardware", expanded=True):
    ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32])
    weight = st.slider("Weight (kg)", 0.8, 5.0, 2.0)

    cpu_brand = st.selectbox(
        "CPU Brand",
        ['Intel Core i3', 'Intel Core i5', 'Intel Core i7',
         'Other Intel Processor', 'AMD Processor']
    )

    gpu_brand = st.selectbox(
        "GPU Brand",
        ['Intel', 'Nvidia', 'AMD']
    )

with st.sidebar.expander("🖥 Display & Storage", expanded=True):
    touchscreen = st.radio("Touchscreen", ["No", "Yes"])
    ips = st.radio("IPS Display", ["No", "Yes"])

    ppi = st.slider("PPI (Pixel Density)", 90.0, 300.0, 150.0)

    hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1000, 2000])
    ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1000])

# Convert Yes/No → 0/1
touchscreen = 1 if touchscreen == "Yes" else 0
ips = 1 if ips == "Yes" else 0

# ----------------------------
# Input DataFrame (FIXED)
# ----------------------------
input_df = pd.DataFrame({
    "Unnamed: 0": [0],          # 🔥 FIX FOR ERROR
    "Company": [company],
    "TypeName": [typename],
    "Ram": [ram],
    "Weight": [weight],
    "Touchscreen": [touchscreen],
    "Ips": [ips],
    "ppi": [ppi],
    "Cpu brand": [cpu_brand],
    "HDD": [hdd],
    "SSD": [ssd],
    "Gpu brand": [gpu_brand],
    "os": [os]
})

# ----------------------------
# Prediction Section
# ----------------------------
st.markdown("## 🔮 Predicted Price")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🚀 Predict Laptop Price", use_container_width=True):
        log_price = model.predict(input_df)
        price = np.exp(log_price)[0]

        st.success(f"💰 Estimated Price: ₹ {price:,.2f}")

        st.caption("Prediction based on historical laptop market data")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>"
    "Built with Streamlit • Machine Learning • Gradient Boosting"
    "</p>",
    unsafe_allow_html=True
)