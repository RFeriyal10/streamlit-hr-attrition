
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
def load_data():
    df = pd.read_csv("HR_FINPRO.csv")  # Sesuaikan dengan path dataset Anda
    return df

@st.cache_resource
def load_model():
    return joblib.load('rf_top_10.joblib')

model = load_model()


df = load_data()

# Mapping label asli dari data yang telah dienkode
label_mapping = {
    "Attrition": {0: "No", 1: "Yes"},
}

# Ubah kembali nilai yang sudah dienkode menjadi label asli
for column, mapping in label_mapping.items():
    if column in df.columns:
        df[column] = df[column].map(mapping)

# Sidebar
st.sidebar.title("HR Attrition Prediction")
menu = st.sidebar.radio("Pilih Menu:", ["Home", "Prediction"])


st.title("HR Attrition Prediction")

# =============================== HOME ===============================
if menu == "Home":
    st.subheader("Dashboard Karyawan Berdasarkan Attrition")
    attrition_status = st.selectbox("Pilih Status Attrition:", df["Attrition"].unique())
    filtered_df = df[df["Attrition"] == attrition_status]

    # Menampilkan diagram Top 10 Feature berdasarkan Feature Importance
    if model is not None and hasattr(model, "feature_importances_"):
        st.write("### 🔍 Top 10 Fitur Berdasarkan Importance")
        try:
            # Harus sesuai urutan input ke model
            feature_names = ['MaritalStatus_Single', 'JobLevelSatisfaction', 'MonthlyIncome', 
                             'StockOptionLevel', 'JobInvolvement', 'EmployeeSatisfaction', 
                             'DailyRate', 'DistanceFromHome', 'Age', 'EnvironmentSatisfaction' 
            ]
            importances = model.feature_importances_
            feature_df = pd.DataFrame({
                "Fitur": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(10)

            # Plot
            plt.figure(figsize=(10, 5))
            sns.barplot(x="Importance", y="Fitur", data=feature_df, palette="viridis")
            plt.title("Top 10 Fitur Berdasarkan Importance")
            plt.tight_layout()
            st.pyplot(plt)

            # Tabel
            st.write("### 📋 Tabel Top 10 Fitur")
            st.dataframe(feature_df.reset_index(drop=True))

        except Exception as e:
            st.error(f"Gagal memuat feature importance: {e}")
    else:
        st.warning("Model belum dimuat atau tidak memiliki atribut feature_importances_.")


# ============================ PREDICTION ============================
elif menu == "Prediction":
    # Load Model Random Forest (safe loading)
    if model is None:
        st.error("Model not loaded.")
    else:
        st.title('Employee Attrition Prediction')
        st.markdown("""
        <div style='background-color: black; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
            <span style='color:white'><b>This app predicts employee attrition based on key work-related factors 🚀. Enter the details and get an instant prediction!</b></span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Enter Employee Details")

    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        with col1:
            MaritalStatus_Single = st.checkbox("Single (centang jika ya)", value=False)
            JobLevelSatisfaction = st.selectbox("Job Level Satisfaction (1–4)", [1, 2, 3, 4])
            MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
            StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
            JobInvolvement = st.selectbox("Job Involvement (1–4)", [1, 2, 3, 4])
        with col2:
            EmployeeSatisfaction = st.selectbox("Employee Satisfaction (1–4)", [1, 2, 3, 4])
            DailyRate = st.number_input("Daily Rate (opsional)", min_value=0, max_value=1500, value=0)
            DistanceFromHome = st.number_input("Distance From Home (km)", 0, 100, 10)
            Age = st.number_input("Age", min_value=18, max_value=60, value=30)
            EnvironmentSatisfaction = st.selectbox("Environment Satisfaction (1–4)", [1, 2, 3, 4])

        # ✅ TOMBOL SUBMIT di dalam form
        submit = st.form_submit_button("Predict")

    # Setelah tombol ditekan
    if submit:
        input_data = pd.DataFrame([[
            MaritalStatus_Single,
            JobLevelSatisfaction,
            MonthlyIncome,
            StockOptionLevel,
            JobInvolvement,
            EmployeeSatisfaction,
            DailyRate,
            DistanceFromHome,
            Age,
            EnvironmentSatisfaction
        ]], columns=[
            'MaritalStatus_Single', 'JobLevelSatisfaction', 'MonthlyIncome', 
            'StockOptionLevel', 'JobInvolvement', 'EmployeeSatisfaction', 
            'DailyRate', 'DistanceFromHome', 'Age', 'EnvironmentSatisfaction',
        ])

        prediction = model.predict(input_data)[0]
        result = "✅ Employee is likely to stay." if prediction == 0 else "⚠️ Employee is likely to leave."
        st.success(f"Prediction Result: {result}")
