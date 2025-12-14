import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Customer Subscription & Churn Dashboard",
    layout="centered"
)

st.title("📊 Customer Subscription & Churn Analysis")
st.write(
    "Aplikasi ini menampilkan hasil **klasifikasi subscription** "
    "dan **regresi churn risk** menggunakan metode ensemble Random Forest."
)

st.markdown("---")

# =========================
# UPLOAD CSV
# =========================
uploaded_file = st.file_uploader(
    "📂 Upload CSV hasil prediksi dari Google Colab",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Hasil Prediksi")
    st.dataframe(df.head())

    # =========================
    # KLASIFIKASI
    # =========================
    if "subscription_pred" in df.columns:
        st.subheader("✅ Hasil Klasifikasi Subscription")

        fig1, ax1 = plt.subplots()
        df["subscription_pred"].value_counts().plot(
            kind="bar",
            ax=ax1
        )
        ax1.set_xlabel("Subscription (0 = Tidak, 1 = Ya)")
        ax1.set_ylabel("Jumlah Pelanggan")
        st.pyplot(fig1)

    # =========================
    # REGRESI
    # =========================
    if "churn_risk_pred" in df.columns:
        st.subheader("📈 Hasil Regresi Churn Risk")

        fig2, ax2 = plt.subplots()
        ax2.plot(df["churn_risk_pred"].values)
        ax2.set_ylabel("Churn Risk")
        ax2.set_xlabel("Index Data")
        st.pyplot(fig2)

    # =========================
    # DOWNLOAD
    # =========================
    st.download_button(
        "⬇️ Download Data Hasil",
        df.to_csv(index=False),
        file_name="hasil_prediksi.csv",
        mime="text/csv"
    )
else:
    st.info("Silakan upload file CSV hasil prediksi untuk melihat analisis.")
