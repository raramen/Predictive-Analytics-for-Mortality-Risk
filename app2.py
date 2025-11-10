import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
from PIL import Image
import shap

# ============================================================== #
# KONFIGURASI HALAMAN
# ============================================================== #
st.set_page_config(
    page_title="Prediksi Mortalitas COVID-19",
    page_icon="🦠",
    layout="wide"
)

# ============================================================== #
# CSS TAMBAHAN UNTUK UI
# ============================================================== #
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f9f9f9;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .risk-high {
        color: #e63946;
        font-weight: bold;
    }
    .risk-low {
        color: #2a9d8f;
        font-weight: bold;
    }
    .highlight-box {
        background-color: #f1f3f4;
        border-left: 5px solid #0068c9;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================== #
# LOAD MODEL & ASSETS
# ============================================================== #
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load('mortality_model.pkl')
        explainer = joblib.load('shap_explainer_small.pkl')
        expected_value = joblib.load('shap_expected_value.pkl')
        original_features = joblib.load('original_feature_names.pkl')
        processed_features = joblib.load('feature_names.pkl')
        return model, explainer, expected_value, original_features, processed_features
    except Exception as e:
        st.error(f"Gagal memuat aset model: {e}")
        st.stop()

@st.cache_data
def load_display_assets():
    img_roc = Image.open('output_roc_curve.png')
    img_cm = Image.open('output_confusion_matrix.png')
    metrics = joblib.load('model_metrics.pkl')
    return img_roc, img_cm, metrics

model, explainer, expected_value, original_features, processed_features = load_model_assets()
img_roc, img_cm, metrics = load_display_assets()

# ============================================================== #
# SIDEBAR INPUT
# ============================================================== #
st.sidebar.title("👨‍⚕️ Input Data Pasien")

opsi_map = {1: 'Ya', 0: 'Tidak'}
opsi_val = [1, 0]

age = st.sidebar.number_input("Usia (AGE)", min_value=0, max_value=120, value=45)
sex = st.sidebar.selectbox("Jenis Kelamin (SEX)", [0, 1], format_func=lambda x: 'Pria' if x == 0 else 'Wanita')
patient_type = st.sidebar.selectbox("Tipe Pasien (PATIENT_TYPE)", [0, 1], format_func=lambda x: 'Rawat Inap' if x == 0 else 'Rawat Jalan')

if sex == 1:
    pregnant = st.sidebar.selectbox("Hamil (PREGNANT)", opsi_val, format_func=lambda x: opsi_map[x], index=1)
else:
    pregnant = 0
    st.sidebar.info("Pasien Pria: kolom 'Hamil' otomatis diatur ke 'Tidak'.")

st.sidebar.markdown("---")
st.sidebar.subheader("Riwayat Kondisi Medis")

col1, col2 = st.sidebar.columns(2)
pneumonia = col1.selectbox("Pneumonia", opsi_val, format_func=lambda x: opsi_map[x], index=1)
diabetes = col2.selectbox("Diabetes", opsi_val, format_func=lambda x: opsi_map[x], index=1)
copd = col1.selectbox("COPD", opsi_val, format_func=lambda x: opsi_map[x], index=1)
asthma = col2.selectbox("Asthma", opsi_val, format_func=lambda x: opsi_map[x], index=1)
inmsupr = col1.selectbox("Imunosupresi", opsi_val, format_func=lambda x: opsi_map[x], index=1)
hipertension = col2.selectbox("Hipertensi", opsi_val, format_func=lambda x: opsi_map[x], index=1)
cardiovascular = col1.selectbox("Kardiovaskular", opsi_val, format_func=lambda x: opsi_map[x], index=1)
obesity = col2.selectbox("Obesitas", opsi_val, format_func=lambda x: opsi_map[x], index=1)
renal_chronic = col1.selectbox("Gagal Ginjal Kronis", opsi_val, format_func=lambda x: opsi_map[x], index=1)
tobacco = col2.selectbox("Perokok", opsi_val, format_func=lambda x: opsi_map[x], index=1)
other_disease = col1.selectbox("Penyakit Lain", opsi_val, format_func=lambda x: opsi_map[x], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Perawatan Kritis")
intubed = st.sidebar.selectbox("Terintubasi", opsi_val, format_func=lambda x: opsi_map[x], index=1)
icu = st.sidebar.selectbox("Masuk ICU", opsi_val, format_func=lambda x: opsi_map[x], index=1)

# ============================================================== #
# TAB UTAMA
# ============================================================== #
st.title("📊 Dashboard Prediksi Mortalitas Pasien COVID-19")

tab_prediksi, tab_performa = st.tabs(["🔮 Prediksi Pasien", "📈 Performa Model"])

# ============================================================== #
# TAB PREDIKSI
# ============================================================== #
with tab_prediksi:
    st.header("Hasil Prediksi")

    if st.sidebar.button("🚀 Jalankan Prediksi", use_container_width=True):
        with st.spinner("Model sedang memproses data pasien... ⏳"):
            input_data = {
                'AGE': age,
                'SEX': sex,
                'PATIENT_TYPE': patient_type,
                'PNEUMONIA': np.nan if pneumonia == 99 else pneumonia,
                'PREGNANT': np.nan if pregnant == 99 else pregnant,
                'DIABETES': np.nan if diabetes == 99 else diabetes,
                'COPD': np.nan if copd == 99 else copd,
                'ASTHMA': np.nan if asthma == 99 else asthma,
                'INMSUPR': np.nan if inmsupr == 99 else inmsupr,
                'HIPERTENSION': np.nan if hipertension == 99 else hipertension,
                'OTHER_DISEASE': np.nan if other_disease == 99 else other_disease,
                'CARDIOVASCULAR': np.nan if cardiovascular == 99 else cardiovascular,
                'OBESITY': np.nan if obesity == 99 else obesity,
                'RENAL_CHRONIC': np.nan if renal_chronic == 99 else renal_chronic,
                'TOBACCO': np.nan if tobacco == 99 else tobacco,
                'INTUBED': np.nan if intubed == 99 else intubed,
                'ICU': np.nan if icu == 99 else icu
            }

            input_df = pd.DataFrame([input_data], columns=original_features)
            processed_input = model.named_steps['preprocessor'].transform(input_df)
            processed_input_df = pd.DataFrame(processed_input, columns=processed_features)
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

        colA, colB = st.columns([1, 2])
        with colA:
            st.markdown("### 🎯 Ringkasan Hasil")

            risk_score = probability * 100
            if risk_score < 33:
                color = "#d4edda"; text_color = "#155724"; emoji = "✅"; message = "Risiko Rendah (Selamat)"
            elif risk_score < 67:
                color = "#fff3cd"; text_color = "#856404"; emoji = "⚠️"; message = "Risiko Sedang (Perlu Perhatian)"
            else:
                color = "#f8d7da"; text_color = "#721c24"; emoji = "🚨"; message = "Risiko Tinggi (Berisiko Tinggi)"

            st.markdown(
                f"""
                <div style='
                    background-color:{color};
                    color:{text_color};
                    padding:20px;
                    border-radius:12px;
                    margin-top:10px;
                    font-size:20px;
                    font-weight:bold;
                    text-align:center;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                '>
                    {emoji} {message}<br>
                    <span style='font-size:16px; font-weight:normal;'>Skor Risiko: {risk_score:.2f}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric("Tingkat Risiko Kematian", f"{risk_score:.2f}%")
            st.progress(probability)

        with colB:
            st.markdown("### 🔍 Faktor Penting (SHAP)")
            shap_values_local = explainer(processed_input_df)
            force_plot = shap.force_plot(
                expected_value, shap_values_local.values[0],
                features=processed_input_df.iloc[0],
                feature_names=processed_features, link="logit"
            )
            st_shap(force_plot, height=200)

        st.markdown("---")
        st.markdown("### 💧 Visualisasi Waterfall (Detail Pengaruh Fitur)")
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_local.values[0],
                base_values=expected_value,
                data=processed_input_df.iloc[0],
                feature_names=processed_features
            ),
            show=False
        )
        st.pyplot(fig, bbox_inches="tight")

    else:
        st.info("Isi data di sidebar, lalu klik **🚀 Jalankan Prediksi** untuk melihat hasil.")

# ============================================================== #
# TAB PERFORMA MODEL (SEJAJAR + AUC SCORE KOTAK)
# ============================================================== #
with tab_performa:
    st.header("📈 Performa Model Prediksi")

    # Ambil metrik dari hasil evaluasi
    auc_score = metrics.get('auc', 0)
    report_str = metrics.get('classification_report_string', 'Laporan tidak ditemukan.')

    # ==============================================================
    # Kotak AUC
    # ==============================================================
    st.markdown("""
        <div style="
            background-color:#f1f3f4;
            padding: 15px 25px;
            border-radius: 10px;
            display: inline-block;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            margin-top: 15px;
        ">
            <span style="font-size:18px; color:#202124;">
                <b>🎯 AUC (Area Under Curve):</b> {:.4f}
            </span>
        </div>
    """.format(auc_score), unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)  # tambahkan jarak ke bawah

    # ==============================================================
    # ROC Curve & Confusion Matrix sejajar (TANPA BURAM)
    # ==============================================================
    col1, col2 = st.columns(2)

    with col1:
        st.image("output_roc_curve.png", caption="Kurva ROC", use_container_width=True)

    with col2:
        st.image("output_confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==============================================================
    # Penjelasan Logistic Regression
    # ==============================================================
    st.subheader("🧠 Penjelasan Model Logistic Regression")
    st.markdown(
        """
        <div class='highlight-box' style='line-height:1.6; font-size:16px;'>
        <b>Logistic Regression</b> digunakan untuk memprediksi probabilitas kejadian biner —
        misalnya apakah pasien <b>meninggal atau selamat</b> akibat COVID-19.
        Model ini menggunakan fungsi <i>sigmoid</i> untuk mengubah kombinasi linier dari fitur menjadi nilai antara 0 dan 1.
        <br><br>
        Rumus dasarnya:
        <br><code>p = 1 / (1 + e<sup>-(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)</sup>)</code>
        <br><br>
        Jika p mendekati 1 → pasien berisiko tinggi<br>
        Jika p mendekati 0 → pasien berisiko rendah
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==============================================================
    # Kurva sigmoid statis
    # ==============================================================
    x = np.linspace(-7, 7, 200)
    y = 1 / (1 + np.exp(-x))

    fig_sig, ax_sig = plt.subplots(figsize=(8, 4), dpi=150)
    ax_sig.plot(x, y, linewidth=2.5, color="#0068c9")
    ax_sig.set_title("Kurva Fungsi Sigmoid (Logistic Regression)", fontsize=12, pad=10)
    ax_sig.set_xlabel("Skor Model (log-odds)")
    ax_sig.set_ylabel("Probabilitas Risiko")
    ax_sig.grid(alpha=0.2)
    st.pyplot(fig_sig)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Laporan Klasifikasi")
    st.code(report_str, language='text')