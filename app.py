import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_shap import st_shap  # Menggunakan library eksternal
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------------------------------------------
# KONFIGURASI HALAMAN & JUDUL
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Mortalitas COVID-19",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# FUNGSI CACHING UNTUK MEMUAT ASET (MODEL, GAMBAR, METRIK)
# -------------------------------------------------------------------

# Menggunakan cache_resource untuk model/explainer (objek kompleks)
@st.cache_resource
def load_model_assets():
    """Memuat model pipeline, explainer, dan nama fitur."""
    try:
        model = joblib.load('mortality_model.pkl')
        explainer = joblib.load('shap_explainer.pkl')
        expected_value = joblib.load('shap_expected_value.pkl')
        original_features = joblib.load('original_feature_names.pkl')
        processed_features = joblib.load('feature_names.pkl')
        # Kita perlu import shap di sini agar joblib bisa me-load explainer-nya
        import shap 
        return model, explainer, expected_value, original_features, processed_features
    except FileNotFoundError:
        st.error("ERROR: File model (.pkl) tidak ditemukan.")
        st.stop()
    except Exception as e:
        # Menangkap error jika 'shap' tidak terinstall saat load
        if 'shap' in str(e):
             st.error("ERROR: Library 'shap' tidak terinstall. Harap install 'shap' (pip install shap) agar bisa memuat file explainer.")
             st.stop()
        st.error(f"Error saat memuat aset: {e}")
        st.stop()


# Menggunakan cache_data untuk data sederhana (kamus, gambar)
@st.cache_data
def load_display_assets():
    """Memuat gambar dan metrik untuk ditampilkan."""
    try:
        img_roc = Image.open('output_roc_curve.png')
        img_cm = Image.open('output_confusion_matrix.png')
        metrics = joblib.load('model_metrics.pkl')
        return img_roc, img_cm, metrics
    except FileNotFoundError:
        st.error("ERROR: File gambar (output_*.png) atau metrik (.pkl) tidak ditemukan.")
        st.stop()

# Memuat semua aset
model, explainer, expected_value, original_features, processed_features = load_model_assets()
img_roc, img_cm, metrics = load_display_assets()

# -------------------------------------------------------------------
# SIDEBAR - INPUT DATA PASIEN
# -------------------------------------------------------------------
st.sidebar.title("👨‍⚕️ Input Data Pasien")
st.sidebar.markdown("Masukkan fitur pasien di bawah ini untuk prediksi.")

# Peta untuk opsi boolean (Ya=1, Tidak=0, Tidak Tahu=99/NaN)
opsi_map = {1: 'Ya', 0: 'Tidak', 99: 'Tidak Tahu'}
opsi_val = [1, 0, 99]

# --- Input Numerik ---
age = st.sidebar.number_input("Usia (AGE)", min_value=0, max_value=120, value=50, step=1)

# --- Input Kategorikal Dasar ---

# Model dilatih: 0 = Pria, 1 = Wanita
sex = st.sidebar.selectbox(
    "Jenis Kelamin (SEX)", 
    [0, 1], # Default 'Pria' (0)
    format_func=lambda x: 'Pria' if x == 0 else 'Wanita'
)

# Model dilatih: 0 = Rawat Inap, 1 = Rawat Jalan
patient_type = st.sidebar.selectbox(
    "Tipe Pasien (PATIENT_TYPE)", 
    [0, 1], # Default 'Rawat Inap' (0)
    format_func=lambda x: 'Rawat Inap' if x == 0 else 'Rawat Jalan'
)

# --- Logika Khusus untuk Hamil ---
if sex == 1: # Jika Wanita (1)
    pregnant = st.sidebar.selectbox(
        "Hamil (PREGNANT)", 
        opsi_val, 
        format_func=lambda x: opsi_map[x],
        index=1 # Default 'Tidak'
    )
else: # Jika Pria (0)
    pregnant = 0 # Otomatis 'Tidak'
    st.sidebar.info("Pasien Pria: 'Hamil' diatur ke 'Tidak'.")

# --- Input Boolean Lainnya (Kondisi Medis) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Riwayat Kondisi Medis")

# Grup 1
col1, col2 = st.sidebar.columns(2)
pneumonia = col1.selectbox("Pneumonia", opsi_val, format_func=lambda x: opsi_map[x], index=1)
diabetes = col2.selectbox("Diabetes", opsi_val, format_func=lambda x: opsi_map[x], index=1)
copd = col1.selectbox("COPD", opsi_val, format_func=lambda x: opsi_map[x], index=1)
asthma = col2.selectbox("Asthma", opsi_val, format_func=lambda x: opsi_map[x], index=1)
inmsupr = col1.selectbox("Imunosupresi (INMSUPR)", opsi_val, format_func=lambda x: opsi_map[x], index=1)
hipertension = col2.selectbox("Hipertensi", opsi_val, format_func=lambda x: opsi_map[x], index=1)
cardiovascular = col1.selectbox("Kardiovaskular", opsi_val, format_func=lambda x: opsi_map[x], index=1)
obesity = col2.selectbox("Obesitas", opsi_val, format_func=lambda x: opsi_map[x], index=1)
renal_chronic = col1.selectbox("Gagal Ginjal Kronis", opsi_val, format_func=lambda x: opsi_map[x], index=1)
tobacco = col2.selectbox("Perokok (TOBACCO)", opsi_val, format_func=lambda x: opsi_map[x], index=1)
other_disease = col1.selectbox("Penyakit Lain", opsi_val, format_func=lambda x: opsi_map[x], index=1)

# --- Kondisi Kritis ---
st.sidebar.markdown("---")
st.sidebar.subheader("Perawatan Kritis (ICU)")
intubed = st.sidebar.selectbox("Terintubasi (INTUBED)", opsi_val, format_func=lambda x: opsi_map[x], index=1)
icu = st.sidebar.selectbox("Masuk ICU", opsi_val, format_func=lambda x: opsi_map[x], index=1)


# -------------------------------------------------------------------
# HALAMAN UTAMA - JUDUL
# -------------------------------------------------------------------
st.title("Dashboard Prediksi Mortalitas Pasien COVID-19")
st.markdown("""
Aplikasi ini menggunakan model *Logistic Regression* untuk memprediksi risiko mortalitas (kematian) 
pasien berdasarkan data klinis. Gunakan panel di sebelah kiri untuk memasukkan data pasien.
""")

# Mengorganisir output ke dalam Tab
tab_prediksi, tab_performa = st.tabs(
    ["📊 Hasil Prediksi Pasien", "📈 Performa Model"]
)


# -------------------------------------------------------------------
# TAB 1: HASIL PREDIKSI & SHAP
# -------------------------------------------------------------------
with tab_prediksi:
    st.header("Hasil Prediksi")

    # Tombol untuk memicu prediksi
    if st.sidebar.button("Prediksi Risiko Pasien", type="primary", use_container_width=True):
        
        # 1. Mengumpulkan data input
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

        # 2. Membuat DataFrame (harus sesuai urutan 'original_features')
        input_df = pd.DataFrame([input_data], columns=original_features)

        # 3. Melakukan Prediksi
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] # Probabilitas kelas '1' (Meninggal)

        # 4. Menampilkan Hasil Prediksi
        st.subheader("Ringkasan Prediksi")
        if prediction == 1:
            st.error(f"**Prediksi: Risiko Tinggi (Meninggal)**", icon="🚨")
        else:
            st.success(f"**Prediksi: Risiko Rendah (Selamat)**", icon="✅")

        st.metric(
            label="Tingkat Risiko Kematian",
            value=f"{probability * 100:.2f} %"
        )
        st.progress(probability)
        st.markdown("---")

        # 5. Menampilkan Penjelasan SHAP
        st.subheader("Faktor Penentu Prediksi (SHAP Analysis)")
        st.markdown("""
        Plot di bawah ini menjelaskan *mengapa* model memberikan prediksi tersebut.
        - **Fitur berwarna merah** (cth: `AGE_scaled > 0.5`) mendorong prediksi ke arah **Risiko Kematian**.
        - **Fitur berwarna biru** mendorong prediksi ke arah **Selamat**.
        - Semakin panjang bar, semakin kuat pengaruhnya.
        """)

        # Proses input data menggunakan preprocessor dari pipeline
        processed_input = model.named_steps['preprocessor'].transform(input_df)
        processed_input_df = pd.DataFrame(processed_input, columns=processed_features)

        # Hitung SHAP values untuk input spesifik ini
        # Kita perlu import shap di sini juga untuk memanggil shap.force_plot dan shap.Explanation
        import shap
        shap_values_local = explainer(processed_input_df)
        
        # Buat Force Plot
        force_plot_object = shap.force_plot(
            base_value=expected_value,
            shap_values=shap_values_local.values[0],
            features=processed_input_df.iloc[0],
            feature_names=processed_features,
            link="logit", # Karena kita pakai Logistic Regression
            out_names="Risiko Kematian"
        )
        
        # Tampilkan plot menggunakan st_shap (tanpa 'scrolling')
        st_shap(force_plot_object, height=160)

        # Tampilkan juga Waterfall Plot (alternatif)
        st.markdown("#### Penjelasan Detail (Waterfall Plot)")
        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_local.values[0],
                base_values=expected_value,
                data=processed_input_df.iloc[0],
                feature_names=processed_features
            ),
            # max_display=15, # Dihapus untuk menghindari bug IndexError
            show=False
        )
        st.pyplot(fig, bbox_inches='tight')

    else:
        st.info("Silakan masukkan data pasien di sidebar kiri dan klik tombol 'Prediksi'.")


# -------------------------------------------------------------------
# TAB 2: PERFORMA MODEL
# -------------------------------------------------------------------
with tab_performa:
    st.header("Evaluasi Kinerja Model (Pada Data Uji)")
    st.markdown("""
    Data ini dihasilkan dari pengujian model pada **20% data (data uji)** yang 
    tidak pernah dilihat sebelumnya saat pelatihan.
    """)
    
    # Memuat metrik
    auc_score = metrics.get('auc', 0)
    report_str = metrics.get('classification_report_string', 'Laporan tidak ditemukan.')
    
    st.metric("Area Under Curve (AUC)", f"{auc_score:.4f}")
    st.markdown("Semakin tinggi (mendekati 1.0) semakin baik model dalam membedakan kelas.")
    st.image(img_roc, caption="Kurva ROC (Receiver Operating Characteristic)")
    
    st.markdown("---")

    st.subheader("Confusion Matrix")
    st.markdown("Menunjukkan seberapa sering model benar atau salah dalam memprediksi.")
    st.image(img_cm, caption="Confusion Matrix (Data Uji)")

    st.markdown("---")
    
    st.subheader("Classification Report")
    st.markdown("Rangkuman metrik presisi, recall, dan f1-score untuk setiap kelas.")
    st.code(report_str, language='text')

    # ==============================================================
    # >>>>>>>> BAGIAN BARU YANG DIPERBARUI <<<<<<<<<<
    # ==============================================================
    
    st.markdown("---") # Garis pemisah
    
    st.subheader("Bagaimana Model Ini Bekerja? (Regresi Logistik)")
    
    # --- Perubahan Teks (Menghilangkan sapaan "Anda") ---
    st.markdown(r"""
    Model yang **digunakan**, **Regresi Logistik**, adalah metode statistik yang sangat baik untuk prediksi 'Ya' atau 'Tidak' (dalam kasus ini, 'Meninggal' atau 'Selamat').
    
    Cara kerjanya dapat disederhanakan menjadi 3 langkah:
    
    1.  **Menghitung Skor:** Model mengambil semua fitur input (Usia, Pneumonia, ICU, dll.) dan memberinya 'bobot' (beberapa menambah risiko, beberapa mengurangi). Ini semua dijumlahkan untuk membuat satu 'skor' (disebut *log-odds*).
    
    2.  **Konversi ke Probabilitas:** Skor ini kemudian 'dipetakan' ke **kurva berbentuk 'S'** (disebut fungsi sigmoid). Kurva ini 'memaksa' skor tersebut menjadi nilai probabilitas antara 0% (pasti selamat) dan 100% (pasti meninggal).
    
    3.  **Membuat Keputusan:** Model menggunakan ambang batas (threshold) 50%. Jika probabilitas risiko kematian yang dihitung **di atas 50%**, model akan memprediksi 'Meninggal' (1). Jika **di bawah 50%**, model akan memprediksi 'Selamat' (0).
    
    Inilah mengapa pada tab 'Hasil Prediksi', **ditampilkan** persentase probabilitas, bukan hanya keputusan akhir.
    """)

    # --- [START] Kode Plot Sigmoid yang Lebih Menarik ---
    
    # 1. Buat data untuk plot
    x_sig = np.linspace(-7, 7, 200)
    y_sig = 1 / (1 + np.exp(-x_sig)) # Fungsi Sigmoid

    # 2. Buat Figure
    fig_sigmoid, ax_sig = plt.subplots(figsize=(10, 4), dpi=150)

    # 3. Plot kurva
    ax_sig.plot(x_sig, y_sig, label="Kurva Probabilitas (Sigmoid)", color="#0068c9", linewidth=2.5)

    # 4. Tambahkan elemen "menarik"
    # Garis ambang batas 50%
    ax_sig.axhline(y=0.5, color="red", linestyle="--", label="Ambang Batas Keputusan (50%)")
    ax_sig.axvline(x=0, color="#666666", linestyle=":", alpha=0.7)

    # Kustomisasi Ticks (Sumbu)
    ax_sig.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_sig.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax_sig.set_xlabel("Skor Gabungan (Semakin ke kanan, risiko semakin tinggi)")
    ax_sig.set_ylabel("Probabilitas Risiko")

    # Tambahkan anotasi zona
    ax_sig.fill_between(x_sig, 0, y_sig, where=(x_sig < 0), color="blue", alpha=0.05)
    ax_sig.fill_between(x_sig, y_sig, 1, where=(x_sig > 0), color="red", alpha=0.05)
    
    # Tambahkan teks anotasi
    ax_sig.text(-3.5, 0.25, "Prediksi: Selamat", ha='center', va='center', color='blue', fontsize=10, weight='bold')
    ax_sig.text(3.5, 0.75, "Prediksi: Meninggal", ha='center', va='center', color='red', fontsize=10, weight='bold')

    # 5. Rapikan
    ax_sig.set_title("Visualisasi Keputusan Regresi Logistik", fontsize=14)
    ax_sig.legend(loc="lower right")
    ax_sig.grid(alpha=0.2)
    plt.tight_layout()

    # 6. Tampilkan plot di Streamlit
    st.pyplot(fig_sigmoid)