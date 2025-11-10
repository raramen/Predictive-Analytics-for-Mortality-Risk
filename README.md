# Prototipe Model Prediksi Mortalitas COVID-19

Ini adalah proyek *machine learning* yang bertujuan untuk memprediksi risiko mortalitas (kematian) pasien COVID-19 berdasarkan data klinis.

Proyek ini mencakup alur kerja *end-to-end*:
1.  **ETL (Extract, Transform, Load):** Membersihkan dan mempersiapkan 1 juta+ data pasien.
2.  **Model Training:** Melatih model Regresi Logistik untuk memprediksi risiko.
3.  **Model Evaluation:** Memvalidasi performa model menggunakan Akurasi, AUC, dan Confusion Matrix.
4.  **Explainable AI (XAI):** Membuat *explainer* (penjelasan) model menggunakan SHAP untuk memahami *mengapa* model membuat keputusan tertentu.

Tujuan dari *repository* ini adalah untuk menyediakan "pabrik" model (`train_model.py`) yang dapat dijalankan untuk menghasilkan semua aset (file `.pkl` dan `.png`) yang diperlukan untuk menjalankan prototipe di aplikasi lain.

## 🚀 1. Cara Menjalankan (Prasyarat)

Untuk mereplikasi hasil dan menghasilkan semua file aset, ikuti langkah-langkah berikut:

### Prasyarat
1.  **Python:** Pastikan Anda memiliki Python 3.9 atau lebih baru.
2.  **Data Mentah:** Unduh file `Covid Data.csv` (58MB) dan letakkan di folder yang sama dengan `train_model.py`.
3.  **Libraries:** Instal semua *library* yang diperlukan menggunakan file `requirements.txt`.

    ```bash
    # Buka terminal Anda di folder proyek
    pip install -r requirements.txt
    ```

### Menjalankan Pelatihan
Setelah semua prasyarat terpenuhi, jalankan skrip `train_model.py` dari terminal Anda:

```bash
python train_model.py
```

Skrip ini akan memakan waktu beberapa menit. Ia akan menjalankan seluruh proses ETL, melatih model, dan menyimpan semua file output di folder yang sama. Anda akan melihat log proses di terminal Anda.

---

## 📦 2. Penjelasan File Output (Aset Model)

Setelah skrip `train_model.py` selesai, Anda akan mendapatkan file-file berikut. Ini adalah "output" dari model Anda.

### A. Visualisasi Performa (Untuk Presentasi / Frontend)

File `.png` ini adalah bukti visual dari performa model. Anda bisa menampilkannya langsung di website atau laporan.

* `output_roc_curve.png`
    * **Apa ini:** Gambar Kurva ROC (Receiver Operating Characteristic).
    * **Artinya:** Menunjukkan seberapa baik model dapat membedakan antara pasien yang akan 'Selamat' (0) vs 'Meninggal' (1). Garis biru yang mendekati sudut kiri atas adalah ideal. Skor **AUC (Area Under Curve)** kita adalah ~0.94, yang berarti **Excellent**.

* `output_confusion_matrix.png`
    * **Apa ini:** Gambar Confusion Matrix (Matriks Kebingungan).
    * **Artinya:** Ini adalah visualisasi dari akurasi. Ini menunjukkan:
        * **Kiri Atas (True Negative):** Pasien 'Selamat' yang diprediksi 'Selamat' (Benar).
        * **Kanan Bawah (True Positive):** Pasien 'Meninggal' yang diprediksi 'Meninggal' (Benar).
        * **Kanan Atas (False Positive):** Pasien 'Selamat' tapi diprediksi 'Meninggal' (Kesalahan Tipe I).
        * **Kiri Bawah (False Negative):** Pasien 'Meninggal' tapi diprediksi 'Selamat' (Kesalahan Tipe II - Ini adalah error paling kritis dalam konteks medis).

### B. Aset Model (Untuk Prediksi Backend)

File `.pkl` ini adalah "otak" dari model Anda. Gunakan ini di *backend* website Anda untuk membuat prediksi baru.

* `mortality_model.pkl`: **(File Paling Penting)** Ini adalah *pipeline* model lengkap. Isinya sudah termasuk *preprocessor* DAN *classifier*. **Anda hanya perlu memuat file ini untuk membuat prediksi.**
* `preprocessor.pkl`: (Opsional) Ini adalah *pipeline* preprocessor saja, jika Anda perlu memproses data tanpa memprediksi.
* `model_metrics.pkl`: File yang berisi semua metrik (AUC, Akurasi, Presisi, Recall, dll) dalam format data (dictionary) untuk dianalisis lebih lanjut.

### C. Aset XAI (Untuk Fitur "Explainable AI")

File-file ini digunakan untuk menjalankan fitur *Explainable AI* (menjelaskan *mengapa* model membuat keputusan).

* `shap_explainer.pkl`: Model *explainer* yang sudah dilatih (untuk membuat plot *waterfall*).
* `shap_expected_value.pkl`: Nilai risiko dasar (rata-rata) dari populasi.
* `shap_values_global.pkl`: Wawasan global tentang fitur apa yang paling berpengaruh (untuk plot *summary*).
* `feature_names.pkl`: Daftar nama fitur setelah diproses (misal: `PATIENT_TYPE_1`).
* `original_feature_names.pkl`: Daftar nama fitur asli (misal: `PATIENT_TYPE`).

---

## 💻 3. Cara Menggunakan Model di Aplikasi Lain (Contoh Backend)

Ini adalah contoh kode Python yang menunjukkan cara memuat model `mortality_model.pkl` di *backend* website Anda (misal: menggunakan Flask, Django, atau FastAPI) untuk memprediksi data pasien baru secara *real-time*.

```python
import joblib
import pandas as pd
import numpy as np

# 1. Muat model pipeline LENGKAP saat aplikasi Anda dimulai
# (Hanya perlu 'mortality_model.pkl')
try:
    model = joblib.load("mortality_model.pkl")
    print("Model prediksi berhasil dimuat.")
except FileNotFoundError:
    print("File 'mortality_model.pkl' tidak ditemukan! Jalankan 'train_model.py' terlebih dahulu.")
    model = None # Handle error di aplikasi Anda

def get_prediction_from_data(data_pasien_dict):
    """
    Fungsi untuk mengambil data input (dictionary) dan mengembalikan probabilitas risiko.
    """
    
    if model is None:
        return {"error": "Model tidak ter-load.", "risk_score_percent": None}

    # 2. Ubah data input (misal dari JSON/form) menjadi DataFrame
    # Pastikan nama kolom SAMA PERSIS dengan 'original_feature_names.pkl'
    try:
        input_df = pd.DataFrame([data_pasien_dict])
        
        # 3. Lakukan Prediksi
        # Model pipeline akan otomatis menjalankan preprocessing DAN prediksi
        # `predict_proba` mengembalikan [prob_selamat, prob_meninggal]
        prediction_proba = model.predict_proba(input_df)
        
        # 4. Ambil skor risiko (probabilitas meninggal)
        risk_score = prediction_proba[0][1] * 100 # (e.g., 92.4%)
        
        return {"error": None, "risk_score_percent": round(risk_score, 2)}
        
    except Exception as e:
        return {"error": f"Error saat prediksi: {str(e)}", "risk_score_percent": None}

# --- Contoh Penggunaan ---
# Ini adalah data simulasi dari user di website Anda
data_input_baru = {
    'AGE': 65,
    'SEX': 1,           # 1=Female, 0=Male
    'PATIENT_TYPE': 1,  # 1=Dirawat, 0=Pulang
    'PNEUMONIA': 1,     # 1=Yes, 0=No
    'PREGNANT': 0,
    'DIABETES': 1,
    'COPD': 0,
    'ASTHMA': 0,
    'INMSUPR': 0,
    'HIPERTENSION': 1,
    'OTHER_DISEASE': 0,
    'CARDIOVASCULAR': 0,
    'OBESITY': 1,
    'RENAL_CHRONIC': 0,
    'TOBACCO': 0,
    'INTUBED': 0,
    'ICU': 0
}

hasil = get_prediction_from_data(data_input_baru)

if hasil["error"]:
    print(f"Gagal melakukan prediksi: {hasil['error']}")
else:
    print(f"Hasil Prediksi Risiko Mortalitas: {hasil['risk_score_percent']}%")

```