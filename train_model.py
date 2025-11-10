import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Opsional: Mempercepat Sklearn
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("Intel(R) Extension for Scikit-learn* diaktifkan.")
except ImportError:
    print("Intel(R) Extension for Scikit-learn* tidak ditemukan. Lanjut tanpa patch.")


print("--- [FASE 1: EXTRACT] ---")
print("Memuat dataset 'Covid Data.csv'...")
try:
    df = pd.read_csv('Covid Data.csv')
    print("Dataset berhasil dimuat.")
except FileNotFoundError:
    print("ERROR: 'Covid Data.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")
    exit()

print("\n--- [FASE 2: TRANSFORM] ---")
print("Memulai proses ETL (Cleaning, Target Engineering)...")

# 2.1. Target Engineering
df['DEATH_EVENT'] = df['DATE_DIED'].apply(lambda x: 0 if x == '9999-99-99' else 1)
df = df.drop('DATE_DIED', axis=1)
print("- Kolom target 'DEATH_EVENT' dibuat.")

# 2.2. Feature Cleaning
bool_cols = [
    'SEX', 'PATIENT_TYPE', 'PNEUMONIA', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 
    'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR',
    'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'INTUBED', 'ICU'
]

# Antisipasi jika di CSV masih ada typo
if 'OTHER_DISESAE' in df.columns:
    df = df.rename(columns={'OTHER_DISESAE': 'OTHER_DISEASE'})
    
num_cols = ['AGE']
features = num_cols + bool_cols

print(f"- Membersihkan {len(bool_cols)} fitur boolean (2->0, 97/99->NaN)...")
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].replace(2, 0)
        df[col] = df[col].replace(97, np.nan)
        df[col] = df[col].replace(99, np.nan)
    else:
        print(f"Peringatan: Kolom '{col}' tidak ditemukan di CSV.")
        features.remove(col) # Hapus dari daftar fitur jika tidak ada

df.loc[df['SEX'] == 0, 'PREGNANT'] = 0
print("- Logika 'PREGNANT' untuk 'Male' diterapkan.")
print("Proses Transform selesai.")

# 2.3. Data Splitting
print("\nMembagi data (80% Train, 20% Test)...")
X = df[features]
y = df['DEATH_EVENT']
X = X[y.notna()]
y = y[y.notna()]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"- Ukuran Data Latih: {X_train.shape[0]} baris")
print(f"- Ukuran Data Uji: {X_test.shape[0]} baris")


print("\n--- [FASE 3: LOAD & TRAIN] ---")
print("Membuat Preprocessing Pipeline...")
# Pipeline untuk 'AGE' (isi NaN dengan rata-rata, lalu scaling)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# Pipeline untuk kolom boolean (isi NaN dengan nilai yg paling sering muncul)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])
# Gabungkan pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, bool_cols)
    ],
    remainder='drop'
)

# Membuat Model Pipeline Lengkap
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'))
])

print("Melatih Model Regresi Logistik...")
model_pipeline.fit(X_train, y_train)
print("Model berhasil dilatih.")


print("\n--- [FASE 4: EVALUATE & SAVE OUTPUTS] ---")
print("Mengevaluasi model pada data uji...")
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# 4.1. Menghitung Metrics
auc = roc_auc_score(y_test, y_pred_proba)
report_dict = classification_report(y_test, y_pred, output_dict=True, target_names=['Selamat', 'Meninggal'])
conf_matrix = confusion_matrix(y_test, y_pred)
metrics = {
    "auc": auc,
    "classification_report_dict": report_dict,
    "classification_report_string": classification_report(y_test, y_pred, target_names=['Selamat', 'Meninggal']),
    "confusion_matrix": conf_matrix
}
print(f"- AUC: {auc:.4f}")
print(f"- Accuracy: {report_dict['accuracy']:.4f}")

# 4.2. Menyimpan Aset Model (.pkl)
joblib.dump(model_pipeline, 'mortality_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(metrics, 'model_metrics.pkl')
print("- File 'mortality_model.pkl', 'preprocessor.pkl', dan 'model_metrics.pkl' disimpan.")

# 4.3. [BARU] Menyimpan Visualisasi AUC (ROC Curve)
print("- Membuat 'output_roc_curve.png'...")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Model Regresi Logistik (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Garis Referensi (Acak)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kurva ROC (Receiver Operating Characteristic)', fontsize=16)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('output_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 4.4. [BARU] Menyimpan Visualisasi Akurasi (Confusion Matrix)
print("- Membuat 'output_confusion_matrix.png'...")
plt.figure(figsize=(10, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Selamat', 'Meninggal'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Validasi Akurasi Model)', fontsize=16)
plt.savefig('output_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 4.5. [BONUS] Menyimpan Aset XAI (SHAP Explainer)
print("- Membuat dan menyimpan aset SHAP Explainer (mungkin perlu waktu)...")
processed_feature_names = num_cols + list(preprocessor.named_transformers_['cat']
                                          .named_steps['imputer']
                                          .get_feature_names_out(bool_cols))
X_train_processed = preprocessor.transform(X_train)
X_train_processed_df = pd.DataFrame(X_train_processed, columns=processed_feature_names)
explainer = shap.LinearExplainer(
    model_pipeline.named_steps['classifier'],
    X_train_processed_df
)
shap_values_global = explainer(X_train_processed_df)

joblib.dump(explainer, 'shap_explainer.pkl')
joblib.dump(explainer.expected_value, 'shap_expected_value.pkl')
joblib.dump(processed_feature_names, 'feature_names.pkl')
joblib.dump(features, 'original_feature_names.pkl')
joblib.dump(shap_values_global, 'shap_values_global.pkl')
print("- Aset SHAP ('shap_*.pkl', 'feature_names.pkl') berhasil disimpan.")


print("\n--- [PROSES SELESAI] ---")
print("Semua file aset (model, metrik, visualisasi) telah berhasil dibuat.")