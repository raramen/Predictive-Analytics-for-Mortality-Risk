import joblib
import shap
import pandas as pd

# 1️⃣ Load model pipeline
model = joblib.load("mortality_model.pkl")

# 2️⃣ Ambil data sample kecil
X = pd.read_csv("Covid Data.csv").sample(100, random_state=42)

# 3️⃣ Dapatkan data setelah preprocessing (karena pipeline punya preprocessor)
X_preprocessed = model.named_steps['preprocessor'].transform(X)

# 4️⃣ Ambil classifier di dalam pipeline
clf = model.named_steps['classifier']

# 5️⃣ Buat explainer yang sesuai
explainer = shap.LinearExplainer(clf, X_preprocessed)
expected_value = explainer.expected_value

# 6️⃣ Simpan versi kecil
joblib.dump(explainer, "shap_explainer_small.pkl")
joblib.dump(expected_value, "shap_expected_value.pkl")

print("✅ SHAP explainer versi kecil berhasil disimpan.")