# Lead Scoring App

Aplikasi web untuk memprediksi peluang sebuah *lead* berubah menjadi pelanggan, lanjutan dari Tugas Besar Machine Learning *Lead Scoring: Prediksi Minat Beli*

- **Data:** Kaggle *Lead Scoring Dataset* (`ashydv/leads-dataset`).
- **Model:** pipeline imputasi + one-hot + scaling + SMOTE + XGBoost (hasil tuning), disimpan di `model/lead_scoring_model.pth` (format joblib dari Google Colab).
- **Output:** probabilitas convert, skor 0-100, kategori **Hot / Warm / Cold**, dan rekomendasi prioritas follow-up.

## Fitur
1. **Prediksi satu lead** lewat form (dropdown dan angka).
2. **Prediksi banyak lead** dengan unggah CSV, ringkasan Hot/Warm/Cold, dan unduh hasil.
3. Sidebar berisi metrik model pada data uji dan aturan kategori.

## Struktur
```
app.py                      # UI Streamlit
lead_utils.py               # cleaning, pemuatan model, scoring
model/lead_scoring_model.pth
model/model_info.json       # kolom, batas outlier, pilihan dropdown, threshold, versi library
data/sample_leads.csv       # 30 lead contoh untuk uji tab CSV
tests/test_app.py           # uji otomatis (pytest)
requirements.txt
```

## Menjalankan
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```