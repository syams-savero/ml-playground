# Prediksi Harga Rumah

Proyek regresi untuk memprediksi harga rumah menggunakan data dari kompetisi Kaggle.

## Ringkasan

- Dataset: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) (Kaggle)
- Pendekatan: klasik ML, dari pembersihan data hingga deployment
- Hasil: model Gradient Boosting (R² = 0.90) diekspos lewat API Flask (`POST /predict`)

## Alur

Setiap langkah diwakili satu file Python, dinomori sesuai urutan pengerjaan:

| # | File | Isi |
|---|------|-----|
| 1 | `01_load_data.py` | Membaca dataset |
| 2 | `02_check_missing.py` | Deteksi nilai kosong |
| 3 | `03_fill_missing.py` | Isi missing (median untuk numerik, mode untuk kategorik) |
| 4 | `04_outlier.py` | Deteksi & hapus outlier (IQR) |
| 5 | `05_standardize.py` | Standardisasi fitur numerik |
| 6 | `06_check_duplicate.py` | Cek duplikasi data |
| 7 | `07_encoding.py` | Encoding kolom kategorik |
| 8 | `08_build_clean_data.py` | Pipeline lengkap → `data_bersih.csv` |
| 9 | `09_eda.py` | Eksplorasi data (histogram, korelasi) |
| 10 | `10_split_data.py` | Split 80/20 → train/test |
| 11 | `11_modelling.py` | Training & evaluasi model |
| 12 | `12_api.py` | API Flask untuk prediksi |

Catatan alur & konsep ada di `flow.md`.

## Hasil

| Model | MAE | R² |
|-------|-----|-----|
| LARS | 41.795 | 0.15 |
| Linear Regression | 13.912 | 0.89 |
| Gradient Boosting | 12.910 | 0.90 |

## Cara Menjalankan

```bash
# pastikan sudah di folder house-price
pip install -r requirements.txt

python 08_build_clean_data.py   # → data_bersih.csv
python 10_split_data.py         # → data train/test
python 11_modelling.py          # → gbr_model.joblib

# jalankan API
python 12_api.py
# server berjalan di http://127.0.0.1:5000

# terminal lain, minta prediksi:
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d @data.json
```

Jika venv tidak diaktifkan: `../venv/bin/python <file>.py`.

## Struktur

```
house-price/
├── 01_load_data.py ... 12_api.py   # langkah-langkah pengerjaan
├── dataset/                         # data mentah (tidak di-commit)
├── data.json                        # contoh input API
├── flow.md                          # ringkasan alur dan konsep
└── requirements.txt
```

File hasil pipeline (`*.csv`, `*.png`, `*.joblib`, `*.pkl`) tidak di-commit.

## Environment

- Python 3.14 (venv, Arch Linux)
- pandas 3.0.5, scikit-learn 1.9.1, Flask 3.1.3

## Sumber

- Dataset milik Kaggle; kode boleh dipakai untuk belajar.