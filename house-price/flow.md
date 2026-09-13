# Alur Machine Learning Workflow

Urutan tahapan yang dipelajari + file latihan tiap tahap di proyek ini.

## Cara Menjalankan

Dari folder proyek ini (`house-price/`), pakai venv dari root `ml-playground`:

```bash
cd house-price
../venv/bin/python 08_build_clean_data.py
../venv/bin/python 10_split_data.py
../venv/bin/python 11_modelling.py
../venv/bin/python 12_api.py     # server API
```

## Pipeline Lengkap

| # | Tahap | Fungsi | File Latihan |
|---|-------|--------|--------------|
| 1 | Data Loading | Baca dataset (CSV → DataFrame) | `01_load_data.py` |
| 2 | Cek Missing Value | Deteksi data kosong & jumlahnya | `02_check_missing.py` |
| 3 | Isi Missing Value | Numerik → median, kategorik → mode | `03_fill_missing.py` |
| 4 | Outlier | Deteksi & hapus data ekstrem (IQR) | `04_outlier.py` |
| 5 | Standardisasi | Samakan skala fitur (mean=0, std=1) | `05_standardize.py` |
| 6 | Duplikasi | Cek/hapus baris ganda | `06_check_duplicate.py` |
| 7 | Encoding | Ubah kolom kategorik jadi angka (One-Hot / Label) | `07_encoding.py` |
| 8 | Simpan data bersih | Konsolidasi semua cleaning jadi 1 pipeline | `08_build_clean_data.py` |
| 9 | EDA | Eksplorasi data: histogram, korelasi | `09_eda.py` |
| 10 | Data Splitting | Bagi data jadi train (80%) & test (20%) | `10_split_data.py` |
| 11 | Modelling | Latih model (fit) + evaluasi (MAE/MSE/R2) | `11_modelling.py` |
| 12 | Deployment | Sajikan model pakai API (Flask) | `12_api.py` |

## Urutan Singkat (Hafalan)

```
Load → Clean → EDA → Split → Train → Evaluate → Deploy
```

Lebih detail:

```
Load data
  → 02 cek missing
  → 03 isi missing (median/mode)
  → 04 hapus outlier (IQR)
  → 05 standardisasi
  → 06 cek duplikat
  → 07 encoding
  → 08 SAVE data_bersih.csv
  → 09 EDA (visualisasi & korelasi)
  → 10 split (x_train, y_train, x_test, y_test)
  → 11 training (model.fit) & evaluasi (MAE, MSE, R2) → simpan model
  → 12 deploy (API Flask)
```

> Catatan: file 01–07 mengajarkan konsep tahap per tahap. File 08 = gabungan semuanya dalam 1 pipeline (hasilnya `data_bersih.csv`). File 09–12 memakai `data_bersih.csv` sebagai input.

## Konsep Penting

- **Fitur (X)** = kolom input (selain target)
- **Target (y)** = kolom yang diprediksi (SalePrice)
- **fit()** = model belajar dari data → menyimpan parameter
- **predict()** = model menebak dari data baru
- **Data test** = data yang TIDAK boleh dilihat model saat belajar (biar fair)

## Urutan Preprocessing vs Splitting

```
Cleaning & Encoding → SPLITTING → Standardisasi
```

- Sebelum split: isi missing, encoding, hapus outlier
- Sesudah split: standardisasi (pakai mean/std training saja, hindari data leakage)

## Hasil Modelling

| Model | MAE | R2 |
|-------|-----|-----|
| LARS | 41.795 | 0.15 |
| Linear Regression | 13.912 | 0.89 |
| **Gradient Boosting (terpilih)** | **12.910** | **0.90** |