# ml-playground

Kumpulan proyek latihan Machine Learning. Setiap proyek berisi langkah-langkah pengerjaan dari data mentah hingga deployment model.

## Daftar Proyek

| Proyek | Deskripsi | Status |
|--------|-----------|--------|
| [house-price](house-price/) | Prediksi harga rumah (regresi) menggunakan data Kaggle | Selesai |

## Struktur

```
ml-playground/
├── house-price/          # proyek prediksi harga rumah
├── venv/                 # virtual environment (dipakai semua proyek)
└── README.md
```

## Cara Menjalankan

```bash
# dari root repo
python -m venv venv
source venv/bin/activate

# masuk ke proyek
cd house-price
pip install -r requirements.txt
python 11_modelling.py
```

Penjelasan per langkah ada di README masing-masing proyek.