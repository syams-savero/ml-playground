import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
train = pd.read_csv("dataset/train.csv")

# Hitung missing value & pisahkan
missing_values = train.isnull().sum()
less = missing_values[missing_values < 1000].index
over = missing_values[missing_values >= 1000].index

# Isi numerik → median
numeric_features = train[less].select_dtypes(include=['number']).columns
train[numeric_features] = train[numeric_features].fillna(train[numeric_features].median())

# Isi kategorik → mode
kategorical_features = train[less].select_dtypes(include=['object', 'string']).columns
for column in kategorical_features:
    train[column] = train[column].fillna(train[column].mode()[0])

# Drop kolom banyak kosong
df = train.drop(columns=over)

# Simpan boxplot ke file PNG
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.savefig(f'boxplot_{feature}.png')
    plt.close()

print("Semua grafik tersimpan!")

# Kolom penting untuk cek outlier
important_features = ['SalePrice', 'LotArea', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']

# Hitung Q1, Q3, IQR untuk kolom penting saja
Q1 = df[important_features].quantile(0.25)
Q3 = df[important_features].quantile(0.75)
IQR = Q3 - Q1

# Hapus baris yang mengandung outlier di kolom penting
condition = ~((df[important_features] < (Q1 - 1.5 * IQR)) | (df[important_features] > (Q3 + 1.5 * IQR))).any(axis=1)
df = df.loc[condition]

# Verifikasi
print(f"Data sebelum: {train.shape[0]} baris")
print(f"Data sesudah: {df.shape[0]} baris")
print(f"Baris dihapus: {train.shape[0] - df.shape[0]}")
