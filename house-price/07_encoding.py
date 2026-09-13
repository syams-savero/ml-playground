import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

# Hapus outlier di kolom penting saja
important_features = ['SalePrice', 'LotArea', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']
Q1 = df[important_features].quantile(0.25)
Q3 = df[important_features].quantile(0.75)
IQR = Q3 - Q1
condition = ~((df[important_features] < (Q1 - 1.5 * IQR)) | (df[important_features] > (Q3 + 1.5 * IQR))).any(axis=1)
df = df.loc[condition]

# Ambil kolom kategorik
category_features = df.select_dtypes(include=['object', 'string']).columns
print("Jumlah kolom kategorik:", len(category_features))
print("Contoh:", list(category_features[:8]))

# ==== 1. ONE-HOT ENCODING ====
print("\n=== ONE-HOT ENCODING ===")
df_one_hot = pd.get_dummies(df, columns=category_features)
print(f"Shape sebelum  : {df.shape}  (baris, kolom)")
print(f"Shape sesudah  : {df_one_hot.shape}  (baris, kolom)")

print("\nContoh kolom baru hasil One-Hot (dari MSZoning):")
print([c for c in df_one_hot.columns if "MSZoning" in c])

# ==== 2. LABEL ENCODING ====
print("\n=== LABEL ENCODING ===")
label_encoder = LabelEncoder()
df_lencoder = pd.DataFrame(df)

for col in category_features:
    df_lencoder[col] = label_encoder.fit_transform(df[col])

print(f"Shape sebelum  : {df.shape}  (baris, kolom)")
print(f"Shape sesudah  : {df_lencoder.shape}  (baris, kolom)  <- kolom TIDAK bertambah")

print("\nMSZoning sebelum (teks):")
print(df['MSZoning'].head(10).tolist())
print("MSZoning sesudah (angka):")
print(df_lencoder['MSZoning'].head(10).tolist())