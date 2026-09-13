import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
train = pd.read_csv("dataset/train.csv")

# 1. Cek & pisahkan kolom missing
missing_values = train.isnull().sum()
less = missing_values[missing_values < 1000].index
over = missing_values[missing_values >= 1000].index

# 2. Isi numerik → median
numeric_features = train[less].select_dtypes(include=['number']).columns
train[numeric_features] = train[numeric_features].fillna(train[numeric_features].median())

# 3. Isi kategorik → mode
kategorical_features = train[less].select_dtypes(include=['object', 'string']).columns
for column in kategorical_features:
    train[column] = train[column].fillna(train[column].mode()[0])

# 4. Drop kolom banyak kosong
df = train.drop(columns=over)

# 5. Hapus outlier (kolom penting)
important_features = ['SalePrice', 'LotArea', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']
Q1 = df[important_features].quantile(0.25)
Q3 = df[important_features].quantile(0.75)
IQR = Q3 - Q1
condition = ~((df[important_features] < (Q1 - 1.5 * IQR)) | (df[important_features] > (Q3 + 1.5 * IQR))).any(axis=1)
df = df.loc[condition]

# 6. Label Encoding → ubah semua kolom kategorik jadi angka
category_features = df.select_dtypes(include=['object', 'string']).columns
label_encoder = LabelEncoder()
for col in category_features:
    df[col] = label_encoder.fit_transform(df[col])

# 7. Simpan hasil
df.to_csv("data_bersih.csv", index=False)

print(f"Data final disimpan ke data_bersih.csv")
print(f"Shape: {df.shape}  (baris, kolom)")
print(f"Kolom kategorik yang di-encode: {list(category_features)}")
print(f"\n5 baris pertama:")
print(df.head(5))