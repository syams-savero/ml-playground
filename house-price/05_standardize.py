import pandas as pd
from sklearn.preprocessing import StandardScaler

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

# ==== SEBELUM STANDARDISASI ====
print("=== SEBELUM STANDARDISASI ===")
print(df[numeric_features].head(3))
print("\nMean tiap kolom:")
print(df[numeric_features].mean().round(2))
print("\nStd tiap kolom:")
print(df[numeric_features].std().round(2))

# ==== STANDARDISASI ====
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# ==== SESUDAH STANDARDISASI ====
print("\n=== SESUDAH STANDARDISASI ===")
print(df[numeric_features].head(3))
print("\nMean tiap kolom (harus ~0):")
print(df[numeric_features].mean().round(4))
print("\nStd tiap kolom (harus ~1):")
print(df[numeric_features].std().round(4))