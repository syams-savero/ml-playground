import pandas as pd

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

# ==== CEK DUPLIKAT ====
duplicates = df.duplicated()

print("Total baris:", len(df))
print("Baris duplikat:", duplicates.sum())
print("\nBaris yang terindikasi duplikat:")
print(df[duplicates])

# Kalau hasilnya 0, berarti tidak ada duplikat → skip
if duplicates.sum() == 0:
    print("\nTidak ada duplikat, lanjut saja tanpa drop_duplicates()")
else:
    df = df.drop_duplicates()
    print("Duplikat dihapus. Baris sekarang:", len(df))