import pandas as pd
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv("dataset/train.csv")
print("Jumlah data sebelum bersih outlier:", len(df_train))

Q1 = df_train.quantile(0.25)
Q3 = df_train.quantile(0.75)
IQR = Q3 - Q1
condition = ~((df_train < (Q1 - 1.5 * IQR)) | (df_train > (Q3 + 1.5 * IQR))).any(axis=1)
df = df_train.loc[condition, df_train.columns]
print("Jumlah data setelah buang outlier:", len(df))

print("\n=== standardisasi fitur numerik ===")
numeric_features = df.select_dtypes(include=["number"]).columns
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
print("Rentang setelah standardisasi (min/max per fitur):")
print(df[numeric_features].agg(["min", "max"]))

duplicates = df.duplicated()
print("\nBaris duplikat:", duplicates.sum(), "(harusnya 0)")

df.to_csv("dataset/df_clean.csv", index=False)
print("\ndf_clean.csv tersimpan:", df.shape)