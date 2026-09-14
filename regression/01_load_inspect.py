import pandas as pd

df_train = pd.read_csv("dataset/train.csv")
print(df_train)
print("\n=== info() ===")
df_train.info()
print("\n=== describe() ===")
print(df_train.describe(include="all"))
print("\n=== missing values ===")
missing = df_train.isnull().sum()
print(missing[missing > 0] if (missing > 0).any() else "Tidak ada missing value.")