import pandas as pd

df = pd.read_csv("dataset/Mall_Customers.csv")
print(df.head())
print("\n=== info() ===")
df.info()
print("\n=== describe() ===")
print(df.describe())