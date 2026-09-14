import pandas as pd

data = pd.read_csv("dataset/churn.csv")

print("\nInformasi dataset:")
print(f"Jumlah baris x kolom: {data.shape}")
print(f"Missing values per fitur:\n{data.isnull().sum().sum()} total")

data.info()

data = data.drop(columns=["RowNumber", "CustomerId", "Surname"])

print("\n=== Data setelah bersih (5 baris pertama) ===")
print(data.head())
print(f"\nShape setelah drop kolom: {data.shape}")