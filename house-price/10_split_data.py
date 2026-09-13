from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("data_bersih.csv")

X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Tampilkan isi data
print("=== x_train (5 baris pertama) ===")
print(x_train.head())
print(f"\nBaris: {len(x_train)}, Kolom: {x_train.shape[1]}")

print("\n=== x_test (5 baris pertama) ===")
print(x_test.head())
print(f"\nBaris: {len(x_test)}, Kolom: {x_test.shape[1]}")

print("\n=== y_train (harga rumah untuk data latih) ===")
print(y_train.head())

print("\n=== y_test (harga rumah untuk data uji) ===")
print(y_test.head())

# Simpan ke file CSV
x_train.to_csv("x_train.csv", index=False)
x_test.to_csv("x_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\n=== FILE TERSIMPAN ===")
print("x_train.csv, x_test.csv, y_train.csv, y_test.csv")