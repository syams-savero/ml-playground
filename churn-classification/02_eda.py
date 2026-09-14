import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

data = pd.read_csv("dataset/churn.csv")
data = data.drop(columns=["RowNumber", "CustomerId", "Surname"])

num_features = data.select_dtypes(include=[np.number])

plt.figure(figsize=(14, 10))
for i, column in enumerate(num_features.columns, 1):
    plt.subplot(3, 4, i)
    sns.histplot(data[column], bins=30, kde=True, color="blue")
    plt.title(f"Distribusi {column}", fontsize=9)
plt.tight_layout()
plt.savefig("eda_distribusi_numerik.png", dpi=90)
plt.close()
print("Tersimpan: eda_distribusi_numerik.png")

cat_features = data.select_dtypes(include=["object"])
plt.figure(figsize=(14, 8))
for i, column in enumerate(cat_features.columns, 1):
    plt.subplot(2, 4, i)
    sns.countplot(y=data[column], palette="viridis")
    plt.title(f"Distribusi {column}", fontsize=9)
plt.tight_layout()
plt.savefig("eda_distribusi_kategorik.png", dpi=90)
plt.close()
print("Tersimpan: eda_distribusi_kategorik.png")

plt.figure(figsize=(12, 10))
sns.heatmap(num_features.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.tight_layout()
plt.savefig("eda_heatmap_korelasi.png", dpi=90)
plt.close()
print("Tersimpan: eda_heatmap_korelasi.png")

plt.figure(figsize=(8, 4))
sns.countplot(x="Exited", data=data, palette="viridis")
plt.title("Distribusi Variabel Target (Exited)")
plt.tight_layout()
plt.savefig("eda_distribusi_exited.png", dpi=90)
plt.close()
print("Tersimpan: eda_distribusi_exited.png")

counts = data["Exited"].value_counts()
print(f"\nProporsi target:\n  Tidak churn (0): {counts[0]} ({counts[0]/len(data)*100:.1f}%)")
print(f"  Churn (1):     {counts[1]} ({counts[1]/len(data)*100:.1f}%)")
print("\nKelas TIDAK seimbang - sebagian besar pelanggan tidak churn.")