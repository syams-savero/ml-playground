import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend('Agg')  # simpan ke file, bukan tampil

df = pd.read_csv("data_bersih.csv")

print("=== 1) CEK MISSING VALUE ===")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
}).sort_values(by='Missing Values', ascending=False)
print("Total missing:", missing_values.sum())

print("\n=== 2) HISTOGRAM SEMUA KOLOM ===")
num_vars = df.shape[1]
n_cols = 4
n_rows = -(-num_vars // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()
for i, column in enumerate(df.columns):
    df[column].hist(ax=axes[i], bins=20, edgecolor='black')
    axes[i].set_title(column, fontsize=9)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig("eda_histogram_semua.png")
print("Tersimpan: eda_histogram_semua.png")

print("\n=== 3) DISTRIBUSI KOLOM PENTING ===")
columns_to_plot = ['OverallQual', 'YearBuilt', 'LotArea', 'SaleType', 'SaleCondition']
plt.figure(figsize=(15, 10))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.savefig("eda_distribusi_penting.png")
print("Tersimpan: eda_distribusi_penting.png")

print("\n=== 4) KORELASI DENGAN SalePrice (TERSORTIR) ===")
target_corr = df.corr()['SalePrice'].drop('SalePrice').abs().sort_values(ascending=False)
print("5 fitur paling berkorelasi dengan harga:")
print(target_corr.head(10).round(3).to_string())

plt.figure(figsize=(12, 6))
target_corr.head(15).plot(kind='bar')
plt.title('Korelasi (absolut) dengan SalePrice')
plt.xlabel('Variables')
plt.ylabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig("eda_korelasi_saleprice.png")
print("\nTersimpan: eda_korelasi_saleprice.png")

plt.figure(figsize=(14, 12))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig("eda_correlation_matrix.png")
print("Tersimpan: eda_correlation_matrix.png")