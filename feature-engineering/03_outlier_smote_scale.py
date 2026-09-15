import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

X_Selected = pd.read_csv("dataset_selected.csv")
y = pd.read_csv("dataset_target.csv")["Target"]

numeric_columns = X_Selected.select_dtypes(include=["float64", "int64"]).columns
numeric_columns = numeric_columns.drop(["Fitur_12", "Fitur_13"])

X_cleaned = X_Selected[numeric_columns].copy()
print("Jumlah data sebelum hapus outlier:", len(X_cleaned))

for col in numeric_columns:
    Q1 = X_Selected[col].quantile(0.25)
    Q3 = X_Selected[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = X_cleaned[(X_cleaned[col] < lower_bound) | (X_cleaned[col] > upper_bound)]
    X_cleaned = X_cleaned.drop(outliers.index)
    y = y.drop(outliers.index)

print("Jumlah data setelah hapus outlier:", len(X_cleaned))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_cleaned, y)
print("Distribusi kelas setelah SMOTE:", Counter(y_resampled))

X_resampled = pd.DataFrame(X_resampled, columns=numeric_columns)
y_resampled = pd.Series(y_resampled, name="Target")

plt.figure(figsize=(10, 6))
for col in X_resampled.columns:
    sns.histplot(X_resampled[col], kde=True, label=col, bins=30, element="step")
plt.title("Distribusi Data Sebelum Scaling (Histogram)")
plt.legend(loc="upper center", fontsize=7, ncol=4)
plt.tight_layout()
plt.savefig("hist_sebelum_scaling.png", dpi=100)
plt.close()

scaler = StandardScaler()
X_resampled[numeric_columns] = scaler.fit_transform(X_resampled[numeric_columns])

plt.figure(figsize=(10, 6))
for col in X_resampled.columns:
    sns.histplot(X_resampled[col], kde=True, label=col, bins=30, element="step")
plt.title("Distribusi Data Setelah Scaling (Histogram)")
plt.legend(loc="upper center", fontsize=7, ncol=4)
plt.tight_layout()
plt.savefig("hist_setelah_scaling.png", dpi=100)
plt.close()

print("\ndescribe() setelah scaling:")
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print(X_resampled.describe().loc[["mean", "std", "min", "max"]])

print("\nDimensi akhir:", X_resampled.shape, y_resampled.shape)
print("Proporsi kelas:", Counter(y_resampled))
print("\n2 PNG tersimpan: hist_sebelum_scaling.png, hist_setelah_scaling.png")