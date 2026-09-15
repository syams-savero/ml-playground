import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from collections import Counter

X, y = make_classification(
    n_samples=1000,
    n_features=15,
    n_informative=10,
    n_redundant=2,
    n_clusters_per_class=1,
    weights=[0.9],
    flip_y=0,
    random_state=42,
)

df = pd.DataFrame(X, columns=[f"Fitur_{i}" for i in range(1, 16)])
df["Target"] = y

df["Fitur_12"] = np.random.choice(["A", "B", "C"], size=1000)
df["Fitur_13"] = np.random.choice(["X", "Y", "Z"], size=1000)

X_full = df.drop("Target", axis=1)
y_full = df["Target"]

print(df.head())
print(f"\nDimensi dataset: {df.shape}")
print("Distribusi kelas sebelum SMOTE:", Counter(y_full))

df.to_csv("dataset.csv", index=False)