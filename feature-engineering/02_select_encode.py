import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("dataset.csv")
X = df.drop("Target", axis=1)
y = df["Target"]

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
X_integer = X.drop(["Fitur_12", "Fitur_13"], axis=1)
rf_model.fit(X_integer, y)

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

threshold = 0.05
important_features_indices = [i for i in range(len(importances)) if importances[i] >= threshold]

print("Fitur yang dipilih dengan Embedded Methods (di atas ambang 5%):")
for i in important_features_indices:
    print(f"  {X_integer.columns[i]}: {importances[i]:.4f}")

important_features = X_integer.columns[important_features_indices]
X_important = X_integer[important_features]
print(f"\nDimensi data fitur penting: {X_important.shape}")

X_Selected = pd.concat([X_important, X["Fitur_12"]], axis=1)
X_Selected = pd.concat([X_Selected, X["Fitur_13"]], axis=1)

label_encoder = LabelEncoder()
X_Selected["Fitur_12"] = label_encoder.fit_transform(X_Selected["Fitur_12"])
X_Selected["Fitur_13"] = label_encoder.fit_transform(X_Selected["Fitur_13"])

print("\nX_Selected setelah encoding (semua numerik):")
print(X_Selected.head())
print(f"\nDimensi X_Selected: {X_Selected.shape}")

X_Selected.to_csv("dataset_selected.csv", index=False)
y.to_csv("dataset_target.csv", index=False)