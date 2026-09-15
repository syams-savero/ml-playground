import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 60)
print("STUDI KASUS 1: OVERFITTING")
print("=" * 60)

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
print(f"\nDataset : California Housing ({X.shape[0]} baris) | Fitur: {X.shape[1]}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
print(f"Split 70/30 -> train: {X_train.shape[0]} | test: {X_test.shape[0]}")

model = DecisionTreeRegressor(max_depth=50, random_state=42)
print("\nTrain DecisionTreeRegressor(max_depth=50) ...")
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("\n--- Deteksi Overfitting ---")
print(f"Training MSE : {train_mse:.4e}")
print(f"Test MSE     : {test_mse:.4f}")
print("\nGap training vs test sangat jauh -> model overfitting!")
print("(Training nyaris sempurna, tapi gagal di data baru)")

train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5,
    scoring='neg_mean_squared_error', n_jobs=-1
)

train_mean = -np.mean(train_scores, axis=1)
test_mean = -np.mean(test_scores, axis=1)
print("\n--- Learning Curve (5-fold CV) ---")
print(f"Train error mengecil cepat ~ {train_mean[-1]:.4e}")
print(f"CV error tetap tinggi           ~ {test_mean[-1]:.4f}")
print("=> gap permanen menandakan overfitting")

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training error")
plt.plot(train_sizes, test_mean, 'o-', color="green", label="Cross-validation error")
plt.title("Learning Curve - Overfitting")
plt.xlabel("Training Set Size")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curve_overfit.png")
print("\nPlot disimpan: learning_curve_overfit.png")