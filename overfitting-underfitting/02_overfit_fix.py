import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

model = DecisionTreeRegressor(max_depth=50, random_state=42)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
base_test_mse = mean_squared_error(y_test, y_test_pred)

print("=" * 60)
print("MENGATASI OVERFITTING (baseline test MSE: %.4f)" % base_test_mse)
print("=" * 60)

print("\n[1] CROSS-VALIDATION")
cross_val_scores = cross_val_score(
    model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'
)
print(f"Cross-Validation MSE : {-cross_val_scores.mean():.4f}")
print("(lebih stabil/konsisten, mendekati test MSE)")

print("\n[2] REGULARIZATION (max_depth=5)")
model_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
model_reg.fit(X_train, y_train)
y_train_pred_reg = model_reg.predict(X_train)
y_test_pred_reg = model_reg.predict(X_test)
print(f"Training MSE (After Regularization): {mean_squared_error(y_train, y_train_pred_reg):.4f}")
print(f"Test MSE (After Regularization)    : {mean_squared_error(y_test, y_test_pred_reg):.4f}")

print("\n[3] PRUNING (Cost Complexity Pruning)")
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
model_pruned = DecisionTreeRegressor(random_state=42, ccp_alpha=ccp_alphas[-2])
model_pruned.fit(X_train, y_train)
y_train_pred_pruned = model_pruned.predict(X_train)
y_test_pred_pruned = model_pruned.predict(X_test)
print(f"Pruned Model Training MSE : {mean_squared_error(y_train, y_train_pred_pruned):.4f}")
print(f"Pruned Model Test MSE     : {mean_squared_error(y_test, y_test_pred_pruned):.4f}")

print("\n[4] DATA AUGMENTATION (tambah noise)")
X_train_aug = X_train + np.random.normal(0, 0.1, X_train.shape)
model_aug = DecisionTreeRegressor(max_depth=10, random_state=42)
model_aug.fit(X_train_aug, y_train)
y_train_pred_aug = model_aug.predict(X_train_aug)
y_test_pred_aug = model_aug.predict(X_test)
print(f"Augmented Data Training MSE : {mean_squared_error(y_train, y_train_pred_aug):.4f}")
print(f"Augmented Data Test MSE     : {mean_squared_error(y_test, y_test_pred_aug):.4f}")

print("\n[5] RANDOM FOREST (100 pohon, max_depth=10)")
model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model_rf.fit(X_train, y_train)
y_train_pred_rf = model_rf.predict(X_train)
y_test_pred_rf = model_rf.predict(X_test)
print(f"Random Forest Training MSE : {mean_squared_error(y_train, y_train_pred_rf):.4f}")
print(f"Random Forest Test MSE     : {mean_squared_error(y_test, y_test_pred_rf):.4f}")

print("\n--- Rangkuman Overfitting ---")
print("Best pada data uji: Random Forest (test MSE %.4f)" % mean_squared_error(y_test, y_test_pred_rf))
print("Terbuka untuk early stopping pada Gradient Boosting.")