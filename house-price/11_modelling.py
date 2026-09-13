import pandas as pd
from sklearn.model_selection import train_test_split

# === 1. Siapkan data (sama seperti latihan split) ===
df = pd.read_csv("data_bersih.csv")
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(f"Data siap: {len(x_train)} latih, {len(x_test)} uji")

# === 2. Melatih 3 model ===
print("\n=== TRAINING 3 MODEL ===")

# Model 1: Larsen (Least Angle Regression)
from sklearn import linear_model
lars = linear_model.Lars(n_nonzero_coefs=1).fit(x_train, y_train)
print("Model 1: Lars               ✔")

# Model 2: Linear Regression
from sklearn.linear_model import LinearRegression
LR = LinearRegression().fit(x_train, y_train)
print("Model 2: Linear Regression  ✔")

# Model 3: Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(random_state=184)
GBR.fit(x_train, y_train)
print("Model 3: GradientBoosting   ✔")

# === 3. Evaluasi ===
print("\n=== EVALUASI 3 MODEL (MAE, MSE, R2) ===")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Lars
pred_lars = lars.predict(x_test)
data = {
    'MAE': [mean_absolute_error(y_test, pred_lars)],
    'MSE': [mean_squared_error(y_test, pred_lars)],
    'R2': [r2_score(y_test, pred_lars)]
}
df_results = pd.DataFrame(data, index=['Lars'])

# Linear Regression
pred_LR = LR.predict(x_test)
df_results.loc['Linear Regression'] = [
    mean_absolute_error(y_test, pred_LR),
    mean_squared_error(y_test, pred_LR),
    r2_score(y_test, pred_LR)
]

# Gradient Boosting
pred_GBR = GBR.predict(x_test)
df_results.loc['GradientBoostingRegressor'] = [
    mean_absolute_error(y_test, pred_GBR),
    mean_squared_error(y_test, pred_GBR),
    r2_score(y_test, pred_GBR)
]

print("\nHasil evaluasi (dibulatkan):")
print(df_results.round(2).to_string())

# === 4. Simpan model terbaik (GBR) ===
print("\n=== MENYIMPAN MODEL GBR ===")
import joblib
joblib.dump(GBR, 'gbr_model.joblib')
print("Model GBR disimpan → gbr_model.joblib")

import pickle
with open('gbr_model.pkl', 'wb') as file:
    pickle.dump(GBR, file)
print("Model GBR disimpan → gbr_model.pkl")