import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

x_train = pd.read_csv("dataset/x_train.csv")
x_test = pd.read_csv("dataset/x_test.csv")
y_train = pd.read_csv("dataset/y_train.csv")["FloodProbability"]
y_test = pd.read_csv("dataset/y_test.csv")["FloodProbability"]

print("===== Model 1: LARS =====")
lars = linear_model.Lars(n_nonzero_coefs=1).fit(x_train, y_train)
pred_lars = lars.predict(x_test)
mae_lars = mean_absolute_error(y_test, pred_lars)
mse_lars = mean_squared_error(y_test, pred_lars)
r2_lars = r2_score(y_test, pred_lars)
print(f"MAE: {mae_lars}")
print(f"MSE: {mse_lars}")
print(f"R²: {r2_lars}")

data = {"MAE": [mae_lars], "MSE": [mse_lars], "R2": [r2_lars]}
df_results = pd.DataFrame(data, index=["Lars"])
print(df_results)

print("===== Model 2: Linear Regression =====")
LR = LinearRegression().fit(x_train, y_train)
pred_LR = LR.predict(x_test)
mae_LR = mean_absolute_error(y_test, pred_LR)
mse_LR = mean_squared_error(y_test, pred_LR)
r2_LR = r2_score(y_test, pred_LR)
print(f"MAE: {mae_LR}")
print(f"MSE: {mse_LR}")
print(f"R²: {r2_LR}")

df_results.loc["Linear Regression"] = [mae_LR, mse_LR, r2_LR]
print(df_results)

print("===== Model 3: GradientBoostingRegressor (train LIVE) =====")
GBR = GradientBoostingRegressor(random_state=184, verbose=True)
GBR.fit(x_train, y_train)
pred_GBR = GBR.predict(x_test)
mae_GBR = mean_absolute_error(y_test, pred_GBR)
mse_GBR = mean_squared_error(y_test, pred_GBR)
r2_GBR = r2_score(y_test, pred_GBR)
print(f"MAE: {mae_GBR}")
print(f"MSE: {mse_GBR}")
print(f"R²: {r2_GBR}")

df_results.loc["GradientBoostingRegressor"] = [mae_GBR, mse_GBR, r2_GBR]
print("\n===== Perbandingan Final =====")
print(df_results)