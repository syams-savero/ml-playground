import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("dataset/churn.csv")
data = data.drop(columns=["RowNumber", "CustomerId", "Surname"])

label_encoder = LabelEncoder()
for column in ["Geography", "Gender"]:
    data[column] = label_encoder.fit_transform(data[column])

scaler = MinMaxScaler()

X = data.drop(columns=["Exited"])
y = data["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_columns = X_train.select_dtypes(include=["int64", "float64"]).columns
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")

X_train.to_csv("x_train.csv", index=False)
X_test.to_csv("x_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("\nX_train, X_test, y_train, y_test tersimpan")