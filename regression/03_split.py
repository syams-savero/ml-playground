import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/df_clean.csv")

X = df.drop(columns=["FloodProbability"])
y = df["FloodProbability"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("Jumlah data:", len(X))
print("Jumlah data latih:", len(x_train))
print("Jumlah data test:", len(x_test))

x_train.to_csv("dataset/x_train.csv", index=False)
x_test.to_csv("dataset/x_test.csv", index=False)
y_train.to_csv("dataset/y_train.csv", index=False)
y_test.to_csv("dataset/y_test.csv", index=False)