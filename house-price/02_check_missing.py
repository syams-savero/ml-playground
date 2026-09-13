import pandas as pd
train = pd.read_csv("dataset/train.csv")

missing_values = train.isnull().sum()
print(missing_values[missing_values > 0])
