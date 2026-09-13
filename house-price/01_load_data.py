import pandas as pd
train = pd.read_csv("dataset/train.csv")

#lihat struktur data
print(train.info())

#mendekskripsikan isi data untuk cek kualitas nya
print(train.describe(include="all"))
