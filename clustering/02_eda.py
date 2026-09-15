import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("dataset/Mall_Customers.csv")

plt.figure(figsize=(7, 7))
plt.pie(df["Gender"].value_counts(), labels=["Female", "Male"], autopct="%1.1f%%", startangle=90)
plt.title("Gender Distribution")
plt.tight_layout()
plt.savefig("eda_gender.png", dpi=100)
plt.close()

age18_25 = df.Age[(df.Age >= 18) & (df.Age <= 25)]
age26_35 = df.Age[(df.Age >= 26) & (df.Age <= 35)]
age36_45 = df.Age[(df.Age >= 36) & (df.Age <= 45)]
age46_55 = df.Age[(df.Age >= 46) & (df.Age <= 55)]
age55above = df.Age[df.Age >= 56]

x = ["18-25", "26-35", "36-45", "46-55", "55+"]
y = [len(age18_25.values), len(age26_35.values), len(age36_45.values), len(age46_55.values), len(age55above.values)]

plt.figure(figsize=(15, 6))
plt.bar(x, y, color=["red", "green", "blue", "cyan", "yellow"])
plt.title("Customer and Their Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
for i in range(len(x)):
    plt.text(i, y[i], y[i], ha="center", va="bottom")
plt.tight_layout()
plt.savefig("eda_age.png", dpi=100)
plt.close()

ai0_30 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 0) & (df["Annual Income (k$)"] <= 30)]
ai31_60 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 31) & (df["Annual Income (k$)"] <= 60)]
ai61_90 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 61) & (df["Annual Income (k$)"] <= 90)]
ai91_120 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 91) & (df["Annual Income (k$)"] <= 120)]
ai121_150 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 121) & (df["Annual Income (k$)"] <= 150)]

aix = ["$ 0 - 30,000", "$ 30,001 - 60,000", "$ 60,001 - 90,000", "$ 90,001 - 120,000", "$ 120,001 - 150,000"]
aiy = [len(ai0_30.values), len(ai31_60.values), len(ai61_90.values), len(ai91_120.values), len(ai121_150.values)]

plt.figure(figsize=(15, 6))
plt.bar(aix, aiy, color=["red", "green", "blue", "cyan", "yellow"])
plt.title("Customer and Their Annual Income")
plt.xlabel("Annual Income")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
for i in range(len(aix)):
    plt.text(i, aiy[i], aiy[i], ha="center", va="bottom")
plt.tight_layout()
plt.savefig("eda_income.png", dpi=100)
plt.close()

print("gender:", dict(df["Gender"].value_counts()))
print("age bins:", list(zip(x, y)))
print("income bins:", list(zip(aix, aiy)))
print("\n3 PNG tersimpan: eda_gender.png, eda_age.png, eda_income.png")