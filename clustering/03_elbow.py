import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("dataset/Mall_Customers.csv")
X = df.iloc[:, [3, 4]].values

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

print("WCSS per jumlah cluster:")
for i, v in enumerate(wcss, start=1):
    print(f"  k={i}: {v:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker="o", linestyle="-")
plt.title("Elbow Method - WCSS per Jumlah Cluster")
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("WCSS (inertia)")
for i, v in enumerate(wcss, start=1):
    plt.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("elbow.png", dpi=100)
plt.close()
print("\nelbow.png tersimpan.")