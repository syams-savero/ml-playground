import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("dataset/Mall_Customers.csv")
X = df.iloc[:, [3, 4]].values

kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_
k = 4


def analyze_clusters(X, labels, k):
    print("Analisis Karakteristik Setiap Cluster:")
    for cluster_id in range(k):
        cluster_data = X[labels == cluster_id]
        mean_income = cluster_data[:, 0].mean()
        mean_spending = cluster_data[:, 1].mean()
        print(f"\nCluster {cluster_id + 1}:")
        print(f"  Rata-rata Annual Income (k$): {mean_income:.2f}")
        print(f"  Rata-rata Spending Score (1-100): {mean_spending:.2f}")
        print(f"  Jumlah anggota: {len(cluster_data)}")


analyze_clusters(X, labels, k)

sil = silhouette_score(X, labels)
print(f"\nSilhouette Score: {sil:.4f}")

centroids = kmeans.cluster_centers_
print("\nNilai Centroids:")
for i, centroid in enumerate(centroids):
    print(f"  Centroid {i+1}: Annual Income = {centroid[0]:.2f}, Spending Score = {centroid[1]:.2f}")

plt.figure(figsize=(12, 8))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=50, alpha=0.6, edgecolors="w", marker="o")
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=200, marker="X", label="Centroids")
for i, centroid in enumerate(centroids):
    plt.text(centroid[0], centroid[1], f"Centroid {i+1}", color="red", fontsize=12, ha="center", va="center")
plt.title("Visualisasi Cluster dengan Centroid")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.tight_layout()
plt.savefig("cluster_result.png", dpi=100)
plt.close()
print("\ncluster_result.png tersimpan.")