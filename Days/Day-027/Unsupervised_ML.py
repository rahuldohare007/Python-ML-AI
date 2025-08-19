# 🚀 Day 27/100 of #100DaysOfCode
# 🎯 Unsupervised ML: KMeans

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. Create sample data (3 clusters)
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)

# 2. Visualize raw data (no labels!)
plt.scatter(X[:, 0], X[:, 1], s=30)
plt.title("Unlabeled Data (Before Clustering)")
plt.show()

# 3. Use Elbow method to choose 'k'
inertia = []
k_values = range(1, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(k_values, inertia, 'bo-')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (Within-cluster SSE)")
plt.title("Elbow Method to choose k")
plt.show()

# 4. Train KMeans with optimal k (from elbow, assume k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 5. Visualize clustered data
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=30, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("Data Grouped by KMeans")
plt.show()
