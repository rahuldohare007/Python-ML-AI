# 🚀 Day 29/100 of #100DaysOfCode
# 🎯 Dimensionality Reduction (PCA, t-SNE) 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load dataset (high-dimensional: 64 features per sample)
digits = load_digits()
X = digits.data
y = digits.target

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- PCA Reduction to 2D ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- t-SNE Reduction to 2D ---
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# --- Plot PCA ---
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='tab10', s=10)
plt.title("PCA: Digits Dataset")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(*scatter.legend_elements(), title="Digits", bbox_to_anchor=(1.05,1), loc='upper left')

# --- Plot t-SNE ---
plt.subplot(1,2,2)
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='tab10', s=10)
plt.title("t-SNE: Digits Dataset")
plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")
plt.legend(*scatter.legend_elements(), title="Digits", bbox_to_anchor=(1.05,1), loc='upper left')

plt.tight_layout()
plt.show()
