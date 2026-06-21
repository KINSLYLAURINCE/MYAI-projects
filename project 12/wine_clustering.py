import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

column_names = ['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_ash',
                'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                'proanthocyanins', 'color_intensity', 'hue',
                'od280_od315', 'proline']

df = pd.read_csv('H:/AI/project 12/wine.data', header=None, names=column_names)

print("=" * 50)
print("WINE DATASET - EXPLORATION")
print("=" * 50)
print(f"\nTotal wines   : {df.shape[0]}")
print(f"Total features: {df.shape[1] - 1}")
print(f"\n--- Wines per class ---")
print(df['class'].value_counts().sort_index())
print(f"\n--- First 5 rows ---")
print(df.head())

X = df.drop('class', axis=1)
y_true = df['class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
silhouettes = []
K_range = range(2, 8)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.axvline(x=3, color='red', linestyle='--', label='K=3')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouettes, 'go-')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.axvline(x=3, color='red', linestyle='--', label='K=3')
plt.legend()

plt.tight_layout()
plt.savefig('H:/AI/project 12/elbow_method.png')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

km_score = silhouette_score(X_scaled, df['KMeans_Cluster'])
km_rand  = adjusted_rand_score(y_true, df['KMeans_Cluster'])

print(f"\n--- K-Means (K=3) ---")
print(f"Silhouette Score    : {km_score:.3f}")
print(f"Adjusted Rand Score : {km_rand:.3f} (how well it matches real classes)")

hierarchical = AgglomerativeClustering(n_clusters=3)
df['Hierarchical_Cluster'] = hierarchical.fit_predict(X_scaled)

hc_score = silhouette_score(X_scaled, df['Hierarchical_Cluster'])
hc_rand  = adjusted_rand_score(y_true, df['Hierarchical_Cluster'])

print(f"\n--- Hierarchical Clustering ---")
print(f"Silhouette Score    : {hc_score:.3f}")
print(f"Adjusted Rand Score : {hc_rand:.3f}")

print(f"\n--- Model Comparison ---")
print(f"K-Means       : Silhouette={km_score:.3f}  Rand={km_rand:.3f}")
print(f"Hierarchical  : Silhouette={hc_score:.3f}  Rand={hc_rand:.3f}")
best = "K-Means" if km_rand > hc_rand else "Hierarchical"
print(f"Best model    : {best}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA: reduced {X.shape[1]} features to 2 dimensions")
print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

plt.figure(figsize=(14, 5))

colors = ['red', 'blue', 'green']

plt.subplot(1, 3, 1)
for i in range(3):
    mask = df['KMeans_Cluster'] == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], s=50, label=f'Cluster {i}')
plt.title(f'K-Means Clusters\nRand={km_rand:.3f}')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()

plt.subplot(1, 3, 2)
for i in range(3):
    mask = df['Hierarchical_Cluster'] == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], s=50, label=f'Cluster {i}')
plt.title(f'Hierarchical Clusters\nRand={hc_rand:.3f}')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()

plt.subplot(1, 3, 3)
for i, cls in enumerate([1, 2, 3]):
    mask = y_true == cls
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], s=50, label=f'Class {cls}')
plt.title('Real Classes (Ground Truth)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()

plt.tight_layout()
plt.savefig('H:/AI/project 12/clusters_pca.png')
plt.show()

plt.figure(figsize=(12, 5))
Z = linkage(X_scaled, method='ward')
dendrogram(Z, truncate_mode='lastp', p=30)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('H:/AI/project 12/dendrogram.png')
plt.show()

print("\n--- Cluster profiles (K-Means) ---")
print(df.groupby('KMeans_Cluster')[['alcohol', 'flavanoids', 'color_intensity', 'proline']].mean().round(2))
