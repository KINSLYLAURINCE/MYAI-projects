import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv('H:/AI/project 1/iris.data', header=None, names=column_names)

print("=" * 50)
print("IRIS CLUSTERING - UNSUPERVISED")
print("=" * 50)
print(f"\nTotal flowers : {df.shape[0]}")
print(f"Features used : 4 (we ignore the species label)")

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y_true = df['species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias    = []
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
plt.axvline(x=3, color='red', linestyle='--', label='K=3')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouettes, 'go-')
plt.axvline(x=3, color='red', linestyle='--', label='K=3')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.legend()

plt.tight_layout()
plt.savefig('H:/AI/project 13/elbow_method.png')
plt.show()

best_k = K_range[np.argmax(silhouettes)]
print(f"\nBest number of clusters: {best_k}")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

km_silhouette = silhouette_score(X_scaled, df['Cluster'])
km_rand       = adjusted_rand_score(y_true, df['Cluster'])

print(f"\n--- K-Means Results ---")
print(f"Silhouette Score    : {km_silhouette:.3f}")
print(f"Adjusted Rand Score : {km_rand:.3f} (vs real species labels)")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

plt.figure(figsize=(14, 5))

colors  = ['red', 'blue', 'green']
species = df['species'].unique()

plt.subplot(1, 3, 1)
for i in range(3):
    mask = df['Cluster'] == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], s=60, label=f'Cluster {i}')
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', s=200, marker='*', label='Centroids')
plt.title(f'K-Means Clusters\nRand={km_rand:.3f}')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(fontsize=8)

plt.subplot(1, 3, 2)
for i, sp in enumerate(species):
    mask = df['species'] == sp
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], s=60, label=sp)
plt.title('Real Species Labels')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(fontsize=8)

plt.subplot(1, 3, 3)
for i in range(3):
    mask = df['Cluster'] == i
    plt.scatter(df[mask]['petal_length'], df[mask]['petal_width'],
                c=colors[i], s=60, label=f'Cluster {i}')
plt.title('Clusters on Petal Features')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(fontsize=8)

plt.tight_layout()
plt.savefig('H:/AI/project 13/clusters.png')
plt.show()

print(f"\n--- Cluster vs Real Species ---")
comparison = pd.crosstab(df['Cluster'], df['species'])
print(comparison)

print(f"\n--- Cluster profiles ---")
print(df.groupby('Cluster')[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].mean().round(2))
