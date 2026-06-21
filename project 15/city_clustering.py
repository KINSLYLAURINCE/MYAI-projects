import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

df = pd.read_csv('H:/AI/project 15/worldcities.csv')

print("=" * 50)
print("WORLD CITIES DATASET - EXPLORATION")
print("=" * 50)
print(f"\nTotal cities  : {df.shape[0]}")
print(f"Total columns : {df.shape[1]}")
print(f"\n--- Sample cities ---")
print(df[['city', 'country', 'lat', 'lng', 'population']].head(10))

df = df.dropna(subset=['lat', 'lng', 'population'])
df = df[df['population'] > 100000]
df = df.reset_index(drop=True)

print(f"\nCities with population > 100k: {df.shape[0]}")

X = df[['lat', 'lng', 'population']].copy()
X['population'] = np.log1p(X['population'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias    = []
silhouettes = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

best_k = K_range[np.argmax(silhouettes)]
print(f"\nBest number of clusters: {best_k}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.axvline(x=best_k, color='red', linestyle='--', label=f'K={best_k}')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouettes, 'go-')
plt.axvline(x=best_k, color='red', linestyle='--', label=f'K={best_k}')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.legend()

plt.tight_layout()
plt.savefig('H:/AI/project 15/elbow_method.png')
plt.show()

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

km_score = silhouette_score(X_scaled, df['KMeans_Cluster'])
print(f"\n--- K-Means (K={best_k}) ---")
print(f"Silhouette Score: {km_score:.3f}")

dbscan = DBSCAN(eps=0.3, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

n_clusters = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'].values else 0)
n_noise    = (df['DBSCAN_Cluster'] == -1).sum()

print(f"\n--- DBSCAN ---")
print(f"Clusters found : {n_clusters}")
print(f"Noise points   : {n_noise} (isolated cities)")

plt.figure(figsize=(16, 8))
colors = ['red','blue','green','orange','purple','brown','pink','cyan']

plt.subplot(2, 1, 1)
for i in range(best_k):
    mask = df['KMeans_Cluster'] == i
    plt.scatter(df[mask]['lng'], df[mask]['lat'],
                c=colors[i % len(colors)], s=10, alpha=0.6, label=f'Cluster {i}')
plt.title(f'K-Means City Clusters (K={best_k})')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(fontsize=7, loc='lower left')

plt.subplot(2, 1, 2)
unique_labels = sorted(set(df['DBSCAN_Cluster']))
for label in unique_labels[:8]:
    mask  = df['DBSCAN_Cluster'] == label
    color = 'black' if label == -1 else colors[label % len(colors)]
    name  = 'Noise' if label == -1 else f'Cluster {label}'
    plt.scatter(df[mask]['lng'], df[mask]['lat'],
                c=color, s=10, alpha=0.6, label=name)
plt.title('DBSCAN City Clusters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(fontsize=7, loc='lower left')

plt.tight_layout()
plt.savefig('H:/AI/project 15/city_clusters.png')
plt.show()

print(f"\n--- K-Means Cluster Profiles ---")
profile = df.groupby('KMeans_Cluster').agg(
    num_cities=('city', 'count'),
    avg_lat=('lat', 'mean'),
    avg_lng=('lng', 'mean'),
    avg_population=('population', 'mean'),
    top_city=('city', 'first')
).round(1)
print(profile)

print(f"\n--- Top 3 Cities per Cluster (by population) ---")
for i in range(best_k):
    top = df[df['KMeans_Cluster'] == i].nlargest(3, 'population')[['city', 'country', 'population']]
    print(f"\nCluster {i}:")
    print(top.to_string(index=False))
