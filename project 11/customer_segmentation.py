import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

df = pd.read_csv('H:/AI/project 11/Mall_Customers.csv')

print("=" * 50)
print("MALL CUSTOMERS - EXPLORATION")
print("=" * 50)
print(f"\nTotal customers : {df.shape[0]}")
print(f"\n--- First 5 rows ---")
print(df.head())
print(f"\n--- Statistics ---")
print(df.describe())

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
df['Age'].hist(bins=20, color='steelblue')
plt.title('Age Distribution')
plt.xlabel('Age')

plt.subplot(1, 3, 2)
df['Annual Income (k$)'].hist(bins=20, color='green')
plt.title('Income Distribution')
plt.xlabel('Annual Income (k$)')

plt.subplot(1, 3, 3)
df['Spending Score (1-100)'].hist(bins=20, color='orange')
plt.title('Spending Score Distribution')
plt.xlabel('Spending Score')

plt.tight_layout()
plt.savefig('H:/AI/project 11/exploration.png')
plt.show()

X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Finding best number of clusters (Elbow Method) ---")
inertias = []
silhouettes = []
K_range = range(2, 11)

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
plt.axvline(x=5, color='red', linestyle='--', label='Best K=5')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouettes, 'go-')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.axvline(x=5, color='red', linestyle='--', label='Best K=5')
plt.legend()

plt.tight_layout()
plt.savefig('H:/AI/project 11/elbow_method.png')
plt.show()

best_k = K_range[np.argmax(silhouettes)]
print(f"Best number of clusters: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\n--- K-Means Clusters ---")
print(df.groupby('KMeans_Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(1))

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i in range(best_k):
    cluster_data = df[df['KMeans_Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'],
                cluster_data['Spending Score (1-100)'],
                s=80, c=colors[i], label=f'Cluster {i}')

plt.scatter(scaler.inverse_transform(kmeans.cluster_centers_)[:, 0],
            scaler.inverse_transform(kmeans.cluster_centers_)[:, 1],
            s=300, c='black', marker='*', label='Centroids')
plt.title('Customer Segments (K-Means)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.tight_layout()
plt.savefig('H:/AI/project 11/kmeans_clusters.png')
plt.show()

dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'].values else 0)
n_noise      = (df['DBSCAN_Cluster'] == -1).sum()

print(f"\n--- DBSCAN Results ---")
print(f"Clusters found : {n_clusters_db}")
print(f"Noise points   : {n_noise} (outliers)")

plt.figure(figsize=(8, 6))
unique_labels = set(df['DBSCAN_Cluster'])
color_map = plt.colormaps['tab10'].resampled(len(unique_labels))

for label in unique_labels:
    cluster_data = df[df['DBSCAN_Cluster'] == label]
    color = 'black' if label == -1 else color_map(label)
    name  = 'Noise' if label == -1 else f'Cluster {label}'
    plt.scatter(cluster_data['Annual Income (k$)'],
                cluster_data['Spending Score (1-100)'],
                s=80, c=[color], label=name)

plt.title('Customer Segments (DBSCAN)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.tight_layout()
plt.savefig('H:/AI/project 11/dbscan_clusters.png')
plt.show()

print(f"\n--- Cluster Interpretation (K-Means) ---")
cluster_summary = df.groupby('KMeans_Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean().round(1)
for i, row in cluster_summary.iterrows():
    income  = row['Annual Income (k$)']
    spending = row['Spending Score (1-100)']
    if income > 70 and spending > 60:
        label = "High Income, High Spenders (VIP customers)"
    elif income > 70 and spending < 40:
        label = "High Income, Low Spenders (Careful savers)"
    elif income < 40 and spending > 60:
        label = "Low Income, High Spenders (Impulsive buyers)"
    elif income < 40 and spending < 40:
        label = "Low Income, Low Spenders (Budget customers)"
    else:
        label = "Average customers"
    print(f"Cluster {i}: Income=${income}k  Spending={spending}/100  -> {label}")
