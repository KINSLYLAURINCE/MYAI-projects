import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

genre_cols = ['unknown','Action','Adventure','Animation','Childrens',
              'Comedy','Crime','Documentary','Drama','Fantasy','FilmNoir',
              'Horror','Musical','Mystery','Romance','SciFi','Thriller',
              'War','Western']

movie_cols = ['movie_id','title','release_date','video_date','imdb_url'] + genre_cols
movies = pd.read_csv('H:/AI/project 14/ml-100k/u.item',
                     sep='|', encoding='latin-1', header=None, names=movie_cols)

rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('H:/AI/project 14/ml-100k/u.data',
                      sep='\t', header=None, names=rating_cols)

print("=" * 50)
print("MOVIELENS DATASET - EXPLORATION")
print("=" * 50)
print(f"\nTotal movies  : {movies.shape[0]}")
print(f"Total ratings : {ratings.shape[0]}")
print(f"Total users   : {ratings['user_id'].nunique()}")
print(f"\n--- Sample movies ---")
print(movies[['title'] + genre_cols[:6]].head())

movie_stats = ratings.groupby('movie_id').agg(
    avg_rating=('rating', 'mean'),
    num_ratings=('rating', 'count')
).reset_index()

df = movies.merge(movie_stats, on='movie_id', how='left')
df = df.dropna(subset=['avg_rating', 'num_ratings'])

print(f"\nMovies with ratings: {df.shape[0]}")

features = genre_cols + ['avg_rating', 'num_ratings']
X = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFeatures used: {len(features)}")
print(f"  - 19 genre flags (0 or 1)")
print(f"  - avg_rating")
print(f"  - num_ratings")

inertias    = []
silhouettes = []
K_range = range(2, 9)

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
plt.savefig('H:/AI/project 14/elbow_method.png')
plt.show()

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

km_score = silhouette_score(X_scaled, df['KMeans_Cluster'])
print(f"\n--- K-Means (K={best_k}) ---")
print(f"Silhouette Score: {km_score:.3f}")

hierarchical = AgglomerativeClustering(n_clusters=best_k)
df['Hierarchical_Cluster'] = hierarchical.fit_predict(X_scaled)

hc_score = silhouette_score(X_scaled, df['Hierarchical_Cluster'])
print(f"\n--- Hierarchical Clustering ---")
print(f"Silhouette Score: {hc_score:.3f}")

print(f"\n--- Model Comparison ---")
print(f"K-Means      : Silhouette={km_score:.3f}")
print(f"Hierarchical : Silhouette={hc_score:.3f}")
best = "K-Means" if km_score > hc_score else "Hierarchical"
print(f"Best model   : {best}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))
colors = ['red','blue','green','orange','purple','brown','pink']

plt.subplot(1, 2, 1)
for i in range(best_k):
    mask = df['KMeans_Cluster'] == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors[i], s=20, alpha=0.6, label=f'Cluster {i}')
plt.title(f'K-Means Movie Clusters (K={best_k})')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(fontsize=8)

plt.subplot(1, 2, 2)
plt.scatter(df['avg_rating'], df['num_ratings'],
            c=df['KMeans_Cluster'].map({i: colors[i] for i in range(best_k)}),
            s=20, alpha=0.6)
plt.title('Movies: Rating vs Popularity')
plt.xlabel('Average Rating')
plt.ylabel('Number of Ratings')

plt.tight_layout()
plt.savefig('H:/AI/project 14/movie_clusters.png')
plt.show()

print(f"\n--- Cluster Profiles ---")
cluster_profile = df.groupby('KMeans_Cluster')[
    ['avg_rating', 'num_ratings'] + genre_cols[:5]].mean().round(2)
print(cluster_profile)

print(f"\n--- Sample Movies per Cluster ---")
for i in range(best_k):
    cluster_movies = df[df['KMeans_Cluster'] == i].nlargest(3, 'num_ratings')['title'].values
    print(f"Cluster {i}: {list(cluster_movies)}")
