import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import joblib
import os

os.makedirs('../models', exist_ok=True)
os.makedirs('../plots', exist_ok=True)

df = pd.read_csv('../data/users_preprocessed.csv')
scaler = joblib.load('../models/scaler.pkl')

features = ['daily_usage_min', 'sessions_per_day', 'avg_session_duration',
            'purchase_count', 'referral_count']
X_raw = df[features].values
X_scaled = scaler.transform(X_raw)

# график для выбора eps
neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(X_scaled)
distances, _ = neigh.kneighbors(X_scaled)
distances = np.sort(distances[:, -1])

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('points')
plt.ylabel('distance')
plt.title('k-distance graph')
plt.savefig('../plots/dbscan_eps_selection.png', dpi=300)
plt.close()

eps = 0.5
min_samples = 15

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
y_dbscan = dbscan.fit_predict(X_scaled)

n_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
n_noise = list(y_dbscan).count(-1)

if n_clusters > 1:
    sil = silhouette_score(X_scaled, y_dbscan)
    ch = calinski_harabasz_score(X_scaled, y_dbscan)
    db = davies_bouldin_score(X_scaled, y_dbscan)
else:
    sil = -1
    ch = -1
    db = -1

print('dbscan results:')
print(f'  eps: {eps}, min_samples: {min_samples}')
print(f'  clusters: {n_clusters}')
print(f'  noise: {n_noise} ({n_noise/len(y_dbscan)*100:.1f}%)')
print(f'  silhouette: {sil:.3f}')
print(f'  calinski-harabasz: {ch:.1f}')
print(f'  davies-bouldin: {db:.3f}')

df['dbscan_cluster'] = y_dbscan
df.to_csv('../data/users_dbscan.csv', index=False)

joblib.dump(dbscan, '../models/dbscan_model.pkl')