import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import os

os.makedirs('../models', exist_ok=True)

df = pd.read_csv('../data/users_preprocessed.csv')
scaler = joblib.load('../models/scaler.pkl')

features = ['daily_usage_min', 'sessions_per_day', 'avg_session_duration',
            'purchase_count', 'referral_count']
X_raw = df[features].values
X_scaled = scaler.transform(X_raw)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, y_kmeans)
ch = calinski_harabasz_score(X_scaled, y_kmeans)
db = davies_bouldin_score(X_scaled, y_kmeans)

print('kmeans results:')
print(f'  clusters: {len(np.unique(y_kmeans))}')
print(f'  sizes: {np.bincount(y_kmeans)}')
print(f'  silhouette: {sil:.3f}')
print(f'  calinski-harabasz: {ch:.1f}')
print(f'  davies-bouldin: {db:.3f}')

df['kmeans_cluster'] = y_kmeans
df.to_csv('../data/users_kmeans.csv', index=False)

joblib.dump(kmeans, '../models/kmeans_model.pkl')