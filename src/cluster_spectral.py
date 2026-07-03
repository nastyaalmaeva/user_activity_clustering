import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
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

spectral = SpectralClustering(n_clusters=4, random_state=42, affinity='rbf')
y_spectral = spectral.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, y_spectral)
ch = calinski_harabasz_score(X_scaled, y_spectral)
db = davies_bouldin_score(X_scaled, y_spectral)

print('spectral clustering results:')
print(f'  clusters: {len(np.unique(y_spectral))}')
print(f'  sizes: {np.bincount(y_spectral)}')
print(f'  silhouette: {sil:.3f}')
print(f'  calinski-harabasz: {ch:.1f}')
print(f'  davies-bouldin: {db:.3f}')

df['spectral_cluster'] = y_spectral
df.to_csv('../data/users_spectral.csv', index=False)

joblib.dump(spectral, '../models/spectral_model.pkl')