import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
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

gmm = GaussianMixture(n_components=4, random_state=42)
y_gmm = gmm.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, y_gmm)
ch = calinski_harabasz_score(X_scaled, y_gmm)
db = davies_bouldin_score(X_scaled, y_gmm)

print('gmm results:')
print(f'  clusters: {len(np.unique(y_gmm))}')
print(f'  sizes: {np.bincount(y_gmm)}')
print(f'  silhouette: {sil:.3f}')
print(f'  calinski-harabasz: {ch:.1f}')
print(f'  davies-bouldin: {db:.3f}')

df['gmm_cluster'] = y_gmm
df.to_csv('../data/users_gmm.csv', index=False)

joblib.dump(gmm, '../models/gmm_model.pkl')