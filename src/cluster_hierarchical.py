import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
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

# дендрограмма
linkage_matrix = linkage(X_scaled[:500], method='ward')

plt.figure(figsize=(15, 10))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title('dendrogram (first 500 samples)')
plt.xlabel('samples')
plt.ylabel('distance')
plt.savefig('plots/dendrogram.png', dpi=300)
plt.close()

# иерархическая кластеризация
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
y_hier = hierarchical.fit_predict(X_scaled)

n_clusters = len(np.unique(y_hier))

if n_clusters > 1:
    sil = silhouette_score(X_scaled, y_hier)
    ch = calinski_harabasz_score(X_scaled, y_hier)
    db = davies_bouldin_score(X_scaled, y_hier)
else:
    sil = -1
    ch = -1
    db = -1

print('hierarchical clustering results:')
print(f'  clusters: {n_clusters}')
print(f'  sizes: {np.bincount(y_hier)}')
print(f'  silhouette: {sil:.3f}')
print(f'  calinski-harabasz: {ch:.1f}')
print(f'  davies-bouldin: {db:.3f}')

df['hierarchical_cluster'] = y_hier
df.to_csv('../data/users_hierarchical.csv', index=False)

joblib.dump(hierarchical, '../models/hierarchical_model.pkl')