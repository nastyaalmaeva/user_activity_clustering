import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib
import os

os.makedirs('../models', exist_ok=True)

df = pd.read_csv('../data/users_data.csv')

features = ['daily_usage_min', 'sessions_per_day', 'avg_session_duration',
            'purchase_count', 'referral_count']
X = df[features].values
y = df['true_cluster'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f'pca explained variance: {pca.explained_variance_ratio_.sum():.2%}')

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

df_processed = df.copy()
df_processed['PC1'] = X_pca[:, 0]
df_processed['PC2'] = X_pca[:, 1]
df_processed['PC1_3d'] = X_pca_3d[:, 0]
df_processed['PC2_3d'] = X_pca_3d[:, 1]
df_processed['PC3_3d'] = X_pca_3d[:, 2]
df_processed['tSNE1'] = X_tsne[:, 0]
df_processed['tSNE2'] = X_tsne[:, 1]

df_processed.to_csv('../data/users_preprocessed.csv', index=False)

joblib.dump(scaler, '../models/scaler.pkl')
joblib.dump(pca, '../models/pca.pkl')

print(f'saved preprocessed data: {df_processed.shape}')