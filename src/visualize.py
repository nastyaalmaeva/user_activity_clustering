import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

os.makedirs('../plots', exist_ok=True)

df_kmeans = pd.read_csv('../data/users_kmeans.csv')
df_dbscan = pd.read_csv('../data/users_dbscan.csv')
df = pd.read_csv('../data/users_preprocessed.csv')

df['kmeans_cluster'] = df_kmeans['kmeans_cluster']
df['dbscan_cluster'] = df_dbscan['dbscan_cluster']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

sc1 = axes[0,0].scatter(df['PC1'], df['PC2'],
                        c=pd.Categorical(df['true_cluster']).codes,
                        cmap='viridis', alpha=0.6, s=10)
axes[0,0].set_title('true clusters')
axes[0,0].set_xlabel('PC1')
axes[0,0].set_ylabel('PC2')

sc2 = axes[0,1].scatter(df['PC1'], df['PC2'],
                        c=df['kmeans_cluster'], cmap='plasma', alpha=0.6, s=10)
axes[0,1].set_title('kmeans')
axes[0,1].set_xlabel('PC1')
axes[0,1].set_ylabel('PC2')

sc3 = axes[0,2].scatter(df['PC1'], df['PC2'],
                        c=df['dbscan_cluster'], cmap='cool', alpha=0.6, s=10)
axes[0,2].set_title('dbscan')
axes[0,2].set_xlabel('PC1')
axes[0,2].set_ylabel('PC2')

axes[1,0].scatter(df['tSNE1'], df['tSNE2'],
                  c=pd.Categorical(df['true_cluster']).codes,
                  cmap='viridis', alpha=0.6, s=10)
axes[1,0].set_title('t-sne (true)')
axes[1,0].set_xlabel('tSNE1')
axes[1,0].set_ylabel('tSNE2')

sns.boxplot(data=df, x='kmeans_cluster', y='daily_usage_min', ax=axes[1,1])
axes[1,1].set_title('daily_usage_min by kmeans cluster')

df['kmeans_cluster'].value_counts().sort_index().plot(kind='bar', ax=axes[1,2])
axes[1,2].set_title('cluster sizes (kmeans)')
axes[1,2].set_xlabel('cluster')
axes[1,2].set_ylabel('count')

plt.tight_layout()
plt.savefig('plots/static_visualization.png', dpi=300)
plt.close()

fig1 = px.scatter_3d(
    df, x='PC1_3d', y='PC2_3d', z='PC3_3d',
    color='true_cluster',
    hover_data=['user_id', 'daily_usage_min', 'purchase_count'],
    title='true clusters (3d pca)',
    color_discrete_sequence=px.colors.qualitative.Set1,
    opacity=0.6
)
fig1.write_html('../plots/3d_true_clusters.html')
fig1.show()

fig2 = px.scatter_3d(
    df, x='PC1_3d', y='PC2_3d', z='PC3_3d',
    color='kmeans_cluster',
    hover_data=['user_id', 'true_cluster'],
    title='kmeans (3d)',
    color_continuous_scale='Plasma',
    opacity=0.6
)
fig2.write_html('../plots/3d_kmeans.html')
fig2.show()

fig3 = px.scatter_3d(
    df, x='PC1_3d', y='PC2_3d', z='PC3_3d',
    color='dbscan_cluster',
    hover_data=['user_id', 'true_cluster'],
    title='dbscan (3d)',
    color_continuous_scale='Turbo',
    opacity=0.6
)
fig3.write_html('../plots/3d_dbscan.html')
fig3.show()

features = ['daily_usage_min', 'sessions_per_day', 'avg_session_duration',
            'purchase_count', 'referral_count']
cluster_profiles = df.groupby('kmeans_cluster')[features].mean().reset_index()

fig4 = px.parallel_coordinates(
    cluster_profiles,
    dimensions=features,
    color='kmeans_cluster',
    color_continuous_scale=px.colors.diverging.Tealrose,
    title='cluster profiles (kmeans)'
)
fig4.write_html('../plots/parallel_coordinates.html')
fig4.show()