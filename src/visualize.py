import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

os.makedirs('plots', exist_ok=True)

df = pd.read_csv('../data/users_preprocessed.csv')
df_kmeans = pd.read_csv('../data/users_kmeans.csv')
df_dbscan = pd.read_csv('../data/users_dbscan.csv')
df_hier = pd.read_csv('../data/users_hierarchical.csv')
df_gmm = pd.read_csv('../data/users_gmm.csv')
df_spectral = pd.read_csv('../data/users_spectral.csv')

df['kmeans'] = df_kmeans['kmeans_cluster']
df['dbscan'] = df_dbscan['dbscan_cluster']
df['hierarchical'] = df_hier['hierarchical_cluster']
df['gmm'] = df_gmm['gmm_cluster']
df['spectral'] = df_spectral['spectral_cluster']

# static visualization
fig, axes = plt.subplots(3, 3, figsize=(20, 18))

methods = ['true_cluster', 'kmeans', 'dbscan', 'hierarchical', 'gmm', 'spectral']
titles = ['true clusters', 'kmeans', 'dbscan', 'hierarchical', 'gmm', 'spectral']

for i, (method, title) in enumerate(zip(methods, titles)):
    ax = axes[i // 3, i % 3]

    if method == 'true_cluster':
        colors = pd.Categorical(df[method]).codes
        sc = ax.scatter(df['PC1'], df['PC2'], c=colors, cmap='viridis', alpha=0.6, s=8)
    else:
        sc = ax.scatter(df['PC1'], df['PC2'], c=df[method], cmap='plasma', alpha=0.6, s=8)

    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

sns.boxplot(data=df, x='kmeans', y='daily_usage_min', ax=axes[2, 0])
axes[2, 0].set_title('daily_usage_min by kmeans')

df['kmeans'].value_counts().sort_index().plot(kind='bar', ax=axes[2, 1])
axes[2, 1].set_title('kmeans cluster sizes')
axes[2, 1].set_xlabel('cluster')
axes[2, 1].set_ylabel('count')

sizes = {}
for method in ['kmeans', 'dbscan', 'hierarchical', 'gmm', 'spectral']:
    sizes[method] = len(df[method].unique())
    if -1 in df[method].unique():
        sizes[method] -= 1

axes[2, 2].bar(sizes.keys(), sizes.values())
axes[2, 2].set_title('number of clusters')
axes[2, 2].set_ylabel('clusters')

plt.tight_layout()
plt.savefig('../plots/static_visualization.png', dpi=300)
plt.close()

# interactive 3d plots
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
    color='kmeans',
    hover_data=['user_id'],
    title='kmeans (3d)',
    color_continuous_scale='Plasma',
    opacity=0.6
)
fig2.write_html('../plots/3d_kmeans.html')
fig2.show()

fig3 = px.scatter_3d(
    df, x='PC1_3d', y='PC2_3d', z='PC3_3d',
    color='dbscan',
    hover_data=['user_id'],
    title='dbscan (3d)',
    color_continuous_scale='Turbo',
    opacity=0.6
)
fig3.write_html('../plots/3d_dbscan.html')
fig3.show()

fig4 = px.scatter_3d(
    df, x='PC1_3d', y='PC2_3d', z='PC3_3d',
    color='hierarchical',
    hover_data=['user_id'],
    title='hierarchical (3d)',
    color_continuous_scale='Viridis',
    opacity=0.6
)
fig4.write_html('../plots/3d_hierarchical.html')
fig4.show()

fig5 = px.scatter_3d(
    df, x='PC1_3d', y='PC2_3d', z='PC3_3d',
    color='gmm',
    hover_data=['user_id'],
    title='gmm (3d)',
    color_continuous_scale='Cividis',
    opacity=0.6
)
fig5.write_html('../plots/3d_gmm.html')
fig5.show()

fig6 = px.scatter_3d(
    df, x='PC1_3d', y='PC2_3d', z='PC3_3d',
    color='spectral',
    hover_data=['user_id'],
    title='spectral (3d)',
    color_continuous_scale='Inferno',
    opacity=0.6
)
fig6.write_html('../plots/3d_spectral.html')
fig6.show()

# parallel coordinates
features = ['daily_usage_min', 'sessions_per_day', 'avg_session_duration',
            'purchase_count', 'referral_count']

for method in ['kmeans', 'dbscan', 'hierarchical', 'gmm', 'spectral']:
    cluster_profiles = df.groupby(method)[features].mean().reset_index()

    fig = px.parallel_coordinates(
        cluster_profiles,
        dimensions=features,
        color=method,
        color_continuous_scale=px.colors.diverging.Tealrose,
        title=f'cluster profiles ({method})'
    )
    fig.write_html(f'plots/parallel_coordinates_{method}.html')
    fig.show()

print('visualization complete')