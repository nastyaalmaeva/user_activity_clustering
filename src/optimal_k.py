import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import os

os.makedirs('../plots', exist_ok=True)

df = pd.read_csv('../data/users_preprocessed.csv')
scaler = joblib.load('../models/scaler.pkl')

features = ['daily_usage_min', 'sessions_per_day', 'avg_session_duration',
            'purchase_count', 'referral_count']
X_raw = df[features].values
X_scaled = scaler.transform(X_raw)

def compute_gap_statistic(X, k_max=10, n_refs=5):
    k_range = range(1, k_max + 1)

    # dispersion for original data
    original_dispersion = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        original_dispersion.append(kmeans.inertia_)

    # dispersion for reference data
    ref_dispersions = []
    for _ in range(n_refs):
        # generate uniform reference data
        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)
        X_ref = np.random.uniform(min_vals, max_vals, size=X.shape)

        ref_disp = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_ref)
            ref_disp.append(kmeans.inertia_)
        ref_dispersions.append(ref_disp)

    ref_dispersions = np.array(ref_dispersions)
    mean_ref_disp = ref_dispersions.mean(axis=0)
    std_ref_disp = ref_dispersions.std(axis=0)

    gap = np.log(mean_ref_disp) - np.log(original_dispersion)
    gap_std = std_ref_disp * np.sqrt(1 + 1 / n_refs)

    return k_range, gap, gap_std


k_range, gap, gap_std = compute_gap_statistic(X_scaled, k_max=10, n_refs=5)

optimal_k_gap = None
for i in range(len(k_range) - 1):
    if gap[i] >= gap[i + 1] - gap_std[i + 1]:
        optimal_k_gap = k_range[i]
        break

# other methods
results = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    results.append({
        'k': k,
        'inertia': kmeans.inertia_,
        'silhouette': silhouette_score(X_scaled, labels),
        'calinski_harabasz': calinski_harabasz_score(X_scaled, labels),
        'davies_bouldin': davies_bouldin_score(X_scaled, labels)
    })

results_df = pd.DataFrame(results)

print('\nelbow method (inertia):')
for i in range(1, len(results_df)):
    diff = results_df.loc[i - 1, 'inertia'] - results_df.loc[i, 'inertia']
    diff_pct = diff / results_df.loc[i - 1, 'inertia'] * 100
    print(f'  k={i}->{i + 1}: {diff:.0f} ({diff_pct:.1f}%)')

best_sil = results_df.loc[results_df['silhouette'].idxmax()]
print(f'\nsilhouette score: best k={best_sil["k"]:.0f}, score={best_sil["silhouette"]:.3f}')

best_ch = results_df.loc[results_df['calinski_harabasz'].idxmax()]
print(f'calinski-harabasz: best k={best_ch["k"]:.0f}, score={best_ch["calinski_harabasz"]:.1f}')

best_db = results_df.loc[results_df['davies_bouldin'].idxmin()]
print(f'davies-bouldin: best k={best_db["k"]:.0f}, score={best_db["davies_bouldin"]:.3f}')

print(f'\ngap statistic: optimal k={optimal_k_gap}')

# voting
votes = [best_sil['k'], best_ch['k'], best_db['k'], optimal_k_gap]
k_counts = pd.Series(votes).value_counts()

print('\nsummary:')
if len(k_counts) > 1:
    print(f'  methods disagree: {dict(k_counts)}')
    print('  recommended: k=4')
else:
    print(f'  all methods agree: k={k_counts.index[0]}')

# visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(results_df['k'], results_df['inertia'], marker='o', linewidth=2)
axes[0, 0].axvline(x=4, color='red', linestyle='--', alpha=0.5, label='k=4')
axes[0, 0].set_xlabel('k')
axes[0, 0].set_ylabel('inertia')
axes[0, 0].set_title('elbow method')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(results_df['k'], results_df['silhouette'], marker='o', color='green', linewidth=2)
axes[0, 1].axvline(x=4, color='red', linestyle='--', alpha=0.5, label='k=4')
axes[0, 1].set_xlabel('k')
axes[0, 1].set_ylabel('silhouette')
axes[0, 1].set_title('silhouette score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(results_df['k'], results_df['calinski_harabasz'], marker='o', color='orange', linewidth=2)
axes[1, 0].axvline(x=4, color='red', linestyle='--', alpha=0.5, label='k=4')
axes[1, 0].set_xlabel('k')
axes[1, 0].set_ylabel('calinski-harabasz')
axes[1, 0].set_title('calinski-harabasz index')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(results_df['k'], results_df['davies_bouldin'], marker='o', color='red', linewidth=2)
axes[1, 1].axvline(x=4, color='green', linestyle='--', alpha=0.5, label='best')
axes[1, 1].set_xlabel('k')
axes[1, 1].set_ylabel('davies-bouldin')
axes[1, 1].set_title('davies-bouldin index (lower is better)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('methods for optimal k selection', fontsize=14)
plt.tight_layout()
plt.savefig('../plots/optimal_k_analysis.png', dpi=300)
plt.close()

print('\nplots saved: plots/optimal_k_analysis.png')