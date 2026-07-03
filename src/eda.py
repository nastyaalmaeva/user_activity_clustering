import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('../plots', exist_ok=True)

df = pd.read_csv('../data/users_data.csv')

print('dataset shape:', df.shape)
print('\ncolumns:', df.columns.tolist())
print('\ndtypes:\n', df.dtypes)
print('\nmissing values:\n', df.isnull().sum())
print('\nstatistics:\n', df.describe().round(2))
print('\ncluster distribution:\n', df['true_cluster'].value_counts())

numeric_cols = ['daily_usage_min', 'sessions_per_day', 'avg_session_duration',
                'purchase_count', 'referral_count']
print('\ncorrelations:\n', df[numeric_cols].corr().round(3))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, col in enumerate(numeric_cols):
    ax = axes[i//3, i%3]
    ax.hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(col)

ax = axes[1, 2]
df.boxplot(column='daily_usage_min', by='true_cluster', ax=ax)
ax.set_title('daily_usage_min by cluster')

plt.tight_layout()
plt.savefig('../plots/eda_plots.png', dpi=300)
plt.close()

print('\noutliers (iqr method):')
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
    print(f'  {col}: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)')