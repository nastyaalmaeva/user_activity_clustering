## Кластеризация пользователей

Анализ и кластеризация синтетических данных о поведении пользователей.

### Данные

10000 пользователей с 5 поведенческими признаками:
- daily_usage_min - минуты в день
- sessions_per_day - сессий за день
- avg_session_duration - средняя длительность сессии
- purchase_count - покупок за месяц
- referral_count - приглашений друзей

Истинные кластеры: Sleeping, Active, Whales, Viral

### Этапы

1. `generate_data.py` - генерация данных
2. `eda.py` - разведочный анализ
3. `preprocess.py` - масштабирование, PCA, t-SNE
4. `cluster_kmeans.py` - кластеризация KMeans
5. `cluster_dbscan.py` - кластеризация DBSCAN
6. `visualize.py` - статические и интерактивные графики

### Результаты

| Метрика | KMeans | DBSCAN |
|---------|--------|--------|
| Кластеры | 4 | 4 |
| Silhouette | 0.667 | 0.656 |
| Calinski-Harabasz | 19661.5 | 11686.3 |
| Davies-Bouldin | 0.530 | 2.238 |

### Запуск

```bash
python generate_data.py
python eda.py
python preprocess.py
python cluster_kmeans.py
python cluster_dbscan.py
python visualize.py