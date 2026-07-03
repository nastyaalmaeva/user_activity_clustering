## Кластеризация пользователей

Сравнительный анализ методов кластеризации на данных о поведении пользователей.

### Данные

10 000 пользователей с 5 поведенческими признаками:
- `daily_usage_min` — минуты в день
- `sessions_per_day` — сессий за день
- `avg_session_duration` — средняя длительность сессии
- `purchase_count` — покупок за месяц
- `referral_count` — приглашений друзей

Истинные кластеры: Sleeping, Active, Whales, Viral

### Методы кластеризации

| Метод | Тип | Особенности |
|-------|-----|-------------|
| KMeans | центроидный | быстрый, требует указать k |
| DBSCAN | плотностной | не требует k, находит шум |
| Иерархическая | агломеративная | дендрограмма, не требует k |
| GMM | вероятностный | мягкая кластеризация |
| Спектральная | графовый | работает со сложными формами |

### Выбор оптимального k

5 методов:
- Elbow Method (инерция)
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index
- Gap Statistic

Результат: все методы сошлись на k=4

### Результаты

| Метод | Кластеры | Шум | Silhouette |
|-------|----------|-----|------------|
| KMeans | 4 | 0% | 0.667 |
| DBSCAN | 4 | 3.8% | 0.656 |
| Иерархическая | 4 | 0% | 0.665 |
| GMM | 4 | 0% | 0.664 |
| Спектральная | 4 | 0% | 0.660 |

### Запуск

```bash
# установка зависимостей
pip install -r requirements.txt

# генерация данных
python src/generate_data.py

# анализ
python src/eda.py

# предобработка
python src/preprocess.py

# выбор оптимального k
python src/optimal_k.py

# кластеризация (по очереди)
python src/cluster_kmeans.py
python src/cluster_dbscan.py
python src/cluster_hierarchical.py
python src/cluster_gmm.py
python src/cluster_spectral.py

# визуализация
python src/visualize.py