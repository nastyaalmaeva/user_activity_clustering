## Кластеризация пользователей

Сравнительный анализ методов кластеризации на синтетических данных о поведении пользователей.

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

### Структура проекта
.
├── src/
│ ├── generate_data.py # генерация синтетических данных
│ ├── eda.py # разведочный анализ
│ ├── preprocess.py # масштабирование, PCA, t-SNE
│ ├── optimal_k.py # выбор оптимального числа кластеров
│ ├── cluster_kmeans.py # KMeans
│ ├── cluster_dbscan.py # DBSCAN
│ ├── cluster_hierarchical.py # иерархическая кластеризация
│ ├── cluster_gmm.py # Gaussian Mixture Models
│ ├── cluster_spectral.py # спектральная кластеризация
│ └── visualize.py # визуализация результатов
├── data/ # csv файлы с данными
├── plots/ # графики (png, html)
├── models/ # сохранённые модели (pkl)
├── requirements.txt # зависимости
└── README.md


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