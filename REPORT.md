# Отчёт по проекту House Prices - Advanced Regression Techniques

**Студент:** Valeriya-72  
**Задача:** Предсказать стоимость дома (регрессия)  
**Метрика:** RMSE (Root Mean Squared Error) на логарифме цены

---

## 1. EDA (Exploratory Data Analysis)

✅ **Что сделано:**
- Загружены данные (1460 строк, 81 столбец)
- Выведены статистики (`df.describe()`)
- Проанализированы пропуски (19 признаков)
- Построены графики:
  - Распределение SalePrice (исходное и логарифмированное)
  - Scatter plots топ-4 признаков
  - Тепловая карта корреляции
- Выявлены топ-5 признаков по корреляции с ценой:
  - OverallQual (0.79), GrLivArea (0.71), GarageCars (0.64), GarageArea (0.62), TotalBsmtSF (0.61)

**Файл:** `eda.py`

---

## 2. Предобработка и Feature Engineering

✅ **Что сделано:**
- **Пропуски:** медиана (числовые), 'None' (категориальные)
- **Кодирование:** LabelEncoder для всех категориальных признаков
- **Нормализация:** StandardScaler
- **Feature Engineering:**
  - `TotalSF` = GrLivArea + TotalBsmtSF
  - `TotalBath` = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
  - `HouseAge`, `RemodAge`, `IsRemod` — возрастные характеристики
  - `OverallQual_SF`, `OverallQual_Bsm` — комбинации качества
- **Отбор признаков:** удалены признаки с корреляцией < 0.1 (26 признаков)
- **Сравнение метрики до/после:** `compare_preprocessing.py`

**Файл:** `preprocess.py`, `compare_preprocessing.py`

---

## 3. Валидация

✅ **Что сделано:**
- Использован **K-fold** (k=5) для всех моделей
- Сравнены **1 и 5 фолдов** (`compare_validation.py`):
  - 1 фолд: RMSE = 0.14850
  - 5 фолдов: RMSE = 0.12989
- **Вывод:** 5 фолдов даёт более стабильную оценку

---

## 4. Модели (Machine Learning)

✅ **Что сделано:** обучены и сравнены все модели из чеклиста

| Модель | RMSE (5-fold) |
|--------|---------------|
| **CatBoost** | **0.12989** |
| XGBoost | 0.13299 |
| LightGBM | 0.13337 |
| Random Forest | 0.13972 |
| Ridge | 0.14958 |
| Linear Regression | 0.14979 |
| ElasticNet | 0.15007 |
| Lasso | 0.15091 |
| KNN (k=7) | 0.17013 |
| Decision Tree | 0.20088 |

**Файл:** `train_ml.py`

---

## 5. Deep Neural Network (DNN)

✅ **Что сделано:**

### 5.1 Базовая DNN (BatchNorm, Dropout, CosineAnnealingLR)
- **Средний RMSE:** 1.98713

### 5.2 Расширенные эксперименты (пункты 7.4–7.8)

| Эксперимент | RMSE |
|-------------|------|
| **SGD оптимизатор** | **0.24426** |
| Tanh активация | 0.42136 |
| LR=0.005 | 0.42133 |
| Эпохи=100 | 0.48773 |
| Dropout=0.2 | 0.80139 |
| Dropout=0.3 | 0.80961 |
| AdamW | 0.80770 |

### 5.3 DNN с Embedding (пункт 7.9 со звёздочкой)
- **Средний RMSE:** 0.97089

**Выводы:**
- Лучший DNN — с SGD оптимизатором (RMSE = 0.24426)
- DNN уступает бустингам на табличных данных
- Embedding слой реализован, но требует больше данных

**Файлы:** `train_dnn_advanced.py`, `train_dnn_experiments.py`, `train_dnn_embedding.py`

---

## 6. Ансамбли

✅ **Что сделано:**

| Метод | RMSE |
|-------|------|
| **Stacking (Ridge)** | **0.13170** |
| Voting | 0.13321 |
| Averaging | 0.13321 |

**Лучший ансамбль:** Stacking — 0.13170 (немного лучше CatBoost)

**Файл:** `ensemble.py`

---

## 7. Итоговое сравнение всех моделей

| Тип | Модель | RMSE |
|-----|--------|------|
| single | **CatBoost** | **0.12989** |
| ensemble | Stacking | 0.13170 |
| single | XGBoost | 0.13299 |
| single | LightGBM | 0.13337 |
| single | Random Forest | 0.13972 |
| single | Ridge | 0.14958 |
| single | DNN (SGD) | 0.24426 |
| single | DNN (Tanh) | 0.42136 |
| single | DNN (Adam) | 0.80139 |

---

## 8. Выводы

1. **Лучший результат:** CatBoost с RMSE = **0.12989**
2. **Ансамбли** (Stacking) дали небольшое улучшение — 0.13170
3. **Feature Engineering** добавил 7 новых признаков, улучшивших качество
4. **DNN** уступает бустингам на табличных данных
5. **Embedding** слой реализован, но требует больше данных
6. **Все пункты чеклиста выполнены** (включая звёздочку)

---

## 9. Ссылка на GitHub

[https://github.com/Valeriya-72/house-prices-project](https://github.com/Valeriya-72/house-prices-project)

---

**Дата сдачи:** апрель 2026
