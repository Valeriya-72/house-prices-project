# House Prices - Advanced Regression Techniques

Проект по предсказанию стоимости домов (регрессия).  
Выполнен в рамках учебного курса по ML/DL.

## 📊 Лучший результат

🏆 **CatBoost** — **RMSE = 0.12989**  
🏆 **DNN с SGD оптимизатором** — **RMSE = 0.24426** (эксперименты)

## 🧠 Ключевые этапы

| Этап | Что сделано |
|------|-------------|
| **EDA** | Анализ пропусков, распределений, корреляций, выбросов |
| **Preprocessing** | Заполнение пропусков, нормализация, кодирование категорий |
| **Feature Engineering** | `TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`, `IsRemod`, `OverallQual_SF`, `OverallQual_Bsm` |
| **Validation** | K-fold (5) + сравнение 1 и 5 фолдов |
| **Models** | Linear Regression, Ridge, Lasso, ElasticNet, KNN, Decision Tree, Random Forest, CatBoost, LightGBM, XGBoost |
| **DNN** | PyTorch: BatchNorm, Dropout, Adam/SGD, CosineAnnealingLR, **Embedding** |
| **Ensembles** | Voting, Averaging, Stacking (Ridge) |

## 📈 Результаты моделей

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
| DNN (SGD) | 0.24426 |
| DNN (Adam) | 0.42136 |

## 📁 Структура

- `eda.py` — разведочный анализ
- `preprocess.py` — предобработка + feature engineering
- `train_ml.py` — обучение ML-моделей
- `train_dnn_advanced.py` — DNN на PyTorch (BatchNorm, Dropout, CosineAnnealingLR)
- `train_dnn_experiments.py` — эксперименты с параметрами DNN
- `train_dnn_embedding.py` — DNN с Embedding (пункт 7.9)
- `ensemble.py` — ансамбли (Voting, Averaging, Stacking)
- `compare_validation.py` — сравнение 1 и 5 фолдов
- `compare_preprocessing.py` — сравнение до/после предобработки

## 🚀 Запуск

1. Установить зависимости:  
   `pip install -r requirements.txt`
2. Запустить EDA:  
   `python eda.py`
3. Обучить ML-модели:  
   `python train_ml.py`
4. Обучить DNN:  
   `python train_dnn_advanced.py`
5. Запустить эксперименты:  
   `python train_dnn_experiments.py`
6. Запустить ансамбли:  
   `python ensemble.py`

## 📎 Ссылки

- [GitHub репозиторий](https://github.com/Valeriya-72/house-prices-project)
- [REPORT.md — подробный отчёт](REPORT.md)

---

**Дата выполнения:** апрель 2026