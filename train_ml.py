# Обучение ML-моделей для регрессии (House Prices)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

import config
from data_loader import load_train_data
from preprocess import preprocess_pipeline, select_features, scale_features, log_transform_target

# ========== 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ==========
print("=" * 60)
print("ОБУЧЕНИЕ ML-МОДЕЛЕЙ (РЕГРЕССИЯ)")
print("=" * 60)

df = load_train_data()
df_processed = preprocess_pipeline(df, is_train=True)
X, y = select_features(df_processed, target_col='SalePrice')

# Логарифмируем целевую переменную
y_log = log_transform_target(y)

print(f"Форма X: {X.shape}")
print(f"Форма y_log: {y_log.shape}")

# Масштабируем
X_scaled, scaler = scale_features(X)

# ========== 2. МЕТРИКА RMSE ==========
def rmse_cv(model, X, y, cv=5):
    """Вычисляет RMSE с кросс-валидацией"""
    scores = -cross_val_score(model, X, y, cv=cv,
                               scoring='neg_mean_squared_error',
                               error_score='raise')
    rmse_scores = np.sqrt(scores)
    return rmse_scores.mean(), rmse_scores.std()

# ========== 3. МОДЕЛИ ==========
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0, random_state=config.RANDOM_SEED),
    'Lasso': Lasso(alpha=0.001, random_state=config.RANDOM_SEED),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=config.RANDOM_SEED),
    'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
    'KNN (k=7)': KNeighborsRegressor(n_neighbors=7),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=config.RANDOM_SEED),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=config.RANDOM_SEED),
    'XGBoost': XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                              random_state=config.RANDOM_SEED, verbosity=0),
    'LightGBM': LGBMRegressor(n_estimators=200, max_depth=5, num_leaves=31,
                               random_state=config.RANDOM_SEED, verbose=-1),
    'CatBoost': CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05,
                                   random_seed=config.RANDOM_SEED, verbose=False)
}

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ МОДЕЛЕЙ (RMSE, кросс-валидация 5 фолдов)")
print("=" * 60)

results = []
for name, model in models.items():
    try:
        mean_rmse, std_rmse = rmse_cv(model, X_scaled, y_log, cv=config.N_FOLDS)
        results.append((name, mean_rmse, std_rmse))
        print(f"{name:20} | RMSE: {mean_rmse:.5f} (+/- {std_rmse:.5f})")
    except Exception as e:
        print(f"{name:20} | Ошибка: {str(e)[:50]}")

# ========== 4. ЛУЧШАЯ МОДЕЛЬ ==========
print("\n" + "=" * 60)
print("ЛУЧШАЯ МОДЕЛЬ")
print("=" * 60)
best_model = min(results, key=lambda x: x[1])
print(f"🏆 {best_model[0]} с RMSE = {best_model[1]:.5f}")

# ========== 5. ОБУЧЕНИЕ ЛУЧШЕЙ МОДЕЛИ НА ВСЕХ ДАННЫХ ==========
print("\n" + "=" * 60)
print("ОБУЧЕНИЕ ЛУЧШЕЙ МОДЕЛИ НА ВСЕХ ДАННЫХ")
print("=" * 60)

if best_model[0] == 'Linear Regression':
    final_model = LinearRegression()
elif best_model[0] == 'Ridge':
    final_model = Ridge(alpha=1.0, random_state=config.RANDOM_SEED)
elif best_model[0] == 'Lasso':
    final_model = Lasso(alpha=0.001, random_state=config.RANDOM_SEED)
elif best_model[0] == 'ElasticNet':
    final_model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=config.RANDOM_SEED)
elif 'KNN' in best_model[0]:
    k = 5 if '5' in best_model[0] else 7
    final_model = KNeighborsRegressor(n_neighbors=k)
elif best_model[0] == 'Decision Tree':
    final_model = DecisionTreeRegressor(max_depth=10, random_state=config.RANDOM_SEED)
elif best_model[0] == 'Random Forest':
    final_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=config.RANDOM_SEED)
elif best_model[0] == 'XGBoost':
    final_model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                random_state=config.RANDOM_SEED, verbosity=0)
elif best_model[0] == 'LightGBM':
    final_model = LGBMRegressor(n_estimators=200, max_depth=5, num_leaves=31,
                                 random_state=config.RANDOM_SEED, verbose=-1)
elif best_model[0] == 'CatBoost':
    final_model = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05,
                                     random_seed=config.RANDOM_SEED, verbose=False)

final_model.fit(X_scaled, y_log)
print(f" {best_model[0]} обучена на всех {X_scaled.shape[0]} примерах")

# Сохраняем модель (позже)
import joblib
joblib.dump(final_model, f"{config.MODELS_DIR}/best_model_{best_model[0].replace(' ', '_')}.pkl")
joblib.dump(scaler, f"{config.MODELS_DIR}/scaler.pkl")
print(f" Модель сохранена в {config.MODELS_DIR}/")