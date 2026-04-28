# Ансамбли для регрессии: Averaging, Voting, Stacking

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')

import config
from data_loader import load_train_data
from preprocess import preprocess_pipeline, select_features, log_transform_target, scale_features

# ========== 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ==========
print("=" * 60)
print("АНСАМБЛИ ДЛЯ РЕГРЕССИИ (House Prices)")
print("=" * 60)

df = load_train_data()
df_processed = preprocess_pipeline(df, is_train=True)
X, y = select_features(df_processed, target_col='SalePrice')
y_log = log_transform_target(y)

X_scaled, scaler = scale_features(X)
X_scaled = np.array(X_scaled)
y_log = np.array(y_log)

print(f"Форма X_scaled: {X_scaled.shape}")
print(f"Форма y_log: {y_log.shape}")

# ========== 2. ОПРЕДЕЛЯЕМ МОДЕЛИ ==========
models = [
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=config.RANDOM_SEED)),
    ('xgb', XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                         random_state=config.RANDOM_SEED, verbosity=0)),
    ('lgbm', LGBMRegressor(n_estimators=200, max_depth=5, num_leaves=31,
                           random_state=config.RANDOM_SEED, verbose=-1)),
    ('cat', CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05,
                              random_seed=config.RANDOM_SEED, verbose=False))
]

# ========== 3. КРОСС-ВАЛИДАЦИЯ ==========
kfold = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)


def evaluate_model(model, X, y, cv=5):
    """Вычисляет RMSE с кросс-валидацией"""
    rmse_scores = []
    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)
    return np.mean(rmse_scores), np.std(rmse_scores)


# Оценка отдельных моделей
print("\n" + "=" * 60)
print("ОТДЕЛЬНЫЕ МОДЕЛИ (5-fold CV)")
print("=" * 60)
single_results = []
for name, model in models:
    mean_rmse, std_rmse = evaluate_model(model, X_scaled, y_log)
    single_results.append((name, mean_rmse, std_rmse))
    print(f"{name:6} | RMSE: {mean_rmse:.5f} (+/- {std_rmse:.5f})")

# ========== 4. VOTING REGRESSOR (ПРАВИЛЬНЫЙ СПОСОБ) ==========
print("\n" + "=" * 60)
print("VOTING REGRESSOR (мягкое голосование)")
print("=" * 60)

voting = VotingRegressor(estimators=models)
voting_rmse = []
for train_idx, val_idx in kfold.split(X_scaled):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y_log[train_idx], y_log[val_idx]

    voting_clone = VotingRegressor(estimators=models)
    voting_clone.fit(X_train, y_train)
    y_pred = voting_clone.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    voting_rmse.append(rmse)

mean_voting = np.mean(voting_rmse)
std_voting = np.std(voting_rmse)
print(f"Voting  | RMSE: {mean_voting:.5f} (+/- {std_voting:.5f})")

# ========== 5. AVERAGING (усреднение предсказаний) ==========
print("\n" + "=" * 60)
print("AVERAGING (усреднение предсказаний)")
print("=" * 60)

avg_rmse = []
for train_idx, val_idx in kfold.split(X_scaled):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y_log[train_idx], y_log[val_idx]

    predictions = []
    for name, model in models:
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train)
        predictions.append(model_clone.predict(X_val))

    avg_pred = np.mean(predictions, axis=0)
    rmse = np.sqrt(mean_squared_error(y_val, avg_pred))
    avg_rmse.append(rmse)

mean_avg = np.mean(avg_rmse)
std_avg = np.std(avg_rmse)
print(f"Averaging | RMSE: {mean_avg:.5f} (+/- {std_avg:.5f})")

# ========== 6. STACKING С RIDGE ==========
print("\n" + "=" * 60)
print("STACKING (метамодель — Ridge)")
print("=" * 60)

stacking_rmse = []
for train_idx, val_idx in kfold.split(X_scaled):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y_log[train_idx], y_log[val_idx]

    stacking = StackingRegressor(estimators=models, final_estimator=Ridge(alpha=1.0))
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    stacking_rmse.append(rmse)

mean_stacking = np.mean(stacking_rmse)
std_stacking = np.std(stacking_rmse)
print(f"Stacking | RMSE: {mean_stacking:.5f} (+/- {std_stacking:.5f})")

# ========== 7. ЛУЧШАЯ МОДЕЛЬ ==========
print("\n" + "=" * 60)
print("ЛУЧШАЯ МОДЕЛЬ")
print("=" * 60)

best_single = min(single_results, key=lambda x: x[1])
print(f"🏆 Лучшая отдельная модель: {best_single[0]} с RMSE = {best_single[1]:.5f}")

all_ensemble = [
    ('Voting', mean_voting),
    ('Averaging', mean_avg),
    ('Stacking', mean_stacking)
]
best_ensemble = min(all_ensemble, key=lambda x: x[1])
print(f"🏆 Лучший ансамбль: {best_ensemble[0]} с RMSE = {best_ensemble[1]:.5f}")

# ========== 8. ИТОГОВАЯ ТАБЛИЦА ==========
print("\n" + "=" * 60)
print("ИТОГОВОЕ СРАВНЕНИЕ ВСЕХ ПОДХОДОВ")
print("=" * 60)

all_results = []
all_results.extend([('single', name, rmse) for name, rmse, _ in single_results])
all_results.extend([('ensemble', name, rmse) for name, rmse in all_ensemble])

for typ, name, rmse in sorted(all_results, key=lambda x: x[2]):
    print(f"{typ:8} | {name:10} | RMSE: {rmse:.5f}")

print("\n" + "=" * 60)
print(" АНСАМБЛИ ЗАВЕРШЕНЫ")
print("=" * 60)