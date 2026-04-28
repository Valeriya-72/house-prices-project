# Сравнение RMSE до и после предобработки

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')

import config
from data_loader import load_train_data
from preprocess import preprocess_pipeline, select_features, log_transform_target

print("=" * 60)
print("СРАВНЕНИЕ МЕТРИКИ ДО И ПОСЛЕ ПРЕДОБРАБОТКИ")
print("=" * 60)

df = load_train_data()

# ========== 1. СЫРЫЕ ДАННЫМИ ==========
print("\n1. МОДЕЛЬ НА СЫРЫХ ДАННЫХ (минимальная предобработка)")
print("-" * 40)

# Минимальная предобработка для сырых данных
df_raw = df.copy()

# Заполняем пропуски простыми значениями
numeric_cols = df_raw.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    df_raw[col].fillna(df_raw[col].median(), inplace=True)

# Выбираем числовые признаки (без кодирования категорий)
feature_cols = numeric_cols.tolist()
if 'SalePrice' in feature_cols:
    feature_cols.remove('SalePrice')
if 'Id' in feature_cols:
    feature_cols.remove('Id')

X_raw = df_raw[feature_cols]
y_raw = df_raw['SalePrice']
y_raw_log = np.log1p(y_raw)

print(f"Форма X_raw: {X_raw.shape}")

# K-fold
kfold = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
rmse_raw = []

for train_idx, val_idx in kfold.split(X_raw):
    X_train, X_val = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
    y_train, y_val = y_raw_log.iloc[train_idx], y_raw_log.iloc[val_idx]

    model = CatBoostRegressor(iterations=100, verbose=False, random_seed=config.RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_raw.append(rmse)

mean_raw = np.mean(rmse_raw)
print(f"CatBoost на сырых данных: RMSE = {mean_raw:.5f}")

# ========== 2. ОБРАБОТАННЫЕ ДАННЫЕ ==========
print("\n2. МОДЕЛЬ НА ОБРАБОТАННЫХ ДАННЫХ (полная предобработка)")
print("-" * 40)

df_processed = preprocess_pipeline(df, is_train=True)
X, y = select_features(df_processed, target_col='SalePrice')
y_log = log_transform_target(y)

print(f"Форма X_processed: {X.shape}")

rmse_proc = []
for train_idx, val_idx in kfold.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

    model = CatBoostRegressor(iterations=100, verbose=False, random_seed=config.RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_proc.append(rmse)

mean_proc = np.mean(rmse_proc)
print(f"CatBoost на обработанных данных: RMSE = {mean_proc:.5f}")

# ========== 3. СРАВНЕНИЕ ==========
print("\n" + "=" * 60)
print("СРАВНЕНИЕ")
print("=" * 60)
print(f"До предобработки:    {mean_raw:.5f}")
print(f"После предобработки: {mean_proc:.5f}")
print(f"Улучшение:           {mean_raw - mean_proc:.5f}")

if mean_proc < mean_raw:
    print(" Предобработка улучшила качество модели!")
else:
    print(" Предобработка не дала улучшения")