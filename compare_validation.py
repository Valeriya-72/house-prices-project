# Сравнение 1 и 5 фолдов

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')

import config
from data_loader import load_train_data
from preprocess import preprocess_pipeline, select_features, log_transform_target

print("=" * 60)
print("СРАВНЕНИЕ 1 ФОЛДА И 5 ФОЛДОВ")
print("=" * 60)

df = load_train_data()
df_processed = preprocess_pipeline(df, is_train=True)
X, y = select_features(df_processed, target_col='SalePrice')
y_log = log_transform_target(y)

print(f"Форма X: {X.shape}")

# ========== 1. ПОДХОД 1: 1 ФОЛД (train_test_split) ==========
print("\n1. ПОДХОД 1: ПРОСТОЕ РАЗДЕЛЕНИЕ (1 ФОЛД)")
print("-" * 40)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_log, test_size=0.2, random_state=config.RANDOM_SEED
)

model = CatBoostRegressor(iterations=100, verbose=False, random_seed=config.RANDOM_SEED)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
rmse_1fold = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE на валидации (1 фолд): {rmse_1fold:.5f}")

# ========== 2. ПОДХОД 2: 5 ФОЛДОВ ==========
print("\n2. ПОДХОД 2: 5-ФОЛДОВАЯ КРОСС-ВАЛИДАЦИЯ")
print("-" * 40)

kfold = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
rmse_folds = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    X_train_f, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
    y_train_f, y_val_f = y_log.iloc[train_idx], y_log.iloc[val_idx]

    model = CatBoostRegressor(iterations=100, verbose=False, random_seed=config.RANDOM_SEED)
    model.fit(X_train_f, y_train_f)
    y_pred_f = model.predict(X_val_f)
    rmse = np.sqrt(mean_squared_error(y_val_f, y_pred_f))
    rmse_folds.append(rmse)
    print(f"  Фолд {fold + 1}: RMSE = {rmse:.5f}")

rmse_5fold = np.mean(rmse_folds)
print(f"\nСредний RMSE по 5 фолдам: {rmse_5fold:.5f}")

# ========== 3. ВЫВОД ==========
print("\n" + "=" * 60)
print("СРАВНЕНИЕ")
print("=" * 60)
print(f"1 фолд (простое разделение):  {rmse_1fold:.5f}")
print(f"5 фолдов (кросс-валидация):   {rmse_5fold:.5f}")
print(f"Разница: {abs(rmse_1fold - rmse_5fold):.5f}")

if rmse_5fold < rmse_1fold:
    print(" 5-фолдовая валидация дала более стабильную оценку")
else:
    print(" Оба подхода показали сопоставимые результаты")
