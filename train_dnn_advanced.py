# DNN на PyTorch для регрессии (House Prices)
# 5 фолдов, BatchNorm, Dropout, CosineAnnealingLR

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

import config
from data_loader import load_train_data
from preprocess import preprocess_pipeline, select_features, log_transform_target


# ========== 1. НЕЙРОСЕТЬ (с BatchNorm и Dropout) ==========
class AdvancedDNN(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze()


# ========== 2. ЗАГРУЗКА И ПРОВЕРКА ДАННЫХ ==========
print("=" * 60)
print("DNN ADVANCED (PyTorch) - House Prices")
print("BatchNorm + Dropout + CosineAnnealingLR + 5 фолдов")
print("=" * 60)

df = load_train_data()
df_processed = preprocess_pipeline(df, is_train=True)
X, y = select_features(df_processed, target_col='SalePrice')
y_log = log_transform_target(y)

print("\nПроверка данных перед масштабированием:")
print(f"NaN в X: {X.isna().sum().sum()}")
print(f"inf в X: {np.isinf(X).sum().sum()}")

# Масштабируем
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Очищаем от NaN
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
y_log = y_log.values.astype(np.float32)
y_log = np.nan_to_num(y_log, nan=y_log.mean())

print(f"\nПосле масштабирования и очистки:")
print(f"NaN в X_scaled: {np.isnan(X_scaled).sum()}")
print(f"Форма X_scaled: {X_scaled.shape}")
print(f"Форма y_log: {y_log.shape}")

# ========== 3. K-FOLD С 5 ФОЛДАМИ ==========
kfold = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
fold_rmse = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
    print(f"\n--- Фолд {fold + 1}/5 ---")

    X_train = X_scaled[train_idx]
    X_val = X_scaled[val_idx]
    y_train = y_log[train_idx]
    y_val = y_log[val_idx]

    # Тензоры
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = AdvancedDNN(input_size=X_scaled.shape[1], dropout_rate=0.3)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.MSELoss()

    # Обучение
    for epoch in range(100):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Эпоха {epoch + 1}/100, Loss: {epoch_loss / len(train_loader):.6f}")

    # Валидация
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_X, _ in val_loader:
            outputs = model(batch_X)
            predictions.extend(outputs.numpy())

    predictions = np.array(predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    fold_rmse.append(rmse)
    print(f"RMSE на фолде {fold + 1}: {rmse:.5f}")

# ========== 4. РЕЗУЛЬТАТЫ ==========
print("\n" + "=" * 60)
print("ИТОГИ DNN ADVANCED (5 фолдов)")
print("=" * 60)
print(f"Средний RMSE по 5 фолдам: {np.mean(fold_rmse):.5f}")
print(f"Стандартное отклонение: {np.std(fold_rmse):.5f}")

# Сравнение с лучшей ML-моделью (CatBoost ~0.131)
best_ml_rmse = 0.13106
best_dnn_rmse = np.mean(fold_rmse)
print(f"\nСравнение с лучшей ML-моделью:")
print(f"  CatBoost (лучшая ML): {best_ml_rmse:.5f}")
print(f"  DNN Advanced:        {best_dnn_rmse:.5f}")