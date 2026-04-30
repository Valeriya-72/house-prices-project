# Расширенные эксперименты DNN: dropout, активации, оптимизаторы, параметры

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

import config
from data_loader import load_train_data
from preprocess import preprocess_pipeline, select_features, log_transform_target


def create_model(input_size, hidden_sizes, dropout_rate, activation='relu'):
    layers = []
    prev = input_size

    for i, size in enumerate(hidden_sizes):
        layers.append(nn.Linear(prev, size))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Dropout(dropout_rate))
        prev = size
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


print("=" * 60)
print("DNN РАСШИРЕННЫЕ ЭКСПЕРИМЕНТЫ")
print("=" * 60)

df = load_train_data()
df_processed = preprocess_pipeline(df, is_train=True)
X, y = select_features(df_processed, target_col='SalePrice')
y_log = log_transform_target(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.nan_to_num(X_scaled, nan=0.0)
y_log = np.array(y_log)

print(f"Форма X_scaled: {X_scaled.shape}")

# Эксперименты
experiments = [
    {"name": "Dropout=0.2", "hidden": [128, 64, 32], "dropout": 0.2, "lr": 0.001, "opt": "Adam", "activation": "relu"},
    {"name": "Dropout=0.3", "hidden": [128, 64, 32], "dropout": 0.3, "lr": 0.001, "opt": "Adam", "activation": "relu"},
    {"name": "Dropout=0.4", "hidden": [128, 64, 32], "dropout": 0.4, "lr": 0.001, "opt": "Adam", "activation": "relu"},
    {"name": "Dropout=0.5", "hidden": [128, 64, 32], "dropout": 0.5, "lr": 0.001, "opt": "Adam", "activation": "relu"},
    {"name": "Tanh активация", "hidden": [128, 64, 32], "dropout": 0.3, "lr": 0.001, "opt": "Adam",
     "activation": "tanh"},
    {"name": "LeakyReLU активация", "hidden": [128, 64, 32], "dropout": 0.3, "lr": 0.001, "opt": "Adam",
     "activation": "leaky_relu"},
    {"name": "SGD оптимизатор", "hidden": [128, 64, 32], "dropout": 0.3, "lr": 0.01, "opt": "SGD",
     "activation": "relu"},
    {"name": "AdamW оптимизатор", "hidden": [128, 64, 32], "dropout": 0.3, "lr": 0.001, "opt": "AdamW",
     "activation": "relu"},
    {"name": "LR=0.0005", "hidden": [128, 64, 32], "dropout": 0.3, "lr": 0.0005, "opt": "Adam", "activation": "relu"},
    {"name": "LR=0.005", "hidden": [128, 64, 32], "dropout": 0.3, "lr": 0.005, "opt": "Adam", "activation": "relu"},
    {"name": "Batch size=64", "hidden": [128, 64, 32], "dropout": 0.3, "lr": 0.001, "opt": "Adam", "activation": "relu",
     "batch_size": 64},
    {"name": "Эпохи=100", "hidden": [128, 64, 32], "dropout": 0.3, "lr": 0.001, "opt": "Adam", "activation": "relu",
     "epochs": 100},
]

kfold = KFold(n_splits=3, shuffle=True, random_state=config.RANDOM_SEED)

for exp in experiments:
    print(f"\n--- {exp['name']} ---")
    rmse_scores = []
    epochs = exp.get("epochs", 50)
    batch_size = exp.get("batch_size", 32)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_log[train_idx], y_log[val_idx]

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

        model = create_model(X_scaled.shape[1], exp["hidden"], exp["dropout"], exp["activation"])

        if exp["opt"] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=exp["lr"])
        elif exp["opt"] == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=exp["lr"])
        else:
            optimizer = optim.SGD(model.parameters(), lr=exp["lr"], momentum=0.9)

        criterion = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_X).squeeze(), batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(X_val_t).numpy().flatten()
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        rmse_scores.append(rmse)

    print(f"  RMSE: {np.mean(rmse_scores):.5f} (+/- {np.std(rmse_scores):.5f})")