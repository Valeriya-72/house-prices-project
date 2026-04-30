# DNN с Embedding слоем для категориальных фичей (пункт 7.9 со звёздочкой)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')

import config
from data_loader import load_train_data

# ========== 1. ЗАГРУЗКА ДАННЫХ ==========
print("=" * 60)
print("DNN С EMBEDDING СЛОЕМ (Пункт 7.9 со звёздочкой)")
print("=" * 60)

df = load_train_data()

# Выбираем категориальные признаки для Embedding
categorical_cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
                    'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
                    'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                    'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                    'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                    'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
                    'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
                    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                    'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType',
                    'SaleCondition']

# Числовые признаки
numerical_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
                  'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                  'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                  'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                  'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
                  'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                  'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

# Заполняем пропуски
for col in categorical_cols:
    df[col] = df[col].fillna('None').astype(str)
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Кодируем категориальные признаки в числа
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Нормализуем числовые признаки
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print(f"Обработано {len(categorical_cols)} категориальных признаков")
print(f"Обработано {len(numerical_cols)} числовых признаков")


# ========== 2. ДАТАСЕТ ==========
class EmbeddingDataset(Dataset):
    def __init__(self, df, categorical_cols, numerical_cols, target_col='SalePrice'):
        self.categorical = torch.tensor(df[categorical_cols].values, dtype=torch.long)
        self.numerical = torch.tensor(df[numerical_cols].values, dtype=torch.float32)
        self.target = torch.tensor(np.log1p(df[target_col].values), dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.categorical[idx], self.numerical[idx], self.target[idx]


# ========== 3. НЕЙРОСЕТЬ С EMBEDDING ==========
class DNNWithEmbedding(nn.Module):
    def __init__(self, categorical_dims, embedding_dim=8, numerical_dim=len(numerical_cols),
                 hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super().__init__()

        # Embedding слои
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim) for dim in categorical_dims
        ])

        # Входная размерность = (число категориальных * embedding_dim) + числовые
        input_dim = len(categorical_dims) * embedding_dim + numerical_dim

        # Полносвязные слои
        layers = []
        prev = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, categorical, numerical):
        # Embedding
        embedded = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded = torch.cat(embedded, dim=1)
        # Объединяем с числовыми
        x = torch.cat([embedded, numerical], dim=1)
        return self.network(x).squeeze()


# ========== 4. ОБУЧЕНИЕ ==========
# Размерности категориальных признаков
categorical_dims = [df[col].nunique() for col in categorical_cols]

dataset = EmbeddingDataset(df, categorical_cols, numerical_cols)
kfold = KFold(n_splits=3, shuffle=True, random_state=config.RANDOM_SEED)

fold_rmse = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
    print(f"\n--- Фолд {fold + 1} ---")

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = DNNWithEmbedding(categorical_dims)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    for epoch in range(50):
        model.train()
        for cat, num, target in train_loader:
            optimizer.zero_grad()
            output = model(cat, num)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for cat, num, target in val_loader:
            output = model(cat, num)
            predictions.extend(output.numpy())
            targets.extend(target.numpy())

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    fold_rmse.append(rmse)
    print(f"RMSE: {rmse:.5f}")

print("\n" + "=" * 60)
print("ИТОГИ DNN С EMBEDDING")
print("=" * 60)
print(f"Средний RMSE: {np.mean(fold_rmse):.5f}")