# Предобработка данных для House Prices + Feature Engineering

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import config
import warnings

warnings.filterwarnings('ignore')


# ========== ЗАПОЛНЕНИЕ ПРОПУСКОВ ==========
def fill_missing_numerical(df, columns, strategy='median'):
    """Заполняет пропуски в числовых колонках"""
    for col in columns:
        if col in df.columns:
            if strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'zero':
                df[col].fillna(0, inplace=True)
    return df


def fill_missing_categorical(df, columns, value='None'):
    """Заполняет пропуски в категориальных колонках"""
    for col in columns:
        if col in df.columns:
            df[col].fillna(value, inplace=True)
    return df


def fill_missing_numerical_for_linear(df):
    """Специальное заполнение пропусков для линейных моделей (медианой)"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    return df


# ========== КОДИРОВАНИЕ ==========
def encode_categorical(df, columns):
    """Кодирует категориальные признаки в числа"""
    df_encoded = df.copy()
    for col in columns:
        if col in df_encoded.columns and df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded


# ========== ОТБОР ПРИЗНАКОВ ==========
def drop_low_importance_features(df, target_corr_threshold=0.1):
    """Удаляет признаки с низкой корреляцией с целевой переменной"""
    if 'SalePrice' in df.columns:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        corr_with_target = df[numeric_cols].corr()['SalePrice'].abs()
        low_corr_features = corr_with_target[corr_with_target < target_corr_threshold].index.tolist()
        if 'Id' in low_corr_features:
            low_corr_features.remove('Id')
        if 'SalePrice' in low_corr_features:
            low_corr_features.remove('SalePrice')
        df = df.drop(columns=low_corr_features, errors='ignore')
        print(f"Удалено {len(low_corr_features)} признаков с низкой корреляцией (<{target_corr_threshold})")
    return df


# ========== ПРЕОБРАЗОВАНИЯ ДЛЯ РЕГРЕССИИ ==========
def log_transform_target(y):
    """Логарифмическое преобразование целевой переменной"""
    return np.log1p(y)


def inverse_log_transform(y_log):
    """Обратное преобразование (exp) для получения исходной цены"""
    return np.expm1(y_log)


def select_features(df, target_col='SalePrice'):
    """Выделяет X и y"""
    if target_col in df.columns:
        y = df[target_col].copy()
        X = df.drop(columns=[target_col, 'Id'], errors='ignore')
        return X, y
    else:
        X = df.drop(columns=['Id'], errors='ignore')
        return X


def scale_features(X_train, X_test=None):
    """Масштабирует признаки (StandardScaler)"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler


# ========== FEATURE ENGINEERING ==========
def create_total_sf(df):
    """Общая площадь дома (надземная + подвал)"""
    df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
    return df


def create_total_bath(df):
    """Общее количество санузлов"""
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    return df


def create_age_features(df):
    """Возраст дома и время с момента ремонта"""
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['IsRemod'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    return df


def create_quality_interactions(df):
    """Комбинации признаков качества"""
    df['OverallQual_SF'] = df['OverallQual'] * (df['GrLivArea'] // 100)
    df['OverallQual_Bsm'] = df['OverallQual'] * df['TotalBsmtSF']
    return df


# ========== ОСНОВНОЙ ПАЙПЛАЙН ==========
def preprocess_pipeline(df, is_train=True):
    """
    Полный пайплайн предобработки для House Prices
    """
    df_processed = df.copy()

    # 1. Заполнение пропусков
    # Числовые признаки
    numeric_with_nulls = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
    for col in numeric_with_nulls:
        if col in df_processed.columns:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)

    # Категориальные признаки (заполняем 'None')
    cat_with_nulls = ['MSZoning', 'Alley', 'Utilities', 'Exterior1st', 'Exterior2nd',
                      'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                      'BsmtFinType2', 'Electrical', 'KitchenQual', 'FireplaceQu',
                      'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                      'PoolQC', 'Fence', 'MiscFeature', 'SaleType']
    for col in cat_with_nulls:
        if col in df_processed.columns:
            df_processed[col].fillna('None', inplace=True)

    # 2. Кодирование ВСЕХ категориальных признаков
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))

    # 3. FEATURE ENGINEERING
    df_processed = create_total_sf(df_processed)
    df_processed = create_total_bath(df_processed)
    df_processed = create_age_features(df_processed)
    df_processed = create_quality_interactions(df_processed)

    # 4. Для линейных моделей заполняем оставшиеся пропуски
    df_processed = fill_missing_numerical_for_linear(df_processed)

    # 5. ЗАПОЛНЯЕМ ВСЕ ОСТАВШИЕСЯ NaN НУЛЯМИ (для линейных моделей и KNN)
    df_processed = df_processed.fillna(0)

    # 6. Если это train, удаляем признаки с низкой корреляцией
    if is_train and 'SalePrice' in df.columns:
        df_processed = drop_low_importance_features(df_processed, target_corr_threshold=0.1)

    return df_processed


# ========== ПРОВЕРКА ==========
if __name__ == "__main__":
    from data_loader import load_train_data

    print("=" * 60)
    print("ПРОВЕРКА ПРЕДОБРАБОТКИ")
    print("=" * 60)

    df = load_train_data()
    print(f"До обработки: {df.shape}")

    df_processed = preprocess_pipeline(df, is_train=True)
    X, y = select_features(df_processed, target_col='SalePrice')

    print(f"После обработки: {df_processed.shape}")
    print(f"Признаки X: {X.shape[1]}")
    print(f"Целевая y: {len(y)}")

    # Логарифмируем целевую переменную
    y_log = log_transform_target(y)
    print(f"\nSalePrice (исходная): мин={y.min():.0f}, макс={y.max():.0f}, средняя={y.mean():.0f}")
    print(f"log(SalePrice+1): мин={y_log.min():.4f}, макс={y_log.max():.4f}")

    # Проверка на NaN
    print(f"\nNaN в X после обработки: {X.isna().sum().sum()}")
    print(f"NaN в y после обработки: {y.isna().sum()}")

    print("\n Предобработка с Feature Engineering работает!")