# Разведочный анализ данных (EDA) для House Prices

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import config
from data_loader import load_train_data

# Настройка стиля графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Загружаем данные
df = load_train_data()

# ========== 1. ОБЩАЯ ИНФОРМАЦИЯ ==========
print("=" * 60)
print("1. ИНФОРМАЦИЯ О ДАННЫХ")
print("=" * 60)
print(df.info())

# ========== 2. СТАТИСТИКИ ЧИСЛОВЫХ ПРИЗНАКОВ ==========
print("\n" + "=" * 60)
print("2. СТАТИСТИКИ ЧИСЛОВЫХ ПРИЗНАКОВ")
print("=" * 60)
print(df.describe())

# ========== 3. ПРОПУСКИ ==========
print("\n" + "=" * 60)
print("3. ПРОПУСКИ В ДАННЫХ")
print("=" * 60)
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)
print(f"\nВсего признаков с пропусками: {len(missing)}")

# ========== 4. РАСПРЕДЕЛЕНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ==========
print("\n" + "=" * 60)
print("4. АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (SalePrice)")
print("=" * 60)
print(f"Минимальная цена: ${df['SalePrice'].min():,.0f}")
print(f"Максимальная цена: ${df['SalePrice'].max():,.0f}")
print(f"Средняя цена: ${df['SalePrice'].mean():,.0f}")
print(f"Медианная цена: ${df['SalePrice'].median():,.0f}")

# График распределения SalePrice
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['SalePrice'], bins=50, kde=True)
plt.title('Распределение SalePrice (исходное)')
plt.xlabel('SalePrice')

plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df['SalePrice']), bins=50, kde=True)
plt.title('Распределение log(SalePrice+1)')
plt.xlabel('log(SalePrice+1)')

plt.tight_layout()
plt.savefig(f"{config.EDA_OUTPUT_DIR}/sale_price_distribution.png")
plt.close()
print(f"\n График сохранён: {config.EDA_OUTPUT_DIR}/sale_price_distribution.png")

# ========== 5. КОРРЕЛЯЦИЯ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ==========
print("\n" + "=" * 60)
print("5. КОРРЕЛЯЦИЯ ПРИЗНАКОВ С SalePrice")
print("=" * 60)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr_with_target = df[numeric_cols].corr()['SalePrice'].sort_values(ascending=False)
print("Топ-10 признаков по корреляции с ценой:")
print(corr_with_target.head(11))

# ========== 6. ГРАФИКИ ТОП-4 ПРИЗНАКОВ ==========
print("\n" + "=" * 60)
print("6. ГРАФИКИ ТОПОВЫХ ПРИЗНАКОВ")
print("=" * 60)

top_features = corr_with_target.index[1:5]  # топ-4 признака после SalePrice
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, feature in enumerate(top_features):
    axes[i].scatter(df[feature], df['SalePrice'], alpha=0.5)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('SalePrice')
    axes[i].set_title(f'Зависимость цены от {feature}')

plt.tight_layout()
plt.savefig(f"{config.EDA_OUTPUT_DIR}/top_features_scatter.png")
plt.close()
print(f" График сохранён: {config.EDA_OUTPUT_DIR}/top_features_scatter.png")

# ========== 7. ТЕПЛОВАЯ КАРТА КОРРЕЛЯЦИИ ==========
print("\n" + "=" * 60)
print("7. ТЕПЛОВАЯ КАРТА КОРРЕЛЯЦИИ (топ-15 признаков)")
print("=" * 60)

top_corr_features = corr_with_target.head(16).index.tolist()
top_corr_data = df[top_corr_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(top_corr_data, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Тепловая карта корреляции топ-15 признаков')
plt.tight_layout()
plt.savefig(f"{config.EDA_OUTPUT_DIR}/correlation_heatmap.png")
plt.close()
print(f" График сохранён: {config.EDA_OUTPUT_DIR}/correlation_heatmap.png")

# ========== 8. ВЫВОДЫ ==========
print("\n" + "=" * 60)
print("КЛЮЧЕВЫЕ ВЫВОДЫ EDA")
print("=" * 60)
print("1. Целевая переменная SalePrice имеет правостороннюю асимметрию")
print("2. Логарифмическое преобразование приближает распределение к нормальному")
print("3. Топ-5 признаков по корреляции с ценой:")
for i, feature in enumerate(corr_with_target.head(6).index[1:6]):
    print(f"   - {feature}: {corr_with_target[feature]:.3f}")
print("4. Много пропусков в признаках: PoolArea, MiscFeature, Alley, Fence, FireplaceQu...")
print("5. Для регрессии (RMSE) логарифмирование целевой переменной улучшит качество")

# ========== 9. АНАЛИЗ ВЫБРОСОВ (IQR) ==========
print("\n" + "=" * 60)
print("9. АНАЛИЗ ВЫБРОСОВ ПО МЕТОДУ IQR")
print("=" * 60)

# Выбираем числовые признаки
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
outliers_summary = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outliers_summary[col] = len(outliers)

# Топ-10 признаков с наибольшим количеством выбросов
top_outliers = sorted(outliers_summary.items(), key=lambda x: x[1], reverse=True)[:10]
print("Топ-10 признаков с наибольшим количеством выбросов:")
for col, count in top_outliers:
    if count > 0:
        print(f"  {col}: {count} выбросов ({count/len(df)*100:.1f}%)")

print("\n EDA ЗАВЕРШЁН!")