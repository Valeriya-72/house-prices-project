# Настройки проекта House Prices

# ===== ПУТИ К ФАЙЛАМ =====
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SAMPLE_SUBMISSION_PATH = "data/sample_submission.csv"
SUBMISSIONS_DIR = "submissions"
EDA_OUTPUT_DIR = "eda_output"
MODELS_DIR = "models"

# ===== НАСТРОЙКИ ВОСПРОИЗВОДИМОСТИ =====
RANDOM_SEED = 42

# ===== ВАЛИДАЦИЯ =====
N_FOLDS = 5
VAL_SIZE = 0.2

# ===== ОСОБЕННОСТИ РЕГРЕССИИ =====
TARGET_COLUMN = "SalePrice"      # целевая переменная (цена дома)
METRIC = "RMSE"                   # метрика качества

# ===== ПАРАМЕТРЫ МОДЕЛЕЙ =====
KNN_NEIGHBORS = [3, 5, 7, 9, 11]

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "random_state": RANDOM_SEED
}

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": RANDOM_SEED
}

LGBM_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "num_leaves": 31,
    "random_state": RANDOM_SEED,
    "verbose": -1
}

CATBOOST_PARAMS = {
    "iterations": 200,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": RANDOM_SEED,
    "verbose": False
}

# ===== DNN (PyTorch) =====
DNN_PARAMS = {
    "input_size": None,           # определится после предобработки
    "hidden_sizes": [128, 64, 32],
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
}