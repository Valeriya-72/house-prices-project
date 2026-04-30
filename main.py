# Главный скрипт для воспроизведения всего пайплайна House Prices

import warnings

warnings.filterwarnings('ignore')


def run_eda():
    print("=" * 60)
    print("1. ЗАПУСК EDA (анализ данных)")
    print("=" * 60)
    import eda


def run_preprocessing():
    print("\n" + "=" * 60)
    print("2. ЗАПУСК ПРЕДОБРАБОТКИ И FEATURE ENGINEERING")
    print("=" * 60)
    import preprocess


def run_ml_models():
    print("\n" + "=" * 60)
    print("3. ЗАПУСК ОБУЧЕНИЯ ML-МОДЕЛЕЙ")
    print("=" * 60)
    import train_ml


def run_dnn():
    print("\n" + "=" * 60)
    print("4. ЗАПУСК ОБУЧЕНИЯ DNN (PyTorch)")
    print("=" * 60)
    import train_dnn_advanced


def run_ensembles():
    print("\n" + "=" * 60)
    print("5. ЗАПУСК АНСАМБЛЕЙ")
    print("=" * 60)
    import ensemble


def compare_validation():
    print("\n" + "=" * 60)
    print("6. СРАВНЕНИЕ 1 И 5 ФОЛДОВ")
    print("=" * 60)
    import compare_validation


if __name__ == "__main__":
    print("=" * 60)
    print("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА ПРОЕКТА HOUSE PRICES")
    print("=" * 60)

    run_eda()
    run_preprocessing()
    run_ml_models()
    run_dnn()
    run_ensembles()
    compare_validation()

    print("\n" + "=" * 60)
    print(" ПРОЕКТ УСПЕШНО ЗАВЕРШЁН!")
    print("=" * 60)