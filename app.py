import argparse
import os
import sys
import pickle
import numpy as np
from typing import List

# Константы для путей
RESOURCES_DIR = 'resources'
MODEL_PATH = os.path.join(RESOURCES_DIR, 'model.pkl')

def predict_salary(npy_path: str) -> List[float]:
    """
    Загружает подготовленные признаки из .npy файла, применяет обученную модель Linear Regression
    и возвращает список предсказанных зарплат в рублях.

    Предполагается, что 'x_data.npy' уже содержит трансформированные и масштабированные признаки,
    подготовленные скриптом 'parse_data.py'.

    Args:
        npy_path: Путь к файлу x_data.npy, содержащему матрицу признаков.

    Returns:
        Список предсказанных зарплат в рублях (List[float]), округленных до двух знаков после запятой.

    Raises:
        FileNotFoundError: Если файл x_data.npy или model.pkl не найдены.
        Exception: В случае других ошибок во время загрузки или предсказания.
    """
    print(f"\n--- Запуск предсказания зарплат для '{os.path.basename(npy_path)}' ---")

    # 1. Проверка наличия файла признаков
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Файл признаков '{npy_path}' не найден. Убедитесь, что он существует.")

    # 2. Проверка наличия обученной модели
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Файл модели '{MODEL_PATH}' не найден. Сначала запустите скрипт 'train_model.py'.")

    # 3. Загрузка массива признаков
    try:
        x_data_to_predict = np.load(npy_path)
    except Exception as e:
        raise Exception(f"Ошибка при загрузке .npy файла: {e}")

    # 4. Загрузка обученной модели
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        raise Exception(f"Ошибка при загрузке модели: {e}")

    # 5. Выполнение предсказания
    predictions = model.predict(x_data_to_predict)
    print(f"Предсказания выполнены для {len(predictions)} образцов.")

    # 6. Преобразование в List[float] с округлением до двух знаков
    # Принудительно приводим к типу float, как того требует задание
    formatted_predictions = [float(round(val, 2)) for val in predictions]
    return formatted_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HH Salary Predictor: Предсказание зарплат на основе файла .npy."
    )
    parser.add_argument(
        "npy_path",
        help="Путь к файлу x_data.npy, содержащему подготовленные признаки для предсказания."
    )
    args = parser.parse_args()

    # Проверка, что входной файл имеет расширение .npy
    if not args.npy_path.endswith('.npy'):
        print(f"Ошибка: Ожидается файл с расширением .npy, но получен '{args.npy_path}'.")
        sys.exit(1)

    try:
        # Вызов функции предсказания и вывод результата
        results = predict_salary(args.npy_path)
        print("\n--- Предсказанные зарплаты (List[float]) ---")
        # Выводим первые 10 значений для примера, если их много, иначе весь список
        if len(results) > 10:
            print(results[:10])
            print(f"... и еще {len(results) - 10} значений.")
        else:
            print(results)
        print("---------------------------------------------")

    except (FileNotFoundError, Exception) as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)