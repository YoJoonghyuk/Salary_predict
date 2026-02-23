import argparse
import os
import sys
from typing import List

from src.npy_loader import NpyLoader
from src.predictor_handler import PredictorHandler

def run_prediction():
    """
    Функция запуска пайплайна предсказания зарплат.
    Парсит аргументы командной строки, конфигурирует цепочку обработчиков (NpyLoader -> PredictorHandler),
    и запускает процесс предсказания.
    """
    parser = argparse.ArgumentParser(
        description="HH Salary Predictor: Предсказание зарплат на основе файла .npy."
    )
    parser.add_argument(
        "npy_path",
        help="Путь к файлу x_data.npy, содержащему подготовленные признаки для предсказания."
    )
    args = parser.parse_args()

    # Предварительные проверки (хотя хендлеры тоже проверяют, это хорошая практика для top-level скрипта)
    if not os.path.exists(args.npy_path):
        print(f"Ошибка: Файл признаков '{args.npy_path}' не найден.")
        sys.exit(1)
    if not args.npy_path.endswith('.npy'):
        print(f"Ошибка: Ожидается файл с расширением .npy, но получен '{args.npy_path}'.")
        sys.exit(1)

    # Создание цепочки обработчиков
    npy_loader = NpyLoader()
    predictor = PredictorHandler() # Использует путь по умолчанию 'resources/model.pkl'
    npy_loader.set_next(predictor)

    try:
        # Запуск пайплайна и получение результатов
        results: List[float] = npy_loader.handle(args.npy_path)

        print("\n--- Предсказанные зарплаты (List[float]) ---")
        if len(results) > 10:
            print(results[:10])
            print(f"... и еще {len(results) - 10} значений.")
        else:
            print(results)
        print("---------------------------------------------")

    except (FileNotFoundError, RuntimeError) as e: # Ловим конкретные ошибки от хендлеров
        print(f"Критическая ошибка: {e}")
        sys.exit(1)
    except Exception as e: # Для любых других непредвиденных ошибок
        print(f"Произошла неизвестная критическая ошибка в пайплайне: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_prediction()