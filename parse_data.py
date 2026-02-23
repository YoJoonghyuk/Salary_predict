import argparse
import os
import sys

from src.loaders import DataLoaderHandler
from src.output import NpySaveHandler
from src.transformation import FeatureExtractionHandler

def run_parse(csv_path: str):
    """Запускает цепочку: Загрузка -> Трансформация -> Сохранение."""
    if not os.path.exists(csv_path):
        print(f"Файл {csv_path} не найден")
        sys.exit(1)

    # Инициализация хендлеров
    loader = DataLoaderHandler()
    transformer = FeatureExtractionHandler()
    saver = NpySaveHandler(output_dir='data')

    # Сборка цепи
    loader.set_next(transformer).set_next(saver)

    try:
        loader.handle(csv_path)
        print("Парсинг успешно завершен.")
    except Exception as e:
        print(f"Критическая ошибка парсинга: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HH Data Parser")
    parser.add_argument("csv_path", help="Путь к hh.csv")
    run_parse(parser.parse_args().csv_path)