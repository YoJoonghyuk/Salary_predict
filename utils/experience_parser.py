import re

import numpy as np

def extract_experience(text: str) -> float:
    """
    Извлекает опыт работы из текстовой строки и конвертирует его в общее количество месяцев.
    Парсит числа перед словами 'год/лет/года' и 'месяц/месяца/месяцев'.
    Возвращает np.nan, если опыт не указан.

    Args:
        text: Входная строка (например, '5 лет 2 месяца').

    Returns:
        Общее количество месяцев опыта (float) или np.nan.
    """
    s = str(text).lower()
    if 'не указано' in s or s == 'nan' or not isinstance(text, str):
        return np.nan

    years = re.search(r'(\d+)\s+(?:год|года|лет)', s)
    months = re.search(r'(\d+)\s+(?:месяц|месяца|месяцев)', s)

    y = int(years.group(1)) if years else 0
    m = int(months.group(1)) if months else 0

    if y == 0 and m == 0 and not (years or months):
        return np.nan

    if y > 100:  # Эвристика против ошибочных дат
        y = 0

    return float(y * 12 + m)