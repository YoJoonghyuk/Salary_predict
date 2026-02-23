import re

import numpy as np

def extract_age(text: str) -> float:
    """
    Извлекает возраст в годах из текстовой строки.
    Парсит первое найденное число, за которым следуют слова 'год', 'года' или 'лет'.
    Возвращает np.nan, если возраст не найден.

    Args:
        text: Входная строка (например, 'Мужчина, 42 года').

    Returns:
        Возраст в годах (float) или np.nan.
    """
    if not isinstance(text, str):
        return np.nan
    match = re.search(r'(\d+)\s+(?:год|года|лет)', text, re.IGNORECASE)
    return float(match.group(1)) if match else np.nan