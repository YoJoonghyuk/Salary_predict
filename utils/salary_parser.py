import re

import numpy as np

def extract_salary(text: str) -> float:
    """
    Извлекает числовое значение зарплаты и конвертирует в рубли по фиксированным курсам.
    Поддерживает RUB, USD, EUR, KZT, UAH.
    Возвращает np.nan, если зарплата не может быть извлечена.

    Args:
        text: Входная строка (например, '200 USD').

    Returns:
        Числовое значение зарплаты в рублях (float) или np.nan.
    """
    if not isinstance(text, str):
        return np.nan

    num_match = re.sub(r'[^\d]', '', text)
    if not num_match:
        return np.nan
    amount = float(num_match)

    text_upper = text.upper()
    if 'USD' in text_upper:
        amount *= 90.0
    elif 'EUR' in text_upper:
        amount *= 98.0
    elif 'KZT' in text_upper:
        amount *= 0.20
    elif 'ГРН' in text_upper or 'UAH' in text_upper:
        amount *= 2.5

    return amount