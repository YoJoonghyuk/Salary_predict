import numpy as np

def remove_salary_outliers(x: np.ndarray, y: np.ndarray,
                           min_sal: int = 10000,
                           max_sal: int = 800000) -> tuple:
    """
    Удаляет строки, где зарплата выходит за пределы [min_sal, max_sal].
    """
    mask = (y > min_sal) & (y < max_sal)
    initial_count = y.shape[0]
    x_clean, y_clean = x[mask], y[mask]

    print(f"[Cleaner] Удалено выбросов: {initial_count - y_clean.shape[0]}")
    return x_clean, y_clean