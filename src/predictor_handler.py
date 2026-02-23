import os
import pickle

import numpy as np

from .base import Handler

class PredictorHandler(Handler):
    """
    Обработчик, использующий обученную модель для предсказания зарплат.
    """
    def __init__(self, model_path: str = 'resources/model.pkl'):
        """
        Инициализирует PredictorHandler с путем к файлу модели.

        Args:
            model_path: Путь к файлу обученной модели (model.pkl).
        """
        self.model_path = model_path

    def handle(self, x_data: np.ndarray) -> list[float]:
        """
        Загружает модель, выполняет предсказания и возвращает их в виде списка float.

        Args:
            x_data: Матрица признаков для предсказания (numpy.ndarray).

        Returns:
            Список предсказанных зарплат (List[float]), округленных до двух знаков.

        Raises:
            FileNotFoundError: Если файл модели не найден.
            RuntimeError: В случае ошибок при загрузке модели или предсказании.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Ошибка: Файл модели '{self.model_path}' не найден. "
                                    f"Убедитесь, что он обучен и находится в '{os.path.dirname(self.model_path)}'.")

        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            raise RuntimeError(f"Ошибка при загрузке модели '{self.model_path}': {e}")
        except Exception as e:
            raise RuntimeError(f"Неизвестная ошибка при загрузке модели: {e}")

        try:
            predictions = model.predict(x_data)
            # Округляем до двух знаков и приводим к List[float]
            formatted_predictions = [float(round(val, 2)) for val in predictions]
            print(f"[{self.__class__.__name__}] Предсказания выполнены для {len(formatted_predictions)} образцов.")
            return super().handle(formatted_predictions)
        except Exception as e:
            raise RuntimeError(f"Ошибка при выполнении предсказаний: {e}")