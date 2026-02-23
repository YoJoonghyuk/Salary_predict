import os

import numpy as np

from .base import Handler

class NpyLoader(Handler):
    """
    Обработчик, отвечающий за загрузку данных из файла формата .npy.
    """
    def handle(self, path: str) -> np.ndarray:
        """
        Загружает массив NumPy из указанного файла.

        Args:
            path: Путь к файлу .npy.

        Returns:
            Загруженный numpy.ndarray.

        Raises:
            FileNotFoundError: Если файл по указанному пути не найден.
            RuntimeError: В случае других ошибок при загрузке.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ошибка: Файл '{path}' не найден.")

        try:
            data = np.load(path)
            print(f"[{self.__class__.__name__}] Файл '{os.path.basename(path)}' загружен.")
            return super().handle(data)
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке .npy файла '{path}': {e}")