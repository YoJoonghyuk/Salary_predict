import os
import pickle
import sys

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from utils.outlier_remover import remove_salary_outliers

def train_pipeline(data_dir='data', res_dir='resources'):
    """Загружает данные, чистит, обучает модель и сохраняет веса."""
    try:
        x = np.load(os.path.join(data_dir, 'x_data.npy'))
        y = np.load(os.path.join(data_dir, 'y_data.npy'))

        # Очистка и разбиение
        x, y = remove_salary_outliers(x, y)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        model = LinearRegression().fit(x_train, y_train)

        # Оценка
        score = r2_score(y_test, model.predict(x_test))
        print(f"[Trainer] Модель обучена. R2 Score: {score:.4f}")

        # Сохранение
        os.makedirs(res_dir, exist_ok=True)
        with open(os.path.join(res_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)

    except Exception as e:
        print(f"Ошибка при обучении: {e}")
        sys.exit(1)


if __name__ == "__main__":
    train_pipeline()