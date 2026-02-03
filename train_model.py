import os
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import sys

# Константы для путей
DATA_DIR = 'data'
RESOURCES_DIR = 'resources'
X_DATA_PATH = os.path.join(DATA_DIR, 'x_data.npy')
Y_DATA_PATH = os.path.join(DATA_DIR, 'y_data.npy')
MODEL_PATH = os.path.join(RESOURCES_DIR, 'model.pkl')

def train_model_pipeline() -> None:
    """
    Запускает пайплайн обучения модели Linear Regression на данных из .npy файлов.
    Оценивает модель на тестовой выборке и сохраняет её в 'resources/model.pkl'.

    Этот скрипт загружает 'x_data.npy' (признаки) и 'y_data.npy' (целевые значения),
    очищает данные от экстремальных выбросов, обучает модель Linear Regression,
    оценивает её с помощью метрик R2 и MAE, а затем сохраняет обученную модель.
    """
    print("\n--- Запуск пайплайна обучения модели Linear Regression ---")

    # Проверка наличия подготовленных данных
    if not os.path.exists(X_DATA_PATH) or not os.path.exists(Y_DATA_PATH):
        print(f"Ошибка: Файлы '{X_DATA_PATH}' или '{Y_DATA_PATH}' не найдены.")
        print("Пожалуйста, сначала запустите скрипт 'parse_data.py', чтобы создать эти файлы.")
        sys.exit(1)

    try:
        X = np.load(X_DATA_PATH)
        y = np.load(Y_DATA_PATH)
        print(
            f"Загружены данные для обучения: X ({X.shape[0]} образцов, {X.shape[1]} признаков), y ({y.shape[0]} образцов).")
    except Exception as e:
        print(f"Ошибка при загрузке .npy файлов для обучения: {e}")
        sys.exit(1)

    # Очистка от экстремальных выбросов для стабильности линейной модели
    # Зарплаты от 10k до 800k руб.
    initial_samples = X.shape[0]
    mask = (y > 10000) & (y < 800000)
    X, y = X[mask], y[mask]
    if initial_samples > X.shape[0]:
        print(f"Удалено {initial_samples - X.shape[0]} выбросов из данных для обучения.")

    if X.shape[0] == 0:
        print("Ошибка: Нет данных для обучения после очистки выбросов.")
        sys.exit(1)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Данные разделены: {X_train.shape[0]} для обучения, {X_test.shape[0]} для тестирования.")

    # Обучение модели Linear Regression
    print("Обучение модели Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Модель Linear Regression обучена.")

    # Оценка модели на тестовой выборке
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n--- Метрики качества модели (Linear Regression) ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:,.2f} руб.")
    print("--------------------------------------------------")

    # Сохранение обученной модели
    os.makedirs(RESOURCES_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Модель успешно сохранена в '{MODEL_PATH}'.")

if __name__ == "__main__":
    train_model_pipeline()