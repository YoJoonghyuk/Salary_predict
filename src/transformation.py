import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from .base import Handler
from utils.parsers import extract_age, extract_experience, extract_salary, extract_city
from utils.helpers import find_column_name

class FeatureExtractionHandler(Handler):
    """
    Обработчик, который извлекает и трансформирует сырые данные из DataFrame
    в числовые признаки (X) и целевую переменную (y).
    Включает парсинг пола/возраста, опыта работы, города, должности и зарплаты.
    Применяет TF-IDF для текстовых признаков и StandardScaler для числовых,
    сохраняя обученные трансформаторы для дальнейшего использования.
    Обрабатывает пропущенные значения, удаляя строки с NaN.
    """
    RESOURCES_DIR = "resources"
    VECTORIZER_PATH = os.path.join(RESOURCES_DIR, "vectorizer.pkl")
    SCALER_PATH = os.path.join(RESOURCES_DIR, "scaler.pkl")

    def __init__(self):
        """
        Инициализирует FeatureExtractionHandler и создает директорию для ресурсов,
        если она не существует.
        """
        os.makedirs(self.RESOURCES_DIR, exist_ok=True)

    def _load_or_fit_vectorizer(self, text_data: pd.Series) -> np.ndarray:
        """
        Загружает или обучает TF-IDF векторизатор для текстовых данных.
        """
        if os.path.exists(self.VECTORIZER_PATH):
            try:
                vectorizer = joblib.load(self.VECTORIZER_PATH)
                print(f"[{self.__class__.__name__}] TF-IDF векторизатор загружен из {self.VECTORIZER_PATH}.")
            except Exception as e:
                print(f"[{self.__class__.__name__}] Ошибка при загрузке векторизатора: {e}. Обучаем заново.")
                vectorizer = TfidfVectorizer(max_features=500, min_df=2)
                vectorizer.fit(text_data)
                joblib.dump(vectorizer, self.VECTORIZER_PATH)
                print(f"[{self.__class__.__name__}] TF-IDF векторизатор обучен и сохранен.")
        else:
            print(f"[{self.__class__.__name__}] Обучение TF-IDF векторизатора для текстовых признаков...")
            vectorizer = TfidfVectorizer(max_features=500, min_df=2)
            vectorizer.fit(text_data)
            joblib.dump(vectorizer, self.VECTORIZER_PATH)
            print(f"[{self.__class__.__name__}] TF-IDF векторизатор обучен и сохранен в {self.VECTORIZER_PATH}.")

        return vectorizer.transform(text_data).toarray()

    def _load_or_fit_scaler(self, numeric_data: pd.DataFrame) -> np.ndarray:
        """
        Загружает или обучает StandardScaler для числовых данных.
        """
        if os.path.exists(self.SCALER_PATH):
            try:
                scaler = joblib.load(self.SCALER_PATH)
                print(f"[{self.__class__.__name__}] StandardScaler загружен из {self.SCALER_PATH}.")
            except Exception as e:
                print(f"[{self.__class__.__name__}] Ошибка при загрузке скалера: {e}. Обучаем заново.")
                scaler = StandardScaler()
                scaler.fit(numeric_data)
                joblib.dump(scaler, self.SCALER_PATH)
                print(f"[{self.__class__.__name__}] StandardScaler обучен и сохранен.")
        else:
            print(f"[{self.__class__.__name__}] Обучение StandardScaler для числовых признаков...")
            scaler = StandardScaler()
            scaler.fit(numeric_data)
            joblib.dump(scaler, self.SCALER_PATH)
            print(f"[{self.__class__.__name__}] StandardScaler обучен и сохранен в {self.SCALER_PATH}.")

        return scaler.transform(numeric_data)

    def handle(self, df: pd.DataFrame) -> dict:
        """
        Выполняет извлечение, трансформацию и масштабирование признаков,
        формируя матрицу признаков (X) и вектор целевой переменной (y).
        Удаляет строки с пропущенными значениями после парсинга.

        Args:
            df: pandas.DataFrame, содержащий исходные данные резюме.

        Returns:
            Словарь, содержащий:
            'x': numpy.ndarray - матрица признаков.
            'y': numpy.ndarray - вектор целевой переменной.
            Все массивы приведены к типу float32.

        Raises:
            KeyError: Если одна из необходимых колонок не найдена.
            ValueError: Если после обработки данных не осталось ни одной строки.
        """
        try:
            # Определение колонок по ключевым словам (A, C, D, E, H)
            col_personal = find_column_name(df, 'Пол, возраст')
            col_salary = find_column_name(df, 'ЗП')  # Столбец C
            col_position = find_column_name(df, 'Ищет работу')  # Столбец D
            col_city = find_column_name(df, 'Город')  # Столбец E
            col_exp = find_column_name(df, 'Опыт')  # Столбец H
        except KeyError as e:
            raise KeyError(f"[{self.__class__.__name__}] Ошибка: Необходимая колонка не найдена. {e}")

        # Извлечение данных с помощью парсеров
        salary = df[col_salary].apply(extract_salary)
        age = df[col_personal].apply(extract_age)
        experience = df[col_exp].apply(extract_experience)
        is_male = df[col_personal].apply(lambda x: 1 if 'Мужчина' in str(x) else 0)
        city = df[col_city].apply(extract_city)
        position = df[col_position].fillna('не указана').astype(str)

        # Формируем DataFrame для удобной очистки
        processed_data = pd.DataFrame({
            'salary': salary, 'age': age, 'experience': experience,
            'is_male': is_male, 'city': city, 'position': position
        })

        # Удаляем строки с NaN в числовых признаках или зарплате
        initial_rows = processed_data.shape[0]
        processed_data.dropna(inplace=True)
        rows_after_drop = processed_data.shape[0]

        if initial_rows > rows_after_drop:
            print(
                f"[{self.__class__.__name__}] Удалено {initial_rows - rows_after_drop} строк из-за пропущенных значений.")

        if processed_data.shape[0] == 0:
            raise ValueError(f"[{self.__class__.__name__}] После обработки данных не осталось ни одной строки.")

        # 1. Текстовые признаки: Должность и Город (TF-IDF)
        # Объединяем город и должность в одну строку для анализа контекста
        text_for_vectorizer = processed_data['position'] + " " + processed_data['city']
        text_features = self._load_or_fit_vectorizer(text_for_vectorizer)
        print(f"[{self.__class__.__name__}] TF-IDF признаков: {text_features.shape[1]}")

        # 2. Масштабирование числовых признаков (age, experience)
        numeric_features_to_scale = processed_data[['age', 'experience']]
        scaled_numeric_features = self._load_or_fit_scaler(numeric_features_to_scale)
        print(f"[{self.__class__.__name__}] Числовые признаки масштабированы.")

        # Пол (is_male) - бинарный, не требует масштабирования
        is_male_feature = processed_data[['is_male']].values

        # Объединение всех признаков: [пол, возраст_scaled, опыт_scaled, tf-idf_признаки...]
        x_data = np.hstack([is_male_feature, scaled_numeric_features, text_features]).astype(np.float32)
        y_data = processed_data['salary'].values.astype(np.float32)

        print(
            f"[{self.__class__.__name__}] Сформированы финальные данные: X ({x_data.shape[0]}, {x_data.shape[1]} признаков), y ({y_data.shape[0]} образцов).")

        return super().handle({'x': x_data, 'y': y_data})