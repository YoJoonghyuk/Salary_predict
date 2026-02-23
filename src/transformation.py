import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from utils.age_parser import extract_age
from utils.city_parser import extract_city
from utils.experience_parser import extract_experience
from utils.helpers import find_column_name
from utils.salary_parser import extract_salary
from utils.transformer_utils import load_or_fit_transformer
from .base import Handler

class FeatureExtractionHandler(Handler):
    """Извлекает признаки и применяет предобученные веса (TF-IDF, Scaler)."""
    RES_DIR = "resources"

    def handle(self, df: pd.DataFrame) -> dict:
        # Парсинг колонок
        data = pd.DataFrame({
            'y': df[find_column_name(df, 'ЗП')].apply(extract_salary),
            'age': df[find_column_name(df, 'Пол, возраст')].apply(extract_age),
            'exp': df[find_column_name(df, 'Опыт')].apply(extract_experience),
            'male': df[find_column_name(df, 'Пол, возраст')].apply(
                lambda x: 1 if 'Мужчина' in str(x) else 0),
            'city': df[find_column_name(df, 'Город')].apply(extract_city),
            'pos': df[find_column_name(df, 'Ищет работу')].fillna('none')
        }).dropna() # Убрали inplace=True (Критическое требование Sonar!)

        # Векторизация текста (Должность + Город)
        v_path = os.path.join(self.RES_DIR, "vectorizer.pkl")
        vec = load_or_fit_transformer(
            TfidfVectorizer(max_features=500), v_path, data['pos'] + " " + data['city']
        )
        text_feats = vec.transform(data['pos'] + " " + data['city']).toarray()

        # Масштабирование чисел
        s_path = os.path.join(self.RES_DIR, "scaler.pkl")
        scaler = load_or_fit_transformer(
            StandardScaler(), s_path, data[['age', 'exp']]
        )
        num_feats = scaler.transform(data[['age', 'exp']])

        # Сборка финальной матрицы
        x_data = np.hstack([data[['male']].values, num_feats, text_feats])
        return super().handle({'x': x_data.astype(np.float32),
                              'y': data['y'].values.astype(np.float32)})