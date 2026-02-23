import os

import joblib

def load_or_fit_transformer(transformer, path, data):
    """
    Загружает трансформер из файла или обучает новый, если файл не найден.
    """
    if os.path.exists(path):
        return joblib.load(path)

    transformer.fit(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(transformer, path)
    return transformer