import numpy as np
import pickle
import os
from .base import Handler

class PredictorHandler(Handler):
    def handle(self, x_data: np.ndarray):
        with open('resources/model.pkl', 'rb') as f:
            model = pickle.load(f)
        preds = model.predict(x_data)
        return super().handle(preds.tolist())

class NpyLoader(Handler):
    def handle(self, path: str):
        data = np.load(path)
        return super().handle(data)