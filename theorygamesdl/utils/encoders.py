"""
Вспомогательные классы для сериализации данных
"""

import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    """
    Кастомный JSON энкодер для работы с numpy типами
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)