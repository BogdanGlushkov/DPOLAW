import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self):
        pass

    def load_data(self, data_path=None, num_samples=1000, num_features=10):
        """
        Загрузка данных из файла или генерация случайных данных.
        :param data_path: Путь к файлу с данными. Если None, то генерируются случайные данные.
        :param num_samples: Число образцов в данных. Используется при генерации данных.
        :param num_features: Число признаков в данных. Используется при генерации данных.
        :return: DataFrame с данными и целевой переменной.
        """
        if data_path is not None:
            # Загрузка данных из файла
            data = pd.read_csv(data_path)
            target = data.pop('target')
        else:
            # Генерация случайных данных
            data = pd.DataFrame(np.random.rand(num_samples, num_features))
            target = pd.Series(np.random.rand(num_samples))

            data.to_csv('generated_data.csv', index=False)
            target.to_csv('Data.csv', index=False)

        return data, target


