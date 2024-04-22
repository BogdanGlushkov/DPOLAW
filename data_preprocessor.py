import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        pass

    def preprocess_data(self, data, target):
        """
        Предварительная обработка данных.
        :param data: DataFrame с данными.
        :param target: Целевая переменная.
        :return: Обработанные данные и целевая переменная.
        """
        # Обработка пропусков
        data = self.handle_missing_values(data)

        # Нормализация данных
        data = self.normalize_data(data)

        # Кодирование категориальных признаков
        data = self.encode_categorical_features(data)

        # Разделение данных на обучающую и тестовую выборки
        train_data, test_data, train_target, test_target = self.split_data(data, target)

        return train_data, test_data, train_target, test_target

    def handle_missing_values(self, data):
        """
        Обработка пропусков в данных.
        :param data: DataFrame с данными.
        :return: DataFrame с обработанными данными.
        """
        # Замена пропусков средним значением
        data = data.fillna(data.mean())

        return data

    def normalize_data(self, data):
        """
        Нормализация данных.
        :param data: DataFrame с данными.
        :return: DataFrame с нормализованными данными.
        """
        # Вычисление среднего значения и стандартного отклонения для каждого признака
        means = data.mean()
        stds = data.std()

        # Нормализация данных с помощью формулы (x - mean) / std
        data = (data - means) / stds

        return data

    def encode_categorical_features(self, data):
        """
        Кодирование категориальных признаков.
        :param data: DataFrame с данными.
        :return: DataFrame с обработанными данными.
        """
        # Определение категориальных признаков
        categorical_features = data.select_dtypes(include=['object', 'category']).columns

        # Кодирование категориальных признаков с помощью one-hot encoding
        data = pd.get_dummies(data, columns=categorical_features)

        return data

    def split_data(self, data, target):
        """
        Разделение данных на обучающую и тестовую выборки.
        :param data: DataFrame с данными.
        :param target: Целевая переменная.
        :return: Обучающая и тестовая выборки данных и целевой переменной.
        """
        # Разделение данных на обучающую и тестовую выборки в соотношении 80/20
        train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

        return train_data, test_data, train_target, test_target
