"""MyTable class"""
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from collections import defaultdict
from utils.ft_math import mean, var


class HogwartsFrame(pd.DataFrame):
    @staticmethod
    def read_csv(filename: str, *args, **kwargs):
        return HogwartsFrame(pd.read_csv(filename, *args, **kwargs))

    def count(self, column):
        return len(self[column].dropna())

    def mean(self, column):
        """
        Среднее значение колонок
        Returns:
            Словарь столбец/строка: посчитанное значение по каждой колонке
        """
        return mean(self[column].dropna())

    def var(self, column):
        """
        Дисперсия объектов колонок
        """
        return var(self[column].dropna())

    def std(self, column):
        """
        Стандартное отклонение колонок
        Returns:
            Словарь столбец/строка: посчитанное значение по каждой колонке
        """
        return self.var(column) ** 0.5

    def min(self, column):
        """
        Минимальное значение колонок
        """
        min_value = np.inf
        for value in self[column].dropna():
            if value < min_value:
                min_value = value
        return min_value

    def max(self, column):
        """
        Максимальное значение колонок
        """
        max_value = - np.inf
        for value in self[column].dropna():
            if value > max_value:
                max_value = value
        return max_value

    def quantile(self, column, qntl=0.5):
        """
        `q`-ый квантиль колонок
        Args:
            qntl - квантиль
        """
        items = sorted(self[column].dropna())
        if not items:
            return np.nan
        k = (len(items) - 1) * qntl
        f = np.floor(k)
        c = np.ceil(k)
        if f == c:
            return items[int(k)]
        return items[int(f)] * (c - k) + items[int(c)] * (k - f)

    def is_number(self, key):
        return np.issubdtype(self[key], np.number) and self[key].notna().sum()

    @property
    def number_columns(self):
        return [column for column in self.columns if self.is_number(column)]

    def describe(self):
        """
            Получить словарь основных статистик таблицы
        Returns:
            Словарь статистик§
        """
        result = defaultdict(dict)
        for column in self.number_columns:
            result['count'][column] = self.count(column)
            result['mean'][column] = self.mean(column)
            result['std'][column] = self.std(column)
            result['min'][column] = self.min(column)
            result['max'][column] = self.max(column)
            result['25%'][column] = self.quantile(column, 0.25)
            result['50%'][column] = self.quantile(column, 0.5)
            result['75%'][column] = self.quantile(column, 0.75)

        return HogwartsFrame(result)
