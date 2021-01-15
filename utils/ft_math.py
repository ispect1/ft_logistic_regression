"""Math utils"""
import numpy as np


def mean(arr: np.ndarray):
    """
    Поиск среднего числовой последовательности
    Args:
        arr: Последовательность чисел
    Returns:
        Среднее
    """
    if len(arr) == 0:
        return np.nan

    return sum(arr) / len(arr)


def var(arr: np.ndarray):
    """
    Поиск дисперсии числовой последовательности
    Args:
        arr: Последовательность чисел
    Returns:
        Дисперсия
    """
    if not len(arr):
        return np.nan
    squared_diff = (arr - mean(arr)) ** 2
    return sum(squared_diff) / len(squared_diff)


def std(arr: np.ndarray):
    """
    Поиск стандартного отклонения числовой последовательности
    Args:
        arr: Последовательность чисел
    Returns:
        Дисперсия
    """
    return var(arr) ** 0.5
