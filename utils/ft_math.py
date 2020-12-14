"""Math utils"""
from typing import Sequence
from utils.ft_typing import Comparable


def mean(arr: Sequence[float]):
    """
    Поиск среднего числовой последовательности
    Args:
        arr: Последовательность чисел
    Returns:
        Среднее
    """
    if len(arr) == 0:
        return float('nan')
    return sum(arr) / len(arr)


def var(arr: Sequence[float]):
    """
    Поиск дисперсии числовой последовательности
    Args:
        arr: Последовательность чисел
    Returns:
        Дисперсия
    """
    if len(arr) == 0:
        return float('nan')
    arr_mean = mean(arr)
    try:
        return 1 / ((len(arr) - 1) or 1) * sum((curr - arr_mean) ** 2
                                               for curr in arr)
    except ZeroDivisionError:
        return float('nan')


def std(arr: Sequence[float]):
    """
    Поиск стандартного отклонения числовой последовательности
    Args:
        arr: Последовательность чисел
    Returns:
        Стандартное оклонение
    """
    return var(arr) ** 0.5


def mmax(arr: Sequence[Comparable]):
    """
    Поиск наибольшего объекта в последовательности
    Args:
        arr: Последовательность сравниваемых друг с другом объектов
    Returns:
        Максимум
    """
    if len(arr) == 0:
        return float('nan')
    max_object = arr[0]
    for num in arr:
        if num > max_object:
            max_object = num
    return max_object


def mmin(arr: Sequence[Comparable]):
    """
    Поиск наименьшего объекта в последовательности
    Args:
        arr: Последовательность сравниваемых друг с другом объектов
    Returns:
        Минимум
    """
    if len(arr) == 0:
        return float('nan')
    min_object = arr[0]
    for num in arr:
        if num < min_object:
            min_object = num
    return min_object


def quantile(arr: Sequence[float], qntl=0.5) -> float:
    """
    Найти квантиль q-ого порядка
    Args:
        arr: Последовательность чисел
        qntl: порядок квантиля, по умолчанию `q=0.5`

    Returns:
        Посчитанный квантиль
    """
    if qntl < 0 or qntl > 1:
        raise ValueError('Квантиль может быть в диапазоне от 0 до 1')
    if len(arr) == 0:
        return float('nan')
    arr = sorted(arr)
    max_indices = len(arr) - 1
    indices = qntl * max_indices
    indices_below = int(indices)
    indices_above = (indices_below + 1
                     if max_indices > indices_below
                     else max_indices)
    weights_above = indices - indices_below
    weights_below = 1 - weights_above

    return (arr[indices_below] * weights_below +
            arr[indices_above] * weights_above)
