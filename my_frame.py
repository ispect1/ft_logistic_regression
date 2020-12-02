"""MyTable class"""
from typing import List, Dict, Union, Optional, Set, Callable, Sequence, Type, Iterator, Any
import csv
import sys
import itertools
from datetime import datetime
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from utils import mean, std, var, mmin, mmax, quantile
from config import BASE_DATETIME_FORMAT, TYPE_MAPPING


def read_csv(filename, sep=',', header=True, dtypes=None,  # pylint: disable=too-many-arguments
             usecols: List[str] = None, index_col=None,
             datetime_format=BASE_DATETIME_FORMAT, error_bad_lines=True) -> 'MyTable':
    """

    Args:
        filename:
        sep:
        header:
        dtypes:
        usecols:
        index_col:
        datetime_format:
        error_bad_lines:

    Returns:

    """
    with open(filename) as file_descriptor:
        reader: Iterator[List[Any]] = csv.reader(file_descriptor, delimiter=sep)
        if header:
            header = next(reader)
        else:
            header = itertools.count()
        if usecols is None:
            usecols = header

        data = []
        for i, row in enumerate(reader):
            try:
                data.append({column: row[i]
                             for i, column in enumerate(header) if column in usecols})
            except IndexError as err:
                if error_bad_lines:
                    raise TypeError(f'Неверный формат данных в строке {i}') from err

    data_frame = MyTable(data, dtypes, datetime_format=datetime_format)
    if index_col:
        data_frame.set_index(index_col)
    return data_frame


def read_csv_saving(filename, **kwargs) -> 'MyTable':
    """
    read_csv + защита от краша
    Args:
        filename: расположение файла
        **kwargs: остальные аргументы read_csv

    Returns:

    """

    try:
        table = read_csv(filename, **kwargs)
    except FileNotFoundError:
        print(f'File {filename} not found')
        sys.exit()
    except TypeError as err:
        print(err.args[0])
        sys.exit()
    except KeyError:
        print('Enter valid index column')
        sys.exit()

    except Exception as err:  # pylint: disable=broad-except
        print("Invalid input. Error '{0}' occured. Arguments {1}.".format(err, err.args))
        sys.exit()

    return table


class MyTable:
    """
    Наивная реализация pd.DataFrame. Данные внутри него хранятся в виде набора трех элементов:
            Кортеж `название - номер` признаков (колонки)
            Кортеж `название - номер` объектов (индексы)
            Список numpy массивов данных
                (n-ый массив соответсвует n-ому признаков,
                n-ый элемент m-ого массива соответует n-ому объектов m-ого признака)

    Названия колонок определяются по первому объекту,
        остальные названия других объектов игнорируются

    Данные для инициализации
        data: Список объектов, каждый объект - словарь `название признака - значение объекта на нем`

        dtypes: словарь типов для явного задания типа некоторых признаков

        datetime_format: формат парсинга даты
    """

    def __init__(self, data: List[Dict[str, Union[float, int, str, datetime]]],
                 dtypes: Optional[Dict[str, Type]] = None,
                 datetime_format: str = BASE_DATETIME_FORMAT):

        self.datetime_format = datetime_format
        if not isinstance(data, list):
            raise TypeError('Invalid data format. Use `List[Dict[str, Union[float, int, str]]]`')
        if not data:
            raise TypeError('Minimal length dataframe - 1')
        if not data[0]:
            raise TypeError('Minimal count columns - 1')

        self.columns = {column: i for i, column in enumerate(list(data[0]))}
        self.indexs: Dict[Union[str, int, float, datetime], 'np.ndarray[int]'] = {
            i: np.array([i]) for i in range(len(data))}

        dtypes = dtypes or {}
        self.dtypes: Dict[str, Type] = {}
        dtypes = self._parse_data(data, dtypes)
        self._transform_to_correct_types(dtypes)

    def __getitem__(self, item):
        if isinstance(item, List):
            return {column: self.arrays[self.columns[column]] for column in item}
        return self.arrays[self.columns[item]]

    def __iter__(self):
        return iter(self.columns)

    def __len__(self):
        return len(self.arrays[0])

    @property
    def index(self):
        """
        Получить словарь индексов
        """
        return self.indexs

    @property
    def shape(self):
        """
        Показыва
        Returns:

        """
        return len(self.arrays[0]), len(self.arrays)

    def _parse_data(self, data, dtypes):
        """
        Парсит данные исходя из их типов
        Args:
            data: Список объектов, каждый объект -
                словарь `название признака - значение объекта на нем`
            dtypes: словарь типов для явного задания типа некоторых признаков
        """
        arrays: List[List[Union[str, datetime, int, float]]] = [[] for _ in self.columns]

        possibles_dtypes: Dict[str, Set[str]] = {column: {'str', 'int', 'float', 'datetime'}
                                                 for column in self.columns if column not in dtypes}

        for obj in data:
            for j, column in enumerate(self.columns):
                if column not in possibles_dtypes:
                    continue

                curr_possible_dtypes = possibles_dtypes[column]
                try:
                    x_i_j = obj[column]
                except KeyError:
                    x_i_j = float('nan')

                # if not isinstance(x_i_j, (float, int, str, np.datetime64, datetime)):
                #     raise TypeError(f'Неправильный тип данных "{obj}: {type(obj)}". '
                #                     f'Возможны только int, float, str, np.datetime64')

                if 'float' in curr_possible_dtypes:
                    try:
                        x_i_j = float(x_i_j) if x_i_j else x_i_j
                    except ValueError:
                        curr_possible_dtypes.remove('float')

                if 'int' in curr_possible_dtypes:
                    try:
                        if isinstance(x_i_j, float) and not x_i_j.is_integer():
                            raise ValueError
                        x_i_j = int(x_i_j)
                    except ValueError:
                        curr_possible_dtypes.remove('int')

                if 'datetime' in curr_possible_dtypes and isinstance(x_i_j, str):
                    try:
                        x_i_j = datetime.strptime(x_i_j, self.datetime_format)
                    except ValueError:
                        curr_possible_dtypes.remove('datetime')

                arrays[j].append(x_i_j)

        dtypes = ({column: sorted(dtypes,
                   key=lambda x: {'int': 0, 'float': 1, 'datetime': 2, 'str': 3}[x])[0]
                   for column, dtypes in possibles_dtypes.items()})
        self.arrays = arrays
        return dtypes

    def _transform_to_correct_types(self, dtypes):
        """
        Преобразовывает признаки и названия типов в корректный формат
        """
        for column, dtype_name in dtypes.items():
            idx_column = self.columns[column]
            if dtype_name == 'datetime':
                self.arrays[idx_column] = list(map(
                    lambda x: datetime.strptime(x, self.datetime_format)
                    if isinstance(x, str) else x,
                    self.arrays[idx_column]))
            elif dtype_name in ('float', 'int'):
                self.arrays[idx_column] = list(map(float, map(
                    lambda x: 'nan' if x == '' or x is None else x, self.arrays[idx_column])))
            else:
                dtypes[column] = 'str'
            if isinstance(dtypes[column], str):
                dtypes[column] = TYPE_MAPPING[dtypes[column]]
        self.astype(dtypes)
        return self

    def astype(self, dtypes: Dict[str, Type]) -> 'MyTable':
        """
        Приводит датафрейм к типам данных из dtypes
        Args:
            dtypes: словарь типов для явного задания типа некоторых признаков
        """

        for column, dtype in dtypes.items():
            idx_column = self.columns[column]
            self.arrays[idx_column] = np.array(self.arrays[idx_column], dtype)
        self.dtypes = {column: self.arrays[idx].dtype
                       for column, idx in self.columns.items()}
        return self

    def set_index(self, key: str) -> 'MyTable':
        """
        Делает объекты с колонкой `key` индексами. Удяляет предыдущие индексы
        Поддерживает несколько обхектов с одним индекском
        Args:
            key: название колонки
        """
        try:
            idx_columns = self.columns.pop(key)
            new_indexs = self.arrays.pop(idx_columns)
            self.indexs = {}
            for i, index in enumerate(new_indexs):
                self.indexs[index] = self.indexs.get(index, []) + [i]
            self.indexs = {name: np.array(idx)
                           for name, idx in self.indexs.items()}
            self.columns = dict(zip(self.columns, range(len(self.columns))))
            self.dtypes.pop(key)
        except KeyError as exc:
            raise KeyError(f'Incorrect index column name "{key}"') from exc
        return self

    def _calc_helper(self):
        pass

    def _calc_params(self, func_name: Callable[[Sequence],
                                               float]) -> Dict[str, float]:
        """
        Применяет функцию к столбцам
        Args:
            func_name: функцию, которую нужно применить к данным
        Returns:
            Словарь столбец/строка: посчитанное значение по каждой колонке
        """
        res = {}
        for column in self.columns:
            if not (np.issubdtype(self.dtypes[column], int)
                    or np.issubdtype(self.dtypes[column], float)):
                continue
            array = self.arrays[self.columns[column]]
            array = array[~np.isnan(array)]
            res[column] = func_name(array)
        return res

    def count(self) -> Dict[str, float]:
        """
        Кол-во объектов колонок
        Returns:
            Словарь столбец/строка: посчитанное значение по каждой колонке
        """
        return self._calc_params(len)

    def mean(self) -> Dict[str, float]:
        """
        Среднее значение колонок
        Returns:
            Словарь столбец/строка: посчитанное значение по каждой колонке
        """
        return self._calc_params(mean)

    def std(self) -> Dict[str, float]:
        """
        Стандартное отклонение колонок
        Returns:
            Словарь столбец/строка: посчитанное значение по каждой колонке
        """
        return self._calc_params(std)

    def var(self) -> Dict[str, float]:
        """
        Дисперсия объектов колонок
        Returns:
            Словарь столбец/строка: посчитанное значение по каждой колонке
        """
        return self._calc_params(var)

    def min(self) -> Dict[str, float]:
        """
        Минимальное значение колонок
        Returns:
            Словарь столбец/строка: посчитанное значение по каждой колонке
        """
        return self._calc_params(mmin)

    def max(self) -> Dict[str, float]:
        """
        Максимальное значение колонок
        Returns:
            Словарь столбец/строка: посчитанное значение по каждой колонке
        """
        return self._calc_params(mmax)

    def quantile(self, qntl=0.5) -> Dict[str, float]:
        """
        `q`-ый квантиль колонок
        Args:
            qntl - квантиль
        Returns:
            Словарь столбец/строка: посчитанное значение по каждой колонке
        """
        return self._calc_params(lambda x: quantile(x, qntl=qntl))

    def get_numbers_features(self) -> List[str]:
        """
            Получить числовые колонки
        Returns:
            Список колонок
        """
        return [column for column, dtype in self.dtypes.items()
                if np.issubsctype(float, dtype) or np.issubsctype(int, dtype)]

    def describe(self) -> Dict[str, Dict[str, float]]:
        """
            Получить словарь основных статистик таблицы
        Returns:
            Словарь статистик
        """
        return {'count': self.count(), 'mean': self.mean(),
                'std': self.std(), 'min': self.min(),
                '25%': self.quantile(0.25), '50%': self.quantile(0.5),
                '75%': self.quantile(0.75),
                'max': self.max()}

    def to_frame(self) -> pd.DataFrame:
        """
        Транформ таблицы в pd.DataFrame формат
        """
        table = pd.DataFrame({column: self.arrays[self.columns[column]]
                              for column in self.columns})
        table.index = table.index.map(
            {i: line for line, lst in self.index.items() for i in lst})
        return table
