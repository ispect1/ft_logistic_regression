from typing import List, Dict, Union, Optional, Type
import numpy as np  # type: ignore
from config import BASE_DATETIME_FORMAT
import csv
from datetime import datetime


MY_DTYPE = Type[Union[float, int, str, np.datetime64]]


def read_csv(filename, sep=',', header=True, dtypes=None, usecols=None, index_col=None,
             datetime_format=BASE_DATETIME_FORMAT):
    with open(filename) as f:
        count_feature = len(f.readline().strip().split(sep))
    with open(filename) as f:
        reader = csv.reader(f, delimiter=sep)
        if header:
            header = next(reader)
        else:
            header = range(count_feature)
        if usecols is None:
            usecols = header

        data = []
        for row in reader:
            obj = {}

            for i, column in enumerate(header):
                if column not in usecols:
                    continue
                obj[column] = row[i]
            data.append(obj)

    df = MyDataFrame(data, dtypes, datetime_format=datetime_format)
    if index_col:
        df.set_index(index_col)
    return df


class MyDataFrame(object):
    """
    Наивная реализация pd.DataFrame. Данные внутри него хранятся в виде набора трех элементов:
            Кортеж `название - номер` признаков (колонки)
            Кортеж `название - номер` объектов (индексы)
            Список numpy массивов данных
                (n-ый массив соответсвует n-ому признаков,
                n-ый элемент m-ого массива соответует n-ому объектов m-ого признака)

    Названия колонок определяются по первому объекту, остальные названия других объектов игнорируются

    Данные для инициализации
        data: List[Dict[str, Union[float, int, str]]]
            Список объектов, каждый объект - словарь `название признака - значение объекта на нем`

        dtypes: Optional[Dict[str, Type[Union[float, int, str]]]] -
            словарь типов для явного задания типа некоторых признаков
    """
    def __init__(self, data: List[Dict[str, Union[float, int, str]]],
                 dtypes: Optional[Dict[str, MY_DTYPE]] = None, datetime_format=BASE_DATETIME_FORMAT):
        self.datetime_format = datetime_format
        if not isinstance(data, list):
            raise TypeError('Invalid data format. Use `List[Dict[str, Union[float, int, str]]]`')
        if not data:
            raise TypeError('Minimal length dataframe - 1')
        if not data[0]:
            raise TypeError('Minimal count columns - 1')

        self.columns = dict(zip(tuple(data[0]), range(len(data[0]))))
        self.indexs = dict(zip(range(len(self.columns)), range(len(self.columns))))
        arrays: List[List[Union[str, float, np.datetime64, int]]] = [[] for _ in self.columns]

        for i, obj in enumerate(data):
            try:
                for j, column in enumerate(self.columns):
                    arrays[j].append(obj[column])
            except KeyError:
                raise KeyError(f'Отсутствует признак "{column}" на {i}-ом объекте')
        dtypes = self._get_columns_types(arrays, dtypes)
        self.arrays = self._transform_to_correct_types(arrays, dtypes)
        self.astype(dtypes)

    def _transform_to_correct_types(self, arrays, dtypes):
        for column, dtype in dtypes.items():
            idx_column = self.columns[column]
            if dtype == np.datetime64:
                arrays[idx_column] = list(map(
                    lambda x: datetime.strptime(x, self.datetime_format), arrays[idx_column]))
            elif dtype in (float, int):
                arrays[idx_column] = list(map(float, map(
                    lambda x: 'nan' if x == '' or x is None else x, arrays[idx_column])))
        return arrays

    def astype(self, dtypes: Dict[str, MY_DTYPE]):
        """
        Приводит датафрейм к типам данных из dtypes
        :param dtypes: словарь типов для явного задания типа некоторых признаков
        """
        for column, dtype in dtypes.items():
            idx_column = self.columns[column]
            self.arrays[idx_column] = np.array(self.arrays[idx_column], dtype)

    def _get_columns_types(self, arrays, dtypes: Dict[str, MY_DTYPE] = None) -> Dict[str, MY_DTYPE]:
        """
        Определяет валидный тип признаков (int, float, str, np.datetime64)
        :param dtypes: словарь типов для явного задания типа некоторых признаков
        :return: словарь типов для явного задания типа всех признаков
        """
        dtypes = dtypes or {}
        for column, i in self.columns.items():
            if column in dtypes:
                continue

            dtype: Optional[MY_DTYPE] = None
            one_feature_objects = arrays[i]
            for obj in one_feature_objects:

                if not isinstance(obj, (float, int, str, np.datetime64)):
                    raise TypeError(f'Неправильный тип данных "{obj}: {type(obj)}". '
                                    f'Возможны только int, float, str, np.datetime64')

                if (not dtype or dtype == int) and isinstance(obj, int):
                    dtype = int
                    continue
                elif not dtype or dtype == np.datetime64:
                    try:
                        datetime.strptime(obj, self.datetime_format)
                        dtype = np.datetime64
                    except ValueError:
                        pass
                if dtype != str:
                    if obj == '' or obj is None or obj != obj:
                        if dtype:
                            dtype = float
                        continue
                    if dtype == np.datetime64:
                        continue
                    try:
                        if isinstance(obj, str):
                            obj = float(obj)
                    except ValueError:
                        dtype = str
                        continue
                    if isinstance(obj, float):
                        if obj.is_integer() and dtype != float:
                            dtype = int
                        else:
                            dtype = float

            dtypes[column] = dtype or str
        return dtypes

    def set_index(self, key):
        try:
            idx_columns = self.columns.pop(key)
            self.indexs = dict(zip(self.arrays.pop(idx_columns), range(len(self.columns))))
            self.columns = dict(zip(self.columns, range(len(self.columns))))
        except KeyError:
            raise KeyError('Incorrect index column name')
        return self