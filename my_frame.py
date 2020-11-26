from typing import List, Dict, Union, Optional
import numpy as np
from config import MY_DTYPE


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
                 dtypes: Optional[Dict[str, MY_DTYPE]] = None):

        if not isinstance(data, list):
            raise TypeError('Invalid data format. Use `List[Dict[str, Union[float, int, str]]]`')
        if not data:
            raise TypeError('Minimal length dataframe - 1')
        if not data[0]:
            raise TypeError('Minimal count columns - 1')

        self.columns = dict(zip(tuple(data[0]), range(len(data[0]))))
        self.indexs = dict(zip(range(len(self.columns)), range(len(self.columns))))
        self.arrays = [[] for _ in self.columns]

        for i, obj in enumerate(data):
            try:
                for j, column in enumerate(self.columns):
                    self.arrays[j].append(obj[column])
            except KeyError:
                raise KeyError(f'Отсутствует признак "{column}" на {i}-ом объекте')
        dtypes = self._get_columns_types(dtypes)
        self._replace_nan_to_float(dtypes)
        self.arrays = [np.array(self.arrays[self.columns[column]], dtype=dtypes[column]) for column in self.columns]

    def _replace_nan_to_float(self, dtypes: Dict[str, MY_DTYPE]):
        """
        Заменяет '' и None на float('nan') в данных (self.arrays)
        :param dtypes: словарь типов для явного задания типа некоторых признаков
        """
        # print(dtypes)
        for column, dtype in dtypes.items():
            # print(column)
            # print(dtype)
            if dtype != float:
                continue
            idx_column = self.columns[column]
            self.arrays[idx_column] = list(map(
                lambda x: float('nan') if x == '' or x is None else x, self.arrays[idx_column]))

    def _get_columns_types(self, dtypes: Optional[Dict[str, MY_DTYPE]] = None) -> Dict[str, MY_DTYPE]:
        """
        Определяет валидный тип признаков (int, float, str)
        :param dtypes: словарь типов для явного задания типа некоторых признаков
        :return: словарь типов для явного задания типа всех признаков
        """
        dtypes = dtypes or {}
        for column, i in self.columns.items():
            if column in dtypes:
                continue

            dtype = int
            one_feature_objects = self.arrays[i]
            for obj in one_feature_objects:

                if not isinstance(obj, (float, int, str)):
                    raise TypeError(f'Неправильный тип данных "{obj}: {type(obj)}". '
                                    f'Возможны только int, float, str')

                if dtype == int and isinstance(obj, int):
                    continue

                elif dtype != str:
                    if obj == '' or obj is None or obj != obj:
                        dtype = float
                        continue
                    try:
                        if isinstance(obj, str):
                            obj = float(obj)
                    except ValueError:
                        dtype = str
                        continue

                    if isinstance(obj, float) and not obj.is_integer():
                        dtype = float
                else:
                    dtype = str
            dtypes[column] = dtype
        return dtypes
