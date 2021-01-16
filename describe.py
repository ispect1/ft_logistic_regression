#!/usr/bin/env python3
"""Print describe matrix"""
import argparse
from models.my_frame import HogwartsFrame


def _transform_string(line):
    return line + '' if len(line) < 7 else line[:7].strip() + '...'


def print_describe_matrix(mdf, is_transpose=False):
    """
    Напечатать describe таблицу класса MyTable
    Args:
        mdf: Таблица MyTable
        is_transpose: Транспонировать таблицу

    Returns:

    """

    data = mdf.describe()
    columns = data.columns
    indexs = data.index

    if is_transpose:
        data = data.T
        indexs, columns = columns, indexs

    row_format = "|{:<12}" * (len(columns) + 1)
    print(row_format.format("", *map(_transform_string, columns)) + '|')
    for name, row in data.iterrows():
        try:
            sorted_row = [f'{value}'[:9] for value in row.values]
            print(row_format.format(
                _transform_string(name), *sorted_row) + '|')
        except Exception:  # pylint: disable=broad-except
            print(f'Невозможно посчитать метрику {name} для {row}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print describe data')

    parser.add_argument('filename', help='CSV filename')
    parser.add_argument('-t', '--transpose', dest='transpose',
                        help='Transpose describe matrix', action='store_true')

    args = parser.parse_args()
    table = HogwartsFrame.read_csv(args.filename)

    try:
        print_describe_matrix(table, is_transpose=args.transpose)
    except Exception as err:  # pylint: disable=broad-except
        print("Невозможно построить матрицу. Проверьте вводные параметры."
              "Error '{0}' occured. Arguments {1}.".format(err, err.args))
