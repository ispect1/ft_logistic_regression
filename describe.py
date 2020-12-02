#!/usr/bin/env python3
"""Print describe matrix"""
import argparse
from my_frame import read_csv_saving


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
    columns = mdf.get_numbers_features()

    data = mdf.describe()
    data = {key: dict(sorted(values.items(), key=lambda x: mdf.columns[x[0]]))
            for key, values in data.items()}

    if is_transpose:
        data, columns = {column: {name: values[column]
                                  for name, values in data.items()}
                         for column in columns}, list(data)
    row_format = "|{:<12}" * (len(columns) + 1)
    print(row_format.format("", *map(_transform_string, columns)) + '|')
    for name, row in data.items():
        try:
            sorted_row = [f'{row[column]}'[:9] for column in row]
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
    table = read_csv_saving(args.filename)

    try:
        print_describe_matrix(table, is_transpose=args.transpose)
    except Exception as err:  # pylint: disable=broad-except
        print("Невозможно построить матрицу. Проверьте вводные параметры."
              "Error '{0}' occured. Arguments {1}.".format(err, err.args))
