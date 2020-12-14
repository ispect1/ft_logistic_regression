"""Programs utils"""
import argparse
from typing import Optional
from matplotlib import pyplot as plt  # type: ignore

from config import TRAIN_CSV_FILE


def add_read_csv_args(parser: Optional[argparse.ArgumentParser] = None):
    """
    Добавляет к парсеру аргументы read_csv
    Args:
        parser: аргумент парсер

    Returns:
        Список аргументов
    """
    if parser is None:
        parser = argparse.ArgumentParser(description='Read dataframe')
    parser.add_argument('--index', '-i', dest='index_col', action='store',
                        help='Index column')
    parser.add_argument('--filename', '-f', dest='filename', action='store',
                        help='CSV Filename',
                        default=TRAIN_CSV_FILE)
    parser.add_argument('--usecols', '-c', dest='usecols', nargs='+',
                        help='Used columns')

    args = parser.parse_args()
    return args


def show_graph(figure: plt.figure, filename: Optional[str]):
    """
    Show or save plot
    Args:
        figure: plt.figure
        filename: Файл для записи

    Returns:

    """
    if filename:
        try:
            figure.savefig(filename)
        except (FileNotFoundError, IsADirectoryError) as err:
            print("Невозможно сохранить файл. Проверьте путь. "
                  "Error '{0}' occured. Arguments {1}.".format(err, err.args))
    else:
        plt.show()


def init_argparser(description: str = ''):
    """
    Args:
        description: Название парсера

    Returns:
        Список аргументов
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--save', '-s', dest='plot_filename', action='store',
                        help='Save plot to file')
    args = add_read_csv_args(parser)
    return args
