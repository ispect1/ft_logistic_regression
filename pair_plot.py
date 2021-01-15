#!/usr/bin/env python3
"""Pair plot"""
import sys
import seaborn as sns  # type: ignore
from config import COURSE_COLUMN
from models.my_frame import HogwartsFrame
from utils.helper import init_argparser, show_graph


if __name__ == '__main__':
    args = init_argparser('Pair plot graphic')
    df = HogwartsFrame.read_csv(filename=args.filename, usecols=args.usecols,
                                index_col=args.index_col)

    df = df[df.number_columns + [COURSE_COLUMN]]
    sns.set_theme(style="ticks")

    try:
        if COURSE_COLUMN not in df:
            print(f'Признак {COURSE_COLUMN} отсутвует. Введите другой файл')
            sys.exit()
        plot = sns.pairplot(data=df, hue=COURSE_COLUMN)
    except Exception as err:  # pylint: disable=broad-except
        print("Невозможно построить график, проверьте вводные параметры. "
              "Error '{0}' occured. Arguments {1}.".format(err, err.args))
        sys.exit()

    show_graph(plot, args.plot_filename)
