#!/usr/bin/env python3
"""Scatter plot"""
import sys
import seaborn as sns  # type: ignore
from models.my_frame import read_csv_saving
from config import X_SCATTER_PLOT_COLUMN, Y_SCATTER_PLOT_COLUMN, COURSE_COLUMN
from utils.helper import init_argparser, show_graph


if __name__ == '__main__':
    args = init_argparser('Scatter plot graphic')

    df = read_csv_saving(filename=args.filename, usecols=args.usecols,
                         index_col=args.index_col)

    try:
        plot = sns.scatterplot(data=df, x=X_SCATTER_PLOT_COLUMN,
                               y=Y_SCATTER_PLOT_COLUMN, hue=COURSE_COLUMN)
    except Exception as err:  # pylint: disable=broad-except
        print("Невозможно построить график, проверьте вводные параметры. "
              "Error '{0}' occured. Arguments {1}.".format(err, err.args))
        sys.exit()

    show_graph(plot.figure, args.plot_filename)
