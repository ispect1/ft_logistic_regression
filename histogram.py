#!/usr/bin/env python3
"""Histogram"""
import math
import sys
from matplotlib import pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from models.my_frame import HogwartsFrame
from config import HISTOGRAM_VALID_COLUMN, COURSE_COLUMN
from utils.helper import init_argparser, show_graph


if __name__ == '__main__':
    args = init_argparser('Histogram plot graphic')
    df = HogwartsFrame.read_csv(filename=args.filename, usecols=args.usecols,
                                index_col=args.index_col)

    columns = df.number_columns
    n_cols = len(columns) // 2 or 1
    n_rows = math.ceil(len(columns) / n_cols)

    fig = plt.figure(figsize=(14, 10))
    fig.subplots_adjust(hspace=0.7)
    fig.tight_layout()

    for plot_num, column in enumerate(columns):
        ax = fig.add_subplot(n_cols, n_rows, plot_num + 1)
        if column == HISTOGRAM_VALID_COLUMN:
            ax.set_xlabel(column, fontsize=10, color='darkred')
            ALPHA_TRANSPARENCY = 0.5
        else:
            ALPHA_TRANSPARENCY = 0.25
        try:
            COURSE_COLUMN = COURSE_COLUMN if COURSE_COLUMN in df else None
            sns.histplot(data=df, x=column, ax=ax, hue=COURSE_COLUMN,
                         alpha=ALPHA_TRANSPARENCY)
            if ax.get_legend():
                plt.setp(ax.get_legend().get_texts(), fontsize=6)
                plt.setp(ax.get_legend().get_title(), fontsize=10)
        except KeyError:
            print('Invalid csv file')
            sys.exit()
        except Exception as err:  # pylint: disable=broad-except
            print("Невозможно построить график, проверьте вводные параметры. "
                  "Error '{0}' occured. Arguments {1}.".format(err, err.args))
            raise err
            sys.exit()

    show_graph(fig, args.plot_filename)
