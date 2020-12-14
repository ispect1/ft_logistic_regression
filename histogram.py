#!/usr/bin/env python3
"""Histogram"""
import math
import sys
from matplotlib import pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from models.my_frame import read_csv_saving
from config import HISTOGRAM_VALID_COLUMN, COURSE_COLUMN
from utils.helper import init_argparser, show_graph


if __name__ == '__main__':
    args = init_argparser('Histogram plot graphic')
    df = read_csv_saving(filename=args.filename, usecols=args.usecols,
                         index_col=args.index_col)

    columns = df.get_numbers_features()
    n_cols = len(columns) // 2 or 1
    n_rows = math.ceil(len(columns) / n_cols)

    fig = plt.figure(figsize=(16, 13))
    fig.subplots_adjust(hspace=0.5)
    fig.tight_layout()

    for plot_num, column in enumerate(columns):
        ax = fig.add_subplot(n_cols, n_rows, plot_num + 1)
        if column == HISTOGRAM_VALID_COLUMN:
            ax.set_xlabel(column, fontsize=17, color='darkred')
            ALPHA_TRANSPARENCY = 0.5
        else:
            ALPHA_TRANSPARENCY = 0.25
        try:
            sns.histplot(data=df, x=column, hue=COURSE_COLUMN, ax=ax,
                         alpha=ALPHA_TRANSPARENCY)
            plt.setp(ax.get_legend().get_texts(), fontsize=6)
            plt.setp(ax.get_legend().get_title(), fontsize=10)
        except KeyError:
            print('Invalid csv file')
            sys.exit()
        except Exception as err:  # pylint: disable=broad-except
            print("Невозможно построить график, проверьте вводные параметры. "
                  "Error '{0}' occured. Arguments {1}.".format(err, err.args))
            sys.exit()

    show_graph(fig, args.plot_filename)
