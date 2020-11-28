"""Histogram"""
import math
import sys
from matplotlib import pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from my_frame import read_csv_saving
from config import HISTOGRAM_VALID_COLUMN, TRAIN_CSV_FILE, COURSE_COLUMN


if __name__ == '__main__':
    df = read_csv_saving(TRAIN_CSV_FILE, index_col='Index')

    columns = df.get_numbers_features()
    n_cols = len(columns) // 2
    n_rows = math.ceil(len(columns) / n_cols)
    fig = plt.figure(figsize=(16, 13))
    fig.subplots_adjust(hspace=0.5)
    fig.tight_layout()
    for plot_num, column in enumerate(columns):
        ax = fig.add_subplot(n_cols, n_rows, plot_num+1)
        if column == HISTOGRAM_VALID_COLUMN:
            ax.set_xlabel(column, fontsize=17, color='darkred')
            ALPHA_TRANSPARENCY = 0.5
        else:
            ALPHA_TRANSPARENCY = 0.25
        try:
            sns.histplot(data=df, x=column, hue=COURSE_COLUMN, ax=ax, alpha=ALPHA_TRANSPARENCY)
        except KeyError:
            print('Invalid csv file')
            sys.exit()
        plt.setp(ax.get_legend().get_texts(), fontsize=6)
        plt.setp(ax.get_legend().get_title(), fontsize=10)

    plt.show()
