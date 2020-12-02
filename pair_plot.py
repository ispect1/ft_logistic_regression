"""Pair plot"""
import sys
import seaborn as sns  # type: ignore
from config import COURSE_COLUMN
from my_frame import read_csv_saving
from utils import init_argparser, show_graph


if __name__ == '__main__':
    args = init_argparser('Pair plot graphic')
    df = read_csv_saving(filename=args.filename, usecols=args.usecols,
                         index_col=args.index_col)

    df = df.to_frame()[df.get_numbers_features() + [COURSE_COLUMN]]
    sns.set_theme(style="ticks")

    try:
        plot = sns.pairplot(data=df, hue=COURSE_COLUMN)
    except Exception as err:  # pylint: disable=broad-except
        print("Невозможно построить график, проверьте вводные параметры. "
              "Error '{0}' occured. Arguments {1}.".format(err, err.args))
        sys.exit()

    show_graph(plot, args.plot_filename)
