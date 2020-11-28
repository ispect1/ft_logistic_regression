"""Histogram"""
import math
import sys
from matplotlib import pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import pandas as pd
from my_frame import read_csv_saving, read_csv, MyTable
from config import HISTOGRAM_VALID_COLUMN, TRAIN_CSV_FILE, COURSE_COLUMN


if __name__ == '__main__':
    df = read_csv_saving(TRAIN_CSV_FILE, index_col='Index')

    sns.pairplot(df, vars=['Arithmancy', 'Astronomy'])
    plt.show()
