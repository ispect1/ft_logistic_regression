"""Test"""
import pandas as pd  # type: ignore
from my_frame import read_csv


mdf = read_csv('./datasets/dataset_train.csv', error_bad_lines=False)


print(pd.read_csv('./datasets/dataset_train.csv').describe())
