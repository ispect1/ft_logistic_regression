#!/usr/bin/env python3
"""Predict model"""
import sys
import json
import argparse
from models.ml import MyLogisticRegression, Scaler
from models.my_frame import HogwartsFrame
from config import WEIGHTS_DATA_PATH, COURSE_COLUMN, PREDICT_CSV


def main(args):
    """

    Args:
        args:

    Returns:

    """
    log_reg = MyLogisticRegression()
    try:
        with open(WEIGHTS_DATA_PATH) as file_descriptor:
            json_data = json.load(file_descriptor)
    except FileNotFoundError:
        print(f'File {args.filename_data} not found')
        sys.exit()
    scaler = Scaler()
    if 'scaler' in json_data:
        scaler.download(json_data['scaler'])
    if 'model' not in json_data:
        print('В файле отсутвуют данные для модели')
        sys.exit()
    log_reg.download(json_data['model'])
    table = HogwartsFrame.read_csv(args.filename_data,
                                   index_col=args.index_col)
    table = table[[feature for feature in table.number_columns
                   if feature != COURSE_COLUMN]].fillna(0).values
    scaler_df = scaler.transform(table)
    predict = log_reg.predict(scaler_df)

    with open(PREDICT_CSV, 'w') as file_descriptor:
        file_descriptor.write(f'Index,{COURSE_COLUMN}\n')
        file_descriptor.write('\n'.join(f'{i},{pred}'
                                        for i, pred in enumerate(predict)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict  model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest='filename_data',
                        help='input data file')
    parser.add_argument('--index_col',
                        dest='index_col', action='store', help='index_col')

    arguments = parser.parse_args()
    try:
        main(arguments)
    except FileNotFoundError:  # pylint: disable=broad-except
        print('Отствует файл с весами и/или таблицей')
    except Exception as err:  # pylint: disable=broad-except
        print(f'Невозможно посчитать предикт модели.'
              f'Проверьте вводные параметры и файлы.\n{err.args}')
