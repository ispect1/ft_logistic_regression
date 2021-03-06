#!/usr/bin/env python3
"""Train model"""
import json
import argparse
import sys
from models.ml import MyLogisticRegression, Scaler
from models.my_frame import HogwartsFrame
from utils.metrics import accuracy
from config import COURSE_COLUMN, MAX_ITER, ETA, WEIGHTS_DATA_PATH


def main(args):
    """

    Args:
        args:

    Returns:

    """
    log_reg = MyLogisticRegression()
    scaler = Scaler()
    try:
        table = HogwartsFrame.read_csv(args.filename_data,
                                       index_col=args.index_col)
    except FileNotFoundError:
        print(f'File {args.filename_data} not found')
        sys.exit()
    y_true = table[args.name_target_column]
    table = table[[feature for feature in table.number_columns
                   if feature != args.name_target_column]].fillna(0)
    table = table.values
    scaler_df = scaler.fit_transform(table)
    if not sum(map(lambda x: x == x, y_true)):  # pylint: disable=R0124
        print('Целевой признак не должен состоять из nan')
        sys.exit()
    log_reg.fit(scaler_df, y_true, mode=args.gradient_mode,
                max_iter=args.max_iter, eta=args.eta)

    json_model = log_reg.save()
    json_scaler = scaler.save()
    with open(WEIGHTS_DATA_PATH, 'w') as file_descriptor:
        json.dump({'scaler': json_scaler, 'model': json_model},
                  file_descriptor, indent=4)
    if args.metric:
        predict = log_reg.predict(scaler_df)
        print(f'Accuracy: {accuracy(y_true, predict)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train  model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest='filename_data', help='input data file')
    parser.add_argument('--metric', '-m', dest='metric', action='store_true',
                        help='calculate metric')
    parser.add_argument('--name_target_column', '-n',
                        dest='name_target_column',
                        action='store', help='target column naming',
                        default=COURSE_COLUMN)
    parser.add_argument('--gradient_mode', '-g', dest='gradient_mode',
                        action='store', help='gradient mode',
                        choices={'full', 'stochastic'}, default='stochastic')
    parser.add_argument('--max_iter', '-i', dest='max_iter', action='store',
                        help='max iter steps', type=int, default=MAX_ITER)
    parser.add_argument('--eta', '-e', dest='eta', action='store', help='eta',
                        type=float, default=ETA)
    parser.add_argument('--index_col', dest='index_col', action='store',
                        help='index_col')

    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as err:  # pylint: disable=broad-except
        print(f'Неаозможно обучить модель. '
              f'Проверьте вводные параметры и файлы.\n{err.args}')
