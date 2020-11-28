"""Configuration"""
from datetime import datetime


BASE_DATETIME_FORMAT = '%Y-%m-%d'
TYPE_MAPPING = {'str': str, 'float': float, 'int': int, 'datetime': datetime}
HISTOGRAM_VALID_COLUMN = 'Care of Magical Creatures'
TRAIN_CSV_FILE = './datasets/dataset_train.csv'
TEST_CSV_FILE = './datasets/dataset_test.csv'
COURSE_COLUMN = 'Hogwarts House'
