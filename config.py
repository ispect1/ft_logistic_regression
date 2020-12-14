"""Configuration"""
from datetime import datetime


BASE_DATETIME_FORMAT = '%Y-%m-%d'
TYPE_MAPPING = {'str': str, 'float': float, 'int': int, 'datetime': datetime}
HISTOGRAM_VALID_COLUMN = 'Care of Magical Creatures'
TRAIN_CSV_FILE = './datasets/dataset_train.csv'
TEST_CSV_FILE = './datasets/dataset_test.csv'
COURSE_COLUMN = 'Hogwarts House'
X_SCATTER_PLOT_COLUMN = 'Astronomy'
Y_SCATTER_PLOT_COLUMN = 'Defense Against the Dark Arts'
ETA = 0.01
MAX_ITER = 10_000
WEIGHTS_DATA_PATH = 'weights.json'
PREDICT_CSV = 'houses.csv'
