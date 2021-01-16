# First logistic regression (no)

## Before start
`python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

## Description
Naive logistic regression without using third party machine learning libraries (only matplotlib and numpy). Supports full and stochastic gradient descent method, calculation of basic classification metrics, visualization for two-dimensional case, CSV parsing


## Describe
`./describe.py filename` - print describe matrix
```shell
usage: describe.py [-h] [-t] filename

Print describe data

positional arguments:
  filename         CSV filename

optional arguments:
  -h, --help       show this help message and exit
  -t, --transpose  Transpose describe matrix
```

## Plot graphics
- `histogram.py` - plot histogram
- `scatter_plot.py` - plot scatter plot
- `pair_plot.py` - plot pair plot
```shell
optional arguments:
  -h, --help            show this help message and exit
  --save PLOT_FILENAME, -s PLOT_FILENAME
                        Save plot to file
  --index INDEX_COL, -i INDEX_COL
                        Index column
  --filename FILENAME, -f FILENAME
                        CSV Filename
  --usecols USECOLS [USECOLS ...], -c USECOLS [USECOLS ...]
                        Used columns
```

## Train model

`./logreg_predict.py` - run train model
```shell
usage: logreg_train.py [-h] [--metric] [--filename_data FILENAME_DATA]
                       [--name_target_column NAME_TARGET_COLUMN]
                       [--gradient_mode {stochastic,full}]
                       [--max_iter MAX_ITER] [--eta ETA]
                       [--index_col INDEX_COL]

Train model

optional arguments:
  -h, --help            show this help message and exit
  --metric, -m          calculate metric (default: False)
  --filename_data FILENAME_DATA, -f FILENAME_DATA
                        input data file (default:
                        ./datasets/dataset_train.csv)
  --name_target_column NAME_TARGET_COLUMN, -n NAME_TARGET_COLUMN
                        target column naming (default: Hogwarts House)
  --gradient_mode {stochastic,full}, -g {stochastic,full}
                        gradient mode (default: stochastic)
  --max_iter MAX_ITER, -i MAX_ITER
                        max iter steps (default: 10000)
  --eta ETA, -e ETA     eta (default: 0.01)
  --index_col INDEX_COL
                        index_col (default: None)

```


## Predict model

`./logreg_predict.py` - run predict model

```
usage: logreg_predict.py [-h] [--filename_data FILENAME_DATA]
                         [--index_col INDEX_COL]

Predict model

optional arguments:
  -h, --help            show this help message and exit
  --filename_data FILENAME_DATA, -f FILENAME_DATA
                        input data file (default: ./datasets/dataset_test.csv)
  --index_col INDEX_COL
                        index_col (default: None)

```
  
## Examples

```shell script
./logreg_train.py.py -m
./logreg_predict.py
```
