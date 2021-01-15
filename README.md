# First logistic regression (no)

## Before start
`python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

## Description
Naive logistic regression without using third party machine learning libraries (only matplotlib and numpy). Supports full and stochastic gradient descent method, calculation of basic classification metrics, visualization for two-dimensional case, CSV parsing
## Train model

`./train.py` - run train model
```
usage: train.py [-h] [--plot] [--metrics] [--filename_data FILENAME_DATA]
                [--num_target_column NUM_TARGET_COLUMN]
                [--gradient_mode {stochastic,full}] [--max_iter MAX_ITER]
                [--eta ETA]

optional arguments:
  -h, --help            show this help message and exit
  --plot, -p            plot mode (default: False)
  --metrics, -m         calculate metrics (default: False)
  --filename_data FILENAME_DATA, -f FILENAME_DATA
                        input data file (default: data.csv)
  --num_target_column NUM_TARGET_COLUMN, -n NUM_TARGET_COLUMN
                        target column number (default: -1)
  --gradient_mode {stochastic,full}, -g {stochastic,full}
                        gradient mode (default: stochastic)
  --max_iter MAX_ITER, -i MAX_ITER
                        max iter steps (default: None)
  --eta ETA, -e ETA     eta (default: None)

```


## Predict model

`./predict.py` - run predict model

```
usage: predict.py [-h] objects [objects ...]

positional arguments:
  objects     Various objects for predict. Separate the signs of one object
              with a comma

optional arguments:
  -h, --help  show this help message and exit
```
  
## Examples

```shell script
./train.py -m -g=full
./predict.py 240000 74000
```

```shell script
./train.py -m -p -f=weights_heights.csv
./predict.py 64.39693 68.02403 71.23661 67.50812 76.92387
```  

```shell script
./train.py -m -f=boston.csv
./predict.py 0.02731,0.0,7.07,0.0,0.469,6.421,78.9,4.9671,2.0,242.0,17.8,396.9,9.14
```  

## Data
__data.csv__ - Dependence of the rates of the car mileage (km.)

__weights_heights.csv__ -  Dependence of the weight from human height (inc.)

__boston.csv__ - Dependence of housing prices on a variety of indicators