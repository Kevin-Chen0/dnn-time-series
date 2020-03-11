import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
from typing import Tuple
# Scikit-learn metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
                            mean_squared_log_error, r2_score


def print_static_rmse(actual: np.ndarray, predicted: np.ndarray, start_from:
                      int = 0, verbose: int = 0) -> Tuple[float, float]:
    """
    this calculates the ratio of the rmse error to the standard deviation of the actuals.
    This ratio should be below 1 for a model to be considered useful.
    The comparison starts from the row indicated in the "start_from" variable.
    """
    rmse = np.sqrt(mean_squared_error(actual[start_from:],predicted[start_from:]))
    std_dev = actual[start_from:].std()
    if verbose == 1:
        print('    RMSE = %0.2f' %rmse)
        print('    Std Deviation of Actuals = %0.2f' %(std_dev))
        print('    Normalized RMSE = %0.1f%%' %(rmse*100/std_dev))
    return rmse, rmse/std_dev


def print_dynamic_rmse(actuals: np.ndarray, predicted: np.ndarray, original:
                       np.ndarray) -> Tuple[float, float]:
    """
    This utility calculates rmse between actuals and predicted. However, it does one more.
    Since in dynamic forecast, we need the longer original, it calculates Normalized RMSE
    using the original array's std deviation. That way, the forecast of 2 values does not
    result in a larger Normalized RMSE since the std deviation of 2 values will be v small.
    """
    rmse = np.sqrt(np.mean((actuals - predicted)**2))
    norm_rmse = rmse/original.std()
    # print('    RMSE = {:,.2f}'.format(rmse))
    # print('    Std Deviation of Originals = {:,.2f}'.format(original.std()))
    # print('    Normalized RMSE = %0.0f%%' %(100*norm_rmse))
    return rmse, norm_rmse


def print_normalized_rmse(actuals: np.ndarray, predicted: np.ndarray,
                          start_from: int = 0) -> Tuple[float, float]:
    """
    This utility calculates rmse between actuals and predicted. However, it does one more.
    If the original is given, it calculates Normalized RMSE using the original array's std deviation.
    """
    actuals = actuals[start_from:]
    predicted = predicted[start_from:]
    rmse = np.sqrt(np.mean(mean_squared_error(actuals, predicted)))
    norm_rmse = rmse/actuals.std()
    # print('RMSE = {:,.2f}'.format(rmse))
    # print('Std Deviation of Actuals = {:,.2f}'.format(actuals.std()))
    # print('Normalized RMSE = %0.0f%%' %(100*norm_rmse))
    return rmse, norm_rmse


def print_rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculating Root Mean Square Error https://en.wikipedia.org/wiki/Root-mean-square_deviation
    """
    return np.sqrt(np.mean((y - y_hat)**2))


def print_mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculating Mean Absolute Error https://en.wikipedia.org/wiki/Mean_absolute_error
    """
    return np.mean(np.abs(y - y_hat))


def print_mape(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculating Mean Absolute Percent Error https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    """
    perc_err = (100*(y - y_hat))/y
    return np.mean(np.abs(perc_err))


# mean absolute percentage error (mape)
def print_smape(y: np.ndarray, y_hat: np.ndarray) -> float:
    pass

