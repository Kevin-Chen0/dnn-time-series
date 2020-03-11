import numpy as np
import pandas as pd
# Scikit-learn metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
                            mean_squared_log_error, r2_score


def calc_rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error (RMSE)
    Source: https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Parameters
    ----------
    y : DESCRIPTION.
    y_hat : DESCRIPTION.

    Returns
    -------
    float: DESCRIPTION.

    """
    return np.sqrt(np.mean((y - y_hat)**2))


def calc_nrmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculate Normalized Root Mean Square Error (NRMSE)

    Parameters
    ----------
    y : DESCRIPTION.
    y_hat : DESCRIPTION.

    Returns
    -------
    float: DESCRIPTION.

    """
    pass


def calc_rmsle(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Logarithmic Error (RMSLE)

    Parameters
    ----------
    y : DESCRIPTION.
    y_hat : DESCRIPTION.

    Returns
    -------
    float: DESCRIPTION.

    """
    pass


def calc_mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE)
    Source: https://en.wikipedia.org/wiki/Mean_absolute_error

    Parameters
    ----------
    y : DESCRIPTION.
    y_hat : DESCRIPTION.

    Returns
    -------
    float: DESCRIPTION.

    """
    return np.mean(np.abs(y - y_hat))


def calc_mape(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percent Error (MAPE)
    Source: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    Parameters
    ----------
    y : DESCRIPTION.
    y_hat : DESCRIPTION.

    Returns
    -------
    float: DESCRIPTION.

    """
    perc_err = (100*(y - y_hat))/y
    return np.mean(np.abs(perc_err))


def calc_smape(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percent Error (SMAPE)
    Source: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Parameters
    ----------
    y : DESCRIPTION.
    y_hat : DESCRIPTION.

    Returns
    -------
    float: DESCRIPTION.

    """
    pass

