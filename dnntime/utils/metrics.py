import numpy as np
# Scikit-learn metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
                            mean_squared_log_error, r2_score


def calc_rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculates Root Mean Square Error (RMSE).
    Source: https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Parameters
    ----------
    y : Actual target dataset.
    y_hat : Predicted or forecasted dataset.

    Returns
    -------
    rmse: RMSE score value.

    """
    return np.sqrt(np.mean((y - y_hat)**2))


def calc_nrmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculates Normalized Root Mean Square Error (NRMSE).

    Parameters
    ----------
    y : Actual target dataset.
    y_hat : Predicted or forecasted dataset.

    Returns
    -------
    rmse: RMSE score value.

    """
    pass


def calc_rmsle(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculates Root Mean Squared Logarithmic Error (RMSLE).

    Parameters
    ----------
    y : Actual target dataset.
    y_hat : Predicted or forecasted dataset.

    Returns
    -------
    rmsle: RMSLE score value.

    """
    pass


def calc_mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculates Mean Absolute Error (MAE).
    Source: https://en.wikipedia.org/wiki/Mean_absolute_error

    Parameters
    ----------
    y : Actual target dataset.
    y_hat : Predicted or forecasted dataset.

    Returns
    -------
    mae: MAE score value.

    """
    return np.mean(np.abs(y - y_hat))


def calc_mape(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculates Mean Absolute Percent Error (MAPE).
    Source: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    Parameters
    ----------
    y : Actual target dataset.
    y_hat : Predicted or forecasted dataset.

    Returns
    -------
    mape: MAPE score value.

    """
    perc_err = (100*(y - y_hat))/y
    return np.mean(np.abs(perc_err))


def calc_smape(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculates Symmetric Mean Absolute Percent Error (SMAPE).
    Source: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Parameters
    ----------
    y : Actual target dataset.
    y_hat : Predicted or forecasted dataset.

    Returns
    -------
    smape: SMAPE score value.

    """
    pass
