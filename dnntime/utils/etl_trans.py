import numpy as np
import pandas as pd
from random import randint
from typing import Union, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler, power_transform
from statsmodels.tsa.seasonal import seasonal_decompose
from tscv import gap_train_test_split


def normalize(df: pd.DataFrame, target: str, scale: str = 'minmax'
              ) -> Tuple[pd.DataFrame, Union[MinMaxScaler, StandardScaler]]:
    """
    Normalize the input df using either MinMaxScaler for values ranging [0, 1]
    or StandardScaler for Gaussian distribution around a mean of 0.

    Parameters
    ----------
    df : The pd.DataFrame to be normalized. Can be either univariate or multivariate.
    target : The target column name of df.
    scale : The type of normalizing func used. Options are 'minmax' or 'standard'.

    Returns
    -------
    df_norm : The normalized dataframe.
    scaler : The normalizing obj itself, either MinMaxScaler or StandardScaler.
             This scaler can be used to reverse-normalize the df.

    """
    if scale == 'minmax':
        scaler = MinMaxScaler()
    elif scale == 'standard':
        scaler = StandardScaler()
    df_norm = df.copy()
    df_norm[target] = scaler.fit_transform(df)
    return df_norm, scaler


def log_power_transform(df: pd.DataFrame, method: str = 'box-cox',
                        standardize: bool = False) -> Tuple[pd.DataFrame, str]:
    """
    Perform log or power transform of the input dataframe. If performing a
    power transformation, user has option to standardize afterwards.

    Parameters
    ----------
    df : The pd.DataFrame to be normalized. Can be either univariate or multivariate.
    method : The type of log/power transformation used. Current options are
             'box-cox', 'yeo-johnson', or 'log'.
    standardize : The option to standardize data after transformation. The default is False.

    Returns
    -------
    df_trans : The transformed dataframe.
    title : The key used to access df_pwr in CheckpointDict during run_package().

    """
    stan = ''
    if standardize:
        stan = 'Standardized'
    if method == 'log':
        df_trans = df.transform(np.log)
    else:
        data_trans = power_transform(df, method=method, standardize=standardize)
        df_trans = pd.DataFrame(data_trans, index=df.index, columns=df.columns)

    title = str(method.title() + ' ' + stan)
    return df_trans, title


def decompose(df: pd.DataFrame, target: str, decom_type: str = 'deseasonalize',
              decom_model: str = 'additive') -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Decompose the time-series dataframe into seasonal, trend, and residual
    components using statsmodels. This is used to detrend or deseasonalize
    the time-series so only the residuals will be modeled by DNN.

    Parameters
    ----------
    df : The pd.DataFrame to be decomposed. Can be either univariate or multivariate.
    target : The target column name of df.
    decom_type : The type of decomposition. Options are 'deseasonalize', 'detrend', or 'both'.
    decom_model : The type of model component. Options are 'additive' or 'multiplicative'.

    Returns
    -------
    df_decom : The decomposed dataframe.
    removed : The stripped-away portion of df, in ndarray format.

    """
    ets = seasonal_decompose(df, decom_model)
    ets_idx = ets.trend[ets.resid.notnull()].index

    if decom_type == 'deseasonalize' and decom_model == 'additive':
        decom = ets.trend[ets_idx] + ets.resid[ets_idx]
        removed = ets.seasonal[ets_idx]
    elif decom_type == 'deseasonalize' and decom_model == 'multiplicative':
        decom = ets.trend[ets_idx] * ets.resid[ets_idx]
        removed = ets.seasonal[ets_idx]
    elif decom_type == 'detrend' and decom_model == 'additive':
        decom = ets.seasonal[ets_idx] + ets.resid[ets_idx]
        removed = ets.trend[ets_idx]
    elif decom_type == 'detrend' and decom_model == 'multiplicative':
        decom = ets.seasonal[ets_idx] * ets.resid[ets_idx]
    elif decom_model == 'additive':
        decom = ets.resid[ets_idx]
        removed = ets.trend[ets_idx] + ets.seasonal[ets_idx]
    else:
        decom = ets.resid[ets_idx]
        removed = ets.trend[ets_idx] * ets.seasonal[ets_idx]

    df_decom = decom.to_frame(name=target)
    return df_decom, removed


def split_data(data: pd.DataFrame, target: str, n_test: int, n_val: int, n_input: int,
               n_output: int = 1, n_feature: int = 1, g_min: int = 0, g_max: int = 0
               ) -> Union[Tuple[Tuple, Tuple, Tuple], Tuple[Tuple, Tuple, Tuple, Tuple]]:
    """
    Split the time-series dataframe into training set, test set, and optionally
    validation set. For each set, use make_supervise() to split between predictor
    columns (X) and target column (y). Using the Walk-Forward Validation method.
    Source: https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/

    Parameters
    ----------
    df : The pd.DataFrame to be split. Can be either univariate or multivariate.
    target : The target column name of df.
    n_test : The num of samples in testset, which will be carved from the end of df.
    n_val : The num of samples in valset, which will be carved from the end of df after
            testset is taken. The remaining samples constitute the training set.
    n_input : The num of input timesteps to be fed into the DNN model.
    n_output : The num of output timesteps forecasted by DNN model. The default is 1.
    n_feature : The number of features df contains. The default is 1 for univariate.
    g_min : The min % of samples used as gap between various sets. The default is 0.
    g_max : The max % of samples used as gap between various sets. The default is 0.
            See: http://www.zhengwenjie.net/tscv/

    Returns
    -------
    orig : The supervised format of the dataframe.
    train : The training set from orig.
    val : The validation set from orig. This part will be omitted if n_val=0.
    test : The test set from orig.

    """
    X, y, t = _make_supervised(data, target, n_input, n_output, n_feature)
    gap = randint(int(g_min*len(X)), int(g_max*len(X)))

    X_train0, X_test, y_train0, y_test = gap_train_test_split(X, y, test_size=n_test,
                                                              gap_size=gap)
    _, _, t_train0, t_test = gap_train_test_split(X, t, test_size=n_test,
                                                  gap_size=gap)
    # Train, test split
    if n_val == 0:
        orig = (X, y, t)
        train = (X_train0, y_train0, t_train0)
        test = (X_test, y_test, t_test)
        return orig, train, test
    # Train, val, test split
    else:
        X_train, X_val, y_train, y_val = gap_train_test_split(X_train0, y_train0, test_size=n_val,
                                                              gap_size=gap)
        _, _, t_train, t_val = gap_train_test_split(X_train0, t_train0, test_size=n_val,
                                                    gap_size=gap)
        orig = (X, y, t)
        train = (X_train, y_train, t_train)
        val = (X_val, y_val, t_val)
        test = (X_test, y_test, t_test)
        return orig, train, val, test


def _make_supervised(data: pd.DataFrame, target: str, n_input: int, n_output: int,
                    n_feature: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose the time-series dataframe into seasonal, trend, and residual
    components using statsmodels. This is used to detrend or deseasonalize
    the time-series so only the residuals will be modeled by DNN.
    Source: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
            and Machine Learning Mastery (Deep Learning in Time-Series Forecasting book)

    Parameters
    ----------
    df : The pd.DataFrame to be decomposed. Can be either univariate or multivariate.
    target : The target column name of df.
    n_input : The type of decomposition. Options are 'deseasonalize', 'detrend', or 'both'.
    n_output : The type of model component. Options are 'additive' or 'multiplicative'.
    n_feature : The type of model component. Options are 'additive' or 'multiplicative'.

    Returns
    -------
    X_sv : X dataset in supervised format.
    y_sv : y dataset in supervised format.
    t_sv : DateTimeIndex in supervised format.

    """
    X, y, t = list(), list(), list()
    in_start = 0
    # Stepover the entire dataset one timestep at a time
    for _ in range(len(data)):
        # Define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_output
        # Ensure there is enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end].to_numpy()
            x_input = x_input.reshape((len(x_input), n_feature))
            X.append(x_input)
            y_output = data[target][in_end:out_end].to_numpy()
            y_output = y_output.reshape(y_output.shape[0])
            y.append(y_output)
            t_index = data.index[in_end].to_numpy()
            t.append(t_index)
        # Move along one time step
        in_start += 1

    X_sv, y_sv, t_sv = np.array(X), np.array(y), np.array(t)
    return X_sv, y_sv, t_sv
