import numpy as np
import pandas as pd
from random import randint
import datetime
import pytz
from typing import Union, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler, power_transform
from statsmodels.tsa.seasonal import seasonal_decompose
from tscv import gap_train_test_split
# Timesteps
from .ts import interval_to_freq


def load_data(data: Union[str, pd.DataFrame], dt_col: str, deln: str = ','
              ) -> pd.DataFrame:
    """
    Loads an input data into a time-series specific pandas dataframe, which
    sets its DateTimeIndex as dt_col. The input data can be either an existing
    raw dataframe or a filename str that contains its full path.

    Parameters
    ----------
    data : The input data to be loaded. Can be a generic pd.DataFrane or file str.
    dt_col : The datetime or time-series column name.
    deln : The delineator type for file extraction. The default is ',' for *.csv files.

    Returns
    -------
    df_load : The loaded time-series specific pd.DataFrane with DateTimeIndex.

    """
    # If input data is in a file format:
    if isinstance(data, str):
        print(f"    - Now loading data from filepath '{data}' ...")
        try:
            df_load = pd.read_csv(data, index_col=dt_col, parse_dates=True, sep=deln)
        except FileNotFoundError:
            print("File could not be loaded. Check the path or filename and try again")
            return None
        except ValueError:
            print("Load failed. Please specify ts_column= and target=, or set dt_index=False.")
            return None
        print(f"    - File loaded successfully. Shape of dataset = {df_load.shape}")
    # Else if input data is already in DatFrame format
    elif isinstance(data, pd.DataFrame):
        df_load = data.set_index(dt_col)
        print("Input is data frame. Performing Time Series Analysis")

    return df_load


def clean_data(df: pd.DataFrame, target: str, time_interval: str, timezone:
               str = '', allow_neg: bool = True, all_num: bool = False, fill:
               str = 'linear', learning_type: str = 'reg') -> pd.DataFrame:
    """
    Cleans the input dataframe, which contains the following steps:
        1) Sort DateTimeIndex is asc order just in case it hasn't been done.
        2) Check no duplicate time-series point, otherwise, keep only first.
        3) Get freq based on the user-specified time interval.
        3) Add freq to time-series col or index if it doesn't exist.
        4) Convert all cols in DataFrame into float64 and removing any special char prior.
        5) Convert only target col type to float64 if not already done by all_num.
        6) Replace any negative number as NaN if target negative numbers are not allowed.
        7) Fill the missing target values via given interpolation in-place.
        8) Drop all columns that still have NaN values (as all their values are NaNs).
        9) Double-check whether there are still any missing NaN in the dataset.

    Parameters
    ----------
    df : The pd.DataFrame to be cleaned. Can be either univariate or multivariate.
    target : The target column name of df.
    time_interval : The time period between a data point and its adjacent one.
    timezone : The timezone of df is specified. The default is '' for no timezone.
    as_freq : The frequency char of df. Can be obtained by interval_to_freq() in ts.py.
    allow_neg : Whether to permit allow negative numbers or not. The default is True.
    all_num : Whether entire df must only be numeric, prohibiting categorical
              columns. The default is False.
    fill : Type of fill used to interpolate missing values. The default is 'linear'.
    learning_type : The ML output type. The options are 'reg' (default) or 'class'.

    Returns
    -------
    df_clean : The cleaned dataframe.

    """
    df_clean = df.copy()

    # 1) Sort DateTimeIndex is asc order just in case it hasn't been done
    df_clean.sort_index(inplace=True)
    print("    - Sorted DateTimeIndex in asc order (just in case).")
    # 2) Check no duplicate time-series point, otherwise, keep only first one
    if df_clean.index.duplicated().sum() > 0:
        df_clean = df_clean.loc[~df_clean.index.duplicated(keep='first')]
        print("    - WARNING: there were duplicate times! Kept only one and rest are discarded.")
    else:
        print("    - Checked that there are no duplicate times.")
    # 3) Get freq based on the user-specified time interval.
    freq = interval_to_freq(time_interval)
    print(f"    - Frequency has been set to {freq}.\n")
    # 4) Add freq to time-series col or index if it doesn't exist
    tz_offset = 0
    if timezone != '':
        tz = datetime.datetime.now(pytz.timezone(timezone))
        tz_offset += int(tz.utcoffset().total_seconds()/60/60)
    if not isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean.index = pd.to_datetime(df_clean.index, utc=True)
    if df_clean.index.freq is None:
        f_idx = pd.date_range(start=df_clean.index.min(), end=df_clean.index.max(), 
                              freq=freq).tz_localize(None)
        # set_index if current index and new freq indexes have same len, reindex otherwise
        if len(df_clean) == len(f_idx):
            df_clean.set_index(f_idx, inplace=True)
        else:
            df_clean = df_clean.reindex(f_idx)
        print(f"    - Added freq '{freq}' to DateTimeIndex.")
    if tz_offset != 0:
        df_clean.index = df_clean.index.shift(tz_offset)
        print("    - Removed timezone by converting to UTC and then reshifting back. ")
    # 4) Convert all cols in DataFrame into float64 and removing any special char prior
    if all_num:
        df_clean.apply(lambda x: x.astype(str).replace('[^\d\.]', '') \
                                              .astype(np.float64))
    # 5) Convert only target col type to float64 if not already done by all_num
    if learning_type == 'reg' and df_clean[target].dtype.kind != 'f' and not all_num:
        df_clean[target] = df_clean[target].astype(str).replace('[^\d\.]', '') \
                                           .astype(np.float64)
        print(f"    - Converted target={target} col to float64 type.")
    # 6) Replace any negative number as NaN if target negative numbers are not allowed
    if not allow_neg:
        df_clean[df_clean < 0] = np.NaN
        print(f"    - Since negative values are unpermitted, all negative "
              "values found in dataset are converted to NaN.")
    # 7) Fill the missing target values via given interpolation in-place
    if fill != '' and df_clean.isnull().sum().sum() > 0:
        df_clean.interpolate(fill, inplace=True)
        print(f"    - filled any NaN value via {fill} interpolation.")
    # 8) Drop all columns that still have NaN values (as all their values are NaNs)
    before_cols = set(df_clean.columns.tolist())
    df_clean.dropna(axis=1, how='all', inplace=True)
    after_cols = set(df_clean.columns.tolist())
    if len(before_cols-after_cols) > 0:
        print("    - The following columns have all NaN values: "
              f"{list(before_cols-after_cols)}. They are therefore dropped.")
    # 9) Double-check whether there are still any missing NaN in the dataset
    if df_clean.isnull().sum().sum() > 0:
        print(f"    - WARNING: Missing values still exist in the dataset.")

    return df_clean


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
    X, y, t = make_supervised(data, target, n_input, n_output, n_feature)
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


def make_supervised(data: pd.DataFrame, target: str, n_input: int, n_output: int,
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
