import ipdb
import numpy as np
import pandas as pd
from random import randint
import datetime
import pytz
import copy
from typing import Union, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler, power_transform
from statsmodels.tsa.seasonal import seasonal_decompose
from tscv import gap_train_test_split


def load_data(data: Union[str, pd.DataFrame], dt_col: str, deln: str = ','
              ) -> pd.DataFrame:
    """
    This function loads a given filename into a pandas dataframe and sets the
    ts_column as a Time Series index. Note that filename should contain the full
    path to the file.
    """
    # if dt_index:
    # If input data is ina file format:
    if isinstance(data, str):
        print(f"    - Now loading data from filepath '{data}' ...")
        try:
            df = pd.read_csv(data, index_col=dt_col, parse_dates=True, sep=deln)
        except FileNotFoundError:
            print("File could not be loaded. Check the path or filename and try again")
            return
        except ValueError:
            print("Load failed. Please specify ts_column= and target=, or set dt_index=False.")
            return
        print(f"    - File loaded successfully. Shape of dataset = {df.shape}")
    # Else if input data is already in DatFrame format
    elif isinstance(data, pd.DataFrame):
        df = data.set_index(dt_col)
        print("Input is data frame. Performing Time Series Analysis")

    return df


def clean_data(df: pd.DataFrame, target: str, timezone: str = '', as_freq: 
               str = 'H', allow_neg: bool = True, fill: str = 'linear',
               learning_type: str = 'reg') -> pd.DataFrame:

    dtc = df.copy()

    # 1) Sort DateTimeIndex is asc order just in case it hasn't been done
    dtc.sort_index(inplace=True)
    print("    - Sorted DateTimeIndex in asc order (just in case).")
    # 2) Check no duplicate time-series point, otherwise keep only first one
    if dtc.index.duplicated().sum() > 0:
        dtc = dtc.loc[~dtc.index.duplicated(keep='first')]
        print("    - WARNING: there were duplicate times! Kept onlyt one and rest are discarded.")
    else:
        print("    - Checked that there are no duplicate times.")
    # 3) Add freq to time-series col or index if it doesn't exist
    tz_offset = 0
    if timezone != '':
        tz = datetime.datetime.now(pytz.timezone(timezone))
        tz_offset += tz.utcoffset().total_seconds()/60/60
    if not isinstance(dtc.index, pd.DatetimeIndex):
        dtc.index = pd.to_datetime(dtc.index, utc=True)
    if dtc.index.freq is None:
        f_idx = pd.date_range(start=dtc.index.min(), end=dtc.index.max(), freq=as_freq) \
                  .tz_localize(None)
        # set_index if current index and new freq indexes have same len, reindex otherwise
        if len(dtc) == len(f_idx):
            dtc.set_index(f_idx, inplace=True)
        else:
            dtc = dtc.reindex(f_idx)
        print(f"    - Added freq '{as_freq}' to DateTimeIndex.")
    if tz_offset != 0:
        dtc.index = dtc.index.shift(tz_offset)
        print("    - Removed timezone by converting to UTC and then reshifting back. ")
    # 4) Convert target col type to float is not already and removing any puncuations
    if learning_type == 'reg' and dtc[target].dtype.kind != 'f':
        dtc[target] = dtc[target].str.replace('[^\d\.]', '').astype(np.float32)
        print(f"    - Converted target={target} col to float3 type.")
    # 5) Replace any negative number as NaN if target negative numbers are not allowed
    if not allow_neg:
        dtc[target][dtc[target] < 0] = np.NaN
        print(f"    - Since negative values are unpermitted, all negative values found in dataset are converted to NaN.")
    # 6) Fill the missing targetvalues via given interpolation in-place
    if dtc[target].isnull().sum() > 0:
        dtc[target].interpolate(fill, inplace=True)
        print(f"    - filled any NaN value via {fill} interpolation.")

    return dtc


def normalize(df: pd.DataFrame, target: str, scale: str = 'minmax'
              ) -> Tuple[pd.DataFrame, Union[MinMaxScaler, StandardScaler]]:
    if scale == 'minmax':
        scaler = MinMaxScaler()
    elif scale == 'standard':
        scaler = StandardScaler()
    df_norm = df.copy()
    df_norm[target] = scaler.fit_transform(df)
    return df_norm, scaler


def log_power_transform(df: pd.DataFrame, target: str, method: str = 'box-cox',
                        standardize: bool = True) -> Tuple[pd.DataFrame, str]:
    stan = ''
    if standardize:
        stan = 'Standardized'
    if method == 'log':
        return df.transform(np.log), str(method.title() + ' ' + stan)
    else:
        df_pwr = df.copy()
        df_pwr[target] = power_transform(df, method=method, standardize=standardize)
        return df_pwr, str(method.title() + ' ' + stan)


def decompose(df: pd.DataFrame, target: str, decom_type: str = 'deseasonalize',
              decom_model: str = 'additive') -> Tuple[pd.DataFrame, np.ndarray]:
    ets = seasonal_decompose(df[target], decom_model)
    ets_idx = ets.trend[ets.resid.notnull()].index
    if decom_type == 'deseasonalize' and decom_model == 'additive':
        trans = ets.trend[ets_idx] + ets.resid[ets_idx]
        removed = ets.seasonal[ets_idx]
    elif decom_type == 'deseasonalize' and decom_model == 'multiplicative':
        trans = ets.trend[ets_idx] * ets.resid[ets_idx]
        removed = ets.seasonal[ets_idx]
    elif decom_type == 'detrend' and decom_model == 'additive':
        trans = ets.seasonal[ets_idx] + ets.resid[ets_idx]
        removed = ets.trend[ets_idx]
    elif decom_type == 'detrend' and decom_model == 'multiplicative':
        trans = ets.seasonal[ets_idx] * ets.resid[ets_idx]
    elif decom_model == 'additive':
        trans = ets.resid[ets_idx]
        removed = ets.trend[ets_idx] + ets.seasonal[ets_idx]
    else:
        trans = ets.resid[ets_idx]
        removed = ets.trend[ets_idx] * ets.seasonal[ets_idx]

    df_trans = trans.to_frame(name=target)
    return df_trans, removed


def split_data(data: pd.DataFrame, n_test: int, n_val: int, n_input: int,
               n_output: int = 1, g_min: int = 0, g_max: int = 0
               ) -> Union[Tuple[Tuple, Tuple, Tuple], Tuple[Tuple, Tuple, Tuple, Tuple]]:
    X, y, t = make_supervised(data, n_input, n_output)
    gap = randint(int(g_min*len(X)), int(g_max*len(X)))
    # ipdb.set_trace()
    X_train0, X_test, y_train0, y_test = gap_train_test_split(X, y, test_size=n_test, \
                                                              gap_size=gap)
    _, _, t_train0, t_test = gap_train_test_split(X, t, test_size=n_test, \
                                                  gap_size=gap)
    # Train, test split
    if n_val == 0:
        orig = (X, y, t)
        train = (X_train0, y_train0, t_train0)
        test = (X_test, y_test, t_test)
        return orig, train, test
    # Train, val, test split
    else:
        X_train, X_val, y_train, y_val = gap_train_test_split(X_train0, y_train0, test_size=n_val, \
                                                              gap_size=gap)
        _, _, t_train, t_val = gap_train_test_split(X_train0, t_train0, test_size=n_val, \
                                                    gap_size=gap)
        orig = (X, y, t)
        train = (X_train, y_train, t_train)
        val = (X_val, y_val, t_val)
        test = (X_test, y_test, t_test)
        return orig, train, val, test


# Credit: Machine Learning Mastery (Deep Learning in Time-Series Forecasting)
def make_supervised(data: pd.DataFrame, n_input: int, n_output: int
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # flatten data
    # data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y, t = list(), list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_output
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end].to_numpy()
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y_output = data[in_end:out_end].to_numpy()
            y_output = y_output.reshape(y_output.shape[0])
            y.append(y_output)
            t_index = data.index[in_end].to_numpy()
            t.append(t_index)
        # move along one time step
        in_start += 1
    return np.array(X), np.array(y), np.array(t)