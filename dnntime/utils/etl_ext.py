import numpy as np
import pandas as pd
import datetime
import pytz
from typing import Union
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
        df_clean.apply(lambda x: x.astype(str).replace('[^\d\.]', '')
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
