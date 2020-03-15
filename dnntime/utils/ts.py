import numpy as np


class timesteps:

    def __init__(self, freq: str) -> None:
        """
        Initializes timesteps class, which can output a number of timesteps with
        ease given the desired period duration in text. Contains mapping of num
        of timesteps to a stated period, relative to each other.

        Parameters
        ----------
        freq : The time-series interval, see: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        """
        map = {'s': 1,
               't': 1/60,
               'min': 1/60,
               'h': 1/3600,
               'd': 1/(3600*24),
               'b': 1/(3600*24),
               'w': 1/(3600*24*7),
               'm': 1/(3600*24*30),
               'ms': 1/(3600*24*30),
               'q': 1/(3600*24*90),
               'qs': 1/(3600*24*90),
               'a': 1/(3600*24*365),
               'y': 1/(3600*24*365),
               'as': 1/(3600*24*365),
               'ys': 1/(3600*24*365),
               }
        convert = map[freq.lower()]
        # time durations (N timesteps or seq steps)
        sec = convert
        min = sec*60
        hr = min*60
        day = hr*24
        week = day*7
        biwk = day*14
        mth = day*30
        quar = day*90
        year = day*365
        # np.NaN is the converted timestep interval is < 1
        self.SEC = int(sec) if sec >= 1 else np.NaN
        self.MIN = int(min) if min >= 1 else np.NaN
        self.HOUR = int(hr) if hr >= 1 else np.NaN
        self.DAY = int(day) if day >= 1 else np.NaN
        self.WEEK = int(week) if week >= 1 else np.NaN
        self.BIWK = int(biwk) if biwk >= 1 else np.NaN
        self.MONTH = int(mth) if mth >= 1 else np.NaN
        self.QUARTER = int(quar) if quar >= 1 else np.NaN
        self.YEAR = int(year) if year >= 1 else np.NaN


def interval_to_freq(time_interval: str) -> str:
    """
    Convert the natural language-based time period into freq char in order to
    standardize user input.

    Parameters
    ----------
    time_interval : Natural lanaguage time period.

    Returns
    -------
    freq : Appropriate frequency char, see: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

    """
    time_interval = time_interval.strip().lower()
    freq = ''
    if time_interval in ['seconds', 'second', 'sec', 's']:
        freq = 'S'
    elif time_interval in ['minutes', 'minute', 'min', 't']:
        freq = 'T'
    elif time_interval in ['hours', 'hour', 'hourly', 'hr', 'h']:
        freq = 'H'
    elif time_interval in ['days', 'day', 'daily', 'd']:
        freq = 'D'
    elif time_interval in ['weeks', 'week', 'weekly', 'w']:
        freq = 'W'
    elif time_interval in ['months', 'month', 'mth', 'm']:
        freq = 'M'
    elif time_interval in ['qtr', 'quarter', 'quar', 'q']:
        freq = 'Q'
    elif time_interval in ['years', 'year', 'annual', 'y', 'a']:
        freq = 'Y'
    else:
        raise ValueError("Parameter time_interval not recognized.")
        return None

    return freq


def period_to_timesteps(period: str, freq: str) -> int:
    """
    Converts the inputted time period in natural language to a number of timesteps,
    given the inputted freq char that states what are the time intervals length.

    Parameters
    ----------
    period : Time period in natural language.
    freq : Frequency char that should be provided by interval_to_freq().
        DESCRIPTION.

    Returns
    -------
    steps : Number of timesteps. If period len matches freq interval, then steps=1.

    """
    ts = timesteps(freq)
    steps = 0
    if period == '':
        steps = steps  # If period is not passed in, then leave steps as 0
    elif period in ['seconds', 'second', 'sec', 's']:
        steps = ts.SEC
    elif period in ['minutes', 'minute', 'min', 't']:
        steps = ts.MIN
    elif period in ['hours', 'hour', 'hourly', 'hr', 'h']:
        steps = ts.HOUR
    elif period in ['days', 'day', 'daily', 'd']:
        steps = ts.DAY
    elif period in ['weeks', 'week', 'weekly', 'w']:
        steps = ts.WEEK
    elif period in ['biwk', 'biweek', 'biweekly', 'bw', 'b']:
        steps = ts.BIWK
    elif period in ['months', 'month', 'mth', 'm']:
        steps = ts.MONTH
    elif period in ['qtr', 'quarter', 'quar', 'q']:
        steps = ts.QUARTER
    elif period in ['years', 'year', 'annual', 'y', 'a']:
        steps = ts.YEAR
    else:
        raise ValueError("Parameter period or freq not recognized.")
        return
    return steps
