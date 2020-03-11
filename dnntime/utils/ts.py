import numpy as np
from typing import Tuple


class timesteps:

    def __init__(self, freq: str):
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


def interval_to_freq(time_interval: str) -> Tuple[str, int]:
    time_interval = time_interval.strip().lower()
    freq = ''
    if time_interval in ['seconds', 'second', 'sec', 's']:
        freq = 'S'
        seasonal_period = 60
    elif time_interval in ['minutes', 'minute', 'min', 't']:
        freq = 'T'
        seasonal_period = 60
    elif time_interval in ['hours', 'hour', 'hourly', 'hr', 'h']:
        freq = 'H'
        seasonal_period = 24
    elif time_interval in ['days', 'day', 'daily', 'd']:
        freq = 'D'
        seasonal_period = 30
    elif time_interval in ['weeks', 'week', 'weekly', 'w']:
        freq = 'W'
        seasonal_period = 52
    elif time_interval in ['months', 'month', 'mth', 'm']:
        freq = 'M'
        seasonal_period = 12
    elif time_interval in ['qtr', 'quarter', 'quar', 'q']:
        freq = 'Q'
        seasonal_period = 4
    elif time_interval in ['years', 'year', 'annual', 'y', 'a']:
        freq = 'Y'
        seasonal_period = 1
    else:
        raise ValueError("Parameter time_interval not recognized.")
        return

    return freq, seasonal_period


def interval_to_timesteps(period: str, freq: str) -> int:
    ts = timesteps(freq)
    steps = 0
    if period == '':
        steps = steps  # if period is not passed in, then leave steps as 0 
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