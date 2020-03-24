import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set(style="white", color_codes=True)
# Plotly Visualisations
import plotly.express as px
import plotly.graph_objects as go
# Add commas to y-axis tick values for graphs
formatter = ticker.StrMethodFormatter('{x:,.0f}')
from typing import List, Tuple


def ts_plot(df: pd.DataFrame, dt_col: str, target: str, title: str, y_label: str,
            x_label: str = "Date", width: int = 10, height: int = 4,
            line_width: float = 0.1) -> None:
    """
    Plot the input time-series dataframe using plotly visualization.

    Parameters
    ----------
    df : The pd.DataFrame to be plotted. Currently must be univariate.
    dt_col : Datetime column or the time-series axis.
    target : Target column or the y-axis.
    title : Title of the displayed plot.
    y_label : y_label of the displayed plot.
    x_label : x_label of the displayed plot. The default is 'Date'.
    width : The width of the plot display. The default width is 10.
    height : The height of the plot display. The default height is 4.
    line_width : How solid the plot line is. The default is 0.1.

    """
    if isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex):
        df = df.copy()  # prevents from modifying original df
        df[dt_col] = df.index
    fig = px.line(df, x=dt_col, y=target, title=title)
    fig.update_traces(line=dict(width=line_width))
    fig.update_layout(autosize=False,
                      width=100*width,
                      height=100*height,
                      margin=go.layout.Margin(
                          l=50,
                          r=50,
                          b=40,
                          t=50,
                          pad=0
                      ),
                      xaxis_title=x_label,
                      yaxis_title=y_label)
    fig.show()
    print()


def ts_sub_plot(df: pd.DataFrame, dt_col: str, target: str, title: str,
                y_label: str, x_label: str = "Date", split: str = 'y',
                line_width: float = 0.1) -> None:
    """
    Plot the input time-series dataframe using plotly visualization into
    multiple subplots. Uses ts_sub_split() to demarcate the time-series.

    Parameters
    ----------
    df : The pd.DataFrame to subplot. Currently must be univariate.
    dt_col : Datetime column or the time-series axis.
    target : Target column or the y-axis.
    title : Title of the displayed plot.
    y_label : y_label of the displayed plot.
    x_label : x_label of the displayed plot. The default is 'Date'.
    split : How the time-series is demarcated. The default is 'y' for year.
    line_width : How solid the plotline is. The default is 0.1.

    """
    sub_ts, idx = ts_sub_split(df, split=split)
    for i, sub in enumerate(sub_ts):
        if split == 'y':
            year = idx[i]
            ts_plot(sub, dt_col, target,
                    title=f"{title} for {year}",
                    y_label=y_label,
                    x_label=x_label,
                    line_width=line_width
                    )
        elif split == 'm':
            year, month = idx[i][0], idx[i][1]
            ts_plot(sub, dt_col, target,
                    title=f"{title} for {year}-{month:02d}",
                    y_label=y_label,
                    x_label=x_label,
                    line_width=line_width
                    )
        elif split == 'q':
            year, month = idx[i][0], idx[i][1]
            ts_plot(sub, dt_col, target,
                    title=f"{title} from {year}-{month:02d} to {year}-{month+2:02d}",
                    y_label=y_label,
                    x_label=x_label,
                    line_width=line_width
                    )


def ts_sub_split(df: pd.DataFrame, split: str = 'y', offset: int = 0
                 ) -> Tuple[List, List]:
    """
    Demarcate the time-series dataframe for ts_sub_plot() func to use to plot
    multiple sub time-series.

    Parameters
    ----------
    df : The pd.DataFrame to subplot. Currently must be univariate.
    split : How the time-series is demarcated. The default is 'y' for year.
    offset : How much to shift this sub time-series period relevant to calendar.
             The default is 0 or split at the beginning of each calendar period.

    Returns
    -------
    sub : The list of all sub time-series.
    idx : The key for each sub time-series depending on how series is split.

    """
    sub = []
    if isinstance(split, str):
        # annual
        if(split.lower() in ['a', 'y']):
            idx = sorted(set(df.index.year))
            for year in idx:
                sub.append(df[f'{year}':f'{year}'])
        # monthly and quarterly
        elif(split.lower() in ['q', 'm']):
            idx = sorted(set(zip(df.index.year, df.index.month)))
            iex = idx.copy()
            for year, month in idx:
                # ipdb.set_trace()
                if split.lower() == 'm':
                    sub.append(df[f'{year}-{month}':f'{year}-{month}'])
                elif split.lower() == 'q' and month%3 == 1:
                    sub.append(df[f'{year}-{month}':f'{year}-{month+2}'])
                else:
                    iex.remove((year, month))
            idx = iex
        # daily and weekly
        elif(split.lower() in ['d', 'w']):
            idx = sorted(set(zip(df.index.year, df.index.month, df.index.day)))
            for year, month, day in idx:
                if split.lower() == 'd':
                    sub.append(df[f'{year}-{month}-{day}':f'{year}-{month}-{day}'])
            # elif split.lower() == 'w' and day%7 == 1:
                # sub.append(df[f'{year}-{month}-{day}':f'{year}-{month}-{day+6}'])

    return sub, idx