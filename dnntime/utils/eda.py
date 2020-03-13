import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set(style="white", color_codes=True)
# Add commas to y-axis tick values for graphs
formatter = ticker.StrMethodFormatter('{x:,.0f}')
from typing import List, Set, Tuple
# TSA from Statsmodels
from statsmodels.tsa.api import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, acovf, pacf, pacf_yw, pacf_ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Facebook's prophet
from fbprophet import Prophet
# Kevin - Plotly Visualisations
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def ts_plot(df: pd.DataFrame, dt_col: str, target: str, title: str, y_label: str,
            x_label: str = "Date", width: int = 10, height: int = 4,
            line_width: float = 0.1) -> None:
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


def ts_sub_plot(df: pd.DataFrame, dt_col: str, target: str, title: str, y_label: str,
                split: str = 'y', line_width: float = 0.1) -> None:
    sub_ts, idx = ts_sub_split(df, split=split)
    for i, sub in enumerate(sub_ts):
        if split == 'y':
            year = idx[i]
            ts_plot(sub, dt_col, target,
                    title=f"{title} for {year}",
                    y_label=y_label,
                    line_width=line_width
                    )
        elif split == 'm':
            year, month = idx[i][0], idx[i][1]
            ts_plot(sub, dt_col, target,
                    title=f"{title} for {year}-{month:02d}",
                    y_label=y_label,
                    line_width=line_width
                    )
        elif split == 'q':
            year, month = idx[i][0], idx[i][1]
            ts_plot(sub, dt_col, target,
                    title=f"{title} from {year}-{month:02d} to {year}-{month+2:02d}",
                    y_label=y_label,
                    line_width=line_width
                    )


def ts_sub_split(df: pd.DataFrame, split: str = 'y', shift: int = 0) -> Tuple[List, Set]:
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


def ets_decomposition_plot(df: pd.DataFrame, dt_col: str, target: str,
                           title: str, y_label: str, x_label: str = "Date",
                           line_width: float = 0.1, model: str = 'additive',
                           plotly: bool = False, prophet: bool = False) -> None:

    pd.plotting.register_matplotlib_converters()
    ets = seasonal_decompose(df[target], model)  # ['additive', 'multiplicative']
    if plotly:
        # Observed
        ts_plot(ets.observed.to_frame(), dt_col, target,
                title=f"{title} (Observed)",
                x_label=x_label,
                y_label=f"Observed {y_label}",
                line_width=line_width
                )
        # Trend
        ts_plot(ets.trend.to_frame(), dt_col, 'trend',
                title=f"{title} (Trend)",
                x_label=x_label,
                y_label=f"Trend {y_label}",
                line_width=line_width*4
                )
        # Seasonality
        ts_plot(ets.seasonal.to_frame(), dt_col, 'seasonal',
                title=f"{title} (Seasonality)",
                x_label=x_label,
                y_label=f"Seasonal {y_label}",
                line_width=line_width/2
                )
        # Residual
        ts_plot(ets.resid.to_frame(), dt_col, 'resid',
                title=f"{title} (Residual)",
                x_label=x_label,
                y_label=f"Residual {y_label}",
                line_width=line_width
                )
        print()
    else:
        ets.plot()
        plt.show()
        print()

    if prophet:
        dfp = df.reset_index()
        dfp.columns = ['ds','y']
        model = Prophet()
        model.fit(dfp);
        forecast = model.predict(dfp)
        model.plot_components(forecast);
        print()

    return ets


def acf_pacf_plot(df: pd.DataFrame, target: str, title: str = "",
                  lags: List[int] = [24], figsize: Tuple[int, int] = (20,8)
                  ) -> None:
    """
    Autocorrelation Function (ACF) and Partial-Autocorrelation (PACF) Analysis
    Source: https://www.kaggle.com/nicholasjhana/univariate-time-series-forecasting-with-keras

    Parameters
    ----------
    df : DESCRIPTION
    target : DESCRIPTION
    title : DESCRIPTION
    lags : DESCRIPTION
    figsize : DESCRIPTION

    """
    for l in lags:
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        plot_acf(df[target], ax=axs[0], lags=l)
        axs[0].set_title(f"Autocorrelation: {title} ({int(l/24)}-Days Lag)", fontsize=14)
        plot_pacf(df[target], ax=axs[1], lags=l)
        axs[1].set_title(f"Partial Autocorrelation: {title} ({int(l/24)}-Days Lag)", fontsize=14)

        plt.plot();
        plt.show();
        print()


def adf_stationary_test(df: pd.DataFrame, alpha: float = 0.05, criterion:
                        str = 'AIC') -> bool:
    """
    Original source: https://www.insightsbot.com/augmented-dickey-fuller-test-in-python/
    """

    # Run Augmented Dickey-Fuller Test (ADF) statistical test:
    adf_test = adfuller(df, autolag=criterion)
    p_value = adf_test[1]

    if (p_value < alpha):
        stationary = True
    else:
        stationary = False

    results = pd.Series(adf_test[0:4], index=['      ADF Test Statistic',
                                              '      P-Value',
                                              '      # Lags Used',
                                              '      # Observations Used'])
    #Add Critical Values
    for key, value in adf_test[4].items():
        results['      Critical Value (%s)'%key] = value
    print('    - Augmented Dickey-Fuller Test Results:\n')
    print(results.to_string() + '\n')

    return stationary
