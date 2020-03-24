import pandas as pd
# Matplotlib and Seaborn Visualizations
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set(style="white", color_codes=True)
# Plotly Visualizations
import plotly.express as px
import plotly.graph_objects as go
# Add commas to y-axis tick values for graphs
formatter = ticker.StrMethodFormatter('{x:,.0f}')
from typing import List, Tuple
# TSA from Statsmodels
from statsmodels.tsa.api import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, acovf, pacf, pacf_yw, pacf_ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Facebook's prophet
from fbprophet import Prophet
# EDA general modules
from .eda_gen import ts_plot


def ets_decomposition_plot(df: pd.DataFrame, dt_col: str, target: str,
                           title: str, y_label: str, x_label: str = "Date",
                           line_width: float = 0.1, model: str = 'additive',
                           figsize: Tuple[int, int] = (20, 8), plotly:
                           bool = False, prophet: bool = False) -> None:
    """
    Plot the error, trend, and seasonality (ETS) decompositions of the input
    time-series dataframe using statsmodels.

    Parameters
    ----------
    df : The pd.DataFrame to be plotted. Currently must be univariate.
    dt_col : Datetime column or the time-series axis.
    target : Target column or the y-axis.
    title : Title of the displayed plot.
    y_label : y_label of the displayed plot.
    x_label : x_label of the displayed plot. The default is 'Date'.
    line_width : How solid the plotline is. The default is 0.1.
    model : Types of seasonal components. Either 'additive' (default) or 'multiplicative'.
    figsize : The dimension size of the plot display. The default is (20, 8).
    plotly : Whether the plot using plotly or matplotlib+seaborn. The default is False.
    prophet : Whether the include results from Facebook's Prophet. The default is False.

    """
    pd.plotting.register_matplotlib_converters()
    ets = seasonal_decompose(df[target], model)

    # Observed
    ts_plot(ets.observed.to_frame(), dt_col, target,
            title=f"{title} (Observed)",
            x_label=x_label,
            y_label=f"Observed {y_label}",
            line_width=line_width,
            figsize=figsize,
            plotly=plotly
            )
    # Trend
    ts_plot(ets.trend.to_frame(), dt_col, 'trend',
            title=f"{title} (Trend)",
            x_label=x_label,
            y_label=f"Trend {y_label}",
            line_width=line_width*4,
            figsize=figsize,
            plotly=plotly
            )
    # Seasonality
    ts_plot(ets.seasonal.to_frame(), dt_col, 'seasonal',
            title=f"{title} (Seasonality)",
            x_label=x_label,
            y_label=f"Seasonal {y_label}",
            line_width=line_width/2,
            figsize=figsize,
            plotly=plotly
            )
    # Residual
    ts_plot(ets.resid.to_frame(), dt_col, 'resid',
            title=f"{title} (Residual)",
            x_label=x_label,
            y_label=f"Residual {y_label}",
            line_width=line_width,
            figsize=figsize,
            plotly=plotly
            )
    print()

    if prophet:
        dfp = df.reset_index()
        dfp.columns = ['ds', 'y']
        model = Prophet()
        model.fit(dfp);
        forecast = model.predict(dfp)
        model.plot_components(forecast);
        print()

    return ets


def acf_pacf_plot(df: pd.DataFrame, target: str, title: str = "",
                  lags: List[int] = [24], figsize: Tuple[int, int] = (20, 8)
                  ) -> None:
    """
    Autocorrelation Function (ACF) and Partial-Autocorrelation (PACF) Analysis.
    Source: https://www.kaggle.com/nicholasjhana/univariate-time-series-forecasting-with-keras

    Parameters
    ----------
    df : The pd.DataFrame to be plotted. Currently must be univariate.
    target : Target column or the y-axis.
    title : Title of the displayed plot.
    lags : Num of timesteps between each tick. The default is 24, or a day for hourly freq.
    figsize : The dimension size of the plot display. The default is (20, 8).

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
    Test whether dataframe is stationary using the Augmented Dickey Fuller (ADF)
    test found in statsmodel.
    Source: https://www.insightsbot.com/augmented-dickey-fuller-test-in-python/

    Parameters
    ----------
    df : The pd.DataFrame to test for stationarity. Currently must be univariate.
    alpha : The number that is (1 - confidence interval). The default is 0.05 for 95% CI.
    criterion : The criterion used to automatically determine lag. The default
                is 'AIC' or Akaike information criterion.

    Returns
    -------
    stationary : Whether the df stationary or not.

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
    # Add Critical Values
    for key, value in adf_test[4].items():
        results[f'      Critical Value ({key})'] = value
    print("    - Augmented Dickey-Fuller Test Results:\n")
    print(results.to_string() + "\n")

    return stationary
