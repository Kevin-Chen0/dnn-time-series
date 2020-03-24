# -*- coding: utf-8 -*-
import ast
import pandas as pd
import re
from typing import Dict
# Blocks base modules
from .base import Block, CheckpointDict
# Tests
from ..tests import eda_test as test
# Utils
from ..utils.eda_gen import ts_plot
from ..utils.eda_stat import ets_decomposition_plot, acf_pacf_plot, \
                             adf_stationary_test


class EDABlock(Block):

    def __init__(self, data_dict: CheckpointDict, params: Dict) -> None:
        """
        EDABlock inherits from Block. It performs any type of visual and/or
        statisical analysis of data. It takes the latest data from the input
        data_dict, plot the data, and print out any time-series statistical
        information that might be useful for the user.

        Parameters
        ----------
        data_dict: The record of data transformations.
        params: Any additional parameters passed from the results of previous Blocks.

        """
        super().__init__(data_dict, params)

    def run_block(self, config: Dict) -> Dict:
        """
        Executes the EDABlock function on the data_dict based on the user's config
        YAML file as well preexisting params from initialization. Unlike ETLBlock,
        the EDABlock does not modify or transform the data but rather plot graphs 
        or output statistical tests for the user to see. It also optionally output
        statistical params that subsequent Blocks can use. Therefore, neither
        data_dict nor model_dict should be modified by an EDABlock execution.
        Here are the following EDA operations:
            1) General: Performs general plots and visualizations.
            2) Statistical: Perform time-series specific statistical analyses.

        Parameters
        ----------
        config: The specified config block from the user YAML file.

        Returns
        -------
        self.params : Any additional generated parameters for subsequent Blocks.

        """
        super().run_block(config)
        test.validate_eda(self.params['key'], config)

        for key in config.keys():
            keyword = re.sub('[^a-zA-Z]+', '', key)
            params = eval(f'self.run_{keyword}(key, config[key])')
            # Check if the params output is not empty
            if not bool(params):
                self.params.update(params)
                print(f"Added the following parameters: {params}")

        self.params['step_number'] += 1
        return self.params

    def run_general(self, key_name: str, config: Dict) -> Dict:
        """
        EDA operation that performs general plots, subplots, and other types
        of visualizations. This can be used to check whether the latest data has
        been properly cleaned, massaged, or transformed.

        Parameters
        ----------
        key_name : The key for this config block, usually as 'general'{#num}.
        config : Specifies the labels and dimensions of the plots.

        Returns
        -------
        params : Any plot output params could be useful in subsequent Blocks.
        
        """
        # VALIDATE necessary general parameters before running procedure
        assert self.data_dict.current_key is not None, "Data_dict is empty, " + \
            "run_statistical() needs existing data to run."
        test.validate_general(key_name, config, self.params)
        # Initialize df as None, therefore, if it fails, then it will return None
        params = {}

        try:
            # INITIALIZE general config variables
            title = config['title']
            x_label = config['x_label']
            y_label = config['y_label']
            figsize = (20, 8)
            if 'figsize' in config.keys():
                figsize = ast.literal_eval(config['figsize'])
            plotly = False
            if 'plotly' in config.keys():
                plotly = config['plotly']
            # Get most current dataset from data_dict
            key = self.data_dict.current_key
            df = self.data_dict.get()[key]
            dt_col = self.params['dt_col']  # column name containing time-series
            target = self.params['target']  # column name containing y_value
            # Prevent errors
            pd.plotting.register_matplotlib_converters()

            # Define data with the defined plot labels in the config file
            print("Plot the entire time-series data:\n")
            ts_plot(df, dt_col, target,
                    title=title,
                    x_label=x_label,
                    y_label=y_label,
                    figsize=figsize,
                    plotly=plotly
                    )
        except Exception as e:
            print(e)
            return None

        return params

    def run_statistical(self, key_name: str, config: Dict) -> Dict:
        """
        EDA operation that performs time-series or other types of statistical
        analyses. This also includes any plot visualizations that come out of
        this analyses. This can be useful to signal how the data may need to
        be further transformed in subsequent Block steps.

        Parameters
        ----------
        key_name : The key for this config block, usually as 'statistical'{#num}.
        config : Specifies the labels and dimensions of the plots.

        Returns
        -------
        params : Any stat output params could be useful for subsequent Blocks.
        
        """
        # VALIDATE necessary statistical parameters before running procedure
        assert self.data_dict.current_key is not None, "Data_dict is empty, " + \
            "run_statistical() needs existing data to run."
        test.validate_statistical(key_name, config, self.params)
        # Initialize df as None, therefore, if it fails, then it will return None
        params = {}

        try:
            # INITIALIZE general config variables
            title = config['title']
            x_label = config['x_label']
            y_label = config['y_label']
            ci = config['confidence_interval']
            figsize = (20, 8)
            if 'figsize' in config.keys():
                figsize = ast.literal_eval(config['figsize'])
            plotly = False
            if 'plotly' in config.keys():
                plotly = config['plotly']
            # Get most current dataset from data_dict
            key = self.data_dict.current_key
            df = self.data_dict.get()[key]
            dt_col = self.params['dt_col']  # column name containing time-series
            target = self.params['target']  # column name containing y_value
            space = self.params['space']
            stepn = self.params['step_number']

            self.print_bold(f"{stepn}.1) {space}Testing stationarity "
                            "using Augmented Dickey-Fuller (ADF).", n_after=1)
            stationarity = adf_stationary_test(df, 1-ci)
            if stationarity:
                print(f"Current data is stationary with {ci*100}% "
                      "confidence interval.\n")
            else:
                print(f"Current data is non-stationary with {ci*100}% "
                      "confidence interval.\n")

            self.print_bold(f"{stepn}.2) {space}Printing out ETS decomposition "
                            "plot.", n_before=1, n_after=1)
            # ets = ets_decomposition_plot(ts_df, ts_column, target, title, y_label,
            #                        prophet=True);
            ets = ets_decomposition_plot(df, dt_col, target,
                                         title=title,
                                         x_label=x_label,
                                         y_label=y_label,
                                         figsize=figsize,
                                         plotly=plotly);

            self.print_bold(f"{stepn}.3) {space}Plot out ACF/PACF graph..",
                            n_before=1, n_after=1)
            title = "Total Electricity Demand"
            lags_7 = 24*7  # 7-days lag
            lags_30 = 24*30  # 30-days lag
            # lags_90 = 24*90  # 90-days lag
            acf_pacf_plot(df, target, title, lags=[lags_7, lags_30],
                          figsize=figsize)

            # print_bold(f"{stepn}.4) {space}Expotential Smoothing Holt-Winters.",
            #            ui, n_before=1, n_after=1
            #            )

            # print_bold(f"{stepn}.5) {space}ARIMA.", ui, n_before=1)

        except KeyError:
            print("'analyze' section is omitted in config, therefore skipping this step.")
            return None

        return params
