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
        EDABlock is a Block that performs visual and/or statisically analysis
        from the input data_dict, typically on the most recent data. Unlike
        ETLBlock, the EDABlock does not modify or transform the data but rather
        plot graphs or output statistical tests for the user to see, and/or
        output statistical params that subsequent Blocks can use. Neither data_dict
        not model_dict will be affected or mofified by an EDABlock's function.
        Here are the following EDA operations:
            1) General: .
            2) Statistical: .

        Parameters
        ----------
        data_dict: The recorded data transformation.
        params: Any additional parameters passed from the results of previous Blocks.

        """
        super().__init__(data_dict, params)

    def run_block(self, config: Dict) -> Dict:
        """
        Execute the EDABlock function on the data_dict and/or model_dict based on
        the user's config YAML file as well preexisting params from initialization.

        Parameters
        ----------
        config: The specified config block from the user YAML file.

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
        EDA operation that loads data from a file source into a pd.DataFrame.
        This is the only op where input data_dict is expected to be empty. All
        other ops expect the data_dict to have some existing data as input.

        Parameters
        ----------
        key_name : The key for this config block, usually as 'extract'{#num}.
        config : Specifies the source file path and how it is delineated.

        Returns
        -------
        params : The extracted dataset from the given source file.
        
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
        EDA operation that loads data from a file source into a pd.DataFrame.
        This is the only op where input data_dict is expected to be empty. All
        other ops expect the data_dict to have some existing data as input.

        Parameters
        ----------
        key_name : The key for this config block, usually as 'extract'{#num}.
        config : Specifies the source file path and how it is delineated.

        Returns
        -------
        params : The extracted dataset from the given source file.

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
