# -*- coding: utf-8 -*-
import pandas as pd
import re
from typing import DefaultDict, Dict, Optional, Tuple, Union
# Block Basic Classes
from .base import Block, CheckpointDict
# Tests
from ..tests import eda_test as test
# Utils
from ..utils.eda import ts_plot, ets_decomposition_plot, acf_pacf_plot, \
                       adf_stationary_test


class EDABlock(Block):

    def __init__(self, data_dict: CheckpointDict, params: Dict) -> None:
        super().__init__(data_dict, params)

    def run_block(self, config: Dict) -> Dict:
        """
        

        Parameters
        ----------
        config : Dict

        Returns
        -------
        df : TYPE

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

    def run_general(self, key_name: str, config: Dict) -> Optional[Dict]:
        """
        

        Parameters
        ----------
        config : Dict

        Returns
        -------
        params : TYPE

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
                    y_label=y_label
                    )
        except Exception as e:
            print(e)
            return None

        return params

    def run_statistical(self, key_name: str, config: Dict) -> pd.DataFrame:
        """
        

        Parameters
        ----------
        config : Dict

        Returns
        -------
        params : TYPE

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
            # ets = ets_decomposition_plot(ts_df, ts_column, target, title, y_label);
            # ets = ets_decomposition_plot(ts_df, ts_column, target, title, y_label,
            #                        prophet=True);
            ets = ets_decomposition_plot(df, dt_col, target,
                                         title=title,
                                         x_label=x_label,
                                         y_label=y_label,
                                         plotly=True);
    
            self.print_bold(f"{stepn}.3) {space}Plot out ACF/PACF graph..", 
                            n_before=1, n_after=1)
            title = "Total Electricity Demand"
            lags_7 = 24*7  # 7-days lag
            lags_30 = 24*30  # 30-days lag
            # lags_90 = 24*90  # 90-days lag
            acf_pacf_plot(df, target, title, lags=[lags_7, lags_30])
    
            # print_bold(f"{stepn}.4) {space}Expotential Smoothing Holt-Winters.", ui,
            #            n_before=1, n_after=1)
    
            # print_bold(f"{stepn}.5) {space}ARIMA.", ui, n_before=1)
    
        except KeyError:
            print("'analyze' section is omitted in config, therefore skipping this step.")
            return None

        return params