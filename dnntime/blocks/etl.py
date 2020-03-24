# -*- coding: utf-8 -*-
import pandas as pd
import re
from typing import Dict, Optional, Tuple
# Blocks base modules
from .base import Block, CheckpointDict
# Tests
from ..tests import etl_test as test
# Utils
from ..utils.etl_ext import load_data, clean_data
from ..utils.etl_trans import log_power_transform, decompose, split_data
from ..utils.ts import period_to_timesteps


class ETLBlock(Block):

    def __init__(self, data_dict: CheckpointDict, params: Dict) -> None:
        """
        ETLBlock inherits from Block. It performs any type of data proprocessing
        including extract, transform, and load (ETL). It takes the pre-existing
        data_dict, modifies the most current data, and stores the newly modified
        data back into the data_dict to be returned by run_block.

        Parameters
        ----------
        data_dict: The record of data transformations.
        params: Any additional parameters passed from the results of previous Blocks.

        """
        super().__init__(data_dict, params)

    def run_block(self, config: Dict) -> Tuple[CheckpointDict, Dict]:
        """
        Executes the ETLBlock function on the data_dict based on the user's config
        YAML file as well preexisting params from initialization. It returns the
        modified data_dict as well as any supplementary params.
        Here are the following ETL operations:
            1) Extract: Load data from a file source into a pd.DataFrame. This is
                        the only op where input data_dict is expected to be empty.
                        See utils.etl_ext.load_data function.
            2) Univariate: Remove all other columns in data other than its target.
            3) Clean: Massage the data, including regarding its DateTimeIndex and
                      NaN data values. See utils.etl_ext.clean_data function.
            4) Transform: Transforming the data in other to make it more digestible
                          for DNNs models, including deseasonalizing and normalization. 
            5) Supervise: Making the dataset as a supervised learning problem.
                          See utils.etl_trans.split_data function.

        Parameters
        ----------
        config: The specified config block from the user YAML file.

        Returns
        -------
        self.data_dict : The new data_dict with the modified data saved in it.
        self.params : Any additional generated parameters for subsequent Blocks.

        """
        super().run_block(config)
        test.validate_etl(self.params['key'], config)

        for key in config.keys():
            keyword = re.sub('[^a-zA-Z]+', '', key)
            data = eval(f'self.run_{keyword}(key, config[key])')
            # assert data is not None, f"'{key}' resulted in no data, please review."
            if data is not None:
                name = key.title()
                if isinstance(config[key], dict):
                    if 'alias' in config[key].keys():
                        name = config[key]['alias']
                self.data_dict.save(data, name)
            else:
                print(f"Data has return None, result for sub-block " + \
                      f"'{self.params['key']}'->'{key}' is not saved.")
            self.substep_counter += 1

        self.params['step_number'] += 1
        return self.data_dict, self.params

    def run_extract(self, key_name: str, config: Dict) -> Optional[pd.DataFrame]:
        """
        ETL operation that loads data from a file source into a pd.DataFrame.
        This is the only op where input data_dict is expected to be empty. All
        other ops expect the data_dict to have some existing data as input.

        Parameters
        ----------
        key_name : The key for this config block, usually as 'extract'{#num}.
        config : Specifies the source file path and how it is delineated.

        Returns
        -------
        df : The extracted dataset from the given source file.

        """
        # VALIDATE necessary extract parameters before running procedure
        test.validate_extract(key_name, config, self.params)
        # Initialize df as None, therefore, if it fails, then it will return None
        df = None

        try:
            # INITIALIZE extract config variables
            file_path = config['file_path']
            delinator = config['delineator'] if config['delineator'] != '' else ','
            dt_col = self.params['dt_col']  # column name containing time-series

            assert file_path.endswith(
                ".csv"
            ), "Dataset CSV file not found. Please check filepath."

            df = load_data(file_path, dt_col, delinator)

        except FileNotFoundError as e:
            print(e)
            # if data is not None:
            #     df = load_data(data, dt_col, delinator)
            #     print(f"'extract' from file path skipped, using direct data input instead.")
            # else:
            #     print("Data input not found in either file source or DataFrame.")
            #     return pd.DataFrame()

        return df

    def run_univariate(self, key_name: str, is_univariate: bool) -> pd.DataFrame:
        """
        ETL operation that strips away all other columns so only the data's
        target column remains.

        Parameters
        ----------
        key_name : The key for this config block, usually as 'univarate': {bool}.
        is_univariate : Specifies whether the make the data univariate. If not,
                        then this function will do nothing.

        Returns
        -------
        df : The univariate dataset with only its target column.

        """
        # VALIDATE necessary univariate parameters before running procedure
        assert self.data_dict.current_key is not None, "Data_dict is empty, " + \
            "run_univariate() needs existing data to run."
        test.validate_univariate(key_name, is_univariate, self.params)
        # Initialize df as None, therefore, if it fails, then it will return None
        df = None

        if is_univariate:
            key = self.data_dict.current_key
            target = self.params['target']
            df = self.data_dict.get()[key][target].to_frame()
            # df = df[target].to_frame()
            print(f"Set the dataset to univarate using target col of {target}.")

        return df

    def run_clean(self, key_name: str, config: Dict) -> pd.DataFrame:
        """
        ETL operation that massages the data, including setting up its DateTimeIndex
        as well as dealing with NaN or missing values.

        Parameters
        ----------
        key_name : The key for this config block, usually as 'clean'{#num}.
        config : Specifies the cleaning procedure, including what cleaning steps
                 to include/exclude.

        Returns
        -------
        df : The "cleaned" dataset.

        """
        # VALIDATE necessary clean parameters before running procedure
        assert self.data_dict.current_key is not None, "Data_dict is empty, " + \
            "run_clean() needs existing data to run."
        test.validate_clean(key_name, config, self.params)
        # Initialize df as None, therefore, if it fails, then it will return None
        df = None
        # INITIALIZE clean config variables
        allow_neg = config['allow_negatives']
        all_num = config['all_numeric']
        fill = config['nan_fill_type']
        out = config['output_type']
        ti =  config['time_interval']
        tz = ''
        if 'timezone' in config:
            tz = config['timezone']
        # Get most current dataset from data_dict
        key = self.data_dict.current_key
        df_curr = self.data_dict.get()[key]
        target = self.params['target']  # column name containing y_value

        try:
            print("Begin initial cleaning of the extract dataset...")
            df = clean_data(df_curr, target, ti, tz, allow_neg, all_num, fill, out)
        except Exception as e:
            print(e)

        return df

    def run_transform(self, key_name: str, config: Dict) -> pd.DataFrame:
        """
        ETL operation that transforms the data in other to make it more
        digestible for DNNs models to train on. Types of transformations
        include seasonal adjustment and/or normalization of data. 

        Parameters
        ----------
        key_name : The key for this config block, usually as 'transform'{#num}.
        config : Specifies the transformation procedure(s).

        Returns
        -------
        df : The transformed data.

        """
        # VALIDATE necessary transform parameters before running procedure
        assert self.data_dict.current_key is not None, "Data_dict is empty, " + \
            "run_transform() needs existing data to run."
        test.validate_transform(key_name, config, self.params)
        # Initialize df as None, therefore, if it fails, then it will return None
        df = None
        # INITIALIZE transform config variables
        method = config['method']
        standardize = False
        if 'standardize' in config.keys():
            standardize = config['standardize']
        decom_model = 'additive'
        if 'decomposition_model' in config.keys():
            decom_model = config['decomposition_model']
        # Get most current dataset from data_dict
        key = self.data_dict.current_key
        df_curr = self.data_dict.get()[key]
        target = self.params['target']  # column name containing y_value
        stepn = self.params['step_number']
        subn = self.substep_counter

        try:
            if method in ['box-cox', 'yeo-johnson', 'log']:
                # Performs log or power transform and then normalize in one function
                info = f"{stepn}.{subn}) Performed"
                if method in ['box-cox', 'yeo-johnson']:
                    info += f" power transformation using {method.title()} method."
                else:
                    info += " log transformation."
                if standardize:
                    info += " Then standardized data."
                df, trans_type = log_power_transform(df_curr, method=method,
                                                     standardize=standardize)
            elif method in ['detrend', 'deseasonalize', 'residual-only']:
                info = f"{stepn}.{subn}) Performed the following adjustment: " + \
                       f"{method.title()}."
                df, decom_type = decompose(df_curr, target, decom_type=method,
                                           decom_model=decom_model)
        except Exception as e:
            print(e)

        self.print_bold(f"{info}")
        # print()
        return df

    def run_supervise(self, key_name: str, config: Dict) -> Dict:
        """
        ETL operation that make the dataset into a supervised learning problem.
        This is usually the final data ETL op before DNN modeling.

        Parameters
        ----------
        key_name : The key for this config block, usually as 'supervise'{#num}.
        config : Specifies how the data is made supervised, including the input
                 and forecast periods as well as how the data is divided into
                 training, validation, and test sets.

        Returns
        -------
        data : The dict of the datasets used for DNN modelings, including the
               train and test sets.

        """
        # VALIDATE necessary supervise parameters before running procedure
        assert self.data_dict.current_key is not None, "Data_dict is empty, " + \
            "run_supervise() needs existing data to run."
        test.validate_supervise(key_name, config, self.params)
        # Initialize df as None, therefore, if it fails, then it will return None
        data = None
        # INITIALIZE supervise config variables
        train_period = config['training_period']
        fcast_period = config['forecast_period']
        val_set = config['validation_set']
        test_set = config['test_set']
        max_gap = config['max_gap']
        # Get most current dataset from data_dict
        key = self.data_dict.current_key
        df_curr = self.data_dict.get()[key]
        target = self.params['target']  # column name containing y_value
        freq = df_curr.index.freqstr

        # Converting all 'str' periods into #timesteps based of the stated 'freq'
        n_input = period_to_timesteps(train_period, freq)  # num input timesteps
        n_output = period_to_timesteps(fcast_period, freq)  # num output timesteps
        n_val = period_to_timesteps(val_set, freq)  # validation dataset size
        n_test = period_to_timesteps(test_set, freq)  # test dataset size
        n_feature = len(df_curr.columns)  # number of feature(s)

        print("Performing walk-forward validation.")

        try:
            _orig, _train, _val, _test = split_data(df_curr, target,
                                                    n_test=n_test,  # size of test set
                                                    n_val=n_val,  # size of validation set
                                                    n_input=n_input,   # input timestep seq
                                                    n_output=n_output, # output timestep seq
                                                    n_feature=n_feature,
                                                    g_min=0,     # min gap ratio
                                                    g_max=max_gap)  # max gap ratio

            X, y, t = _orig  # original data tuple in supervised format
            X_train, y_train, t_train = _train
            X_val, y_val, t_val = _val
            X_test, y_test, t_test = _test

            print("Converted time-series into supervised leraning problem "
                  "using walk-forward validation:"
                  )
            print(f"    Time-series frequency: '{freq}'.")
            print(f"    Input period: {X.shape[1]} timesteps, or 'bikweek'.")
            print(f"    Output (forecast) period: {y.shape[1]} timesteps, or 'day'.")
            print(f"    Original dataset: {df_curr.shape[0]} observations.")
            print(f"    Supervised dataset: {X.shape[0]} observations.")
            print(f"    Training dataset: {X_train.shape[0]} observations.")
            print(f"    Validation dataset: {X_val.shape[0]} observations, or '{val_set}'.")
            print(f"    Testing dataset: {X_test.shape[0]} observations, or '{test_set}'.")

            train_prct = len(X_train)/len(X)*100
            val_prct = len(X_val)/len(X)*100
            test_prct = len(X_test)/len(X)*100
            gap_prct = 100 - train_prct - val_prct - test_prct

            print("\nSplit %:")
            print(f"Train: {train_prct:.2f}%, Val: {val_prct:.2f}%, Test: "
                  f"{test_prct:.2f}%, Gap: {gap_prct:.2f}%")

            print("\nDataset shapes:")
            print(f"    Original:")
            print(f"        data shape = {df_curr.shape}")
            print(f"    Supervised:")
            print(f"        X.shape = {X.shape}")
            print(f"        y.shape = {y.shape}")
            print(f"        t.shape = {t.shape}")
            print(f"    Training:")
            print(f"        X_train.shape = {X_train.shape}")
            print(f"        y_train.shape = {y_train.shape}")
            print(f"        t_train.shape = {t_train.shape}")
            print(f"    Validation:")
            print(f"        X_val.shape = {X_val.shape}")
            print(f"        y_val.shape = {y_val.shape}")
            print(f"        t_val.shape = {t_val.shape}")
            print(f"    Testing:")
            print(f"        X_test.shape = {X_test.shape}")
            print(f"        y_test.shape = {y_test.shape}")
            print(f"        t_test.shape = {t_test.shape}")

            data = {
                'X_train': X_train,
                'y_train': y_train,
                't_train': t_train,
                'X_val': X_val,
                'y_val': y_val,
                't_val': t_val,
                'X_test': X_test,
                'y_test': y_test,
                't_test': t_test
            }

        except Exception as e:
            print(e)

        return data
