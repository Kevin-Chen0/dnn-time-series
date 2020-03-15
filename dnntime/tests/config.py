# -*- coding: utf-8 -*-
import pandas as pd
import pytz


def validate_config(config: pd.DataFrame) -> None:
    """
    Checks that the contents in the config file are indeed valid to ensure the
    parameters inputted by the user will not cause unexpected errors. The config
    MUST pass all Assertions mentioned here in order to be able to proceed with
    the rest of the run_package() function. Note that the user may freely omit
    blocks in the config YAML file, so those blocks will not be checked if removed.

    Parameters
    ----------
    config : The contents in the config file in DataFrame format.

    """
    # Before STEPS ###########################################################
    if 'meta' in config.keys():
        meta = config['meta']
        path = "'meta'->"
        # Validate variable <ui>
        assert (
            'user_interface' in meta.keys()
        ), f"{path}'user_interface' key is not found in the config file."
        assert isinstance(
            meta['user_interface'], str
        ), f"{path}'user_interface' value must be a str type."
        valid_ui = [
            'notebook',
            'console'
        ]
        assert len(set([meta['user_interface']]) - set(valid_ui)) == 0, \
            f"{path}'user_interface' value must contain one of the " + \
            f"following options: {valid_ui}."

    # For STEP 1) Extract Data from Source ###################################
    if 'extract' in config.keys():
        extract = config['extract']
        path = "'extract'->"
        # Validate variable <file_path>
        assert (
            'file_path' in extract.keys()
        ), f"'extract'->'file_path' key is not found in the config file."
        assert isinstance(
            extract['file_path'], str
        ), f"{path}'file_path' value must be a str type."
        # Validate variable <delineator>
        assert (
            'delineator' in extract.keys()
        ), f"{path}'delineator' key is not found in the config file."
        assert isinstance(
            extract['delineator'], str
        ), "'delineator' value must be a str type."
        # Validate variable <dt_col>
        assert (
            'datetime_column' in extract.keys()
        ), f"{path}'datetime_column' key is not found in the config file."
        assert isinstance(
            extract['datetime_column'], str
        ), "'datetime_column' value must be a str type."
        # Validate <target>
        assert (
            'target_column' in extract.keys()
        ), f"{path}'target_column' key is not found in the config file."
        assert isinstance(
            extract['target_column'], str
        ), "'target_column' value must be a str type."

    # For STEP 2) Preprocessing I (Cleaning) #################################
    if 'preprocess' in config.keys():
        preprocess = config['preprocess']
        path = "'preprocess'->"
        # Validate variable <univariate>
        assert (
            'univariate' in preprocess.keys()
        ), f"{path}'univariate' key is not found in the config file."
        assert isinstance(
            preprocess['univariate'], bool
        ), "'univariate' value must be a bool type."
        # Validate variable <time_interval>
        assert (
            'time_interval' in preprocess.keys()
        ), f"{path}'time_interval' key is not found in the config file."
        assert isinstance(
            preprocess['time_interval'], str
        ), "'time_interval' value must be a str type."
        # Validate sub-section <auto_clean> given that it is not omitted
        if 'auto_clean' in preprocess.keys():
            clean = preprocess['auto_clean']
            path += "'auto_clean'->"
            # Validate variable <timezone>
            assert (
                'timezone' in clean.keys()
            ), f"{path}'timezone' key is not found in the config file."
            assert isinstance(
                clean['timezone'], str
            ), "'timezone' value must be a str type."
            assert (
                clean['timezone'] in pytz.all_timezones or clean['timezone'] == ''
            ), "'timezone' value must be a valid pytz timezone or is left ''."
            # Validate variable <allow_neg>
            assert (
                'allow_negatives' in clean.keys()
            ), f"{path}'allow_negatives' key is not " + \
               "found in the config file."
            assert isinstance(
                clean['allow_negatives'], bool
            ), "'allow_negatives' value must be a bool type."
            # Validate variable <all_num>
            assert (
                'all_numeric' in clean.keys()
            ), f"{path}'all_numeric' key is not " + \
               "found in the config file."
            assert isinstance(
                clean['all_numeric'], bool
            ), "'all_numeric' value must be a bool type."
            # Validate variable <fill>
            assert (
                'nan_fill_type' in clean.keys()
            ), f"{path}'nan_fill_type' key is not " + \
               "found in the config file."
            assert isinstance(
                clean['nan_fill_type'], str
            ), "'nan_fill_type' value must be a str type."
            valid_fills = [
                '', 'linear', 'time', 'index', 'pad', 'nearest', 'zero', 'slinear',
                'quadratic', 'cubic', 'krogh', 'pchip', 'akima', 'from_derivatives'
            ]
            assert len(set([clean['nan_fill_type']]) - set(valid_fills)) == 0, \
                f"{path}'nan_fill_type' value must be '' or within DataFrame.interpolate: " + \
                "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html"
            # Validate variable <output_type>
            assert (
                'output_type' in clean.keys()
            ), f"{path}'output_type' key is not " + \
               "found in the config file."
            assert isinstance(
                clean['output_type'], str
            ), "'output_type' value must be a str type."
            valid_outputs = [
                'reg',
                'regression',
                'class',
                'classification',
            ]
            assert len(set([clean['output_type']]) - set(valid_outputs)) == 0, \
                f"{path}'output_type' value must contain one of the " + \
                f"following options: {valid_outputs}."

    # For STEPS 3-4) EDA I & II (General & Time-Series Stats) ################
    if 'analyze' in config.keys():
        analyze = config['analyze']
        path = "'analyze'->"
        # Validate variable <title>
        assert (
            'title' in analyze.keys()
        ), f"{path}'title' key is not found in the config file."
        assert isinstance(
            analyze['title'], str
        ), "'file_path' value must be a str type."
        # Validate variable <x_label>
        assert (
            'x_label' in analyze.keys()
        ), f"{path}'x_label' key is not found in the config file."
        assert isinstance(
            analyze['x_label'], str
        ), "'x_label' value must be a str type."
        # Validate variable <y_label>
        assert (
            'y_label' in analyze.keys()
        ), f"{path}'y_label' key is not found in the config file."
        assert isinstance(
            analyze['y_label'], str
        ), "'y_label' value must be a str type."
        # Validate variable <ci>
        assert (
            'confidence_interval' in analyze.keys()
        ), f"{path}'confidence_interval' key is not found in the config file."
        assert isinstance(
            analyze['confidence_interval'], float
        ), "'y_label' value must be a float type."

    # For STEP 5) Preprocessing II (Transformations) #########################
    if 'transform' in config.keys():
        transform = config['transform']
        path = "'transform'->"
        # Validate variable <trans_steps>
        assert (
            'steps' in transform.keys()
        ), f"{path}'steps' key is not found in the config file."
        assert isinstance(
            transform['steps'], list
        ), f"{path}'steps' value must be a list type."
        valid_steps = [
            'box-cox',
            'yeo-johnson',
            'log',
            'deseasonalize',
            'detrend',
            'residual-only',
        ]
        assert len(set(transform['steps']) - set(valid_steps)) == 0, \
            "{path}'steps' values must be within the following: {valid_steps}."
        # Validate variable <decom_model>
        assert (
            'decomposition_model' in transform.keys()
        ), f"{path}'decomposition_model' key is not found in the config file."
        assert isinstance(
            transform['decomposition_model'], str
        ), f"{path}'decomposition_model' value must be a str type."
        valid_decoms = [
            'additive',
            'multiplicative'
        ]
        assert len(set([transform['decomposition_model']]) - set(valid_decoms)) == 0, \
            f"{path}'decomposition_model' value must only contain the " + \
            f"following: {valid_decoms}."
        # Validate variable <standardize>
        assert (
            'standardize' in transform.keys()
        ), f"{path}'standardize' key is not found in the config file."
        assert isinstance(
            transform['standardize'], bool
        ), f"{path}'standardize' value must be a bool type."

    # For STEP 6) Preprocessing III (Make Supervised) ########################
    if 'supervise' in config.keys():
        supervise = config['supervise']
        path = "'supervise'->"
        # Validate variable <train_period>
        assert (
            'training_period' in supervise.keys()
        ), f"{path}'training_period' key is not found in the config file."
        assert isinstance(
            supervise['training_period'], str
        ), "'training_period' value must be a str type."
        # Validate variable <fcast_period>
        assert (
            'forecast_period' in supervise.keys()
        ), f"{path}'forecast_period' key is not found in the config file."
        assert isinstance(
            supervise['forecast_period'], str
        ), "'forecast_period' value must be a str type."
        # Validate variable <val_set>
        assert (
            'validation_set' in supervise.keys()
        ), f"{path}'validation_set' key is not found in the config file."
        assert isinstance(
            supervise['validation_set'], str
        ), "'validation_set' value must be a str type."
        # Validate <test_set>
        assert (
            'test_set' in supervise.keys()
        ), f"{path}'test_set' key is not found in the config file."
        assert isinstance(
            supervise['test_set'], str
        ), "'test_set' value must be a str type."
        # Validate <max_gap>
        assert (
            'max_gap' in supervise.keys()
        ), f"{path}'max_gap' key is not found in the config file."
        assert isinstance(
            supervise['max_gap'], float
        ), "'max_gap' value must be a float type."

    # For STEP 7) Model Search (NNs) #########################################
    if 'dnn' in config.keys():
        dnn = config['dnn']
        path = "'dnn'->"
        # Validate variable <gpu>
        assert (
            'enable_gpu' in dnn.keys()
        ), f"{path}'enable_gpu' key is not found in the config file."
        assert isinstance(
            dnn['enable_gpu'], bool
        ), "'enable_gpu' value must be a bool type."
        # Validate variable <model>
        assert (
            'model_type' in dnn.keys()
        ), f"{path}'model_type' key is not found in the config file."
        assert isinstance(
            dnn['model_type'], str
        ), "'model_type' value must be a str type."
        # Validate variable <epochs>
        assert (
            'epochs' in dnn.keys()
        ), f"{path}'epochs' key is not found in the config file."
        assert isinstance(
            dnn['epochs'], int
        ), "'epochs' value must be an int type."
        # Validate variable <batch_size>
        assert (
            'batch_size' in dnn.keys()
        ), f"{path}'batch_size' key is not found in the config file."
        assert isinstance(
            dnn['batch_size'], int
        ), "'batch_size' value must be an int type."
        # Validate <n_unit>
        assert (
            'number_units' in dnn.keys()
        ), f"{path}'number_units' key is not found in the config file."
        assert isinstance(
            dnn['number_units'], int
        ), "'number_units' value must be an int type."
        # Validate <d_rate>
        assert (
            'dropout_rate' in dnn.keys()
        ), f"{path}'dropout_rate' key is not found in the config file."
        assert isinstance(
            dnn['dropout_rate'], float
        ), "'dropout_rate' value must be an float type."
        # Validate variable <opt>
        assert (
            'optimizer' in dnn.keys()
        ), f"{path}'optimizer' key is not found in the config file."
        assert isinstance(
            dnn['optimizer'], str
        ), "'optimizer' value must be a str type."
        # Validate variable <loss>
        assert (
            'objective_function' in dnn.keys()
        ), f"{path}'objective_function' key is not found in the config file."
        assert isinstance(
            dnn['objective_function'], str
        ), "'objective_function' value must be a str type."
        # Validate <verbose>
        assert (
            'verbose' in dnn.keys()
        ), f"{path}'verbose' key is not found in the config file."
        assert isinstance(
            dnn['verbose'], int
        ), "'verbose' value must be an int type."
        # Validate variable <score_type>
        assert (
            'evaluate' in config.keys()
        ), "'evaluate' key is not found in the config file."
        assert (
            'score_type' in config['evaluate'].keys()
        ), "'evaluate'->'score_type' key is not found in the config file."
        assert isinstance(
            config['evaluate']['score_type'], str
        ), "'score_type' value must be a str type."
