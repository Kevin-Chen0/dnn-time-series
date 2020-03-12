# -*- coding: utf-8 -*-
import pandas as pd
import pytz


def validate_config(config: pd.DataFrame) -> None:

    # Before STEPS ###########################################################
    # Validate variable <ui>
    assert (
        'meta_conf' in config.keys()
    ), "'meta_conf' key is not found in the config file."
    assert (
        'user_interface' in config['meta_conf'].keys()
    ), "'meta_conf'-->'user_interface' key is not found in the config file."
    assert isinstance(
        config['meta_conf']['user_interface'], str
    ), "'meta_conf'-->'user_interface' value must be a str type."
    valid_ui = [
        'notebook',
        'console'
    ]
    assert len(set([config['meta_conf']['user_interface']]) - set(valid_ui)) == 0, \
           "'meta_conf'-->'user_interface' value must contain one of the " + \
           f"following options: {valid_ui}."
    # Validate all the non_conf bool values
    assert (
        'extract' in config.keys()
    ), "'extract' key is not found in the config file."
    assert isinstance(
        config['extract'], bool
    ), "'extract' value must be a bool type."
    assert (
        'preprocess' in config.keys()
    ), "'preprocess' key is not found in the config file."
    assert isinstance(
        config['preprocess'], bool
    ), "'preprocess' value must be a bool type."
    assert (
        'analyze' in config.keys()
    ), "'analyze' key is not found in the config file."
    assert isinstance(
        config['analyze'], bool
    ), "'analyze' value must be a bool type."
    assert (
        'transform' in config.keys()
    ), "'transform' key is not found in the config file."
    assert isinstance(
        config['transform'], bool
    ), "'transform' value must be a bool type."
    assert (
        'supervise' in config.keys()
    ), "'supervise' key is not found in the config file."
    assert isinstance(
        config['supervise'], bool
    ), "'supervise' value must be a bool type."
    assert (
        'dnn' in config.keys()
    ), "'dnn' key is not found in the config file."
    assert isinstance(
        config['dnn'], bool
    ), "'dnn' value must be a bool type."
    assert (
        'evaluate' in config.keys()
    ), "'evaluate' key is not found in the config file."
    assert isinstance(
        config['evaluate'], bool
    ), "'evaluate' value must be a bool type."

    # For STEP 1) Extract Data from Source ###################################
    if config['extract']:
        assert (
            'extract_conf' in config.keys()
        ), "'extract_conf' key is not found in the config file."
        # Validate variable <file_path>
        assert (
            'file_path' in config['extract_conf'].keys()
        ), "'extract_conf'->'file_path' key is not found in the config file."
        assert isinstance(
            config['extract_conf']['file_path'], str
        ), "'file_path' value must be a str type."
        # Validate variable <delineator>
        assert (
            'delineator' in config['extract_conf'].keys()
        ), "'extract_conf'->'delineator' key is not found in the config file."
        assert isinstance(
            config['extract_conf']['delineator'], str
        ), "'delineator' value must be a str type."
        # Validate variable <dt_col>
        assert (
            'datetime_column' in config['extract_conf'].keys()
        ), "'extract_conf'->'datetime_column' key is not found in the config file."
        assert isinstance(
            config['extract_conf']['datetime_column'], str
        ), "'datetime_column' value must be a str type."
        # Validate <target>
        assert (
            'target_column' in config['extract_conf'].keys()
        ), "'extract_conf'->'target_column' key is not found in the config file."
        assert isinstance(
            config['extract_conf']['target_column'], str
        ), "'target_column' value must be a str type."

    # For STEP 2) Preprocessing I (Cleaning) #################################
    if config['preprocess']:
        assert (
            'preprocess_conf' in config.keys()
        ), "'preprocess_conf' key is not found in the config file."
        # Validate variable <univariate>
        assert (
            'univariate' in config['preprocess_conf'].keys()
        ), "'preprocess_conf'->'univariate' key is not found in the config file."
        assert isinstance(
            config['preprocess_conf']['univariate'], bool
        ), "'univariate' value must be a bool type."
        # Validate variable <time_interval>
        assert (
            'time_interval' in config['preprocess_conf'].keys()
        ), "'preprocess_conf'->'time_interval' key is not found in the config file."
        assert isinstance(
            config['preprocess_conf']['time_interval'], str
        ), "'time_interval' value must be a str type."
        # Validate variable <auto_clean>
        assert (
            'auto_clean' in config['preprocess_conf'].keys()
        ), "'auto_clean' key is not found in the config file."
        assert isinstance(
            config['preprocess_conf']['auto_clean'], bool
        ), "'preprocess_conf'->'auto_clean' value must be a bool type."
        if config['preprocess_conf']['auto_clean']:
            # Validate variable <timezone>
            assert (
                'auto_clean_conf' in config['preprocess_conf'].keys()
            ), "'preprocess_conf'->'auto_clean_conf' key is not found in the config file."
            assert (
                'timezone' in config['preprocess_conf']['auto_clean_conf'].keys()
            ), "'preprocess_conf'->'auto_clean_conf'->'timezone' key is not found " + \
               "in the config file."
            assert isinstance(
                config['preprocess_conf']['auto_clean_conf']['timezone'], str
            ), "'timezone' value must be a str type."
            assert (
                config['preprocess_conf']['auto_clean_conf']['timezone'] in pytz.all_timezones or
                config['preprocess_conf']['auto_clean_conf']['timezone'] == ''
            ), "'timezone' value must be a valid pytz timezone or is left ''."
            # Validate variable <allow_neg>
            assert (
                'negative_values' in config['preprocess_conf']['auto_clean_conf'].keys()
            ), "'preprocess_conf'->'auto_clean_conf'->'negative_values' key is not " + \
               "found in the config file."
            assert isinstance(
                config['preprocess_conf']['auto_clean_conf']['negative_values'], bool
            ), "'negative_values' value must be a bool type."
            # Validate variable <fill>
            assert (
                'nan_fill_type' in config['preprocess_conf']['auto_clean_conf'].keys()
            ), "'preprocess_conf'->'auto_clean_conf'->'nan_fill_type' key is not " + \
               "found in the config file."
            assert isinstance(
                config['preprocess_conf']['auto_clean_conf']['nan_fill_type'], str
            ), "'nan_fill_type' value must be a str type."

    # For STEPS 3-4) EDA I & II (General & Time-Series Stats) ################
    if config['analyze']:
        assert (
            'analyze_conf' in config.keys()
        ), "'analyze_conf' key is not found in the config file."
        # Validate variable <title>
        assert (
            'title' in config['analyze_conf'].keys()
        ), "'analyze_conf'->'title' key is not found in the config file."
        assert isinstance(
            config['analyze_conf']['title'], str
        ), "'file_path' value must be a str type."
        # Validate variable <x_label>
        assert (
            'x_label' in config['analyze_conf'].keys()
        ), "'analyze_conf'->'x_label' key is not found in the config file."
        assert isinstance(
            config['analyze_conf']['x_label'], str
        ), "'x_label' value must be a str type."
        # Validate variable <y_label>
        assert (
            'y_label' in config['analyze_conf'].keys()
        ), "'analyze_conf'->'y_label' key is not found in the config file."
        assert isinstance(
            config['analyze_conf']['y_label'], str
        ), "'y_label' value must be a str type."
        # Validate variable <ci>
        assert (
            'confidence_interval' in config['analyze_conf'].keys()
        ), "'analyze_conf'->'confidence_interval' key is not found in the config file."
        assert isinstance(
            config['analyze_conf']['confidence_interval'], float
        ), "'y_label' value must be a float type."

    # For STEP 5) Preprocessing II (Transformations) #########################
    if config['transform']:
        assert (
            'transform_conf' in config.keys()
        ), "'transform_conf' key is not found in the config file."
        # Validate variable <trans_steps>
        assert (
            'steps' in config['transform_conf'].keys()
        ), "'transform_conf'-->'steps' key is not found in the config file."
        assert isinstance(
            config['transform_conf']['steps'], list
        ), "'transform_conf'-->'steps' value must be a list type."
        valid_steps = [
            'box-cox',
            'yeo-johnson',
            'log',
            'deseasonalize',
            'detrend',
            'residual-only',
        ]
        assert len(set(config['transform_conf']['steps']) - set(valid_steps)) == 0, \
               f"'transform_conf'-->'steps' valuess must be within the following: {valid_steps}."
        # Validate variable <decom_model>
        assert (
            'decomposition_model' in config['transform_conf'].keys()
        ), "'transform_conf'-->'decomposition_model' key is not found in the config file."
        assert isinstance(
            config['transform_conf']['decomposition_model'], str
        ), "'transform_conf'-->'decomposition_model' value must be a str type."
        valid_decompositions = [
            'additive',
            'multiplicative'
        ]
        assert len(set([config['transform_conf']['decomposition_model']]) - set(valid_decompositions)) == 0, \
               "'transform_conf'-->'decomposition_model' value must only contain the following: " + \
               f"{valid_decompositions}."
        # Validate variable <standardize>
        assert (
            'standardize' in config['transform_conf'].keys()
        ), "'transform_conf'-->'standardize' key is not found in the config file."
        assert isinstance(
            config['transform_conf']['standardize'], bool
        ), "'transform_conf'-->'standardize' value must be a bool type."

    # For STEP 6) Preprocessing III (Make Supervised) ########################
    if config['supervise']:
        assert (
            'supervise_conf' in config.keys()
        ), "'supervise_conf' key is not found in the config file."
        # Validate variable <train_period>
        assert (
            'training_period' in config['supervise_conf'].keys()
        ), "'supervise_conf'->'training_period' key is not found in the config file."
        assert isinstance(
            config['supervise_conf']['training_period'], str
        ), "'training_period' value must be a str type."
        # Validate variable <fcast_period>
        assert (
            'forecast_period' in config['supervise_conf'].keys()
        ), "'supervise_conf'->'forecast_period' key is not found in the config file."
        assert isinstance(
            config['supervise_conf']['forecast_period'], str
        ), "'forecast_period' value must be a str type."
        # Validate variable <val_set>
        assert (
            'validation_set' in config['supervise_conf'].keys()
        ), "'supervise_conf'->'validation_set' key is not found in the config file."
        assert isinstance(
            config['supervise_conf']['validation_set'], str
        ), "'validation_set' value must be a str type."
        # Validate <test_set>
        assert (
            'test_set' in config['supervise_conf'].keys()
        ), "'supervise_conf'->'test_set' key is not found in the config file."
        assert isinstance(
            config['supervise_conf']['test_set'], str
        ), "'test_set' value must be a str type."
        # Validate <max_gap>
        assert (
            'max_gap' in config['supervise_conf'].keys()
        ), "'supervise_conf'->'max_gap' key is not found in the config file."
        assert isinstance(
            config['supervise_conf']['max_gap'], float
        ), "'max_gap' value must be a float type."

    # For STEP 7) Model Search (NNs) #########################################
    if config['dnn']:
        assert (
            'dnn_conf' in config.keys()
        ), "'dnn_conf' key is not found in the config file."
        # Validate variable <model>
        assert (
            'model_type' in config['dnn_conf'].keys()
        ), "'dnn_conf'->'model_type' key is not found in the config file."
        assert isinstance(
            config['dnn_conf']['model_type'], str
        ), "'model_type' value must be an str type."
        # Validate variable <epochs>
        assert (
            'epochs' in config['dnn_conf'].keys()
        ), "'dnn_conf'->'epochs' key is not found in the config file."
        assert isinstance(
            config['dnn_conf']['epochs'], int
        ), "'epochs' value must be an int type."
        # Validate variable <batch_size>
        assert (
            'batch_size' in config['dnn_conf'].keys()
        ), "'dnn_conf'->'batch_size' key is not found in the config file."
        assert isinstance(
            config['dnn_conf']['batch_size'], int
        ), "'batch_size' value must be an int type."
        # Validate variable <n_features>
        assert (
            'n_features' in config['dnn_conf'].keys()
        ), "'dnn_conf'->'n_features' key is not found in the config file."
        assert isinstance(
            config['dnn_conf']['n_features'], int
        ), "'n_features' value must be an int type."
        # Validate <n_units>
        assert (
            'n_units' in config['dnn_conf'].keys()
        ), "'dnn_conf'->'n_units' key is not found in the config file."
        assert isinstance(
            config['dnn_conf']['n_units'], int
        ), "'n_units' value must be an int type."
        # Validate <d_rate>
        assert (
            'd_rate' in config['dnn_conf'].keys()
        ), "'dnn_conf'->'d_rate' key is not found in the config file."
        assert isinstance(
            config['dnn_conf']['d_rate'], float
        ), "'d_rate' value must be an float type."
        # Validate variable <opt>
        assert (
            'optimizer' in config['dnn_conf'].keys()
        ), "'dnn_conf'->'optimizer' key is not found in the config file."
        assert isinstance(
            config['dnn_conf']['optimizer'], str
        ), "'optimizer' value must be an str type."
        # Validate variable <loss>
        assert (
            'objective_function' in config['dnn_conf'].keys()
        ), "'dnn_conf'->'objective_function' key is not found in the config file."
        assert isinstance(
            config['dnn_conf']['objective_function'], str
        ), "'objective_function' value must be an str type."
        # Validate <verbose>
        assert (
            'verbose' in config['dnn_conf'].keys()
        ), "'dnn_conf'->'verbose' key is not found in the config file."
        assert isinstance(
            config['dnn_conf']['verbose'], int
        ), "'verbose' value must be an int type."
        # Validate variable <score_type>
        assert (
            'evaluate_conf' in config.keys()
        ), "'evaluate_conf' key is not found in the config file."
        assert (
            'score_type' in config['evaluate_conf'].keys()
        ), "'evaluate_conf'->'score_type' key is not found in the config file."
        assert isinstance(
            config['evaluate_conf']['score_type'], str
        ), "'score_type' value must be a str type."