# -*- coding: utf-8 -*-
import pandas as pd
import pytz


def validate_config(config: pd.DataFrame) -> None:

    # Before STEPS ###########################################################
    # Validate variable <ui>
    assert (
        'meta' in config.keys()
    ), "'meta' key is not found in the config file."
    assert (
        'user_interface' in config['meta'].keys()
    ), "'meta'-->'user_interface' key is not found in the config file."
    assert isinstance(
        config['meta']['user_interface'], str
    ), "'meta'-->'user_interface' value must be a str type."
    valid_ui = [
        'notebook',
        'console'
    ]
    assert len(set([config['meta']['user_interface']]) - set(valid_ui)) == 0, \
           "'meta'-->'user_interface' value must contain one of the following options: " + \
           f"{valid_ui}."

    # For STEP 1) Extract Data from Source ###################################
    # Validate variable <file_path>
    assert (
        'extract' in config.keys()
    ), "'extract' key is not found in the config file."
    assert (
        'file_path' in config['extract'].keys()
    ), "'extract'->'file_path' key is not found in the config file."
    assert isinstance(
        config['extract']['file_path'], str
    ), "'file_path' value must be a str type."
    # Validate variable <delineator>
    assert (
        'delineator' in config['extract'].keys()
    ), "'extract'->'delineator' key is not found in the config file."
    assert isinstance(
        config['extract']['delineator'], str
    ), "'delineator' value must be a str type."
    # Validate variable <dt_col>
    assert (
        'datetime_column' in config['extract'].keys()
    ), "'extract'->'datetime_column' key is not found in the config file."
    assert isinstance(
        config['extract']['datetime_column'], str
    ), "'datetime_column' value must be a str type."
    # Validate <target>
    assert (
        'target_column' in config['extract'].keys()
    ), "'extract'->'target_column' key is not found in the config file."
    assert isinstance(
        config['extract']['target_column'], str
    ), "'target_column' value must be a str type."
    
    # For STEP 2) Preprocessing I (Cleaning) #################################
    # Validate variable <univariate>
    assert (
        'preprocess' in config.keys()
    ), "'preprocess' key is not found in the config file."
    assert (
        'univariate' in config['preprocess'].keys()
    ), "'preprocess'->'univariate' key is not found in the config file."
    assert isinstance(
        config['preprocess']['univariate'], bool
    ), "'univariate' value must be a bool type."
    # Validate variable <time_interval>
    assert (
        'time_interval' in config['preprocess'].keys()
    ), "'preprocess'->'time_interval' key is not found in the config file."
    assert isinstance(
        config['preprocess']['time_interval'], str
    ), "'time_interval' value must be a str type."
    # Validate variable <auto_clean>
    assert (
        'auto_clean' in config['preprocess'].keys()
    ), "'auto_clean' key is not found in the config file."
    assert isinstance(
        config['preprocess']['auto_clean'], bool
    ), "'preprocess'->'auto_clean' value must be a bool type."
    if config['preprocess']['auto_clean']:
        # Validate variable <timezone>
        assert (
            'auto_clean_conf' in config['preprocess'].keys()
        ), "'preprocess'->'auto_clean_conf' key is not found in the config file."
        assert (
            'timezone' in config['preprocess']['auto_clean_conf'].keys()
        ), "'preprocess'->'auto_clean_conf'->'timezone' key is not found " + \
           "in the config file."
        assert isinstance(
            config['preprocess']['auto_clean_conf']['timezone'], str
        ), "'timezone' value must be a str type."
        assert (
            config['preprocess']['auto_clean_conf']['timezone'] in pytz.all_timezones or
            config['preprocess']['auto_clean_conf']['timezone'] == ''
        ), "'timezone' value must be a valid pytz timezone or is left ''."
        # Validate variable <allow_neg>
        assert (
            'negative_values' in config['preprocess']['auto_clean_conf'].keys()
        ), "'preprocess'->'auto_clean_conf'->'negative_values' key is not " + \
           "found in the config file."
        assert isinstance(
            config['preprocess']['auto_clean_conf']['negative_values'], bool
        ), "'negative_values' value must be a bool type."
        # Validate variable <fill>
        assert (
            'nan_fill_type' in config['preprocess']['auto_clean_conf'].keys()
        ), "'preprocess'->'auto_clean_conf'->'nan_fill_type' key is not " + \
           "found in the config file."
        assert isinstance(
            config['preprocess']['auto_clean_conf']['nan_fill_type'], str
        ), "'nan_fill_type' value must be a str type."

    # For STEP 3) EDA I (General) ############################################
    # Validate variable <title>
    assert (
        'analyze' in config.keys()
    ), "'analyze' key is not found in the config file."
    assert (
        'title' in config['analyze'].keys()
    ), "'analyze'->'title' key is not found in the config file."
    assert isinstance(
        config['analyze']['title'], str
    ), "'file_path' value must be a str type."
    # Validate variable <x_label>
    assert (
        'x_label' in config['analyze'].keys()
    ), "'analyze'->'x_label' key is not found in the config file."
    assert isinstance(
        config['analyze']['x_label'], str
    ), "'x_label' value must be a str type."
    # Validate variable <y_label>
    assert (
        'y_label' in config['analyze'].keys()
    ), "'analyze'->'y_label' key is not found in the config file."
    assert isinstance(
        config['analyze']['y_label'], str
    ), "'y_label' value must be a str type."

    # For STEP 4) EDA II (Time-Series Stats) #################################
    # Validate variable <ci>
    assert (
        'confidence_interval' in config['analyze'].keys()
    ), "'analyze'->'confidence_interval' key is not found in the config file."
    assert isinstance(
        config['analyze']['confidence_interval'], float
    ), "'y_label' value must be a float type."

    # For STEP 5) Preprocessing II (Transformations) #########################
    # Validate variable <trans_steps>
    assert (
        'transform' in config.keys()
    ), "'transform' key is not found in the config file."
    assert (
        'steps' in config['transform'].keys()
    ), "'transform'-->'steps' key is not found in the config file."
    assert isinstance(
        config['transform']['steps'], list
    ), "'transform'-->'steps' value must be a list type."
    valid_steps = [
        'box-cox',
        'yeo-johnson',
        'log',
        'deseasonalize',
        'detrend',
        'residual-only',
    ]
    assert len(set(config['transform']['steps']) - set(valid_steps)) == 0, \
           f"'transform'-->'steps' valuess must be within the following: {valid_steps}."
    # Validate variable <decom_model>
    assert (
        'decomposition_model' in config['transform'].keys()
    ), "'transform'-->'decomposition_model' key is not found in the config file."
    assert isinstance(
        config['transform']['decomposition_model'], str
    ), "'transform'-->'decomposition_model' value must be a str type."
    valid_decompositions = [
        'additive',
        'multiplicative'
    ]
    assert len(set([config['transform']['decomposition_model']]) - set(valid_decompositions)) == 0, \
           "'transform'-->'decomposition_model' value must only contain the following: " + \
           f"{valid_decompositions}."
    # Validate variable <standardize>
    assert (
        'standardize' in config['transform'].keys()
    ), "'transform'-->'standardize' key is not found in the config file."
    assert isinstance(
        config['transform']['standardize'], bool
    ), "'transform'-->'standardize' value must be a bool type."

    # For STEP 6) Preprocessing III (Make Supervised) ########################
    # Validate variable <train_period>
    assert (
        'extract' in config.keys()
    ), "'extract' key is not found in the config file."
    assert (
        'file_path' in config['extract'].keys()
    ), "'extract'->'file_path' key is not found in the config file."
    assert isinstance(
        config['extract']['file_path'], str
    ), "'file_path' value must be a str type."
    # Validate variable <fcast_period>
    assert (
        'delineator' in config['extract'].keys()
    ), "'extract'->'delineator' key is not found in the config file."
    assert isinstance(
        config['extract']['delineator'], str
    ), "'delineator' value must be a str type."
    # Validate variable <val_set>
    assert (
        'datetime_column' in config['extract'].keys()
    ), "'extract'->'datetime_column' key is not found in the config file."
    assert isinstance(
        config['extract']['datetime_column'], str
    ), "'datetime_column' value must be a str type."
    # Validate <test_set>
    assert (
        'target_column' in config['extract'].keys()
    ), "'extract'->'target_column' key is not found in the config file."
    assert isinstance(
        config['extract']['target_column'], str
    ), "'target_column' value must be a str type."
    # Validate <max_gap>
    assert (
        'target_column' in config['extract'].keys()
    ), "'extract'->'target_column' key is not found in the config file."
    assert isinstance(
        config['extract']['target_column'], str
    ), "'target_column' value must be a str type."

    # For STEP 7) Model Search (NNs) #########################################
    # Validate variable <model>
    assert (
        'dnn' in config.keys()
    ), "'dnn' key is not found in the config file."
    assert (
        'model_type' in config['dnn'].keys()
    ), "'dnn'->'model_type' key is not found in the config file."
    assert isinstance(
        config['dnn']['model_type'], str
    ), "'model_type' value must be an str type."
    # Validate variable <epochs>
    assert (
        'epochs' in config['dnn'].keys()
    ), "'dnn'->'epochs' key is not found in the config file."
    assert isinstance(
        config['dnn']['epochs'], int
    ), "'epochs' value must be an int type."
    # Validate variable <batch_size>
    assert (
        'batch_size' in config['dnn'].keys()
    ), "'dnn'->'batch_size' key is not found in the config file."
    assert isinstance(
        config['dnn']['batch_size'], int
    ), "'batch_size' value must be an int type."
    # Validate variable <n_features>
    assert (
        'n_features' in config['dnn'].keys()
    ), "'dnn'->'n_features' key is not found in the config file."
    assert isinstance(
        config['dnn']['n_features'], int
    ), "'n_features' value must be an int type."
    # Validate <n_units>
    assert (
        'n_units' in config['dnn'].keys()
    ), "'dnn'->'n_units' key is not found in the config file."
    assert isinstance(
        config['dnn']['n_units'], int
    ), "'n_units' value must be an int type."
    # Validate <d_rate>
    assert (
        'd_rate' in config['dnn'].keys()
    ), "'dnn'->'d_rate' key is not found in the config file."
    assert isinstance(
        config['dnn']['d_rate'], float
    ), "'d_rate' value must be an float type."
    # Validate <verbose>
    assert (
        'verbose' in config['dnn'].keys()
    ), "'dnn'->'verbose' key is not found in the config file."
    assert isinstance(
        config['dnn']['verbose'], int
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