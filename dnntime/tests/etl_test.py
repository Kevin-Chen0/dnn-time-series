# -*- coding: utf-8 -*-
import pytz
import re
from typing import Dict


def validate_etl(key_name: str, config: Dict) -> None:
    """

    Parameters
    ----------
    config : The contents in the config file in DataFrame format.

    """
    # TEST 1) Validate config's keys #########################################
    # Get the given ETL config keys and remove all non-alphabetical chars
    # etl_keys = set(config.keys())
    etl_keys = {re.sub('[^a-zA-Z]+', '', key) for key in config.keys()}
    # Assert that etl keys only contain prefixes:
    valid_etls = {'extract', 'univariate', 'clean', 'transform', 'supervise'}
    # Assert all etl keys are subset of valid etls, otherwise process will fail.
    assert len(etl_keys - valid_etls) == 0, "ETL config keys must contain " + \
           f"only the following subkeys: {valid_etls}. '{key_name}' " + \
           f"contains the following: {list(config.keys())}."


def validate_datetime_target(params: Dict) -> None:

    # TEST 1) Validate 'dt_col' param key ####################################
    path = "'meta'->"
    assert (
        'dt_col' in params.keys()
    ), f"{path}'dt_col' key is not found in the config file."
    assert isinstance(
        params['dt_col'], str
    ), f"{path}'dt_col' value must be a str type."

    # TEST 2) Validate 'target' param key ####################################
    assert (
        'target' in params.keys()
    ), f"{path}'target' key is not found in the config file."
    assert isinstance(
        params['target'], str
    ), f"{path}'target' value must be a str type."


def validate_extract(key_name: str, ext_conf: Dict, params: Dict) -> None:
    """

    Parameters
    ----------
    ext_conf : The extract sub-config dict or config['extract'].

    """
    # TEST 1) Validate root 'key' in params ##################################
    assert (
        'key' in params.keys()
    ), "Missing the root key for this ETL block."
    assert isinstance(
        params['key'], str
    ), "'key' value in param must be a str type."
    assert params['key'] != '', "'key' value in param cannot be blank."
    path = f"'{params['key']}'->'{key_name}'->"

    # TEST 2) Validate 'alias' sub-config key (optional) #####################
    if 'alias' in ext_conf.keys():
        assert isinstance(
            ext_conf['alias'], str
        ), f"{path}'alias' value must be a str type."
        assert ext_conf['alias'] != '', f"{path}'alias' value cannot be blank."

    # TEST 3) Validate 'file_path' sub-config key ############################
    assert (
        'file_path' in ext_conf.keys()
    ), f"'extract'->'file_path' key is not found in the config file."
    assert isinstance(
        ext_conf['file_path'], str
    ), f"{path}'file_path' value must be a str type."

    # TEST 4) Validate 'delineator' sub-config key ###########################
    assert (
        'delineator' in ext_conf.keys()
    ), f"{path}'delineator' key is not found in the config file."
    assert isinstance(
        ext_conf['delineator'], str
    ), f"{path}'delineator' value must be a str type."

    validate_datetime_target(params)


def validate_univariate(key_name: str, uni_conf: bool, params: Dict) -> None:

    # TEST 1) Validate root 'key' in params ##################################
    assert (
        'key' in params.keys()
    ), "Missing the root key for this ETL block."
    assert isinstance(
        params['key'], str
    ), "'key' value in param must be a str type."
    assert params['key'] != '', "'key' value in param cannot be blank."
    path = f"'{params['key']}'->"

   # TEST 2) Validate 'univariate' sub-config key ###########################
    assert isinstance(
        uni_conf, bool
    ), f"{path}'{key_name}' value must be a bool type."

    validate_datetime_target(params)


def validate_clean(key_name: str, cln_conf: Dict, params: Dict) -> None:

    # TEST 1) Validate root 'key' in params ##################################
    assert (
        'key' in params.keys()
    ), "Missing the root key for this ETL block."
    assert isinstance(
        params['key'], str
    ), "'key' value in param must be a str type."
    assert params['key'] != '', "'key' value in param cannot be blank."
    path = f"'{params['key']}'->'{key_name}'->"

    # TEST 2) Validate 'alias' sub-config key (optional) #####################
    if 'alias' in cln_conf.keys():
        assert isinstance(
            cln_conf['alias'], str
        ), f"{path}'alias' value must be a str type."
        assert cln_conf['alias'] != '', f"{path}'alias' value cannot be blank."

    # TEST 3) Validate 'time_interval' sub-config key ########################
    assert (
        'time_interval' in cln_conf.keys()
    ), f"{path}'time_interval' key is not found in the config file."
    assert isinstance(
        cln_conf['time_interval'], str
    ), f"{path}'time_interval' value must be a str type."

    # TEST 4) Validate 'timezone' sub-config key (optional) ##################
    if 'timezone' in cln_conf.keys():
        assert isinstance(
            cln_conf['timezone'], str
        ), f"{path}'timezone' value must be a str type."
        assert (
            cln_conf['timezone'] in pytz.all_timezones or cln_conf['timezone'] == ''
        ), f"{path}'timezone' value must be a valid pytz timezone or is left ''."

    # TEST 5) Validate 'allow_negatives' sub-config key (optional) ###########
    assert (
        'allow_negatives' in cln_conf.keys()
    ), f"{path}'allow_negatives' key is not " + \
       "found in the config file."
    assert isinstance(
        cln_conf['allow_negatives'], bool
    ), f"{path}'allow_negatives' value must be a bool type."

    # TEST 6) Validate 'all_numeric' sub-config key ##########################
    assert (
        'all_numeric' in cln_conf.keys()
    ), f"{path}'all_numeric' key is not " + \
       "found in the config file."
    assert isinstance(
        cln_conf['all_numeric'], bool
    ), f"{path}'all_numeric' value must be a bool type."

    # TEST 7) Validate 'nan_fill_type' sub-config key ########################
    assert (
        'nan_fill_type' in cln_conf.keys()
    ), f"{path}'nan_fill_type' key is not " + \
       "found in the config file."
    assert isinstance(
        cln_conf['nan_fill_type'], str
    ), f"{path}'nan_fill_type' value must be a str type."
    valid_fills = {
        '', 'linear', 'time', 'index', 'pad', 'nearest', 'zero', 'slinear',
        'quadratic', 'cubic', 'krogh', 'pchip', 'akima', 'from_derivatives'
    }
    assert len(set([cln_conf['nan_fill_type']]) - valid_fills) == 0, f"{path}" + \
        "'nan_fill_type' value must be '' or within DataFrame.interpolate: " + \
        "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html"

    # TEST 8) Validate 'output_type' sub-config key ##########################
    assert (
        'output_type' in cln_conf.keys()
    ), f"{path}'output_type' key is not found in the config file."
    assert isinstance(
        cln_conf['output_type'], str
    ), f"{path}'output_type' value must be a str type."
    valid_outputs = {
        'reg',
        'regression',
        'class',
        'classification',
    }
    assert len(set([cln_conf['output_type']]) - valid_outputs) == 0, \
        f"{path}'output_type' value must contain one of the " + \
        f"following options: {valid_outputs}."

    validate_datetime_target(params)


def validate_transform(key_name: str, tran_conf: Dict, params: Dict) -> None:

    # TEST 1) Validate root 'key' in params ##################################
    assert (
        'key' in params.keys()
    ), "Missing the root key for this ETL block."
    assert isinstance(
        params['key'], str
    ), "'key' value in param must be a str type."
    assert params['key'] != '', "'key' value in param cannot be blank."
    path = f"'{params['key']}'->'{key_name}'->"

    # TEST 2) Validate 'alias' sub-config key (optional) #####################
    if 'alias' in tran_conf.keys():
        assert isinstance(
            tran_conf['alias'], str
        ), f"{path}'alias' value must be a str type."
        assert tran_conf['alias'] != '', f"{path}'alias' value cannot be blank."

    # TEST 3) Validate 'method' sub-config key ###############################
    assert (
        'method' in tran_conf.keys()
    ), f"{path}'method' key is not found in the config file."
    assert isinstance(
        tran_conf['method'], str
    ), f"{path}'method' value must be a str type."
    valid_methods = {
        'box-cox',
        'yeo-johnson',
        'log',
        'deseasonalize',
        'detrend',
        'residual-only',
    }
    assert len(set([tran_conf['method']]) - valid_methods) == 0, \
        f"{path}'method' value must only contain the following: {valid_methods}."

    # TEST 4) Validate 'standardize' sub-config key (optional) ###############
    if 'standardize' in tran_conf.keys():
        assert (
            'standardize' in tran_conf.keys()
        ), f"{path}'standardize' key is not found in the config file."
        assert isinstance(
            tran_conf['standardize'], bool
        ), f"{path}'standardize' value must be a bool type."

    # TEST 5) Validate 'decomposition_model' sub-config key (optional) #######
    if 'decomposition_model' in tran_conf.keys():
        assert (
            'decomposition_model' in tran_conf.keys()
        ), f"{path}'decomposition_model' key is not found in the config file."
        assert isinstance(
            tran_conf['decomposition_model'], str
        ), f"{path}'decomposition_model' value must be a str type."
        valid_decoms = {
            'additive',
            'multiplicative'
        }
        assert len(set([tran_conf['decomposition_model']]) - valid_decoms) == 0, \
            f"{path}'decomposition_model' value must only contain the " + \
            f"following: {valid_decoms}."

    validate_datetime_target(params)


def validate_supervise(key_name: str, sup_conf: Dict, params: Dict) -> None:

    # TEST 1) Validate root 'key' in params ##################################
    assert (
        'key' in params.keys()
    ), "Missing the root key for this ETL block."
    assert isinstance(
        params['key'], str
    ), "'key' value in param must be a str type."
    assert params['key'] != '', "'key' value in param cannot be blank."
    path = f"'{params['key']}'->'{key_name}'->"

    # TEST 2) Validate 'alias' sub-config key (optional) #####################
    if 'alias' in sup_conf.keys():
        assert isinstance(
            sup_conf['alias'], str
        ), f"{path}'alias' value must be a str type."
        assert sup_conf['alias'] != '', f"{path}'alias' value cannot be blank."

    # TEST 3) Validate 'training_period' sub-config key ######################
    assert (
        'training_period' in sup_conf.keys()
    ), f"{path}'training_period' key is not found in the config file."
    assert isinstance(
        sup_conf['training_period'], str
    ), f"{path}'training_period' value must be a str type."

    # TEST 4) Validate 'forecast_period' sub-config key ######################
    assert (
        'forecast_period' in sup_conf.keys()
    ), f"{path}'forecast_period' key is not found in the config file."
    assert isinstance(
        sup_conf['forecast_period'], str
    ), f"{path}'forecast_period' value must be a str type."

    # TEST 5) Validate 'validation_set' sub-config key (optional) ############
    if 'validation_set' in sup_conf.keys():
        assert (
            'validation_set' in sup_conf.keys()
        ), f"{path}'validation_set' key is not found in the config file."
        assert isinstance(
            sup_conf['validation_set'], str
        ), f"{path}'validation_set' value must be a str type."

    # TEST 6) Validate 'test_set' sub-config key #############################
    assert (
        'test_set' in sup_conf.keys()
    ), f"{path}'test_set' key is not found in the config file."
    assert isinstance(
        sup_conf['test_set'], str
    ), f"{path}'test_set' value must be a str type."

    # TEST 7) Validate 'max_gap' sub-config key ##############################
    assert (
        'max_gap' in sup_conf.keys()
    ), f"{path}'max_gap' key is not found in the config file."
    assert isinstance(
        sup_conf['max_gap'], float
    ), f"{path}'max_gap' value must be a float type."

    validate_datetime_target(params)
