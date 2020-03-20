# -*- coding: utf-8 -*-
import re
from typing import Dict
# Validate existence of datetime and target columns in the etl_test file
from .etl_test import validate_datetime_target


def validate_eda(key_name: str, config: Dict) -> None:
    """

    Parameters
    ----------
    config : The contents in the config file in DataFrame format.

    """
    # TEST 1) Validate config's keys #########################################
    # Get the given EDA config keys and remove all non-alphabetical chars
    # eda_keys = set(config.keys())
    eda_keys = {re.sub('[^a-zA-Z]+', '', key) for key in config.keys()}
    # Assert that eda keys only contain prefixes:
    valid_edas = {'general', 'statistical'}
    # Assert all eda keys are subset of valid edas, otherwise process will fail.
    assert len(eda_keys - valid_edas) == 0, "EDA config keys must contain " + \
           f"only the following subkeys: {valid_edas}. '{key_name}' " + \
           f"contains the following: {list(config.keys())}."


def validate_general(key_name: str, gen_conf: Dict, params: Dict) -> None:
    """

    Parameters
    ----------
    gen_conf : The extract sub-config dict or config['extract'].

    """
    # TEST 1) Validate root 'key' in params ##################################
    assert (
        'key' in params.keys()
    ), "Missing the root key for this EDA block."
    assert isinstance(
        params['key'], str
    ), "'key' value in param must be a str type."
    assert params['key'] != '', "'key' value in param cannot be blank."
    path = f"'{params['key']}'->'{key_name}'->"

    # TEST 2) Validate 'alias' sub-config key (optional) #####################
    if 'alias' in gen_conf.keys():
        assert isinstance(
            gen_conf['alias'], str
        ), f"{path}'alias' value must be a str type."
        assert gen_conf['alias'] != '', "{path}'alias' value cannot be blank."

    # TEST 2) Validate 'title' sub-config key ################################
    assert (
        'title' in gen_conf.keys()
    ), f"{path}'title' key is not found in the config file."
    assert isinstance(
        gen_conf['title'], str
    ), f"{path}'title' value must be a str type."

    # TEST 3) Validate 'x_label' sub-config key ##############################
    assert (
        'x_label' in gen_conf.keys()
    ), f"{path}'x_label' key is not found in the config file."
    assert isinstance(
        gen_conf['x_label'], str
    ), f"{path}'x_label' value must be a str type."

    # TEST 4) Validate 'y_label' sub-config key ################################
    assert (
        'y_label' in gen_conf.keys()
    ), f"{path}'y_label' key is not found in the config file."
    assert isinstance(
        gen_conf['y_label'], str
    ), f"{path}'y_label' value must be a str type."

    validate_datetime_target(params)


def validate_statistical(key_name: str, stat_conf: bool, params: Dict) -> None:

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
    if 'alias' in stat_conf.keys():
        assert isinstance(
            stat_conf['alias'], str
        ), f"{path}'alias' value must be a str type."
        assert stat_conf['alias'] != '', f"{path}'alias' value cannot be blank."

    # TEST 2) Validate 'title' sub-config key ################################
    assert (
        'title' in stat_conf.keys()
    ), f"{path}'title' key is not found in the config file."
    assert isinstance(
        stat_conf['title'], str
    ), f"{path}'title' value must be a str type."

    # TEST 3) Validate 'x_label' sub-config key ##############################
    assert (
        'x_label' in stat_conf.keys()
    ), f"{path}'x_label' key is not found in the config file."
    assert isinstance(
        stat_conf['x_label'], str
    ), f"{path}'x_label' value must be a str type."

    # TEST 4) Validate 'y_label' sub-config key ##############################
    assert (
        'y_label' in stat_conf.keys()
    ), f"{path}'y_label' key is not found in the config file."
    assert isinstance(
        stat_conf['y_label'], str
    ), f"{path}'y_label' value must be a str type."

   # TEST 5) Validate 'confidence_interval' sub-config key ###################
    assert (
        'confidence_interval' in stat_conf.keys()
    ), f"{path}'confidence_interval' key is not found in the config file."
    assert isinstance(
        stat_conf['confidence_interval'], float
    ), f"{path}'confidence_interval' value must be a float type."

    validate_datetime_target(params)
