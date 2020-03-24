# -*- coding: utf-8 -*-
import ast
import re
from typing import Dict
# Validate existence of datetime and target columns in the etl_test file
from .etl_test import validate_datetime_target


def validate_eda(key_name: str, config: Dict) -> None:
    """
    Validate the subkeys of this particular EDA config block.

    Parameters
    ----------
    key_name : The root key that maps to this EDA config block.
    config : The contents in the config file in DataFrame format.

    """
    # TEST 1) Validate config's keys #########################################
    # Get the given EDA config keys and remove all non-alphabetical chars
    eda_keys = {re.sub('[^a-zA-Z]+', '', key) for key in config.keys()}
    # Assert that eda keys only contain prefixes:
    valid_edas = {'general', 'statistical'}
    # Assert all eda keys are subset of valid edas, otherwise process will fail.
    assert len(eda_keys - valid_edas) == 0, "EDA config keys must contain " + \
           f"only the following subkeys: {valid_edas}. '{key_name}' " + \
           f"contains the following: {list(config.keys())}."


def validate_general(key_name: str, gen_conf: Dict, params: Dict) -> None:
    """
    Validate the general config subblock to be used for EDABlock general operation.

    Parameters
    ----------
    key_name : The subkey that maps to this extract subblock.
    gen_conf : The extract sub-config dict or config['extract'].
    params : Any additional passed-in params including datetime and target cols.

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

    # TEST 4) Validate 'y_label' sub-config key ##############################
    assert (
        'y_label' in gen_conf.keys()
    ), f"{path}'y_label' key is not found in the config file."
    assert isinstance(
        gen_conf['y_label'], str
    ), f"{path}'y_label' value must be a str type."

    # TEST 5) Validate 'figsize' sub-config key (optional) ###################
    if 'figsize' in gen_conf.keys():
        gen_conf_figsize = ast.literal_eval(gen_conf['figsize'])
        assert isinstance(
            gen_conf_figsize, tuple
        ), f"{path}'figsize' value must be a tuple type."
        assert len(gen_conf_figsize) == 2, f"{path}'figsize' must have length of 2."
        assert isinstance(
            gen_conf_figsize[0], int
        ), f"{path}'figsize[0]' value must be an int type."
        assert isinstance(
            gen_conf_figsize[1], int
        ), f"{path}'figsize[1]' value must be an int type."

   # TEST 6) Validate 'plotly' sub-config key (optional) #####################
    if 'plotly' in gen_conf.keys():
        assert isinstance(
            gen_conf['plotly'], bool
        ), f"{path}'plotly' value must be a bool type."

    validate_datetime_target(params)


def validate_statistical(key_name: str, stat_conf: Dict, params: Dict) -> None:
    """
    Validate the statistical config subblock to be used for EDABlock
    statistical operation.

    Parameters
    ----------
    key_name : The subkey that maps to this extract subblock.
    stat_conf : The extract sub-config dict or config['statistical'].
    params : Any additional passed-in params including datetime and target cols.

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

    # TEST 6) Validate 'figsize' sub-config key (optional) ####################
    if 'figsize' in stat_conf.keys():
        stat_conf_figsize = ast.literal_eval(stat_conf['figsize'])
        assert isinstance(
            stat_conf_figsize, tuple
        ), f"{path}'figsize' value must be a tuple type."
        assert len(stat_conf_figsize) == 2, f"{path}'figsize' must have length of 2."
        assert isinstance(
            stat_conf_figsize[0], int
        ), f"{path}'figsize[0]' value must be an int type."
        assert isinstance(
            stat_conf_figsize[1], int
        ), f"{path}'figsize[1]' value must be an int type."

   # TEST 7) Validate 'plotly' sub-config key (optional) #####################
    if 'plotly' in stat_conf.keys():
        assert isinstance(
            stat_conf['plotly'], bool
        ), f"{path}'plotly' value must be a bool type."

    validate_datetime_target(params)
