# -*- coding: utf-8 -*-
import re
from typing import Dict


def validate_model(key_name: str, config: Dict) -> None:
    """
    Validate the subkeys of this particular Model config block.

    Parameters
    ----------
    key_name : The root key that maps to this EDA config block.
    config : The contents in the config file in DataFrame format.

    """
    # TEST 1) Validate config's keys #########################################
    # Get the given ETL config keys and remove all non-alphabetical chars
    model_keys = {re.sub('[^a-zA-Z_]+', '', key) for key in config.keys()}
    # Assert that etl keys only contain prefixes:
    valid_models = {'enable_gpu', 'dnn'}
    assert len(model_keys - valid_models) == 0, "Model config keys must contain " + \
           f"only the following subkeys: {valid_models}. '{key_name}' " + \
           f"contains the following: {list(config.keys())}."
    path = f"'{key_name}'-> "

    # TEST 2) Validate 'enable_gpu' sub-config key (optional)  ###############
    if 'enable_gpu' in config.keys():
        assert isinstance(
            config['enable_gpu'], bool
        ), f"{path}'enable_gpu' value must be a bool type."


def validate_dnn(key_name: str, dnn_conf: Dict, params: Dict) -> None:
    """
    Validate the dnn config subblock to be used for ModelBlock dnn operation.

    Parameters
    ----------
    key_name : The subkey that maps to this extract subblock.
    dnn_conf : The extract sub-config dict or config['dnn'].
    params : Any additional passed-in params including datetime and target cols.

    """
    # TEST 1) Validate root 'key' in params ##################################
    assert (
        'key' in params.keys()
    ), "Missing the root key for this Model block."
    assert isinstance(
        params['key'], str
    ), "'key' value in param must be a str type."
    assert params['key'] != '', "'key' value in param cannot be blank."
    path = f"'{params['key']}'->'{{key_name}}'->"

    # TEST 2) Validate 'alias' sub-config key (optional) #####################
    if 'alias' in dnn_conf.keys():
        assert isinstance(
            dnn_conf['alias'], str
        ), f"{path}'alias' value must be a str type."
        assert dnn_conf['alias'] != '', f"{path}'alias' value cannot be blank."

    # TEST 3) Validate 'model_type' sub-config key ##################################
    assert (
        'model_type' in dnn_conf.keys()
    ), f"{path}'model_type' key is not found in the config file."
    assert isinstance(
        dnn_conf['model_type'], str
    ), f"{path}'model_type' value must be a str type."

    # TEST 4) Validate 'epochs' sub-config key ##################################
    assert (
        'epochs' in dnn_conf.keys()
    ), f"{path}'epochs' key is not found in the config file."
    assert isinstance(
        dnn_conf['epochs'], int
    ), f"{path}'epochs' value must be an int type."

    # TEST 5) Validate 'batch_size' sub-config key ##################################
    assert (
        'batch_size' in dnn_conf.keys()
    ), f"{path}'batch_size' key is not found in the config file."
    assert isinstance(
        dnn_conf['batch_size'], int
    ), f"{path}'batch_size' value must be an int type."

    # TEST 6) Validate 'number_layers' sub-config key ##################################
    assert (
        'number_layers' in dnn_conf.keys()
    ), f"{path}'number_layers' key is not found in the config file."
    assert isinstance(
        dnn_conf['number_layers'], int
    ), f"{path}'number_layers' value must be an int type."

    # TEST 7) Validate 'number_units' sub-config key ##################################
    assert (
        'number_units' in dnn_conf.keys()
    ), f"{path}'number_units' key is not found in the config file."
    assert isinstance(
        dnn_conf['number_units'], int
    ), f"{path}'number_units' value must be an int type."

    # TEST 8) Validate 'dropout_rate' sub-config key ##################################
    assert (
        'dropout_rate' in dnn_conf.keys()
    ), f"{path}'dropout_rate' key is not found in the config file."
    assert isinstance(
        dnn_conf['dropout_rate'], float
    ), f"{path}'dropout_rate' value must be an float type."

    # TEST 9) Validate 'optimizer' sub-config key ##################################
    assert (
        'optimizer' in dnn_conf.keys()
    ), f"{path}'optimizer' key is not found in the config file."
    assert isinstance(
        dnn_conf['optimizer'], str
    ), f"{path}'optimizer' value must be a str type."

    # TEST 10) Validate 'objective_function' sub-config key ##################################
    assert (
        'objective_function' in dnn_conf.keys()
    ), f"{path}'objective_function' key is not found in the config file."
    assert isinstance(
        dnn_conf['objective_function'], str
    ), f"{path}'objective_function' value must be a str type."

    # TEST 11) Validate 'verbose' sub-config key ##################################
    assert (
        'verbose' in dnn_conf.keys()
    ), f"{path}'verbose' key is not found in the config file."
    assert isinstance(
        dnn_conf['verbose'], int
    ), f"{path}'verbose' value must be an int type."

    # TEST 12) Validate 'score_type' sub-config key ##################################
    assert (
        'score_type' in dnn_conf.keys()
    ), f"{path}'score_type' key is not found in the config file."
    assert isinstance(
        dnn_conf['score_type'], str
    ), f"{path}'score_type' value must be a str type."
