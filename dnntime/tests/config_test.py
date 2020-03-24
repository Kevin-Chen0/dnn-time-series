# -*- coding: utf-8 -*-
import pandas as pd
import pytz
import re
from typing import Dict


def validate_config(config: Dict) -> None:
    """
    Validate the main root keys of the user config YAML file.

    Parameters
    ----------
    config : The contents in the config file in DataFrame format.

    """
    # TEST 1) Validate config's keys #########################################
    # Get the root config keys and remove all non-alphabetical chars
    root_keys = {re.sub('[^a-zA-Z]+', '', key) for key in config.keys()}
    # Assert that root keys only contain 'eda', 'etl', 'meta', and 'model'
    valid_roots = {'eda', 'etl', 'meta', 'model'}
    # Assert all root keys are subset of valid roots, otherwise process will fail.
    assert len(root_keys - valid_roots) == 0, f"{list(config.keys())} keys " + \
           f"must have only the following prefixes: {valid_roots}."

    # TEST 2) Validate 'meta' key ############################################
    assert (
        'meta' in config.keys()
    ), f"'meta' key is not found in the config file."
    meta = config['meta']
    path = "'meta'->"

    # TEST 3) Validate 'user_interface' subkey ###############################
    assert (
        'user_interface' in meta.keys()
    ), f"{path}'user_interface' key is not found in the config file."
    assert isinstance(
        meta['user_interface'], str
    ), f"{path}'user_interface' value must be a str type."
    valid_ui = {'notebook', 'console'}
    assert len(set([meta['user_interface']]) - valid_ui) == 0, \
        f"{path}'user_interface' value must contain one of the " + \
        f"following options: {valid_ui}."

    # TEST 4) Validate 'datetime_column' subkey ##############################
    assert (
        'datetime_column' in meta.keys()
    ), f"{path}'datetime_column' key is not found in the config file."
    assert isinstance(
        meta['datetime_column'], str
    ), "'datetime_column' value must be a str type."

    # TEST 5) Validate 'target_column' subkey ################################
    assert (
        'target_column' in meta.keys()
    ), f"{path}'target_column' key is not found in the config file."
    assert isinstance(
        meta['target_column'], str
    ), "'target_column' value must be a str type."