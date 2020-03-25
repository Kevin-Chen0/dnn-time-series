# -*- coding: utf-8 -*-
import art
import pandas as pd
import operator
import time
import yaml
import warnings
warnings.filterwarnings("ignore")
from typing import DefaultDict, Dict, Tuple

#################################################

# Tests
from .tests import config_test as test
# Blocks
from .blocks import (
    ETLBlock,
    EDABlock,
    ModelBlock,
    CheckpointDict
)


def run_package(
    config_file: str, data: pd.DataFrame = None
) -> Tuple[DefaultDict[str, Dict], DefaultDict[str, Dict]]:
    """
    The MASTER function of the entire dnntime package. The runs the entire DL
    pipeline end-to-end from a) loading data from source, b) ETL preprocessing,
    c) statistical EDA and visualization and d) model search and evaluations.

    Parameters
    ----------
    config_file : Provides all the (hyper)parameters needed to this function.
                  Each of its components will be validated. Must be a *.yaml file!
    data : DataFrame version of the data source. Not needed if the config_file
           already specifies a file_path the data source.

    Returns
    -------
    data_dict : A custom CheckpointDict dict that saves a copy of the data during
                during each and every point of its transformation.
    model_dict : A custom CheckpointDict dict that saves all the DNN models used,
                their forecasts, and scores.

    """

    start_time = time.time()

    # Load config YAML file
    assert config_file.endswith(
        ".yaml"
    ), "Config YAML file not found. Please check filepath."

    try:
        with open(config_file, "r") as file:
            content = file.read()
            config = yaml.safe_load(content)
    except FileNotFoundError as e:
        print(e)
        return None, None
    # Check the config dict to ensure it passes all of the assertions
    test.validate_config(config)
    # Initializing meta config variable(s) prior to STEPS
    try:
        ui = config['meta']['user_interface']
        space = "&nbsp" if ui == 'notebook' else " "
        dt_col = config['meta']['datetime_column']
        target = config['meta']['target_column']
    except KeyError as e:
        print(e)
        return None, None
    # Remove 'meta' key as it is unneeded
    del config['meta']
    # Initialize STEP counter
    step_counter = 1

    # Introductory texts
    art.tprint("Running   DNN\ntime-series\npackage...")
    print("-------------------------------------------------------------------")
    print("-------------------------------------------------------------------\n")
    print("SUMMARY STEPS:")
    for key in config.keys():
        print(f"    STEP {step_counter}) {config[key]['description']}")
        step_counter += 1

    # Reset counter
    step_counter = 1
    # Add options for space
    space = "&nbsp" if ui == 'notebook' else " "
    # Store all of the data and models checkpoints in a dict to be returned
    data_dict = CheckpointDict('data')
    model_dict = CheckpointDict('model')
    # Add additional parameter from the 'meta' block initially
    params = {'ui': ui,
              'space': space,
              'dt_col': dt_col,
              'target': target,
              'step_number': step_counter
              }

    # Now do each step one by one
    for key in config.keys():
        params['key'] = key
        if 'etl' in key:
            data_dict, params = ETLBlock(data_dict, params).run_block(config[key])
        elif 'eda' in key:
            params = EDABlock(data_dict, params).run_block(config[key])
        elif 'model' in key:
            model_dict, params = ModelBlock(data_dict, params).run_block(config[key])
        else:
            print(f"{key} is not 'etl', 'eda', or 'model'. Stopping program "
                  "now, please fix.")
            return data_dict.get(), model_dict.get()

    # Print out the best DL model based on lowest given score_type
    final_stats = {}
    score_type = params['score_type']
    for key in model_dict.get().keys():
        final_stats[key] = model_dict.get()[key][score_type]
    best_model_name = min(final_stats.items(), key=operator.itemgetter(1))[0]

    print("\n-----------------------------------------------------------------")
    print("-----------------------------------------------------------------")
    print("\nThe most accurate deep learning model is:")
    print(f"    {best_model_name}")
    best_score = model_dict.get()[best_model_name][score_type]
    print(f"    {score_type.upper()} score: {best_score:.4f}")

    end_time = time.time()
    run_time = end_time - start_time
    print(f"\nTotal package runtime: {(run_time/60):.2f} min")

    return data_dict.get(), model_dict.get()
