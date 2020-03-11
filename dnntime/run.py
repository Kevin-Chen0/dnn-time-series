# -*- coding: utf-8 -*-
from IPython.display import display, HTML
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import yaml
import ipdb
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Optional, Set, Tuple, cast
import operator
import time
import art
import tensorflow as tf

################################################

# Models
from .models import (
    RNNWrapper,
    LSTMWrapper,
    GRUWrapper,
    ConvLSTMWrapper,
)
# Tests
from .tests import validate_config
# Utils
from .utils.etl import load_data, clean_data
# time_series_plot, print_static_rmse, print_dynamic_rmse
from .utils.etl import log_power_transform, decompose, split_data
from .utils.ts import interval_to_freq, interval_to_timesteps
from .utils.eda import ts_plot, ets_decomposition_plot, acf_pacf_plot, \
                       adf_stationary_test


class CheckpointDict:

    def __init__(self, cp_type: str):
        self.dict = defaultdict(dict)
        self.type = cp_type
        if self.type == 'data':
            self.counter = 0
        else:
            self.counter = 1

    def save(self, obj: Any, name: str) -> None:
        new_key = f'{self.counter}) {name}'
        self.dict[new_key] = obj
        if isinstance(obj, pd.DataFrame):
            print(f"\n--> {name} {self.type} saved in {self.type}" + \
                  f"_dict[{new_key}]. See head below:")
            display(HTML(obj.head().to_html()))
            print("\n    See tail below:")
            display(HTML(obj.tail().to_html()))
            print()
        else:
            print(f"\n--> {name} {self.type} saved in {self.type}" + \
                  f"_dict[{new_key}].\n")
        self.counter += 1

    def get(self) -> DefaultDict[str, Dict]:
        return self.dict


def print_bold(
    text: str, ui: str = 'notebook', n_before: int = 0, n_after: int = 0
) -> None:

    if ui == 'notebook':
        nl = "<br>"
        display(HTML(f"{nl*n_before}<b>{text}</b>{nl*(n_after+1)}"))
    elif ui == 'console':
        nl = "\n"
        print(f"\033[1m{nl*n_before}{text}{nl*n_after}\033[0m")


def run_package(
    config_file: str, data: pd.DataFrame = None
) -> Tuple[DefaultDict[str, Dict], DefaultDict[str, Dict]]:
    """
    ABOUT: ff
    --------------------------------------------------------------------------
    INPUTS:
    --------------------------------------------------------------------------
    data: 
    ts_column: 
    sep:
    target: 
    """

    start_time = time.time()

    ### Introductory texts:
    art.tprint("Running   deep\ntime-series\npackage...")
    print("-------------------------------------------------------------------")
    print("-------------------------------------------------------------------\n")
    print("SUMMARY STEPS:")
    print("    STEP 1) Extract Data from Source")
    print("    STEP 2) Preprocessing I (Cleaning)")
    print("    STEP 3) EDA I (General)")
    print("    STEP 4) EDA II (Time-Series Stats)")
    print("    STEP 5) Preprocessing II (Transformations)")
    print("    STEP 6) Preprocessing III (Make Supervised)")
    print("    STEP 7) Model Search (DNNs)")
    print("    STEP 8) Model Results Output")

    assert config_file.endswith(
        ".yaml"
    ), "Config YAML file not found. Please check filepath."

    try:
        with open(config_file, "r") as file:
            content = file.read()
            config = yaml.safe_load(content)
    except:
        print("Something wrong with YAML file, please check.")
        return

    # Check the config dict to ensure it passes all of the assertions
    validate_config(config)
    # Initializing meta config variable(s) prior to STEPS
    ui = config['meta']['user_interface']
    space = "&nbsp" if ui == 'notebook' else " "
    # Store all of the data ETL checkpoints in a dict that will be returned
    data_dict = CheckpointDict('data')


    ########## This is where we start the loading of the data file ######################
    print("\n\n-------------------------------------------------------------------")
    print_bold(f"STEP 1) {space}Extract Data from Source", ui)
    print("-------------------------------------------------------------------\n")

    # Initializing immutable config variables for STEP 1) and beyond
    extract = config['extract']
    file_path = extract['file_path']
    delinator = extract['delineator'] if extract['delineator'] != '' else ','
    dt_col = extract['datetime_column']  # data column that contains time-series
    target = extract['target_column']    # data column that contains y-value

    if file_path != '':
        assert file_path.endswith(
            ".csv"
        ), "Dataset CSV file not found. Please check filepath."
        # print(f"Extracting data from file source: '{FILE_PATH}'.")
        df = load_data(file_path, dt_col, delinator)
    elif data is not None:
        df = load_data(data, dt_col, delinator)
        print(f"Using inputted DataFrame.")
    else:
        print("Data input not found in either file source or DataFrame.")
        return
    
    data_dict.save(df, 'Original')


    print("\n\n-------------------------------------------------------------------")
    print_bold(f"STEP 2) {space}Preprocessing I (Cleaning)", ui)
    print("-------------------------------------------------------------------\n")

    # Initializing immutable config variables for STEP 2) and beyond
    preprocess = config['preprocess']
    univariate = preprocess['univariate']
    time_interval = preprocess['time_interval']
    auto_clean = preprocess['auto_clean']

    if univariate:
        print(f"Set the dataset to univarate using target col of {target}.")
        df = df[target].to_frame()
        data_dict.save(df, 'Univarate')

    freq, _ = interval_to_freq(time_interval)
    print(f"Frequency has been set to {freq}.\n")

    if auto_clean:
        # Initializing immutable config variables for Auto-Clean specifically
        conf = preprocess['auto_clean_conf']
        timezone = conf['timezone']
        allow_neg = conf['negative_values']
        fill = conf['nan_fill_type']

        print("Begin initial cleaning of the extract dataset...")
        df = clean_data(df, target, timezone, freq, allow_neg, fill)
        data_dict.save(df, 'Clean')
    else:
        print("Automatic cleaning of data has been skipped.")


    print("\n\n-------------------------------------------------------------------")
    print_bold(f"STEP 3) {space}EDA I (General)", ui)
    print("-------------------------------------------------------------------\n")

    # Initializing immutable config variables for STEP 3) and beyond
    analyze = config['analyze']
    ci = analyze['confidence_interval']
    title = analyze['title']
    x_label = analyze['x_label']
    y_label = analyze['y_label']

    # Prevent errors
    pd.plotting.register_matplotlib_converters()

    # Define data with the defined plot labels in the config file
    print("Plot the entire time-series data:\n")
    ts_plot(df, dt_col, target,
            title=title,
            x_label=x_label,
            y_label=y_label
            )


    print("\n\n-------------------------------------------------------------------")
    print_bold(f"STEP 4) {space}EDA II (Time-Series Stats)", ui)
    print("-------------------------------------------------------------------\n")

    # Initializing immutable config variables for STEP 4) and beyond
    ci = analyze['confidence_interval']

    print_bold(f"4.1) {space}Testing stationarity using Augmented Dickey-Fuller (ADF).", 
               ui, n_after=1)
    stationarity = adf_stationary_test(df, 1-ci)
    if stationarity:
        print(f"Current data is stationary with {ci*100}% " + \
              "confidence interval.\n")
    else:
        print(f"Current data is non-stationary with {ci*100}% " + \
              "confidence interval.\n")

    print_bold(f"4.2) {space}Printing out ETS decomposition plot.", ui, n_before=1, n_after=1)
    # ets = ets_decomposition_plot(ts_df, ts_column, target, title, y_label);
    # ets = ets_decomposition_plot(ts_df, ts_column, target, title, y_label,
    #                        prophet=True);
    ets = ets_decomposition_plot(df, dt_col, target,
                                 title=title,
                                 x_label=x_label,
                                 y_label=y_label,
                                 plotly=True);

    print_bold(f"4.3) {space}Plot out ACF/PACF graph..", ui, n_before=1, n_after=1)
    title = "Total Electricity Demand"
    lags_7 = 24*7  # 7-days lag
    lags_30 = 24*30  # 30-days lag
    lags_90 = 24*90  # 90-days lag
    acf_pacf_plot(df, target, title, lags=[lags_7, lags_30])

    # print_bold(f"4.4) {space}Expotential Smoothing Holt-Winters.", ui,
    #            n_before=1, n_after=1)

    # print_bold(f"4.5) {space}ARIMA.", ui, n_before=1)


    print("\n\n-------------------------------------------------------------------")
    print_bold(f"STEP 5) {space}Preprocessing II (Transformations)", ui)
    print("-------------------------------------------------------------------\n")

    # Initializing immutable config variables for STEP 5) and beyond
    transform = config['transform']
    trans_steps = transform['steps']
    decom_model = transform['decomposition_model']
    standardize = transform['standardize']

    substep = 1

    for step in trans_steps:
        standardize_note = ''
        if step in ['box-cox', 'yeo-johnson', 'log']:
            # Performs log or power transform and then normalize in one function
            info = f"5.{substep}) Performed"
            if step in ['box-cox', 'yeo-johnson']:
                info += f" power transformation using {step.title()} method."
            else:
                info += " log transformation."
            if standardize:
                info += " Then standardized data."
                standardize_note = ' Standardized'
            df, trans_type = log_power_transform(df, target, method=step,
                                                 standardize=standardize)
        elif step in ['detrend', 'deseasonalize', 'residual-only']:
            info = f"5.{substep}) Performed the following adjustment: {step.title()}." 
            df, decom_type = decompose(df, target, decom_type=step, decom_model=decom_model)

        print_bold(f"{info}", ui)
        data_dict.save(df, step.title()+standardize_note) 
        substep += 1
        print()

    ##################################################################################################
    ### Transform dataset into supervised ML problem with walk-forward validation.
    ### Shifting time-steps
    ##################################################################################################
    print("\n-------------------------------------------------------------------")
    print_bold(f"STEP 6) {space}Preprocessing III (Make Supervised)", ui)
    print("-------------------------------------------------------------------\n")

    # Initializing immutable config variables for STEP 6) and beyond
    supervise = config['supervise']
    train_period = supervise['training_period']
    fcast_period = supervise['forecast_period']
    val_set = supervise['validation_set']
    test_set = supervise['test_set']
    max_gap = supervise['max_gap']

    # Converting all 'str' periods into #timesteps based of the stated 'freq'
    n_input = interval_to_timesteps(train_period, freq)  # num input timesteps
    n_output = interval_to_timesteps(fcast_period, freq)  # num output timesteps
    n_val = interval_to_timesteps(val_set, freq)  # validation dataset size
    n_test = interval_to_timesteps(test_set, freq)  # test dataset size
    print("Performing walk-forward validation.")

    orig, train, val, test = split_data(df,
                                        n_test=n_test,  # size of test set
                                        n_val=n_val,  # size of validation set
                                        n_input=n_input,   # input timestep seq
                                        n_output=n_output, # output timestep seq
                                        g_min=0,     # min gap ratio
                                        g_max=max_gap)  # max gap ratio

    X, y, t = orig  # original data tuple in supervised format
    X_train, y_train, t_train = train
    X_val, y_val, t_val = val
    X_test, y_test, t_test = test

    print("Converted time-series into supervised leraning problem using walk-forward validation:")
    print(f"    Time-series frequency: '{freq}'.")
    print(f"    Input period: {X.shape[1]} timesteps, or 'bikweek'.")
    print(f"    Output (forecast) period: {y.shape[1]} timesteps, or 'day'.")
    print(f"    Original dataset: {df.shape[0]} observations.")
    print(f"    Supervised dataset: {X.shape[0]} observations.")
    print(f"    Training dataset: {X_train.shape[0]} observations.")
    print(f"    Validation dataset: {X_val.shape[0]} observations, or '{val_set}'.")
    print(f"    Testing dataset: {X_test.shape[0]} observations, or '{test_set}'.")

    train_prct = len(X_train)/len(X)*100
    val_prct = len(X_val)/len(X)*100
    test_prct = len(X_test)/len(X)*100
    gap_prct = 100 - train_prct - val_prct - test_prct
    print("\nSplit %:")
    print(f"Train: {train_prct:.2f}%, Val: {val_prct:.2f}%, Test: " + \
          f"{test_prct:.2f}%, Gap: {gap_prct:.2f}%")

    print("\nDataset shapes:")
    print(f"    Original:")
    print(f"        data shape = {df.shape}")
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

    data_dict.save(data, 'Make Supervised')


    print("\n\n-------------------------------------------------------------------")
    print_bold(f"STEP 7) {space}Model Search (NNs)", ui)
    print("-------------------------------------------------------------------\n")

    if len(tf.config.list_physical_devices('GPU')) > 0:
        print("GPU is enabled.")
    else:
        print("Running on CPU as GPU is not enabled.")

    # Initializing immutable config variables for STEP 7) and beyond
    dnn = config['dnn']
    model_type = dnn['model_type']
    n_epoch = dnn['epochs']
    n_batch = dnn['batch_size']
    n_features = 1 if univariate else dnn['n_features']  # number of features
    n_units = dnn['n_units']  # number of units per layer
    d_rate = dnn['d_rate']  # dropout rate
    opt = dnn['optimizer']
    loss = dnn['objective_function']
    verbose = dnn['verbose']

    evaluate = config['evaluate']
    score_type = evaluate['score_type']

    # Store all of the data ETL checkpoints in a dict that will be returned
    model_dict = CheckpointDict('model')

    # Begin the DL model running process #####################################
    if model_type.lower() in ['rnn', 'all']:
        name = 'RNN'
        print_bold(f"7.1) {space}Running a RNN Model...", ui, n_before=1, n_after=1)
        rnn = RNNWrapper(n_input, n_output, n_features, n_units, d_rate, opt, loss)
        rnn.fit(X_train, y_train, n_epoch, n_batch, verbose)
        model, pred, score = rnn.evaluate(X_test, y_test, score_type)
        model_store = {
            'model': model,
            'forecast': pred,
            f'{score_type}': score
            }
        model_dict.save(model_store, name)

    if model_type.lower() in ['lstm', 'all']:
        name = 'LSTM'
        print_bold(f"7.2) {space}Running a LSTM Model...", ui, n_before=1, n_after=1)
        lstm = LSTMWrapper(n_input, n_output, n_features, n_units, d_rate, opt, loss)
        lstm.fit(X_train, y_train, n_epoch, n_batch, verbose)
        model, pred, score = lstm.evaluate(X_test, y_test, score_type)
        model_store = {
            'model': model,
            'forecast': pred,
            f'{score_type}': score
            }
        model_dict.save(model_store, name)

    if model_type.lower() in ['gru', 'all']:
        name = 'GRU'
        print_bold(f"7.3) {space}Running a GRU Model...", ui, n_before=1, n_after=1)
        gru = GRUWrapper(n_input, n_output, n_features, n_units, d_rate, opt, loss)
        gru.fit(X_train, y_train, n_epoch, n_batch, verbose)
        model, pred, score = gru.evaluate(X_test, y_test, score_type)
        model_store = {
            'model': model,
            'forecast': pred,
            f'{score_type}': score
            }
        model_dict.save(model_store, name)

    if model_type.lower() in ['convlstm', 'all']:
        name = 'CONVLSTM'
        print_bold(f"7.4) {space}Running a ConvLSTM Model...", ui, n_before=1, n_after=1)
        conv = ConvLSTMWrapper(n_steps=int(n_input/n_output), # num of steps
                               l_subseq=n_output, # len of subsequence
                               n_row=1,  # len of "image" row, can be left as 1
                               n_col=n_output,    # len of "image" col
                               n_features=n_features, n_units=n_units,
                               d_rate=d_rate, optimizer=opt, loss=loss
                               )
        conv.fit(X_train, y_train, n_epoch, n_batch, verbose)
        model, pred, score = conv.evaluate(X_test, y_test, score_type)
        model_store = {
            'model': model,
            'forecast': pred,
            f'{score_type}': score
            }
        model_dict.save(model_store, name)

    # Print out the best DL model based on lowest given score_type
    final_stats = {}
    for key, val in model_dict.get().items():
        final_stats[key] = model_dict.get()[key][score_type]
    best_model_name = min(final_stats.items(), key=operator.itemgetter(1))[0]

    print("\n-----------------------------------------------------------------")
    print("-----------------------------------------------------------------")
    print_bold("The best deep learning model is:", ui, n_before=1)
    print(f"    {best_model_name}")
    best_score = model_dict.get()[best_model_name][score_type]
    print(f"    {score_type.upper()} score: {best_score:.4f}")
    end_time = time.time()
    run_time = end_time - start_time
    print(f"\nTotal package runtime: {(run_time/60):.2f} min")

    return data_dict.get(), model_dict.get()
