# -*- coding: utf-8 -*-
import pandas as pd
import re
import tensorflow as tf
from typing import Dict, Tuple
# Blocks base modules
from .base import Block, CheckpointDict
# Tests
from ..tests import model_test as test
# DNN Models
from ..models import (
    RNNWrapper,
    LSTMWrapper,
    GRUWrapper,
    ConvLSTMWrapper,
)


class ModelBlock(Block):

    def __init__(self, data_dict: CheckpointDict, params: Dict) -> None:
        """
        ModelBlock inherits from Block. It performs the DNN model training and
        prediction of the processed time-series data. This is usually the final
        or penultimate Block of the dnntime run_package, once all the ETL and
        EDA preprocessing steps have been completed.

        Parameters
        ----------
        data_dict: The record of data transformations.
        params: Any additional parameters passed from the results of previous Blocks.

        """
        super().__init__(data_dict, params)

    def run_block(self, config: Dict) -> Tuple[CheckpointDict, Dict]:
        """
        Executes the ModelBlock function on the data_dict based on the user's
        config YAML file as well as preexisting params from initialization. It
        takes the latest data from the input data_dict, trains the specified DNN
        model(s), and saves these model architecture(s) and result(s) into
        model_dict. Finally, it returns this newly modified model_dict as well
        as any supplementary params.
        Here is the following Model operation:
            1) Dnn: Trains and get results of the specified DNN model(s).

        Parameters
        ----------
        config: The specified config block from the user YAML file.

        Returns
        -------
        self.model_dict : The latest model_dict including the specified DNNs models.
        self.params : Any additional generated parameters for subsequent Blocks.
        
        """
        super().run_block(config)

        test.validate_model(self.params['key'], config)

        gpu = False
        if 'enable_gpu' in config.keys():
            gpu = config['enable_gpu']
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) == 0:
            print("GPU cannot be physically found, running on CPU...")
        elif not gpu:
            tf.config.set_visible_devices([], 'GPU')
            print("GPU is disabled by user, running on CPU...")
        else:
            print("GPU is enabled, running on GPU...")
        del config['enable_gpu']

        for key in config.keys():
            keyword = re.sub('[^a-zA-Z]+', '', key)
            model = eval(f'self.run_{keyword}(key, config[key])')
            # assert data is not None, f"'{key}' resulted in no data, please review."
            if model is not None:
                # Iterate through and save all the DNN models into data_dict
                if config[key]['model_type'] == 'all':
                    for model_key in model.keys():
                        name = model_key.upper()
                        self.model_dict.save(model, name)
                # Save only the specified DNN model into data_dict
                else:
                    name = key.title()
                    if isinstance(config[key], dict):
                        if 'alias' in config[key].keys():
                            name = config[key]['alias']
                    self.model_dict.save(model, name)
            else:
                print(f"Model has return None, result for sub-block " + \
                      f"'{self.params['key']}'->'{key}' is not saved.")

        self.params['step_number'] += 1
        return self.model_dict, self.params

    def run_dnn(self, key_name: str, config: Dict) -> Dict:
        """
        Model operation that trains and gathers results of either a specified
        DNN model or all DNN models.

        Parameters
        ----------
        key_name : The key for this config block, usually as 'dnn'{#num}.
        config : Specifies the DNN model as well as its hyperparameters.

        Returns
        -------
        model_store : The model architectured and results to be stored in the
                      Block's model_dict.

        """
        # VALIDATE necessary transform parameters before running procedure
        assert self.data_dict.current_key is not None, "Data_dict is empty, " + \
            "run_dnn() needs existing data to run."
        test.validate_dnn(key_name, config, self.params)
        # Initialize model as None, therefore, if it fails, then it will return None
        model_store = None
        # INITIALIZE transform config variables
        model_type = config['model_type']
        # Get most current dataset from data_dict
        key = self.data_dict.current_key
        data_curr = self.data_dict.get()[key]
        # Get params variables

        try:
            if model_type == 'all':
                model_store = self._run_all_models(data_curr, config)
            else:
                model_store = self._run_custom_model(model_type, data_curr, config)
        except Exception as e:
            print(e)

        return model_store

    def _run_all_models(self, data_curr: Dict, config: Dict):
        """
        Helper function to iteratively run through all DNN models.

        Parameters
        ----------
        data_curr : The latest data from data_dict to be trained and tested on.
        config : Specifies the hyperparameters shared by all the DNN models.

        Returns
        -------
        model_all : Stores all DNN model architectures and their predictive results.

        """
        model_all = {}
        model_list = ['rnn', 'lstm', 'gru', 'convlstm']

        for model_type in model_list:
            model_store = self._run_custom_model(model_type, data_curr, config)
            model_all[model_type] = model_store

        return model_all

    def _run_custom_model(self, model_type: str, data_curr: Dict, config: Dict):
        """
        Helper function to run through a particular DNN model.

        Parameters
        ----------
        model_type : The name of the DNN model to run.
        data_curr : The latest data from data_dict to be trained and tested on.
        config : Specifies the hyperparameters shared by all the DNN models.

        Returns
        -------
        model_store : Stores the specified DNN model architecture and its
                      predictive results.

        """
        # INITIALIZE transform config variables
        n_epoch = config['epochs']
        n_batch = config['batch_size']
        n_layer = config['number_layers']  # number of layers in the DNN
        n_unit = config['number_units']  # number of units per layer
        d_rate = config['dropout_rate']  # dropout rate
        opt = config['optimizer']
        loss = config['objective_function']
        verbose = config['verbose']
        score_type = config['score_type']
        # Get most current dataset from data_dict
        X_train = data_curr['X_train']
        y_train = data_curr['y_train']
        X_test = data_curr['X_test']
        y_test = data_curr['y_test']
        n_input = X_train.shape[1]
        n_output = y_train.shape[1]
        n_feature = X_train.shape[2]

        space = self.params['space']
        stepn = self.params['step_number']
        subn = self.substep_counter
        name = model_type.upper()

        self.print_bold(f"{stepn}.{subn}) {space}Running a {name} Model...",
                        n_before=1, n_after=1)

        if model_type == 'rnn':
            wrapper = RNNWrapper(n_input, n_output, n_feature, n_layer, n_unit,
                                 d_rate, opt, loss
                                 )
        elif model_type == 'lstm':
            wrapper = LSTMWrapper(n_input, n_output, n_feature, n_layer, n_unit,
                                  d_rate, opt, loss
                                  )
        elif model_type == 'gru':
            wrapper = GRUWrapper(n_input, n_output, n_feature, n_layer, n_unit,
                                 d_rate, opt, loss
                                 )
        elif model_type == 'convlstm':
            wrapper = ConvLSTMWrapper(n_steps=int(n_input/n_output),  # num of steps
                                      l_subseq=n_output,  # len of subsequence
                                      n_row=1,  # len of "image" row, can be left as 1
                                      n_col=n_output,   # len of "image" col
                                      n_feature=n_feature, n_layer=n_layer,
                                      n_unit=n_unit, d_rate=d_rate,
                                      optimizer=opt, loss=loss
                                      )

        wrapper.fit(X_train, y_train, n_epoch, n_batch, verbose)
        model, pred, score = wrapper.evaluate(X_test, y_test, score_type)
        model_store = {
            'model': model,
            'forecast': pred,
            f'{score_type}': score
            }

        self.substep_counter += 1
        return model_store
