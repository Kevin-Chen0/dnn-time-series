# -*- coding: utf-8 -*-
import pandas as pd
import re
import tensorflow as tf
from typing import DefaultDict, Dict, Optional, Tuple, Union
# Block Basic Classes
from .base import Block, CheckpointDict
# Tests
from ..tests import model_test as test
# Models
from ..models import (
    RNNWrapper,
    LSTMWrapper,
    GRUWrapper,
    ConvLSTMWrapper,
)


class ModelBlock(Block):

    def __init__(self, data_dict: CheckpointDict, params: Dict) -> None:
        super().__init__(data_dict, params)

    def run_block(self, config: Dict) -> CheckpointDict:
        """
        

        Parameters
        ----------
        config : Dict

        Returns
        -------
        df : TYPE

        """
        super().run_block(config)

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

        test.validate_model(self.params['key'], config)

        for key in config.keys():
            keyword = re.sub('[^a-zA-Z]+', '', key)
            model = eval(f'self.run_{keyword}(key, config[key])')
            # assert data is not None, f"'{key}' resulted in no data, please review."
            if model is not None:
                name = key.title()
                if isinstance(config[key], dict):
                    if 'alias' in config[key].keys():
                        name = config[key]['alias']
                self.model_dict.save(model, name)
            else:
                print(f"Model has return None, result for sub-block " + \
                      f"'{self.params['key']}'->'{key}' is not saved.")
            self.substep_counter += 1

        self.params['step_number'] += 1
        return self.model_dict, self.params

    def run_dnn(self, key_name: str, config: Dict) -> pd.DataFrame:

        # VALIDATE necessary transform parameters before running procedure
        assert self.data_dict.current_key is not None, "Data_dict is empty, " + \
            "run_dnn() needs existing data to run."
        test.validate_dnn(key_name, config, self.params)
        # Initialize model as None, therefore, if it fails, then it will return None
        model_store = None
        # INITIALIZE transform config variables
        model_type = config['model_type']
        n_epoch = config['epochs']
        n_batch = config['batch_size']
        n_unit = config['number_units']  # number of units per layer
        d_rate = config['dropout_rate']  # dropout rate
        opt = config['optimizer']
        loss = config['objective_function']
        verbose = config['verbose']
        score_type = config['score_type']
        # Get most current dataset from data_dict
        key = self.data_dict.current_key
        data_curr = self.data_dict.get()[key]
        X_train = data_curr['X_train']
        y_train = data_curr['y_train']
        X_test = data_curr['X_test']
        y_test = data_curr['y_test']
        n_input = X_train.shape[1]
        n_output = y_train.shape[1]
        n_feature = X_train.shape[2]
        # Get params variables
        space = self.params['space']
        stepn = self.params['step_number']
        subn = self.substep_counter

        try:
            name = model_type.upper()
            self.print_bold(f"{stepn}.{subn}) {space}Running a {name} Model...", 
                            n_before=1, n_after=1)
            if model_type == 'rnn':
                wrapper = RNNWrapper(n_input, n_output, n_feature, n_unit,
                                     d_rate, opt, loss
                                     )
            elif model_type == 'lstm':
                wrapper = LSTMWrapper(n_input, n_output, n_feature, n_unit, 
                                      d_rate, opt, loss
                                      )
            elif model_type == 'gru':
                wrapper = GRUWrapper(n_input, n_output, n_feature, n_unit, 
                                     d_rate, opt, loss
                                     )
            elif model_type == 'convlstm':
                wrapper = ConvLSTMWrapper(n_steps=int(n_input/n_output),  # num of steps
                                          l_subseq=n_output,  # len of subsequence
                                          n_row=1,  # len of "image" row, can be left as 1
                                          n_col=n_output,   # len of "image" col
                                          n_feature=n_feature, n_unit=n_unit,
                                          d_rate=d_rate, optimizer=opt, loss=loss
                                          )
            wrapper.fit(X_train, y_train, n_epoch, n_batch, verbose)
            model, pred, score = wrapper.evaluate(X_test, y_test, score_type)
            model_store = {
                'model': model,
                'forecast': pred,
                f'{score_type}': score
                }
        except Exception as e:
            print(e)

        return model_store