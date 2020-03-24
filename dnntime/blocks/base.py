# -*- coding: utf-8 -*-
import pandas as pd
from collections import defaultdict
from tensorflow import keras
from typing import DefaultDict, Dict, Union
from IPython.display import display, HTML


class CheckpointDict:

    def __init__(self, cp_type: str) -> None:
        """
        CheckpointDict is a custom class that stores all of the checkpoints or
        snapshots of either the data or models. Its internal defaultdict is returned
        in the run_package() function to be used for later analyses or debugging.

        Parameters
        ----------
        cp_type : Checkpoint type. Options are 'data' or 'model'.

        """
        self.dict = defaultdict(dict)
        self.type = cp_type
        self.current_key = None
        if self.type == 'data':
            self.counter = 0
        else:
            self.counter = 1

    def save(self, obj: Union[pd.DataFrame, keras.Sequential], name: str) -> None:
        """
        Saves the inputted obj, whether data (pd.DataFrame or np.ndarray format)
        or model (keras.Sequential format).

        Parameters
        ----------
        obj : The actual data or model object.
        name : The key used to lookup this obj in the underlying defaultdict.

        """
        new_key = f'{self.counter}) {name}'
        self.dict[new_key] = obj
        if isinstance(obj, pd.DataFrame):
            print(f"\n--> {name} {self.type} saved in {self.type}"
                  f"_dict[{new_key}]. See head below:")
            display(HTML(obj.head().to_html()))
            print("\n    See tail below:")
            display(HTML(obj.tail().to_html()))
            print()
        else:
            print(f"\n--> {name} {self.type} saved in {self.type}"
                  f"_dict[{new_key}].\n")

        self.current_key = new_key
        self.counter += 1

    def get(self) -> DefaultDict[str, Dict]:
        """
        Retrieves CheckpointDict's internal defaultdict.

        Returns
        -------
        self.dict : The internal defaultdict that store of obj checkpoints.

        """
        return self.dict


class Block:

    def __init__(self, data_dict: CheckpointDict, params: Dict,
                 model_dict: CheckpointDict = CheckpointDict('model')
                 ) -> None:
        """
        Block is the most basic building module of the dnntime.
        It structures how each STEP is executed, its input parameters from prior
        Block, and the outputs that can be used to subsequent Block. dnntime's
        run_package() is basically made up of a "chain" or Blocks that is 
        configured by the user's YAML file. More specific types of Blocks,
        including ETLBlock, ETABlock, and ModelBlock inherit from Block.

        Parameters
        ----------
        data_dict: The recorded data transformation.
        params: Any additional parameters passed from the results of previous Blocks.
        model_dict: The recorded model architectures, params, and results.
        cp_type : Checkpoint type. Options are 'data' or 'model'.

        """
        self.data_dict = data_dict
        self.model_dict = model_dict
        self.params = params
        self.substep_counter = 1

    def run_block(self, config: Dict) -> None:
        """
        Execute the Block function on the data_dict and/or model_dict based on
        the user's config YAML file as well preexisting params from initialization.

        Parameters
        ----------
        config: The specified config block from the user YAML file.


        """
        print("\n\n-------------------------------------------------------------------")
        self.print_bold(f"STEP {self.params['step_number']}) {config['description']}")
        print("-------------------------------------------------------------------\n")
        del config['description']

    def print_bold(self, text: str, n_before: int = 0, n_after: int = 0) -> None:
        """
        A global function that prints out a given text in bold format. It uses
        two different fonts depending on whether the user interface or ui is a
        'console' or 'notebook'.

        Parameters
        ----------
        text : The text to printed out.
        ui : Either 'console' (default) or 'notebook'.
        n_before : Number of newlines added before the actual text for formatting.
        n_after : Number of newlines added after the actual text for formatting.

        """
        if self.params['ui'] == 'notebook':
            nl = "<br>"
            display(HTML(f"{nl*n_before}<b>{text}</b>{nl*(n_after+1)}"))
        elif self.params['ui'] == 'console':
            nl = "\n"
            print(f"\033[1m{nl*n_before}{text}{nl*n_after}\033[0m")
