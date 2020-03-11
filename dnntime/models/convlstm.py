import ipdb
import time
import numpy as np
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, LSTM, Dense, Flatten, Dropout, \
                                    RepeatVector, TimeDistributed
# helper functions
from ..utils.metrics import calc_rmse, calc_mae, calc_mape


class ConvLSTMWrapper:

    def __init__(self, n_steps: int, l_subseq: int = 1, n_row: int = 1, 
                 n_col: int = 1, n_features: int = 1, n_units: int = 64,
                 d_rate: int = 0.15, optimizer: str = 'adam', loss: str = "mse"):
        
        self.conv_model = StackedConvLSTM(n_steps, n_row, n_col, n_units,
                                          n_features, d_rate)
        self.conv_model.compile(optimizer, loss)
        self.l_subseq = l_subseq
        self.n_row = n_row
        self.n_col = n_col
        self.n_features = n_features
        self.run_time = 0.0

    def reshape(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ConvLSTM Hyperparameters
        Source: https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/

        Parameters
        ----------
        X : [samples, n_input, n_features]
        y : [samples, n_output]

        Returns
        -------
        X : [samples, n_input/length_subsequence, n_row, n_col, n_features] or
            [samples, subseq, rows, cols, channel]
        y : [samples, n_output, features]
            [samples, target, channel]

        """
        X = X.reshape((X.shape[0], int(X.shape[1]/self.l_subseq),
                       self.n_row, self.n_col, self.n_features
                       ))
        y = y.reshape((y.shape[0], self.n_col, self.n_features))

        return X, y

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_epoch: int = 10,
            n_batch: int = 1, verbose: int = 0) -> None:

        X_train, y_train = self.reshape(X_train, y_train)

        start_time = time.time()
        self.conv_model.fit(X_train, y_train, epochs=n_epoch, batch_size=n_batch,
                           verbose=verbose)
        end_time = time.time()
        self.run_time = end_time - start_time

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, score_type: str = 'rmse', 
                 verbose: int = 0) -> Tuple[Sequential, np.ndarray, float, float]:

        X_test, y_test = self.reshape(X_test, y_test)
        conv_pred = self.conv_model.predict(X_test, verbose=verbose)

        rmse = calc_rmse(y_test, conv_pred)
        mae = calc_mae(y_test, conv_pred)
        mape = calc_mape(y_test, conv_pred)

        print("\n-----------------------------------------------------------------")
        print("ConvLSTM SUMMARY:")
        print("-----------------------------------------------------------------")
        print(f"MAE Score: {round(mae, 4)}")
        print(f"MAPE Score: {round(mape, 4)}")
        print(f"RMSE Score: {round(rmse, 4)}")
        print(f"Total Training Time: {round(self.run_time/60, 2)} min")
    
        return self.conv_model, conv_pred, rmse


def StackedConvLSTM(n_steps: int, n_row: int, n_col: int, n_units: int, 
                    n_features: int, d_rate: float = 0.5) -> Sequential:

    model = Sequential()
    # define encoder
    model.add(ConvLSTM2D(64, (1,3), activation='relu', \
                         input_shape=(n_steps, n_row, n_col, n_features)))
    model.add(Flatten())
    # repeat encoding
    model.add(RepeatVector(n_col))
    # define decoder
    model.add(LSTM(n_units, activation='relu', return_sequences=True))
    # define output model
    model.add(TimeDistributed(Dense(n_units, activation='relu')))
    model.add(TimeDistributed(Dropout(d_rate)))
    model.add(TimeDistributed(Dense(n_features)))  # 1 unit if univariate time-series
    print("Stacked ConvLSTM model summary:")
    model.summary()
    return model


def CustomConvLSTM(n_input: int, n_output: int, n_units: int,
                   n_features: int) -> Sequential:

    pass
