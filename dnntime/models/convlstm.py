import ipdb
import time
import numpy as np
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, LSTM, Dense, Flatten, Dropout, \
                                    RepeatVector, TimeDistributed
# helper functions
from ..utils import print_dynamic_rmse, print_mae, print_mape


def build_convlstm_model(X_train: np.ndarray, y_train: np.ndarray, 
                         X_test: np.ndarray, y_test: np.ndarray,
                         n_input: int, n_output: int = 1, l_subseq: int = 1,
                         n_row: int = 1, n_col: int = 1, n_features: int = 1,
                         n_units: int = 64, d_rate: int = 0.15, n_epoch: int = 10,
                         n_batch: int = 1, verbose: int = 0
                         ) -> Tuple[Sequential, np.ndarray, float, float]:

    # ConvLSTM Hyperparameters
    # [samples, subseq, rows, cols, channel]

    X_train = X_train.reshape((X_train.shape[0], int(X_train.shape[1]/l_subseq),
                               n_row, n_col, n_features))
    y_train = y_train.reshape((y_train.shape[0], n_col, n_features))
    X_test = X_test.reshape((X_test.shape[0], int(X_test.shape[1]/l_subseq),
                             n_row, n_col, n_features))
    y_test = y_test.reshape((y_test.shape[0], n_col, n_features))
    n_steps = X_train.shape[1]

    conv_model = StackedConvLSTM(n_steps, n_row, n_col, n_units, n_features, d_rate)
    conv_model.compile(optimizer="adam", loss="mse")
    start_time = time.time()
    conv_model.fit(X_train, y_train, epochs=n_epoch, batch_size=n_batch, verbose=verbose)
    end_time = time.time()

    conv_pred = conv_model.predict(X_test, verbose=verbose)
    rmse, norm_rmse = print_dynamic_rmse(y_test, conv_pred, y_train)
    mae = print_mae(y_test, conv_pred)
    mape = print_mape(y_test, conv_pred)
    run_time = end_time - start_time
    print("\n-----------------------------------------------------------------")
    print("ConvLSTM SUMMARY:")
    print("-----------------------------------------------------------------")
    print("X: [#samples, #subseq, #rows, #cols (outputs), #channel (features)]")
    print("y: [#samples, #cols (outputs), #channel (features)]")
    print('    Conv X_train.shape = ', X_train.shape)
    print('    Conv y_train.shape = ', y_train.shape)
    print('    Conv X_test.shape = ',  X_test.shape)
    print('    Conv y_test.shape = ', y_test.shape)
    print("-----------------------------------------------------------------")
    print(f"MAE Score: {round(mae, 4)}")
    print(f"MAPE Score: {round(mape, 4)}")
    print(f"RMSE Score: {round(rmse, 4)}")
    print(f"Normalized RMSE Score: {round(norm_rmse, 4)*100}%")
    print(f"Total Training Time: {round(run_time/60, 2)} min")

    return conv_model, conv_pred, rmse, norm_rmse


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
