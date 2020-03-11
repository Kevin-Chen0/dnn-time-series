import ipdb
import time
import numpy as np
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
# helper functions
from ..utils import print_dynamic_rmse, print_mae, print_mape


def build_lstm_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                    y_test: np.ndarray, n_input: int, n_output: int = 1,
                    n_features: int = 1, n_units: int = 64, d_rate: int = 0.15, 
                    n_epoch: int = 10, n_batch: int = 1, verbose: int = 0
                    ) -> Tuple[Sequential, np.ndarray, float, float]:

    lstm_model = StackedLSTM(n_input, n_output, n_units, n_features, d_rate)
    lstm_model.compile(optimizer="adam", loss="mse")
    start_time = time.time()
    lstm_model.fit(X_train, y_train, epochs=n_epoch, batch_size=n_batch, verbose=verbose)
    end_time = time.time()

    lstm_pred = lstm_model.predict(X_test, verbose=verbose)

    rmse, norm_rmse = print_dynamic_rmse(y_test, lstm_pred, y_train)
    mae = print_mae(y_test, lstm_pred)
    mape = print_mape(y_test, lstm_pred)
    run_time = end_time - start_time
    print("\n-----------------------------------------------------------------")
    print("LSTM SUMMARY:")
    print("-----------------------------------------------------------------")
    print(f"MAE Score: {round(mae, 4)}")
    print(f"MAPE Score: {round(mape, 4)}")
    print(f"RMSE Score: {round(rmse, 4)}")
    print(f"Normalized RMSE Score: {round(norm_rmse, 4)*100}%")
    print(f"Total Training Time: {round(run_time/60, 2)} min")

    return lstm_model, lstm_pred, rmse, norm_rmse


def VanillaLSTM(n_input: int, n_output: int, n_units: int, n_features: int) -> Sequential:

    model = Sequential()
    model.add(LSTM(n_units, activation="tanh", return_sequences=False,
                   input_shape=(n_input, n_features)))
    model.add(Dense(n_output))
    print("Vanilla LSTM model summary:")
    model.summary()
    return model


def StackedLSTM(n_input: int, n_output: int, n_units: int, n_features: int,
                d_rate: float = 0.5) -> Sequential:

    model = Sequential()
    model.add(LSTM(n_units, activation="tanh", return_sequences=True,
                   input_shape=(n_input, n_features)))
    model.add(Dropout(d_rate))
    model.add(LSTM(n_units, activation="tanh", return_sequences=True))
    model.add(Dropout(d_rate))
    model.add(LSTM(n_units, activation="tanh", return_sequences=False))
    model.add(Dropout(d_rate))
    model.add(Dense(n_output))
    print("Stacked LSTM model summary:")
    model.summary()
    return model


def CustomRNN(n_input: int, n_output: int, n_units: int,
              n_features: int) -> Sequential:
    
    pass
