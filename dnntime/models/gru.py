import ipdb
import time
import numpy as np
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
# helper functions
from ..utils.metrics import calc_rmse, calc_mae, calc_mape


class GRUWrapper:

    def __init__(self, n_input: int, n_output: int = 1, n_features: int = 1, 
                 n_units: int = 64, d_rate: int = 0.15, optimizer: str = 'adam',
                 loss: str = "mse"):
        
        self.gru_model = StackedGRU(n_input, n_output, n_units, n_features, d_rate)
        self.gru_model.compile(optimizer, loss)
        self.run_time = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_epoch: int = 10,
            n_batch: int = 1, verbose: int = 0) -> None:

        start_time = time.time()
        self.gru_model.fit(X_train, y_train, epochs=n_epoch, batch_size=n_batch,
                           verbose=verbose)
        end_time = time.time()
        self.run_time = end_time - start_time

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, score_type: str = 'rmse', 
                 verbose: int = 0) -> Tuple[Sequential, np.ndarray, float, float]:

        gru_pred = self.gru_model.predict(X_test, verbose=verbose)

        rmse = calc_rmse(y_test, gru_pred)
        mae = calc_mae(y_test, gru_pred)
        mape = calc_mape(y_test, gru_pred)

        print("\n-----------------------------------------------------------------")
        print("GRU SUMMARY:")
        print("-----------------------------------------------------------------")
        print(f"MAE Score: {round(mae, 4)}")
        print(f"MAPE Score: {round(mape, 4)}")
        print(f"RMSE Score: {round(rmse, 4)}")
        print(f"Total Training Time: {round(self.run_time/60, 2)} min")
    
        return self.gru_model, gru_pred, rmse


# def build_gru_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
#                     y_test: np.ndarray, n_input: int, n_output: int = 1,
#                     n_features: int = 1, n_units: int = 64, d_rate: int = 0.15, 
#                     n_epoch: int = 10, n_batch: int = 1, verbose: int = 0
#                     ) -> Tuple[Sequential, np.ndarray, float, float]:

#     gru_model = StackedGRU(n_input, n_output, n_units, n_features, d_rate)
#     gru_model.compile(optimizer="adam", loss="mse")
#     start_time = time.time()
#     gru_model.fit(X_train, y_train, epochs=n_epoch, batch_size=n_batch, verbose=verbose)
#     end_time = time.time()

#     gru_pred = gru_model.predict(X_test, verbose=verbose)

#     rmse, norm_rmse = print_dynamic_rmse(y_test, gru_pred, y_train)
#     mae = calc_mae(y_test, gru_pred)
#     mape = calc_mape(y_test, gru_pred)
#     run_time = end_time - start_time
#     print("\n-----------------------------------------------------------------")
#     print("GRU SUMMARY:")
#     print("-----------------------------------------------------------------")
#     print(f"MAE Score: {round(mae, 4)}")
#     print(f"MAPE Score: {round(mape, 4)}")
#     print(f"RMSE Score: {round(rmse, 4)}")
#     print(f"Normalized RMSE Score: {round(norm_rmse, 4)*100}%")
#     print(f"Total Training Time: {round(run_time/60, 2)} min")

#     return gru_model, gru_pred, rmse, norm_rmse


def VanillaGRU(n_input: int, n_output: int, n_units: int, n_features: int) -> Sequential:

    model = Sequential()
    model.add(GRU(n_units, activation="tanh", return_sequences=False,
                  input_shape=(n_input, n_features)))
    model.add(Dense(n_output))
    print("Vanilla GRU model summary:")
    model.summary()
    return model


def StackedGRU(n_input, n_output, n_units, n_features, d_rate=0.5):

    model = Sequential()
    model.add(GRU(n_units, activation="tanh", return_sequences=True,
                  input_shape=(n_input, n_features)))
    model.add(Dropout(d_rate))
    model.add(GRU(n_units, activation="tanh", return_sequences=True))
    model.add(Dropout(d_rate))
    model.add(GRU(n_units, activation="tanh", return_sequences=False))
    model.add(Dropout(d_rate))
    model.add(Dense(n_output))
    print("Stacked GRU model summary:")
    model.summary()
    return model


def CustomGRU(n_input: int, n_output: int, n_units: int,
              n_features: int) -> Sequential:
    pass
