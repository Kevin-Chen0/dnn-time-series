import time
import numpy as np
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
# helper functions
from ..utils.metrics import calc_rmse, calc_mae, calc_mape


class RNNWrapper:

    def __init__(self, n_input: int, n_output: int = 1, n_feature: int = 1,
                 n_layer: int = 1, n_unit: int = 64, d_rate: float = 0.15,
                 optimizer: str = 'adam', loss: str = "mse") -> None:
        """
        Wrapper that abstracts the underlying RNN Model in order to better
        decouple the actual model specification from DNN package execution.

        Parameters
        ----------
        n_input : Number of input timesteps used to generate a forecast.
        n_output : Number of output timesteps or the forecast horizon.
        n_feature : Number of features, a univariate time-series has n_feature=1.
        n_unit : Number of neural units per layer.
        d_rate : Dropout rate for each layer, see: https://keras.io/layers/core/#dropout
        optimizer : How model learns (i.e. SDG), see: https://keras.io/optimizers/
        loss : The loss or error function for model to minimize, see: https://keras.io/losses/

        """
        self.rnn_model = CustomRNN(n_input, n_output, n_layer, n_unit,
                                   n_feature, d_rate
                                   )
        self.rnn_model.compile(optimizer, loss)
        self.run_time = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_epoch: int = 10,
            n_batch: int = 1, verbose: int = 0) -> None:
        """
        Wraps the RNN model.fit() function and includes the time it takes to
        finish running.

        Parameters
        ----------
        X_train : Training set with predictor columns used to fit the model.
        y_train : Training set with the target column used to fit the model.
        n_epoch : Num of passovers over the training set.
        n_batch : Batch size, or set of N data-points.
        verbose : Whether or not to display fit status, 1 is yes and 0 is no.

        """
        start_time = time.time()
        self.rnn_model.fit(X_train, y_train, epochs=n_epoch, batch_size=n_batch,
                           verbose=verbose)
        end_time = time.time()
        self.run_time = end_time - start_time

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, score_type: str = 'rmse',
                 verbose: int = 0) -> Tuple[Sequential, np.ndarray, float]:
        """
        Wraps the RNN model's forecast of the test set and evaluation of its
        accuracy into one function.

        Parameters
        ----------
        X_test : Test set with predictor columns used to make model forecast.
        y_test : Training set with the target column used to evaluate model forecast.
        score_type : Type of scoring metric used to measure model's forecast error.
        verbose : Whether or not to display predict status, 1 is yes and 0 is no.

        Returns
        -------
        self.rnn_model : The trained RNN model itself.
        rnn_pred : The RNN's forecasted data or y_hat based on X_test.
        rmse : The root mean-squared error score used as default evaluation metric.

        """
        rnn_pred = self.rnn_model.predict(X_test, verbose=verbose)
        rmse = calc_rmse(y_test, rnn_pred)
        mae = calc_mae(y_test, rnn_pred)
        mape = calc_mape(y_test, rnn_pred)

        print("\n-----------------------------------------------------------------")
        print("RNN SUMMARY:")
        print("-----------------------------------------------------------------")
        print(f"MAE Score: {round(mae, 4)}")
        print(f"MAPE Score: {round(mape, 4)}")
        print(f"RMSE Score: {round(rmse, 4)}")
        print(f"Total Training Time: {round(self.run_time/60, 2)} min")

        return self.rnn_model, rnn_pred, rmse


def VanillaRNN(n_input: int, n_output: int, n_unit: int, n_feature: int) -> Sequential:
    """
    A basic version of the RNN model without any "bells and whistles".

    Parameters
    ----------
    n_input : Number of input timesteps used to generate a forecast.
    n_output : Number of output timesteps or the forecast horizon.
    n_unit : Number of neural units per layer.
    n_feature : Number of features, a univariate time-series has n_feature=1.

    Returns
    -------
    model : The keras.Sequential model architecture itself to be fitted with data.

    """
    model = Sequential()
    model.add(SimpleRNN(n_unit, activation="tanh", return_sequences=False,
                        input_shape=(n_input, n_feature)))
    model.add(Dense(n_output))
    print("Vanilla RNN model summary:")
    model.summary()
    return model


def StackedRNN(n_input: int, n_output: int, n_unit: int, n_feature: int,
               d_rate: float = 0.5) -> Sequential:
    """
    A standard, 3-layer deep RNN model that includes dropout rates.

    Parameters
    ----------
    n_input : Number of input timesteps used to generate a forecast.
    n_output : Number of output timesteps or the forecast horizon.
    n_unit : Number of neural units per layer.
    n_feature : Number of features, a univariate time-series has n_feature=1.
    d_rate : Dropout rate for each layer, see: https://keras.io/layers/core/#dropout

    Returns
    -------
    model : The keras.Sequential model architecture itself to be fitted with data.

    """
    model = Sequential()
    model.add(SimpleRNN(n_unit, activation="tanh", return_sequences=True,
                        input_shape=(n_input, n_feature)))
    model.add(Dropout(d_rate))
    model.add(SimpleRNN(n_unit, activation="tanh", return_sequences=True))
    model.add(Dropout(d_rate))
    model.add(SimpleRNN(n_unit, activation="tanh", return_sequences=False))
    model.add(Dropout(d_rate))
    model.add(Dense(n_output))
    print("Stacked RNN model summary:")
    model.summary()
    return model


def CustomRNN(n_input: int, n_output: int, n_layer: int, n_unit: int,
              n_feature: int, d_rate: float = 0.5) -> Sequential:
    """
    A customized, n-layer deep RNN model that includes dropout rates.

    Parameters
    ----------
    n_input : Number of input timesteps used to generate a forecast.
    n_output : Number of output timesteps or the forecast horizon.
    n_layer : Number of layers in the NN, excluding the input layer.
    n_unit : Number of neural units per layer.
    n_feature : Number of features, a univariate time-series has n_feature=1.
    d_rate : Dropout rate for each layer, see: https://keras.io/layers/core/#dropout

    Returns
    -------
    model : The keras.Sequential model architecture itself to be fitted with data.

    """
    model = Sequential()
    for l in range(n_layer):
        ret_seq = True if l < (n_layer-1) else False
        model.add(SimpleRNN(n_unit, activation="tanh",
                            return_sequences=ret_seq,
                            input_shape=(n_input, n_feature)
                            )
                  )
        model.add(Dropout(d_rate))
    model.add(Dense(n_output))

    print("Stacked RNN model summary:")
    model.summary()
    return model
