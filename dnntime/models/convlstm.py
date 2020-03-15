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
                 n_col: int = 1, n_feature: int = 1, n_unit: int = 64,
                 d_rate: float = 0.15, optimizer: str = 'adam', loss: str = "mse"
                 ) -> None:
        """
        Wrapper that abstracts the underlying ConvLSTM Model in order to better
        decouple the actual model specification from DNN package execution.
        Although ConvLSTM is developed for extract 2D data, such as images,
        it can also be used in extract "stacked" time-series data in parallel,
        either by entire series and its multiple features axes or equal
        subsequences of a single series. This wrapper's design uses the latter.

        Parameters
        ----------
        n_steps : Num of steps, where each step is a subseq. Default num is n_input/n_output.
        l_subseq : Len of each subseq. Default len is n_output, thus n_steps*l_subseq = n_input.
        n_row : Num of rows, representing image's height. Default is 1.
        n_col : Num of columns, representing image's width. Default is n_output.
        n_feature : Number of features, a univariate time-series has n_feature=1.
        n_unit : Number of neural units per layer.
        d_rate : Dropout rate for each layer, see: https://keras.io/layers/core/#dropout
        optimizer : How model learns (i.e. SDG), see: https://keras.io/optimizers/
        loss : The loss or error function for model to minimize, see: https://keras.io/losses/

        """
        self.conv_model = StackedConvLSTM(n_steps, n_row, n_col, n_unit,
                                          n_feature, d_rate)
        self.conv_model.compile(optimizer, loss)
        self.l_subseq = l_subseq
        self.n_row = n_row
        self.n_col = n_col
        self.n_feature = n_feature
        self.run_time = 0.0

    def reshape(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshapes ConvLSTM hyperparameters so X is 5-dim and y is 3-dim in order
        to input into the ConvLSTM model.
        Source: https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/

        Parameters
        ----------
        X : [samples, n_input, n_feature]
        y : [samples, n_output]

        Returns
        -------
        X : [samples, n_input/length_subsequence, n_row, n_col, n_feature] or
            [samples, subseq, rows, cols, channel]
        y : [samples, n_output, features] or
            [samples, target, channel]

        """
        X = X.reshape((X.shape[0], int(X.shape[1]/self.l_subseq),
                       self.n_row, self.n_col, self.n_feature
                       ))
        y = y.reshape((y.shape[0], self.n_col, 1))

        return X, y

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_epoch: int = 10,
            n_batch: int = 1, verbose: int = 0) -> None:
        """
        Wraps ConvLSTM model.fit() func, including the time it takes to
        finish running.

        Parameters
        ----------
        X_train : Training set with predictor columns used to fit the model.
        y_train : Training set with the target column used to fit the model.
        n_epoch : Num of passovers over the training set.
        n_batch : Batch size, or set of N data-points.
        verbose : Whether or not to display fit status, 1 is yes and 0 is no.

        """
        X_train, y_train = self.reshape(X_train, y_train)

        start_time = time.time()
        self.conv_model.fit(X_train, y_train, epochs=n_epoch, batch_size=n_batch,
                            verbose=verbose)
        end_time = time.time()
        self.run_time = end_time - start_time

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, score_type: str = 'rmse',
                 verbose: int = 0) -> Tuple[Sequential, np.ndarray, float]:
        """
        Wraps the ConvLSTM model's forecast of the test set and evaluation of
        its accuracy into one function.

        Parameters
        ----------
        X_test : Test set with predictor columns used to make model forecast.
        y_test : Training set with the target column used to evaluate model forecast.
        score_type : Type of scoring metric used to measure model's forecast error.
        verbose : Whether or not to display predict status, 1 is yes and 0 is no.

        Returns
        -------
        self.conv_model : The trained ConvLSTM model itself.
        gru_pred : The ConvLSTM's forecasted data or y_hat based on X_test.
        rmse : The root mean-squared error score used as default evaluation metric.

        """
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


def StackedConvLSTM(n_steps: int, n_row: int, n_col: int, n_unit: int,
                    n_feature: int, d_rate: float = 0.5) -> Sequential:
    """
    A standard, encoder-decoder ConvLSTM model that includes dropout rates.

    Parameters
    ----------
    n_steps : Num of steps, where each step is a subseq. Default num is n_input/n_output.
    n_row : Num of rows, representing an image's height. Default is 1.
    n_col : Num of columns, representing an image's width. Default is n_output.
    n_unit : Number of neural units per layer.
    n_feature : Number of features, a univariate time-series has n_feature=1.
    d_rate : Dropout rate for each layer, see: https://keras.io/layers/core/#dropout

    Returns
    -------
    model : The keras.Sequential model architecture itself to be fitted with data.

    """
    model = Sequential()
    # define encoder
    model.add(ConvLSTM2D(n_unit, (1, 3), activation='relu',
                         input_shape=(n_steps, n_row, n_col, n_feature)))
    model.add(Flatten())
    # repeat encoding
    model.add(RepeatVector(n_col))
    # define decoder
    model.add(LSTM(n_unit, activation='relu', return_sequences=True))
    # define output model
    model.add(TimeDistributed(Dense(n_unit, activation='relu')))
    model.add(TimeDistributed(Dropout(d_rate)))
    model.add(TimeDistributed(Dense(1)))
    print("Stacked ConvLSTM model summary:")
    model.summary()
    return model


def CustomConvLSTM(n_input: int, n_output: int, n_unit: int,
                   n_feature: int) -> Sequential:

    pass
