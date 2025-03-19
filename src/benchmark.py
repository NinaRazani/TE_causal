import pmdarima as pmd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
from tensorflow.keras.losses import MSE, MAE, MAPE # type: ignore

from pmdarima.model_selection import train_test_split

def arima(arr):
    mod = sm.tsa.arima.ARIMA(
        arr,
        order = (1, 0, 1)
    )
    model_train = mod.fit()
    y_pred = model_train.predict()
    metrics = {
    'mse' : MSE(arr.flatten(), y_pred.flatten()).numpy(),
    'mae' : MAE(arr.flatten(), y_pred.flatten()).numpy(),
    'r2' : r2_score(arr, model_train.predict()),
    'mape' : MAPE(arr.flatten(), y_pred.flatten()).numpy()
        }
    return metrics

