import os
import numpy as np
import pandas as pd
import datetime as dt
import talib
from typing import Callable
from typing import Union, Tuple

from fredapi import Fred
fred = Fred(api_key='fdf7fbad89ee632fd8abe48fed980983') 

import yfinance as yf

def fred_down(series_id, col1, col2, start, end, log=False):
    """_summary_

    Args:
        series_id (string): the name of fred symbol
        col1 (string): the name of date column, default: date
        col2 (string): the name of specific column to extract, default: rate
        start (string): start date 
        end (string):  end date
        log (bool, optional): if true calculate the log return of rate column. Defaults to False.

    Returns:
        dataframe: the dataframe of mentioned series_id
    """
    data = fred.get_series(series_id) 
    data = pd.DataFrame(data)
    data.reset_index(inplace=True)
    data.rename(columns={"index": col1, 0:col2}, inplace=True) 
    data = data.loc[(data[col1] >= start) & (data[col1]<=end)] 
    if log==True:
        data['log_ret'] = np.log(data[col2]) - np.log(data[col2].shift(1))
        data= data.fillna(method='bfill')
        data = pd.DataFrame().assign(date=data['date'], rate=data['log_ret'])
    return data


def yfin_down(series_id, start, end, interval, log=False, *useless_cols):
    """_summary_

    Args:
        series_id (string): the name of yahoo finance history table
        start (string): _description_
        end (string): _description_
        interval (string): Valid intervals: 1m(minute),2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo(month),3mo
        log (bool, optional): if true calculate log return of Close colum. Defaults to False.
        useless_cols: there are some columns in yahoo fin history table that are unnecessary 

    Returns:
        dataframe: the dataframe of mentioned series_id
    """
    obj = yf.Ticker(series_id)
    data = obj.history(interval=interval, start=start, end=end) 
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.tz_localize(None) 
    # data = pd.DataFrame().assign(date=data['Date'], Close=data['Close'])
    data.rename(columns={"Date": "date"}, inplace=True)
    if log==True:
        data['Close'] = np.log(abs(data['Close'])) - np.log(abs(data['Close'].shift(1)))
        data = data.fillna(method='bfill') # type: ignore
    data = pd.DataFrame().assign(date=data['date'], Close=data['Close']) 
    # data = pd.DataFrame().assign(date=data['date'], Close=data['Close'], Open= data['Open'], High=data['High'], Low= data['Low']) 
    return data
 
def h_forex_data(currency_pair):
   filename = f"combined_{currency_pair}_data.csv"
   base_path = r"C:\ninap\causal_pred\causal_prediction"
   path = os.path.join(base_path, filename)
   h_data = pd.read_csv(path)
   h_data[['date', 'time']] = h_data['Local time'].apply(lambda x: pd.Series(extract_date_time(x)))
   h_data = pd.DataFrame().assign(date=h_data['date'], Close=h_data['Close'])
   h_data['date'] = pd.to_datetime(h_data['date']) 
   return h_data

def extract_date_time(timestamp):
    date_time = dt.datetime.strptime(timestamp.split(" ")[0] + " " + timestamp.split(" ")[1], "%d.%m.%Y %H:%M:%S.%f")
    return date_time.date(), date_time.time()

def extend_concat(df1, df2, scale=''): 
    """concatenate fred dataframe(df1) with yahoo finance dataframe(df2) and return a dataframe with date, rate, and Close columns

    Args:
        df1 (string): fred dataframe, as out merge how parameter fixed 'left'
        df2 (string): yahoo finance dataframe 
        scale (str, optional): if its value is "M", then extend fred df from Month interval to day interval else 
        remain default df with month frequency. Defaults to ''.

    Returns:
        dataframe: _description_
    """
    if scale == 'M':
        df1['month'] = df1['date'].dt.to_period(scale)
        df2['month'] = df2['date'].dt.to_period(scale)
        merged_df = pd.merge(df1, df2, on='month', how='left')
        merged_df = pd.DataFrame().assign(date=merged_df['date_y'], rate=merged_df['rate'], Close=merged_df['Close'])
    else:
        df1.set_index('date', inplace=True)
        df2.set_index('date', inplace=True)
        merged_df = pd.concat([df1, df2], axis=1)
        merged_df.reset_index(inplace=True)
        merged_df = pd.DataFrame().assign(date=merged_df['date'], rate=merged_df['rate'], Close=merged_df['Close']) 
    merged_df.set_index(['date'], inplace=True)
    return merged_df

def concat(data, data1):
    """_summary_

    Args:
        data (_type_): other yahoo finance dataframes
        data1 (_type_): forex dataframe

    Returns:
        _type_: merge two yahoo finance dataframes 
    """

    merged_df = pd.merge(data, data1, on='date', how='right').fillna(method="bfill")   # type: ignore 
    merged_df.set_index('date', inplace=True)
    return merged_df 
     

def apply_talib_indicator(df: pd.DataFrame, 
                          indicator_func: Callable, 
                          input_columns: Union[str, Tuple[str, ...]], 
                          output_column: Union[str, Tuple[str, ...]], 
                          **kwargs) -> pd.DataFrame:
    """
    Apply a TA-Lib indicator function to a DataFrame and add the result as a new column.

    Parameters:
    df (pd.DataFrame): Input DataFrame with OHLCV data.
    indicator_func (Callable): TA-Lib indicator function to apply.
    input_columns (str or tuple of str): Column name(s) to use as input for the indicator function.
    output_column (str or tuple of str): Name(s) for the output column(s).
    **kwargs: Additional keyword arguments to pass to the indicator function.

    Returns:
    pd.DataFrame: DataFrame with the new indicator column(s) added.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Prepare input data
    if isinstance(input_columns, str):
        input_data = df_copy[input_columns].values
    else:
        input_data = [df_copy[col].values for col in input_columns]

    # Apply the indicator function
    result = indicator_func(input_data, **kwargs)

    # Add the result to the DataFrame
    if isinstance(output_column, str):
        df_copy[output_column] = result
    else:
        for i, col in enumerate(output_column):
            df_copy[col] = result[i]

    return df_copy


#prepare data: create feature and target
def to_supervised_multi(df, n_input, n_out, n_features, target): 
    """_summary_

    Args:
        train_df (dataframe): the dataset name to train network based on it 
        n_input (integer): the lookbach window size that used to train (feature)
        n_out (_type_): the length of targer
        n_features(integer): number of features want to be invlved in model
        target(-1 or -4): you want to classify (-1 last column as target column) or regression(-4 4th last column that is ret_close column)

    Returns:
        two array: train dataset and target dataset
    """
    X, y = list(), list()
    in_start = 0
    for _ in range(len(df)):
        in_end = in_start+n_input
        out_end = in_end+n_out
        if out_end<len(df):
            x_input = df.iloc[in_start:in_end, :n_features].values.flatten() 
            X.append(x_input)
            y.append(df.iloc[in_end:out_end, target].values) 
        in_start+=1
    return X, y