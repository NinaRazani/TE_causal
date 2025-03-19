
import math
import numpy as np
import pandas as pd
import tensorflow as tf

# for i, gpu in enumerate(tf.config.list_physical_devices('GPU')):
#     print(f"TensorFlow GPU:{i} -> {gpu}")

def strategy_reg(y_pred, ret, start_capital=1000):
    """
    Generate trading signals based on predicted values.
    The signal remains 'hold' until the signal changes from 0 to 1 or 1 to 0.
    """

    trades = 0
    balance = [] 
    balance.append(start_capital) 
    strategy_result = ['hold'] * len(y_pred)

    # Example thresholds
    buy_threshold = 0.0045  # Predicted return > 1%
    sell_threshold = -0.0045  # Predicted return < -1%
    for i in range(1, len(y_pred)):
        # Decision based on predicted return
        if y_pred[i] > buy_threshold:  #buy signal
            if strategy_result[i-1] == 'buy':
                strategy_result[i] = "hold"
                balance.append(balance[-1]+balance[-1]*ret[i])  
            else:
                strategy_result[i] = "Buy"
                trades += 1
                balance.append(balance[-1]+balance[-1]*ret[i]) 
        elif y_pred[i] < sell_threshold:   #sell signal
            if strategy_result[i-1] == 'sell':
                strategy_result[i] = "hold"
            else: 
                strategy_result[i] = "Sell"
                balance.append(balance[-1]+balance[-1]*ret[i])
                trades += 1

        elif sell_threshold < y_pred[i] < buy_threshold:
            if strategy_result[i-1] == 'sell':
                strategy_result[i] = 'hold'
                balance.append(balance[-1])
            elif strategy_result[i-1] == 'buy':
                strategy_result[i] = 'hold'
                balance.append(balance[-1]+balance[-1]*ret[i])
            else:       # when previous is 2: it differ when the hold is the continuous of 1 or 0 
                j=i
                while j-2>0:
                    if strategy_result[j-2] == 'buy':
                        strategy_result[i] = 'hold'
                        balance.append(balance[-1]+balance[-1]*ret[i])
                        break
                    elif strategy_result[j-2] == 'sell':
                        strategy_result[i] = 'hold'
                        balance.append(balance[-1])
                        break
                    else:
                        strategy_result[i] = 'hold'
                        # balance.append(balance[-1]) 
                    j-=1
        else:
            strategy_result[i] = "Hold"
        if len(balance) < i + 1:
            balance.append(balance[-1])

    return np.array(strategy_result), trades, np.array(balance) 


def strategy_acc(y_pred, ret, start_capital=1000):
    """
    Generate trading signals based on predicted values.
    The signal remains 'hold' until the signal changes from 0 to 1 or 1 to 0.
    """

    balance = [] 
    balance.append(start_capital) 
    trades = 0
    strategy_result = ['hold'] * len(y_pred)
    if y_pred[0] == 0:
        strategy_result[0] = 'sell'
        balance.append(balance[-1]+balance[-1]*ret[0])
        trades += 1 
    elif y_pred[0] == 1:
        strategy_result[0] = 'buy'
        balance.append(balance[-1]+balance[-1]*ret[0])
        trades += 1
    else:
        strategy_result[0] = 'hold'
        balance.append(balance[-1])

    for i in range(1, len(y_pred)):  #the start must be 1????? 
 
        if y_pred[i] == 0:  # Sell signal
            if y_pred[i-1] == 0:
                strategy_result[i] = 'hold'
                balance.append(balance[-1]) 
            else:
                strategy_result[i] = 'sell'
                trades += 1
                balance.append(balance[-1]+balance[-1]*ret[i]) #ret[i-1] when we want to trade a day before but as our prediction is in term of close
                #price, we can trade in predicted day 
        elif y_pred[i] == 1:  # Buy signal 
            if y_pred[i-1] == 1:
                strategy_result[i] = 'hold'
                balance.append(balance[-1]+balance[-1]*ret[i])
            else:
                strategy_result[i] = 'buy'
                balance.append(balance[-1]+balance[-1]*ret[i]) 
                trades += 1
        elif y_pred[i] == 2: 
            if y_pred[i-1] == 0:
                strategy_result[i] = 'hold'
                balance.append(balance[-1])
            elif y_pred[i-1] == 1: 
                strategy_result[i] = 'hold'
                balance.append(balance[-1]+balance[-1]*ret[i])
            else:       # when previous is 2: it differ when the hold is the continuous of 1 or 0 
                j=i
                while j-2>0:
                    if y_pred[j-2] == 1:
                        strategy_result[i] = 'hold'
                        balance.append(balance[-1]+balance[-1]*ret[i]) 
                        break
                    elif y_pred[j-2] == 0:
                        strategy_result[i] = 'hold'
                        balance.append(balance[-1])
                        break
                    else:
                        strategy_result[i] = 'hold'
                    j-=1
        else:
            strategy_result[i] = 'hold'
        if len(balance) < i + 1:
            balance.append(balance[-1])  

    return np.array(strategy_result), trades, np.array(balance)


def econ_metrics(y_pred, X_ret, risk_free=0.0031):  # Risk free rate must be given
    
    action, trades, x = strategy_acc(y_pred, X_ret) 
    # action , trades, x = strategy_reg(y_pred, X_ret)
    n = 244 

    portfolio_return = (x[-1] / x[0]) - 1                # total return
    mu_annum = ((x[-1]/x[0]) ** (n / x.shape[0])) - 1    # annualized return 
    Std_of_returns = np.std(x[1:] / x[:-1])              # standard deviation of returns
    std_annum = (np.std(x[1:] / x[:-1])) / np.sqrt(n)      # annualized standard deviation of returns
    
           
    sharpe_ratio = ((x[-1] / x[0]) - 1 - risk_free) / np.std(x[1:] / x[:-1])    # total sharp ratio
    # Sharpe_ratio_per_annum_lowtotop = (mu_annum - risk_free) / std_annum   # this formula is for when the tatal sharpe ratio is calculated in a smaller time period
    # sharpe_ratio_per_annum_toptolow = mu_annum / std_annum 
    annualized_sharpratio = sharpe_ratio / (np.sqrt(n/x.shape[0])) 

    # sum_daily_excess_return =0
    # for i in range(1, len(y_pred)):
    #     sum_daily_excess_return += (x[i]-x[i-1])/x[i-1]
    # mean_daily_excess_return = sum_daily_excess_return / len(y_pred) 
    # annualized_return = mean_daily_excess_return * n 
    # annualized_std_return = Std_of_returns * np.sqrt(n) 
    # annualized_sr = annualized_return / annualized_std_return 

    return portfolio_return, Std_of_returns, sharpe_ratio, mu_annum, std_annum, annualized_sharpratio 


def accumulated_return(y_pred, ret, start_capital=1000, dtype= None): 
    assert len(y_pred) == len(ret)
    balance = [start_capital]
    signal = 0
    for i, d in enumerate(y_pred[:-1]): 
        signal = signal if d == 2 else d
        if signal == 0:
            balance.append(balance[-1])
        elif signal == 1:
            balance.append(balance[-1] * ret[i])
        else:
            raise ValueError()
    balance = np.array(balance, dtype=dtype)
    return balance 