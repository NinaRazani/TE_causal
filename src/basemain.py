
from itertools import chain
import numpy as np
from scipy import stats
from collect_features import prepare_features
from collect_data import to_supervised_multi
from enc_net import train_and_evaluate
from deepModel import train_and_evaluate_deep, train_and_evaluate_rec, train_and_evaluate_deep_regression, train_and_evaluate_rec_reg
from deepModel3class import train_and_evaluate_deep_three, train_and_evaluate_rec_three
from benchmark import arima
from strategy import strategy, strategy_acc, accumulated_return, econ_metrics, strategy_reg

def measure_imbalance_ratio(y):
    unique, counts = np.unique(y, return_counts=True)
    
    # Calculate imbalance ratio (majority class / minority class)
    imbalance_ratio = max(counts) / min(counts)
    
    print("Class Distribution:")
    for cls, count in zip(unique, counts):
        percentage = count / len(y) * 100
        print(f"Class {cls}: {count} samples ({percentage:.2f}%)")
    
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")
    
    return imbalance_ratio

def model_input(features,  lookback, n_features, goal, volatility=True): 
    X, y = to_supervised_multi(features, lookback, 1, n_features, goal) 
    target = []
    for i in range(0, len(y)):
        if volatility ==True:
            # print("predicting volatility")
            # # 3 class
            target.append(0 if (features.loc[i+lookback, 'target'] < features.loc[i+lookback, 'Mean'] - features.loc[i+lookback, 'Std']) 
                           else 1 if (features.loc[i+lookback, 'target'] > features.loc[i+lookback, 'Mean'] + features.loc[i+lookback, 'Std'])
                            else 2) 
        else:
            # print("predicting return")
            target.append(1 if features.iloc[i+lookback, -4] > features.iloc[i+lookback-1, -4] else 0)
    X = np.array(X)
    y = np.array(y)
    target = np.array(target)
    v = measure_imbalance_ratio(target)
    return X, y, target,v


file_path = "regression.txt"
def append_to_file(value):
    with open(file_path, "a") as file:
        file.write(str(value) + "\n")

def get_user_input():
    inputs = []
    while True:
        forex_id = input("enter currency pair id among ('AUDUSD=X', 'EURUSD=X', 'GBPUSD=X', 'NZDUSD=X', 'USDCAD=X', 'USDCHF=X', 'USDJPY=X'):")
        inputs.append(forex_id)
        start_date = input("Enter start date:(in format for example:2004-01-01)")
        inputs.append(start_date)
        end_date = input("Enter end date:")
        inputs.append(end_date)
        modelExp = input("which model(LSTM, MLP or ENCODER or mlp_reg or lstm_reg or ARIMA)?") 
        inputs.append(modelExp)
        lookback_window = input("Input lookback window: ") 
        inputs.append( lookback_window)
        features_number = input("Input features number: ")
        inputs.append( features_number)
        reg = input("insert the index of target column:(-1 classification, -4 regression)")
        inputs.append(reg)
        # volatility_prediction = input("Input True if volatility prediction else False: ")
        # inputs.append(volatility_prediction)
        
        done_check = input("Type 'done' to continue or anything else to add another entry: ")
        if done_check.lower() == 'done':
            break
        
    return inputs


def main():
    arguments = get_user_input()
    if len(arguments) == 0:
        print("No inputs were provided.") 
    else:
               
        features = prepare_features(arguments[0], arguments[1], arguments[2]) 
        X, y, target, v = model_input(features, int(arguments[4]), int(arguments[5]), int(arguments[6])) 
        # print(arima(np.array(features['ret_Close'])))
        econometric = []
        protfo_return = []
        for i in range(10):
            print(f"running model for repitition {i}")
            if arguments[3] == "encoder":
              print("running encoder") 
              all_final_metrics, metric_stats = train_and_evaluate(X, target) 
            elif arguments[3] =="mlp": 
              print("running MLP")            
              # all_final_metrics, lr, y_pred, X_test, y_test, plotting = train_and_evaluate_deep(X, target) 
              #3 class
              all_final_metrics, lr, y_pred, X_test, y_test, plotting = train_and_evaluate_deep_three(X, target) 
            elif arguments[3] == "lstm":
              print("running lstm")
              X = X.reshape((X.shape[0], int(arguments[4]), features.iloc[0:1, :int(arguments[5])].shape[1])) 
              # all_final_metrics, lr, y_pred, X_test, y_test, plotting  = train_and_evaluate_rec(X, target, int(arguments[4]), features.iloc[0:int(arguments[4]), :int(arguments[5])].shape[1]) 
              # 3 class
              all_final_metrics, lr, y_pred, X_test, y_test, plotting = train_and_evaluate_rec_three(X, target, int(arguments[4]), features.iloc[0:int(arguments[4]), :int(arguments[5])].shape[1])
            elif arguments[3] == "mlp_reg":
              print("running mlp regression")
              all_final_metrics, lr, y_pred, X_test, y_test, plotting = train_and_evaluate_deep_regression(X, y)    
            elif arguments[3] == "lstm_reg":
              X = X.reshape((X.shape[0], int(arguments[4]), features.iloc[0:1, :int(arguments[5])].shape[1])) 
              print("running lstm regression")
              all_final_metrics, lr, y_pred, X_test, y_test, plotting = train_and_evaluate_rec_reg(X, y, int(arguments[4]), features.iloc[0:int(arguments[4]), :int(arguments[5])].shape[1]) #????
            elif arguments[3] == "arima":
               arima_out = arima(np.array(features['ret_Close'])) 
            else:
               print("no model entered")

            # ///////////// 
            # when classificaion is used
            last_column_list = []
            for inner_list in X_test:
              matrix = np.array(inner_list).reshape(int(arguments[4]), int(arguments[5]))   # the 66 and 24 must replaced by customised values
              last_column = matrix[:, -1] #extract the last column of each reshaped matrix
              last_column_list.append(last_column) #save all last columns in a list
            # concatenated = np.concatenate(last_column_list) 
            seen = set()
            unique_elements = []
            #concate all arrays in last_column_list
            for item in chain.from_iterable(last_column_list):
              if item not in seen:
                unique_elements.append(item)
                seen.add(item)
            ret_Close = unique_elements[int(arguments[4])-1:] 
            #///////////////

         
            # classificaion
            # action, trades, balance = strategy_acc(y_pred, ret_Close) 
            # econometrics = econ_metrics(y_pred, np.array(ret_Close)) 
            # econometric.append(econometrics[5])
            # protfo_return.append(econometrics[3]) 

            #regression
            # print(np.quantile(y_test, 0.25)) 
            # print(np.quantile(y_test, 0.5)) 
            # print(np.quantile(y_test, 0.75))  
            # print(np.min(y_test))  
            # print(np.max(y_test))
            y_test = y_test.reshape(-1) 
            action, trades , balance = strategy_reg(y_pred, y_test)     
            econometrics = econ_metrics(y_pred, y_test) 
            econometric.append(econometrics[5])
            protfo_return.append(econometrics[3])

        print(v)
        append_to_file("currancy pair:"+arguments[0])
        append_to_file("running model is:"+arguments[3])
        append_to_file("lookback window:" + arguments[4]) 
        append_to_file("best learning rate:"+ str(lr))  
        append_to_file(all_final_metrics)   
        append_to_file("trading days:"+ str(y_pred.shape[0]))
        append_to_file("trades:"+ str(trades))
        append_to_file("last balance:" + str(balance[-1]))  
        append_to_file("portfolio_return:"+ str(econometrics[0])) 
        append_to_file("Std_of_returns:" + str(econometrics[1]))
        append_to_file("sharpe_ratio:"+ str(econometrics[2]))
        append_to_file("Return_per_annum:" + str(econometrics[3]))
        append_to_file("Std_of_returns_per_annum:"+ str(econometrics[4])) 
        append_to_file("Sharpe_ratio_per_annum:" + str(econometrics[5]))  
        append_to_file("sharp ratios over repetitions:" + str(econometric)) 
        append_to_file("mean of anual sharp ratio:"+ str(np.mean(econometric)))
        append_to_file("mean of anual return:"+ str(np.mean(protfo_return)))
        # append_to_file(metric_stats)
        
        # append_to_file(arima_out)
        
 #*********GPU*************
# # Check if GPU is available
#         print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) 

# # Print GPU details
#         gpu_devices = tf.config.list_physical_devices('GPU')
#         for device in gpu_devices:
#              print(device) 

#         gpus = tf.config.list_physical_devices('GPU')
#         if gpus:
#              try:
#                   # Prevent TensorFlow from consuming all GPU memory
#                   for gpu in gpus:
#                        tf.config.experimental.set_memory_growth(gpu, True)
#              except RuntimeError as e:
#                   print(e)

if __name__ == "__main__":
    main()

