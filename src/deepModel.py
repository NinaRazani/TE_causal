import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hinge_loss
from sklearn.model_selection import train_test_split
import tensorflow 
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, LSTM, Dropout, BatchNormalization# type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.optimizers import Adam # type: ignore 
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore # Import EarlyStopping
from tensorflow.keras.losses import get as get_loss # type: ignore 


def train_val_test_split(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    n_samples = len(X)
    
    # Calculate split indices
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # Sequential splitting
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_model(input_shape, learning_rate):
    # model_cls=Sequential()
    # model_cls.add(Flatten(input_shape=input_shape))
    # model_cls.add(Dense(10,activation="relu"))
    # model_cls.add(Dense(20,activation="relu"))
    # model_cls.add(Dense(30,activation="relu"))
    # model_cls.add(Dense(1, activation="linear"))
    # optimizer = Adam(learning_rate = learning_rate)
    # model_reg.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc']) 
    model_cls=Sequential()
    model_cls.add(Flatten(input_shape=input_shape))
    model_cls.add(Dropout(0.2)) # type: ignore
    model_cls.add(Dense(64,activation="relu", kernel_regularizer=l2(0.001))) # type: ignore
    model_cls.add(BatchNormalization()) # type: ignore
    model_cls.add(Dropout(0.3)) # type: ignore
    model_cls.add(Dense(32,activation="relu"))
    model_cls.add(BatchNormalization()) # type: ignore
    model_cls.add(Dropout(0.3)) # type: ignore
    model_cls.add(Dense(16,activation="relu"))
    model_cls.add(Dense(1, activation="sigmoid"))
    optimizer = Adam(learning_rate = learning_rate) # type: ignore
    model_cls.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model_cls

def create_model_reg(input_shape, learning_rate):
    # model_reg=Sequential()
    # model_reg.add(Flatten(input_shape=input_shape))
    # model_reg.add(Dense(10,activation="relu"))
    # model_reg.add(Dense(20,activation="relu"))
    # model_reg.add(Dense(30,activation="relu"))
    # model_reg.add(Dense(1, activation="linear"))
    # optimizer = Adam(learning_rate = learning_rate)
    # model_reg.compile(loss='mse', optimizer=optimizer, metrics=['mse']) 

    model_reg=Sequential()
    model_reg.add(Flatten(input_shape=input_shape))
    model_reg.add(Dropout(0.2)) # type: ignore
    model_reg.add(Dense(64,activation="relu", kernel_regularizer=l2(0.001))) # type: ignore
    model_reg.add(BatchNormalization()) # type: ignore
    model_reg.add(Dropout(0.3)) # type: ignore
    model_reg.add(Dense(32,activation="relu"))
    model_reg.add(BatchNormalization()) # type: ignore
    model_reg.add(Dropout(0.3)) # type: ignore
    model_reg.add(Dense(16,activation="relu"))
    model_reg.add(Dense(1, activation="linear"))
    optimizer = Adam(learning_rate= learning_rate)
    model_reg.compile(loss='mse', optimizer=optimizer, metrics=['mae']) 
    return model_reg

def train_and_evaluate_deep(X, y, epochs=50, n_repetitions=20, batch_size=32, learning_rates=[0.1, 0.01, 0.001, 0.0001, 0.00001]):
    loss_function='binary_crossentropy'
    loss_fn = get_loss(loss_function)
    all_results = {} # save the all_final_metrics and metric_states for all learning rates here
    best_result = {} # save the all_final_metrics and metric_states of best learning rate here
    best_lr = None
    best_metric = -float("inf")  # Adjust based on the metric you want to optimize (e.g., accuracy, F1 score)

    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        metric_stats = []  # save the statistics of evaluation metrics over n_repitition in this
        all_final_metrics = []  # save the value of evaluation metrics for each repetition in this 
        
        for i in range(n_repetitions):
            print(f"Repetition {i + 1}/{n_repetitions}")
            # Split the data
            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
            model = create_model(X_train.shape[1:], learning_rate=lr)
            early_stopping = EarlyStopping(
                monitor='val_loss',  # Metric to monitor
                patience=5,         # Number of epochs to wait before stopping
                restore_best_weights=True  # Restore the best weights
            )
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=False)  # type: ignore
            
            loss_threshold = 0.75
            loss_values = history.history['loss']
            if any(loss < loss_threshold for loss in loss_values):
                plot = plot_training_metrics(history, title=f' return prediction (no flow)')  # type: ignore 

            # Evaluate on test set
            y_pred = model.predict(X_test).round()
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            # loss = hinge_loss(y_test, y_pred)
            test_loss = loss_fn(y_test, y_pred).numpy() 

            all_final_metrics.append({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1 score': f1,
                'loss': test_loss[-1].item()
            })

        metrics = ['accuracy', 'precision', 'recall', 'f1 score', 'loss']
        for metric in metrics:
            values = [m[metric] for m in all_final_metrics]
            mean = np.mean(values)
            std_dev = np.std(values)
            conf_interval = stats.t.interval(confidence=0.95, df=len(values) - 1, loc=mean, scale=stats.sem(values))
            metric_stats.append({
                'metric': metric,
                'Mean': np.mean(values),
                'Std Dev': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values),
                '95 percent confidence Interval': (conf_interval[0], conf_interval[1])
            })
        
        # Store results for this learning rate
        all_results[lr] = {
            'metrics': all_final_metrics,
            'summary_stats': metric_stats  
        }
        
        # Determine the best model based on a specific metric (e.g., F1 score)
        avg = np.mean([m['accuracy'] for m in all_final_metrics])
        if avg > best_metric:
            best_metric = avg 
            best_lr = lr 
    
    best_result = {
            'metrics of best lr': all_results[best_lr]['metrics'],
            'summary' : all_results[best_lr]['summary_stats']
        }
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    model = create_model(X_train.shape[1:], learning_rate=best_lr)
    early_stopping = EarlyStopping(
                monitor='val_loss',  # Metric to monitor
                patience=5,         # Number of epochs to wait before stopping
                restore_best_weights=True  # Restore the best weights
            )
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=False)  # type: ignore
    y_pred = model.predict(X_test).round()
    print(f"Best learning rate: {best_lr}, best_metric: {best_metric}") 
    return best_result, best_lr, y_pred, X_test, y_test, plot


def train_and_evaluate_deep_regression(X, y, epochs=50, n_repetitions=20, batch_size=32, learning_rates=[0.1, 0.01, 0.001, 0.0001, 0.00001]):
    all_results = {}
    best_result = {}
    best_lr = None
    best_metric = +float("inf")
    metric_stats = [] 

    for lr in learning_rates:
          print(f"Training with learning rate: {lr}") 
          metric_stats = []
          all_final_metrics = []
          for i in range(n_repetitions):
            print(f"Repetition {i + 1}/{n_repetitions}")
            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
            model = create_model_reg(X_train.shape[1:], learning_rate=lr)
            early_stopping = EarlyStopping(
                monitor='val_loss',  # Metric to monitor
                patience=5,         # Number of epochs to wait before stopping
                restore_best_weights=True  # Restore the best weights
            )
            history = model.fit(X_train, y_train,validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=False)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            R2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            all_final_metrics.append({
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': R2,
                'MAPE': mape
            })

          metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE']
          for metric in metrics:
                values = [m[metric] for m in all_final_metrics]
                mean = np.mean(values)
                std_dev = np.std(values)
                conf_interval = stats.t.interval(confidence=0.95, df=len(values) - 1, loc=mean, scale=stats.sem(values))
                metric_stats.append({
                'metric': metric,
                'Mean': np.mean(values),
                'Std Dev': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values),
                '95 percent confidence Interval': (conf_interval[0], conf_interval[1])
                 })
        
            # Store results for this learning rate
          all_results[lr] = {
            'metrics': all_final_metrics,
            'summary_stats': metric_stats  
             }
        
            # Determine the best model based on a specific metric (e.g., F1 score)
          avg = np.mean([m['MSE'] for m in all_final_metrics])
          if avg < best_metric:
                best_metric = avg
                best_model = model 
                best_lr = lr 
    
    best_result = {
            'metrics of best lr': all_results[best_lr]['metrics'],
            'summary' : all_results[best_lr]['summary_stats']
        }
    # Final model training on the whole dataset with the best learning rate
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    model = create_model_reg(X_train.shape[1:], learning_rate=best_lr)
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=5,         # Number of epochs to wait before stopping
        restore_best_weights=True  # Restore the best weights
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stopping], verbose=False  # type: ignore
    )
    y_pred_probs = model.predict(X_test)
    # y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class labels 
    plot = plot_regression_metrics(history, title=f'regression Prediction (Flow)')  # type: ignore 
    print(f"Best learning rate: {best_lr}, best_metric: {best_metric}") 

    return best_result, best_lr, y_pred, X_test, y_test, plot

def create_recurrent_model(lookback, input_shape, learning_rate, output_dim=1): 
    lstm_model = Sequential([
    Input(shape=(lookback, input_shape)), 
    LSTM(64, activation='tanh', return_sequences=True), 
    LSTM(64, activation='tanh', return_sequences=True, dropout=0, recurrent_dropout=0),
    LSTM(32, activation='tanh', dropout=0.2),
    Dense(16, activation='tanh'),
    Dense(output_dim, activation='sigmoid')
      ])
    optimizer = Adam(learning_rate = learning_rate)
    lstm_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc']) 
    return lstm_model

def create_recurrent_model_reg(lookback, input_shape, learning_rate, output_dim=1):
    lstm_model = Sequential([
    Input(shape=(lookback, input_shape)), 
    LSTM(64, activation='tanh', return_sequences=True), 
    LSTM(64, activation='tanh', return_sequences=True, dropout=0, recurrent_dropout=0),
    LSTM(32, activation='tanh', dropout=0.2),
    Dense(16, activation='tanh'),
    Dense(output_dim, activation='linear')
      ])
    optimizer = Adam(learning_rate = learning_rate)
    lstm_model.compile(loss='mse', optimizer=optimizer, metrics=['mse']) 
    return lstm_model

def train_and_evaluate_rec(X, y, lookback, n_features, epochs=50, n_repetitions=20, batch_size=32, learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]): 
    loss_function='binary_crossentropy'
    loss_fn = get_loss(loss_function)
    all_results = {}
    best_result = {}
    best_lr = None
    best_metric = -float("inf")
    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        metric_stats = []
        all_final_metrics = []
        for i in range(n_repetitions):
           print(f"Repetition {i+1}/{n_repetitions}")
            # Split the data
           # X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42+i) 
           # X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42+i)
           X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
           model = create_recurrent_model(lookback, n_features, lr)
           early_stopping = EarlyStopping(
                monitor='val_loss',  # Metric to monitor
                patience=5,         # Number of epochs to wait before stopping
                restore_best_weights=True  # Restore the best weights
            )
           history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=False)  # type: ignore
            
           loss_threshold = 0.75
           loss_values = history.history['loss']
           if any(loss < loss_threshold for loss in loss_values):
               plot = plot_training_metrics(history, title=f' return prediction (flow)')  # type: ignore    
           # Evaluate on test set
           y_pred = model.predict(X_test).round()
           accuracy = accuracy_score(y_test, y_pred)
           precision = precision_score(y_test, y_pred)
           recall = recall_score(y_test, y_pred)
           f1 = f1_score(y_test, y_pred)
           # loss = hinge_loss(y_test, y_pred)
           test_loss = loss_fn(y_test, y_pred).numpy()    
           all_final_metrics.append({
               'accuracy': accuracy,
               'precision': precision,
               'recall': recall,
               'f1 score': f1,
               'loss': test_loss[-1].item()
           }) 
        metrics = ['accuracy', 'precision', 'recall', 'f1 score', 'loss']
        for metric in metrics:
            values = [m[metric] for m in all_final_metrics]
            mean = np.mean(values)
            std_dev = np.std(values)
            conf_interval = stats.t.interval(confidence=0.95, df=len(values) - 1, loc=mean, scale=stats.sem(values))
            metric_stats.append({
                'metric': metric,
                'Mean': np.mean(values),
                'Std Dev': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values),
                '95 percent confidence Interval': (conf_interval[0], conf_interval[1])
            })
        
        # Store results for this learning rate
        all_results[lr] = {
            'metrics': all_final_metrics,
            'summary_stats': metric_stats  
        }
        
        # Determine the best model based on a specific metric (e.g., F1 score)
        avg = np.mean([m['accuracy'] for m in all_final_metrics])
        if avg > best_metric:
            best_metric = avg
            best_model = model 
            best_lr = lr 

    best_result = {
            'metrics of best lr': all_results[best_lr]['metrics'],
            'summary' : all_results[best_lr]['summary_stats']
        }
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    model = create_recurrent_model(lookback, n_features, best_lr) 
    early_stopping = EarlyStopping(
                monitor='val_loss',  # Metric to monitor
                patience=5,         # Number of epochs to wait before stopping
                restore_best_weights=True  # Restore the best weights
            )
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=False)  # type: ignore
    y_pred = model.predict(X_test).round()
    print(f"Best learning rate: {best_lr}, best_metric: {best_metric}") 
    
    return best_result, best_lr, y_pred, X_test, y_test, plot 

def train_and_evaluate_rec_reg(X, y, lookback, n_features, epochs=50, n_repetitions=20, batch_size=32, learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]): 
    loss_function='binary_crossentropy'
    loss_fn = get_loss(loss_function)
    all_results = {}
    best_result = {}
    best_model = None
    best_lr = None
    best_metric = -float("inf")
    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        metric_stats = []
        all_final_metrics = []
        for i in range(n_repetitions):
           print(f"Repetition {i+1}/{n_repetitions}")
            # Split the data
           # X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42+i) 
           # X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42+i)
           X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
           model = create_recurrent_model_reg(lookback, n_features, lr) 
           early_stopping = EarlyStopping(
                monitor='val_loss',  # Metric to monitor
                patience=5,         # Number of epochs to wait before stopping
                restore_best_weights=True  # Restore the best weights
            )
           history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=False)  # type: ignore    
           # Evaluate on test set
           y_pred = model.predict(X_test)
           mse = mean_squared_error(y_test, y_pred)
           rmse = np.sqrt(mean_squared_error(y_test, y_pred))
           mae = mean_absolute_error(y_test, y_pred)
           R2 = r2_score(y_test, y_pred)
           mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100    
           all_final_metrics.append({
               'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': R2,
                'MAPE': mape
           }) 
        metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE']
        for metric in metrics:
                values = [m[metric] for m in all_final_metrics]
                mean = np.mean(values)
                std_dev = np.std(values)
                conf_interval = stats.t.interval(confidence=0.95, df=len(values) - 1, loc=mean, scale=stats.sem(values))
                metric_stats.append({
                'metric': metric,
                'Mean': np.mean(values),
                'Std Dev': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values),
                '95 percent confidence Interval': (conf_interval[0], conf_interval[1])
                 })
        
        # Store results for this learning rate
        all_results[lr] = {
            'metrics': all_final_metrics,
            'summary_stats': metric_stats  
        }
        
        # Determine the best model based on a specific metric (e.g., F1 score)
        avg = np.mean([m['MSE'] for m in all_final_metrics])
        if avg > best_metric:
            best_metric = avg
            best_model = model 
            best_lr = lr 

    best_result = {
            'metrics of best lr': all_results[best_lr]['metrics'],
            'summary' : all_results[best_lr]['summary_stats'] 
        }
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    model = create_recurrent_model_reg(lookback, n_features, best_lr)
    early_stopping = EarlyStopping(
                monitor='val_loss',  # Metric to monitor
                patience=5,         # Number of epochs to wait before stopping
                restore_best_weights=True  # Restore the best weights
            )
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=False)  # type: ignore
    y_pred = model.predict(X_test)
    plot = plot_regression_metrics(history, title=f' return prediction (flow)')  # type: ignore
    print(f"Best learning rate: {best_lr}, best_metric: {best_metric}") 

    return best_result, best_lr, y_pred, X_test, y_test, plot 

def plot_training_metrics(history, test_metrics=None, title=None, save_path=r'C:\ninap\causal_pred\causal_prediction\fig.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
        # Set a title if provided
        if title:
            fig.suptitle(title, fontsize=16)
        
        # Plot Loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        
        # Add test loss if provided
        if test_metrics and 'test_loss' in test_metrics:
            ax1.axhline(y=test_metrics['test_loss'], color='r', label='test loss') 
        
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot Accuracy
        ax2.plot(history.history['acc'], label='Training Accuracy')
        ax2.plot(history.history['val_acc'], label='Validation Accuracy')
        
        # Add test accuracy if provided
        if test_metrics and 'test_accuracy' in test_metrics:
            ax2.axhline(y=test_metrics['test_accuracy'], color='r', label='test accuracy')
        
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path)
        
        return fig

def plot_regression_metrics(history, test_loss=None, title=None, save_path=r'C:\ninap\causal_pred\causal_prediction\fig.png'):
    """
    Plot training/validation loss and actual vs predicted values for regression models.

    Parameters:
        history: Training history object from Keras or similar libraries.
        y_actual: Actual target values (optional, for scatter plot).
        y_predicted: Predicted values (optional, for scatter plot).
        test_loss: Loss on the test set (optional).
        title: Title for the entire plot (optional).
        save_path: Path to save the plot (optional).

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Create subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    # Set a title if provided
    if title:
        fig.suptitle(title, fontsize=16)

    # Plot Loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')

    # Add test loss if provided
    if test_loss is not None:
        ax1.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')

    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend() 

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)

    return fig