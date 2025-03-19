from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hinge_loss
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Input, LSTM, Dropout, BatchNormalization# type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.optimizers import Adam # type: ignore 
from tensorflow.keras.callbacks import EarlyStopping  # Import EarlyStopping
from tensorflow.keras.losses import get as get_loss


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

def create_model_three(input_shape, learning_rate):
    model_cls=Sequential()
    # model_cls.add(Flatten(input_shape=input_shape))
    # model_cls.add(Dense(10,activation="relu"))
    # model_cls.add(Dense(20,activation="relu"))
    # model_cls.add(Dense(30,activation="relu"))
    # model_cls.add(Dense(3, activation="linear"))
    model_cls = Sequential()
    model_cls.add(Flatten(input_shape=input_shape))
    model_cls.add(Dropout(0.2))  # type: ignore
    model_cls.add(Dense(64, activation="relu", kernel_regularizer=l2(0.001)))  # type: ignore
    model_cls.add(BatchNormalization())  # type: ignore
    model_cls.add(Dropout(0.3))  # type: ignore
    model_cls.add(Dense(32, activation="relu"))
    model_cls.add(BatchNormalization())  # type: ignore
    model_cls.add(Dropout(0.3))  # type: ignore
    model_cls.add(Dense(16, activation="relu"))
    model_cls.add(Dense(3, activation="softmax"))  # Updated for 3-class classification

    optimizer = Adam(learning_rate=learning_rate)  # type: ignore
    model_cls.compile(
        loss='sparse_categorical_crossentropy',  # Use this if labels are integer-encoded
        optimizer=optimizer,
        metrics=['accuracy']  # Use 'accuracy' for multiclass classification
    )
    return model_cls

def train_and_evaluate_deep_three(X, y, epochs=50, n_repetitions=20, batch_size=32, learning_rates=[0.1, 0.01, 0.001, 0.0001, 0.00001]):
    # loss_function = 'sparse_categorical_crossentropy'  # Updated for 3-class classification
    # loss_fn = get_loss(loss_function)
    all_results = {}
    best_result = {}
    best_model = None
    best_lr = None
    best_metric = -float("inf")  # Adjust based on the metric you want to optimize (e.g., accuracy, F1 score)
    plot = None 
    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        metric_stats = []
        all_final_metrics = []
        
        for i in range(n_repetitions):
            print(f"Repetition {i + 1}/{n_repetitions}")
            # Split the data
            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
            model = create_model_three(X_train.shape[1:], learning_rate=lr)
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

            # Evaluate on test set
            y_pred_probs = model.predict(X_test)  # Get probabilities
            y_pred = np.argmax(y_pred_probs, axis=1)  # Convert to class labels
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            test_loss = model.evaluate(X_test, y_test, verbose=False)[0] 

            all_final_metrics.append({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1 score': f1, 
                'loss': test_loss
            })

        metrics = ['accuracy', 'precision', 'recall', 'f1 score', 'loss']
        for metric in metrics:
            values = [m[metric] for m in all_final_metrics]
            mean = np.mean(values)
            std_dev = np.std(values)
            conf_interval = stats.t.interval(confidence=0.95, df=len(values) - 1, loc=mean, scale=stats.sem(values))
            metric_stats.append({
                'metric': metric,
                'Mean': mean,
                'Std Dev': std_dev,
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
        'summary': all_results[best_lr]['summary_stats']
    }

    # Final model training on the whole dataset with the best learning rate
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    model = create_model_three(X_train.shape[1:], learning_rate=best_lr)
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
    y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class labels
    # loss_threshold = 0.75
    # loss_values = history.history['loss']
    # if any(loss < loss_threshold for loss in loss_values):
    plot = plot_training_metrics(history, title=f'volatility Prediction (no Flow)')  # type: ignore 
    print(f"Best learning rate: {best_lr}, best_metric: {best_metric}") 

    return best_result, best_lr, y_pred, X_test, y_test, plot

def create_recurrent_three(lookback, input_shape, learning_rate, output_dim=3): 
    lstm_model = Sequential([
        Input(shape=(lookback, input_shape)), 
        LSTM(64, activation='tanh', return_sequences=True), 
        LSTM(64, activation='tanh', return_sequences=True, dropout=0, recurrent_dropout=0),
        LSTM(32, activation='tanh', dropout=0.2),
        Dense(16, activation='tanh'),
        Dense(output_dim, activation='softmax')  # Updated for 3-class classification
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    lstm_model.compile(
        loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy for integer-encoded labels
        optimizer=optimizer,
        metrics=['accuracy']  # Accuracy metric for multiclass classification
    )
    
    return lstm_model

def train_and_evaluate_rec_three(X, y, lookback, n_features, epochs=50, n_repetitions=20, batch_size=32, learning_rates=[0.1, 0.01, 0.001, 0.0001, 0.00001]): 
    loss_function = 'sparse_categorical_crossentropy'  # Updated for multi-class classification
    loss_fn = get_loss(loss_function)
    all_results = {}
    best_result = {}
    best_model = None
    best_lr = None
    best_metric = -float("inf")
    for device in tf.config.experimental.list_logical_devices(): # type: ignore
         print(device)
    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        metric_stats = []
        all_final_metrics = []
        for i in range(n_repetitions):
            print(f"Repetition {i+1}/{n_repetitions}")
            
            # Split the data
            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
            
            # Create the model
            model = create_recurrent_three(lookback, n_features, lr, output_dim=3)
            
            # Add early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',  
                patience=5,         
                restore_best_weights=True
            )
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=False
            )
            
                   
            # Evaluate on the test set
            y_pred = np.argmax(model.predict(X_test), axis=1)  # Use argmax for multi-class predictions
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')  # Use 'macro' for multi-class
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            test_loss = model.evaluate(X_test, y_test, verbose=False)[0]

            all_final_metrics.append({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1 score': f1,
                'loss': test_loss
            })
        
        # Calculate statistics for the metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1 score', 'loss']
        for metric in metrics:
            values = [m[metric] for m in all_final_metrics]
            mean = np.mean(values)
            std_dev = np.std(values)
            conf_interval = stats.t.interval(confidence=0.95, df=len(values) - 1, loc=mean, scale=stats.sem(values))
            metric_stats.append({
                'metric': metric,
                'Mean': mean,
                'Std Dev': std_dev,
                'Min': np.min(values),
                'Max': np.max(values),
                '95 percent confidence Interval': (conf_interval[0], conf_interval[1])
            })
        
        # Store results for this learning rate
        all_results[lr] = {
            'metrics': all_final_metrics,
            'summary_stats': metric_stats  
        }
        
        # Determine the best model based on the average accuracy
        avg = np.mean([m['accuracy'] for m in all_final_metrics])
        if avg > best_metric:
            best_metric = avg
            best_model = model 
            best_lr = lr 

    # Final evaluation with the best learning rate
    best_result = {
        'metrics of best lr': all_results[best_lr]['metrics'],
        'summary': all_results[best_lr]['summary_stats']
    }
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y) 
    model = create_recurrent_three(lookback, n_features, best_lr, output_dim=3)
    early_stopping = EarlyStopping(
        monitor='val_loss',  
        patience=5,         
        restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=False
    )
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    plot = plot_training_metrics(history, title=f'volatility Prediction (Flow)') 
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
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        
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