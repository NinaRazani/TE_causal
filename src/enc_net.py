import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hinge_loss
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import tensorflow 
from tensorflow import keras

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

def create_transformer_encoder(input_shape, num_layers=4, d_model=64, num_heads=6, dff=128, dropout_rate=0.1): 
    inputs = keras.Input(shape=(input_shape)) 

    x = keras.layers.Reshape((input_shape, 1))(inputs)
    x = keras.layers.Dense(d_model)(x)

    x += keras.layers.Embedding(input_dim=input_shape, output_dim=d_model)(tensorflow.range(start=0, limit=input_shape, delta=1))  #positional encoding

    for _ in range(num_layers): 
        attn_output = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim = d_model)(x, x)
        attn_output = keras.layers.Dropout(dropout_rate)(attn_output)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x+attn_output)

        ffn = keras.Sequential([ # type: ignore
            keras.layers.Dense(dff, activation="relu"),
            keras.layers.Dense(d_model)
        ])
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x+ffn(x))

    x = keras.layers.GlobalAveragePooling1D()(x) # it is used because we get the output of encoder to a classifier # type: ignore
    return keras.Model(inputs=inputs, outputs=x)
 
def encoder_features(features):
     input_sahpe = features.shape[1]
     encoder = create_transformer_encoder(input_sahpe)
     encoder.compile(optimizer='adam', loss='mse')
     #Use the encoder to get the new feature representations
     X_encoded = encoder.predict(features)  
     return X_encoded

def create_model(input_shape):
    model = SVC(kernel='rbf', probability=True)
    return model

def train_and_evaluate(X, y, n_repetitions=20):
    metric_stats = []
    all_final_metrics = []
    X_encoded = encoder_features(X) 
    for i in range(n_repetitions):
        print(f"Repetition {i+1}/{n_repetitions}")
        # X_encoded = encoder_features(X) 
        # Split the data
        # X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42+i)
        # X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X_encoded, y) 
        model = create_model(X_train.shape[1:])
        history = model.fit(X_train, y_train) # type: ignore 
        
        # Evaluate on test set
        y_pred = model.predict(X_test).round()
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, labels=np.unique(y_pred))
        loss = hinge_loss(y_test, y_pred)

        # all_histories.append(history.history) 
        all_final_metrics.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1 score': f1,
            'loss': loss
        })
    
    metrics = ['accuracy', 'precision', 'recall', 'f1 score', 'loss']
    for metric in metrics:
        values = [m[metric] for m in all_final_metrics]
        mean = np.mean(values)
        std_dev = np.std(values)
        conf_interval = stats.t.interval(confidence=0.95, df=len(values)-1, loc=mean, scale=stats.sem(values))
        metric_stats.append({
            'metric': metric,
            'Mean': np.mean(values),
            'Std Dev': np.std(values),
            'Min': np.min(values),
            'Max': np.max(values),
            '95 percent confidence Interval': (conf_interval[0], conf_interval[1])
        })
        
    return all_final_metrics, metric_stats
    
