import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yaml

with open('config/model_params.yaml') as f:
    params = yaml.safe_load(f)

class LSTMModel:
    def __init__(self, target_horizon=1, sequence_length=50):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_horizon = target_horizon
        self.sequence_length = sequence_length
        self.params = params['lstm']
        
    def prepare_data(self, df: pd.DataFrame):
        """Prepare sequences for LSTM."""
        # Select features and scale
        feature_cols = [col for col in df.columns 
                       if not col.startswith('target_') 
                       and not col.startswith('mid_price_change_')
                       and not col in ['timestamp', 'event_type', 'order_id', 'direction']]
        
        X = df[feature_cols]
        y = df[f'target_{self.target_horizon}']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - self.sequence_length):
            X_seq.append(X_scaled[i:i+self.sequence_length])
            y_seq.append(y.iloc[i+self.sequence_length])
            
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape):
        """Build LSTM model architecture."""
        self.model = Sequential([
            LSTM(units=self.params['units'], 
                 return_sequences=True, 
                 input_shape=input_shape),
            Dropout(self.params['dropout']),
            LSTM(units=self.params['units']//2),
            Dropout(self.params['dropout']),
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=self.params['learning_rate'])
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X, y):
        """Train the LSTM model."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        self.model = self.build_model(X_train.shape[1:])
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X).flatten()
    
    def save(self, model_path, scaler_path):
        """Save model and scaler."""
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
    
    def load(self, model_path, scaler_path):
        """Load model and scaler."""
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        return self