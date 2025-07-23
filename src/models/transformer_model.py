import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yaml

with open('config/model_params.yaml') as f:
    params = yaml.safe_load(f)

class TransformerModel:
    def __init__(self, target_horizon=1, sequence_length=50):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_horizon = target_horizon
        self.sequence_length = sequence_length
        self.params = params['transformer']
        
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = Dropout(dropout)(x)
        res = x + inputs
        
        # Feed Forward Part
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Dense(ff_dim, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)
        return x + res
    
    def build_model(self, input_shape):
        inputs = Input(shape=input_shape)
        x = inputs
        
        for _ in range(self.params['num_layers']):
            x = self.transformer_encoder(
                x, 
                self.params['head_size'], 
                self.params['num_heads'], 
                self.params['ff_dim'], 
                self.params['dropout']
            )
            
        x = GlobalAveragePooling1D()(x)
        x = Dropout(self.params['dropout'])(x)
        outputs = Dense(1, activation="sigmoid")(x)
        
        self.model = Model(inputs, outputs)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        return self.model
    
    def prepare_data(self, df: pd.DataFrame):
        """Prepare sequences for Transformer."""
        # Same as LSTM data preparation
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
    
    def train(self, X, y):
        """Train the Transformer model."""
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
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        return self