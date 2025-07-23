from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import yaml
import numpy as np

with open('config/model_params.yaml') as f:
    params = yaml.safe_load(f)

class XGBoostModel:
    def __init__(self, target_horizon=1):
        self.model = None
        self.target_horizon = target_horizon
        self.params = params['xgboost']
        # Move early_stopping to constructor params
        if 'early_stopping_rounds' in self.params:
            self.early_stopping = self.params.pop('early_stopping_rounds')
        else:
            self.early_stopping = None
        
    def prepare_data(self, df):
        feature_cols = [col for col in df.columns 
                       if not col.startswith('target_') 
                       and not col.startswith('mid_price_change_')
                       and col not in ['timestamp', 'event_type', 'order_id', 'direction']]
        
        target_col = f'target_{self.target_horizon}'
        X = df[feature_cols]
        y = df[target_col]
        return X, y
    
    # def train(self, X, y):

    #     # Adjust class weights
    #     class_weights = len(y) / (2 * np.bincount(y))
    #     self.params['scale_pos_weight'] = class_weights[1]/class_weights[0]  # ~3.2 for your data

    #     X_train, X_val, y_train, y_val = train_test_split(
    #         X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42
    #     )
        
    #     # Initialize model with all parameters
    #     self.model = XGBClassifier(**self.params)
        
    #     # Fit with validation set if early stopping is enabled
    #     if self.early_stopping:
    #         self.model.fit(
    #             X_train, y_train,
    #             eval_set=[(X_val, y_val)],
    #             early_stopping_rounds=self.early_stopping,
    #             verbose=10
    #             #verbose=True
    #         )
    #     else:
    #         self.model.fit(X_train, y_train)
        
    #     # Evaluate
    #     print("\n=== Validation Metrics ===")
    #     preds = self.model.predict(X_val)
    #     print(classification_report(y_val, preds))
        
    #     return self.model

    def train(self, X, y):
        # Calculate class weights
        class_counts = np.bincount(y)
        if len(class_counts) >= 2:
            self.params['scale_pos_weight'] = class_counts[0]/class_counts[1]
    
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    
        # Initialize model with all parameters (including early_stopping_rounds)
        self.model = XGBClassifier(**self.params)
    
        # Fit without early_stopping_rounds in fit()
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=10
        )
    
        preds = self.model.predict(X_val)
        print(classification_report(y_val, preds))
    
        return self.model

    def predict_proba(self, X):
        """Return class probabilities"""
        return self.model.predict_proba(X)
    
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    def load(self, path):
        self.model = joblib.load(path)
        return self