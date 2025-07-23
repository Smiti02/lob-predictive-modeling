import pandas as pd
import numpy as np
from typing import List

def calculate_order_flow_imbalance(df: pd.DataFrame, levels: int = 5) -> pd.DataFrame:
    """Calculate order flow imbalance features."""
    for i in range(1, levels+1):
        df[f'ofi_{i}'] = df[f'bid_size_{i}'] - df[f'ask_size_{i}']
    
    # Total order flow imbalance
    df['total_ofi'] = df[[f'ofi_{i}' for i in range(1, levels+1)]].sum(axis=1)
    return df

def calculate_volatility_features(df: pd.DataFrame, windows: List[int] = [10, 50, 100]) -> pd.DataFrame:
    """Calculate rolling volatility features."""
    for window in windows:
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
        df[f'microprice_volatility_{window}'] = df['microprice'].rolling(window).std()
    
    return df

def calculate_price_change_features(df: pd.DataFrame, horizons: List[int] = [1, 5, 10]) -> pd.DataFrame:
    """Calculate future price changes for prediction targets."""
    for horizon in horizons:
        df[f'mid_price_change_{horizon}'] = df['mid_price'].shift(-horizon) - df['mid_price']
        df[f'target_{horizon}'] = (df[f'mid_price_change_{horizon}'] > 0).astype(int)
    
    return df

def calculate_volume_features(df: pd.DataFrame, levels: int = 5) -> pd.DataFrame:
    """Calculate volume-related features."""
    # Total volume at each level
    for i in range(1, levels+1):
        df[f'total_volume_{i}'] = df[f'bid_size_{i}'] + df[f'ask_size_{i}']
    
    # Volume imbalance
    for i in range(1, levels+1):
        df[f'volume_imbalance_{i}'] = (
            (df[f'bid_size_{i}'] - df[f'ask_size_{i}']) / 
            (df[f'bid_size_{i}'] + df[f'ask_size_{i}'])
        )
    
    return df

def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate time-based features."""
    # Time since last event
    df['time_since_last'] = df['timestamp'].diff()
    
    # Cumulative events
    df['cumulative_events'] = np.arange(len(df))
    
    # Event rate
    df['event_rate'] = 1 / df['time_since_last'].rolling(100).mean()
    
    return df

def create_all_features(df: pd.DataFrame, levels: int = 5) -> pd.DataFrame:
    """Create all features for the model."""
    df = calculate_order_flow_imbalance(df, levels)
    df = calculate_volatility_features(df)
    df = calculate_volume_features(df, levels)
    df = calculate_time_features(df)
    df = calculate_price_change_features(df)
    
    # Drop rows with NaN values from rolling calculations
    df = df.dropna()
    
    return df

# # Add these to your feature_engineering.py
# def calculate_improved_features(df):
#     # Price momentum features
#     df['price_trend_5'] = df['mid_price'].pct_change(5)
#     df['price_trend_10'] = df['mid_price'].pct_change(10)
    
#     # Volume features
#     df['volume_imbalance'] = (df['bid_size_1'] - df['ask_size_1']) / (df['bid_size_1'] + df['ask_size_1'])
#     df['total_volume'] = df['bid_size_1'] + df['ask_size_1']
    
#     # Order book depth features
#     for i in range(1, 6):
#         df[f'depth_imbalance_{i}'] = (df[f'bid_size_{i}'] - df[f'ask_size_{i}']) / (df[f'bid_size_{i}'] + df[f'ask_size_{i}'])
    
#     return df

def calculate_improved_features(df):
    # Price momentum
    df['price_change_5'] = df['mid_price'].pct_change(5)
    df['price_accel'] = df['price_change_5'].diff()
    
    # Volume features
    df['volume_ratio'] = df['bid_size_1'] / (df['ask_size_1'] + 1e-6)
    df['volatility'] = df['mid_price'].rolling(20).std()
    
    # Order book imbalance
    df['depth_imbalance'] = (df['bid_size_1'] - df['ask_size_1']) / (df['bid_size_1'] + df['ask_size_1'])
    
    # Time-based features
    df['time_since_last'] = df['timestamp'].diff()
    df['trade_intensity'] = 1 / df['time_since_last'].rolling(10).mean()
    
    return df