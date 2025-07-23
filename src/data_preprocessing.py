import pandas as pd
import numpy as np
import yaml
from pathlib import Path

with open('config/paths.yaml') as f:
    paths = yaml.safe_load(f)

DATA_DIR = Path(paths['data_directory'])

def load_lobster_data(symbol: str, date: str, levels: int = 10):
    """Load LOBSTER data for a specific symbol and date."""
    message_file = DATA_DIR / 'raw' / f'{symbol}_{date}_34200000_57600000_message_{levels}.csv'
    orderbook_file = DATA_DIR / 'raw' / f'{symbol}_{date}_34200000_57600000_orderbook_{levels}.csv'
    
    # Load message data
    message_cols = ['timestamp', 'event_type', 'order_id', 'size', 'price', 'direction']
    messages = pd.read_csv(message_file, header=None, names=message_cols)
    
    # Load order book data
    orderbook = pd.read_csv(orderbook_file, header=None)
    return messages, orderbook

def process_orderbook(orderbook: pd.DataFrame, levels: int = 5):
    """Process raw orderbook data into structured format."""
    # Create column names for bid/ask prices and sizes
    price_cols = []
    size_cols = []
    for i in range(1, levels+1):
        price_cols.extend([f'bid_price_{i}', f'ask_price_{i}'])
        size_cols.extend([f'bid_size_{i}', f'ask_size_{i}'])
    
    # Assign column names
    orderbook.columns = price_cols + size_cols
    
    # Calculate microprice and other basic features
    orderbook['microprice'] = (
        orderbook['bid_price_1'] * orderbook['ask_size_1'] + 
        orderbook['ask_price_1'] * orderbook['bid_size_1']
    ) / (orderbook['bid_size_1'] + orderbook['ask_size_1'])
    
    orderbook['spread'] = orderbook['ask_price_1'] - orderbook['bid_price_1']
    orderbook['mid_price'] = (orderbook['ask_price_1'] + orderbook['bid_price_1']) / 2

    # --- ADD THIS NEW FILTERING CODE ---
    # Remove outliers and stale periods
    #orderbook = orderbook[
       # (orderbook['spread'] > 0) & 
       # (orderbook['spread'] < orderbook['mid_price'] * 0.01) &  # Filter spreads >1% of price
       # (orderbook['time_since_last'] < 1.0)  # Remove periods with >1sec gaps
    #].copy()
    
    return orderbook

def merge_messages_orderbook(messages: pd.DataFrame, orderbook: pd.DataFrame):
    """Merge message and orderbook data."""
    # Add orderbook snapshots to each message
    merged = pd.concat([messages, orderbook], axis=1)
    
    # Calculate time differences
    merged['time_diff'] = merged['timestamp'].diff()
    
    return merged

def save_processed_data(df: pd.DataFrame, symbol: str, date: str):
    """Save processed data to parquet file."""
    output_path = DATA_DIR / 'processed' / f'{symbol}_{date}_processed.parquet'
    df.to_parquet(output_path)
    return output_path