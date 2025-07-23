import argparse
from src.data_preprocessing import load_lobster_data, process_orderbook, merge_messages_orderbook, save_processed_data
from src.feature_engineering import create_all_features
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.backtesting import Backtester
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='LOB Predictive Modeling')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--date', type=str, default='2012-06-21', help='Date in YYYY-MM-DD format')
    parser.add_argument('--model', type=str, default='xgboost', 
                       choices=['xgboost', 'lstm', 'transformer'], help='Model to use')
    parser.add_argument('--levels', type=int, default=5, help='Number of LOB levels to use')
    parser.add_argument('--horizon', type=int, default=5, help='Prediction horizon')
    args = parser.parse_args()

    # 1. Data Loading and Preprocessing
    print("Loading and preprocessing data...")
    messages, orderbook = load_lobster_data(args.symbol, args.date, args.levels)
    orderbook = process_orderbook(orderbook, args.levels)
    merged_data = merge_messages_orderbook(messages, orderbook)
    
    # 2. Feature Engineering
    print("Creating features...")
    feature_data = create_all_features(merged_data, args.levels)
    
    # 3. Model Training
    print(f"Training {args.model} model...")
    if args.model == 'xgboost':
        model = XGBoostModel(target_horizon=args.horizon)
        X, y = model.prepare_data(feature_data)
        model.train(X, y)
        if args.model == 'xgboost':
            print("Feature Importances:")
            print(model.model.feature_importances_)

        try:
            # Plot calibration curve
            from sklearn.calibration import calibration_curve
            prob_true, prob_pred = calibration_curve(y, model.predict_proba(X)[:,1], n_bins=10)
            plt.figure(figsize=(8, 4))
            plt.plot(prob_pred, prob_true, marker='o')
            plt.plot([0, 1], [0, 1], linestyle='--')  # Perfect calibration line
            plt.xlabel("Predicted Probability")
            plt.ylabel("Actual Probability")
            plt.title("XGBoost Probability Calibration")
            plt.show()

            # Additional diagnostic: Prediction distribution
            plt.figure(figsize=(8, 4))
            plt.hist(model.predict_proba(X)[:,1], bins=50)
            plt.title("Predicted Probability Distribution")
            plt.show()

        except AttributeError as e:
            print(f"Cannot plot calibration: {e}")

            predictions = model.predict(X)  
    elif args.model == 'lstm':
        model = LSTMModel(target_horizon=args.horizon)
        X, y = model.prepare_data(feature_data)
        model.train(X, y)
        predictions = model.predict(X)
    else:  # transformer
        model = TransformerModel(target_horizon=args.horizon)
        X, y = model.prepare_data(feature_data)
        model.train(X, y)
        predictions = model.predict(X)


    # Ensure predictions are probabilities between 0-1
    if args.model == 'xgboost':
        predictions = model.model.predict_proba(X)[:, 1]  # Get class 1 probabilities
    elif args.model in ['lstm', 'transformer']:
        predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())  # Scale 0-1
    
    # 4. Backtesting
    # print("Running backtest...")
    # backtester = Backtester()
    # results = backtester.run_backtest(predictions, feature_data['mid_price'])
    

    print("Running backtest...")

    # Get prediction probabilities (for XGBoost)
    if args.model == 'xgboost':
        predictions = model.model.predict_proba(X)[:, 1]  # Class 1 probabilities
    else:
        predictions = predictions.flatten()  # For LSTM/Transformer

    backtester = Backtester(initial_capital=100000, trade_size=1000)
    results = backtester.run_backtest(
        predictions=predictions,
        prices=feature_data['mid_price'],
        #buy_threshold=0.55,  # Adjusted thresholds
        #buy_threshold=0.7,
        buy_threshold=0.8,  # Much stricter threshold
        #sell_threshold=0.45,
        #sell_threshold=0.3,  # Only trade on strong sell signals
        sell_threshold=0.2,
        #execution_lag=1,
        #execution_lag=5,
        execution_lag=3,
        spread_cost=0.0001
    ) 

    # 5. Results - ENHANCED OUTPUT
    print("\n=== Backtest Results ===")
    print(f"Initial Capital: ${backtester.initial_capital:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    #print(f"Avg Trade Risk: ${results['avg_trade_risk']:,.2f}")

    # # 5. Results
    # print("\n=== Backtest Results ===")
    # print(f"Initial Capital: ${backtester.initial_capital:,.2f}")
    # print(f"Final Value: ${results['final_value']:,.2f}")
    # print(f"Total Return: {results['total_return']:.2f}%")
    # print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    # print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    # print(f"Number of Trades: {results['num_trades']}")
    
    # Plot results
    if 'portfolio_values' in results and len(results['portfolio_values']) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(results['portfolio_values'])
        #plt.title('Portfolio Value Over Time')
        plt.title(f'Portfolio Value Over Time ({args.model.upper()} Model)')
        plt.xlabel('Time Step')
        plt.ylabel('Portfolio Value ($)')
        plt.grid()
        plt.show()

    else:
        print("Warning: No portfolio values to plot")

if __name__ == '__main__':
    main()