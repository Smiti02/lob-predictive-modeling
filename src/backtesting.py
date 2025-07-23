# import numpy as np
# import pandas as pd
# from typing import Dict

# class Backtester:
#     # def __init__(self, initial_capital=100000, trade_size=1000):
        
#     #     self.initial_capital = initial_capital
#     #     self.trade_size = trade_size
#     #     self.current_prices = []  # Initialize as regular Python list
#     #     self.portfolio_values = []
#     #     self.position = 0
#     #     self.capital = initial_capital
#     #     self.trades = []
#     #     self.reset()
        
#     # def reset(self):
#     #     """Reset backtester state."""
#     #     self.capital = self.initial_capital
#     #     self.position = 0
#     #     self.trades = []
#     #     self.portfolio_values = []
#     #     self.current_prices = None


#     def __init__(self, initial_capital=100000, trade_size=1000):
#         self.initial_capital = initial_capital
#         self.trade_size = trade_size
#         self.reset()  # Initialize all values through reset()

#     def reset(self):
#         """Reset backtester state."""
#         self.capital = self.initial_capital
#         self.position = 0
#         self.trades = []
#         self.portfolio_values = []
#         self.current_prices = []  # Initialize as empty list, not None


#     def run_backtest(self, predictions: np.ndarray, prices: pd.Series, 
#                 execution_lag: int = 1, spread_cost: float = 0.0001):
#         self.reset()
    
#         # Convert prices to numpy array for consistent handling
#         price_values = prices.values if isinstance(prices, pd.Series) else np.array(prices)
    
#         for i in range(len(predictions)):
#             if i < execution_lag:
#                 continue
            
#             current_price = float(price_values[i])  # Explicit conversion to float
#             pred = predictions[i - execution_lag]
        
#             # Execute trades
#             if pred > 0.6 and self.position <= 0:
#                 self.execute_trade(1, current_price, spread_cost)
#             elif pred < 0.4 and self.position >= 0:
#                 self.execute_trade(-1, current_price, spread_cost)
            
#             # Record portfolio value
#             self.record_portfolio_value(current_price)
    
#         return self.get_results()
        
#     # def run_backtest(self, predictions: np.ndarray, prices: pd.Series, 
#     #                 execution_lag: int = 1, spread_cost: float = 0.0001):
#     #     """
#     #     Run backtest on predictions.
        
#     #     Args:
#     #         predictions: Array of prediction probabilities (0-1)
#     #         prices: Series of mid prices
#     #         execution_lag: Number of steps between signal and execution
#     #         spread_cost: Cost per trade as fraction of spread
#     #     """
#     #     self.reset()
#     #     self.current_prices = prices
        
#     #     for i in range(len(predictions)):
#     #         if i < execution_lag:
#     #             continue
                
#     #         # Get current price and prediction
#     #         current_price = prices.iloc[i]  if hasattr(prices, 'iloc') else prices[i]
#     #         current_price = float(current_price)

#     #         pred = predictions[i - execution_lag]
            
#     #         # Execute trades based on predictions
#     #         if pred > 0.6 and self.position <= 0:  # Buy signal
#     #             self.execute_trade(1, current_price, spread_cost)
#     #         elif pred < 0.4 and self.position >= 0:  # Sell signal
#     #             self.execute_trade(-1, current_price, spread_cost)
                
#     #         # Record portfolio value
#     #         self.record_portfolio_value(current_price)
            
#     #     return self.get_results()
    
#     # def execute_trade(self, direction: int, price: float, spread_cost: float):
#     #     """Execute a trade with volatility-based position sizing"""
#     #     # Calculate recent price volatility (standard deviation of last 100 prices)
#     #     recent_volatility = np.std(self.current_prices[-100:]) if len(self.current_prices) >= 100 else 0

#     #     # Trade frequency filter (NEW)
#     #     if len(self.trades) > 0:
#     #         last_trade_time = self.trades[-1]['timestamp']
#     #         if (len(self.portfolio_values) - last_trade_time) < 10:  # 10-period cooldown
#     #             return

#     #     # Dynamic position sizing - scales with volatility up to 10% of capital
#     #     max_risk_pct = 0.02  # 2% of initial capital at risk
#     #     position_size = min(
#     #         #self.initial_capital * max_risk_pct / (recent_vol + 1e-6),  # Volatility scaling
#     #         self.initial_capital * max_risk_pct / (recent_volatility + 1e-6),  # Volatility scaling
#     #         # self.trade_size * (1 + recent_volatility * 10),  # Base size adjusted by volatility
#     #         self.trade_size * 3, # Maximum 3x base trade size
#     #         self.capital * 0.1  # Never risk more than 10% of capital
#     #     )


#     #     # Calculate execution price with spread cost
#     #     execution_price = price * (1 + direction * spread_cost/2)
        
#     #     # Calculate number of shares (fixed trade size)
#     #     #shares = self.trade_size / execution_price
#     #     shares = position_size / execution_price
        
#     #     # Update position and capital
#     #     self.position += direction * shares
#     #     self.capital -= direction * shares * execution_price
        
#     #     # Record trade
#     #     self.trades.append({
#     #         'direction': direction,
#     #         'price': execution_price,
#     #         'shares': shares,
#     #         'timestamp': len(self.portfolio_values),
#     #         'volatility': recent_volatility,
#     #         'position_size': position_size
#     #     })
    

#     def execute_trade(self, direction: int, price: float, spread_cost: float):
#         """Execute a trade with volatility-based position sizing"""
#         # Calculate recent price volatility
#         recent_vol = np.std(self.current_prices[-100:]) if len(self.current_prices) >= 100 else 0
    
#         # Trade frequency filter
#         if len(self.trades) > 0:
#             last_trade_time = self.trades[-1]['timestamp']
#             if (len(self.portfolio_values) - last_trade_time) < 10:
#                 return

#         # Dynamic position sizing
#         position_size = min(
#             self.initial_capital * 0.02 / (recent_vol + 1e-6),
#             self.trade_size * 3,
#             self.capital * 0.1
#         )
    
#         # Rest of the method remains the same...

#     # def record_portfolio_value(self, current_price: float):
#     #     """Record current portfolio value and track prices"""

#     #     # # Convert float to list if needed (handles both scalar and Series inputs)
#     #     # if hasattr(current_price, '__iter__'):  # If it's a Series/array-like
#     #     #     self.current_prices.extend(current_price.tolist())
#     #     # else:  # If it's a single float
#     #     #     self.current_prices.append(float(current_price))

#     #     # Ensure we're working with a scalar value
#     #     if hasattr(current_price, 'item'):  # Handles pandas Series/DataFrame
#     #         current_price = current_price.item()
#     #     elif hasattr(current_price, '__iter__'):  # Handles numpy arrays
#     #         current_price = float(current_price[0])

#     #     self.current_prices.append(current_price)
#     #     position_value = self.position * current_price
#     #     total_value = self.capital + position_value
#     #     self.portfolio_values.append(total_value)


#     def record_portfolio_value(self, current_price: float):
#         """Record current portfolio value."""
#         # Ensure current_price is a float
#         current_price = float(current_price)
    
#         # Track price history
#         self.current_prices.append(current_price)
    
#         # Calculate portfolio value
#         position_value = self.position * current_price
#         total_value = self.capital + position_value
#         self.portfolio_values.append(total_value)
    
#     def get_results(self) -> Dict:
#         """Get backtest results."""
#         returns = pd.Series(self.portfolio_values).pct_change().dropna()
        
#         return {
#             'final_value': self.portfolio_values[-1],
#             'total_return': (self.portfolio_values[-1] / self.initial_capital - 1) * 100,
#             'sharpe_ratio': self.calculate_sharpe(returns),
#             'max_drawdown': self.calculate_max_drawdown(),
#             'num_trades': len(self.trades),
#             'returns': returns,
#             'portfolio_values': self.portfolio_values,
#             'trades': self.trades
#         }
    
#     def calculate_sharpe(self, returns: pd.Series, risk_free_rate=0.0) -> float:
#         """Calculate annualized Sharpe ratio."""
#         excess_returns = returns - risk_free_rate / 252
#         return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
#     def calculate_max_drawdown(self) -> float:
#         """Calculate maximum drawdown."""
#         peak = self.initial_capital
#         max_drawdown = 0
        
#         for value in self.portfolio_values:
#             if value > peak:
#                 peak = value
#             drawdown = (peak - value) / peak
#             if drawdown > max_drawdown:
#                 max_drawdown = drawdown
                
#         return max_drawdown * 100  # as percentage


import numpy as np
import pandas as pd
from typing import Dict, Optional

class Backtester:
    def __init__(self, initial_capital: float = 100000, trade_size: float = 1000):
        self.initial_capital = initial_capital
        self.trade_size = trade_size
        self.reset()

    def reset(self):
        """Reset all backtesting state variables."""
        self.capital = self.initial_capital
        self.position = 0  # Current position in shares
        self.trades = []  # List of executed trades
        self.portfolio_values = []  # Track portfolio value over time
        self.current_prices = []  # Track price history for volatility calculations

    def run_backtest(
        self,
        predictions: np.ndarray,
        prices: pd.Series,
        execution_lag: int = 1,
        spread_cost: float = 0.0001,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.4,
    ) -> Dict:
        """
        Run backtest on predictions.
        
        Args:
            predictions: Array of prediction probabilities (0-1).
            prices: Series of mid prices.
            execution_lag: Steps between signal and execution.
            spread_cost: Fractional cost per trade (bid-ask spread).
            buy_threshold: Minimum probability to trigger a buy.
            sell_threshold: Maximum probability to trigger a sell.
        """
        self.reset()
        price_values = prices.values if isinstance(prices, pd.Series) else np.array(prices)

        for i in range(len(predictions)):
            if i < execution_lag:
                continue

            current_price = float(price_values[i])
            pred = predictions[i - execution_lag]

            # Dynamic thresholds (optional: adjust based on volatility)
            if len(self.current_prices) > 20:
                recent_vol = np.std(self.current_prices[-20:])
                dynamic_buy_thresh = buy_threshold - 0.1 * recent_vol  # Lower threshold in high volatility
                dynamic_sell_thresh = sell_threshold + 0.1 * recent_vol
            else:
                dynamic_buy_thresh, dynamic_sell_thresh = buy_threshold, sell_threshold

            # Execute trades
            if pred > dynamic_buy_thresh and self.position <= 0:
                self.execute_trade(1, current_price, spread_cost)
            elif pred < dynamic_sell_thresh and self.position >= 0:
                self.execute_trade(-1, current_price, spread_cost)

            # Record portfolio value
            self.record_portfolio_value(current_price)

        return self.get_results()

    def execute_trade(self, direction: int, price: float, spread_cost: float):
        """Execute a trade with dynamic position sizing and risk management."""
        # Calculate recent volatility (last 100 prices)
        recent_vol = np.std(self.current_prices[-100:]) if len(self.current_prices) >= 100 else 0.01  # Default 1% if insufficient data

        # Skip trades in low-volatility conditions
        if recent_vol < 0.005:  # 0.5% minimum daily volatility
            return

        # Trade cooldown (avoid overtrading)
        if len(self.trades) > 0:
            last_trade_step = self.trades[-1]["timestamp"]
            if (len(self.portfolio_values) - last_trade_step) < 5:  # 5-step cooldown
                return

        # Dynamic position sizing (scales with volatility and available capital)
        max_risk_per_trade = 0.02  # Risk 2% of capital per trade
        position_size = min(
            # self.initial_capital * max_risk_per_trade / (recent_vol + 1e-6),  # Volatility scaling
            # self.trade_size * 3,  # Max 3x base size
            # self.capital * 0.9,  # Never use >90% of remaining capital
            self.initial_capital * 0.10,  # Risk up to 10% of capital
            self.trade_size * 20,         # Larger base position
            self.capital * 0.50           # Use up to 50% of remaining capital
        )

        # Adjust for spread cost
        execution_price = price * (1 + direction * spread_cost / 2)
        shares = position_size / execution_price

        # Update position and capital
        self.position += direction * shares
        self.capital -= direction * shares * execution_price

        # Record trade
        self.trades.append({
            "direction": direction,
            "price": execution_price,
            "shares": shares,
            "timestamp": len(self.portfolio_values),
            "volatility": recent_vol,
            "position_size": position_size,
        })

    def record_portfolio_value(self, current_price: float):
        """Track portfolio value and price history."""
        current_price = float(current_price)
        self.current_prices.append(current_price)
        total_value = self.capital + (self.position * current_price)
        self.portfolio_values.append(total_value)

    # def get_results(self) -> Dict:
    #     """Compile backtest results."""
    #     returns = pd.Series(self.portfolio_values).pct_change().dropna()
        
    #     return {
    #         "initial_capital": self.initial_capital,
    #         "final_value": self.portfolio_values[-1],
    #         "total_return": (self.portfolio_values[-1] / self.initial_capital - 1) * 100,
    #         "sharpe_ratio": self.calculate_sharpe(returns),
    #         "max_drawdown": self.calculate_max_drawdown(),
    #         "num_trades": len(self.trades),
    #         "avg_trade_risk": np.mean([trade["position_size"] for trade in self.trades]) if self.trades else 0,
    #         "win_rate": self.calculate_win_rate(),
    #     }

    def get_results(self) -> Dict:
        """Get backtest results."""
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
    
        return {
            'final_value': self.portfolio_values[-1] if self.portfolio_values else self.initial_capital,
            'total_return': (self.portfolio_values[-1] / self.initial_capital - 1) * 100 if self.portfolio_values else 0.0,
            'sharpe_ratio': self.calculate_sharpe(returns),
            'max_drawdown': self.calculate_max_drawdown(),
            'num_trades': len(self.trades),
            'win_rate': self.calculate_win_rate(),
            'portfolio_values': self.portfolio_values,  # THIS IS THE CRITICAL ADDITION
            'trades': self.trades
        }

    def calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Annualized Sharpe ratio with handling for zero volatility."""
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0.0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_max_drawdown(self) -> float:
        """Maximum drawdown as a percentage."""
        peak = self.initial_capital
        max_dd = 0.0
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd * 100

    def calculate_win_rate(self) -> float:
        """Calculate percentage of profitable trades."""
        if not self.trades:
            return 0.0
        profitable = 0
        for i in range(1, len(self.trades)):
            prev_trade = self.trades[i - 1]
            curr_trade = self.trades[i]
            if (curr_trade["price"] - prev_trade["price"]) * prev_trade["direction"] > 0:
                profitable += 1
        return profitable / len(self.trades) * 100