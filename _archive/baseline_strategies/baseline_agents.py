import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class BaselineAgent(ABC):
    """
    Abstract base class for baseline trading strategies.
    All baseline strategies should implement this interface.
    """
    def __init__(self, state_dim, action_dim, **kwargs):
        """Initialize the baseline agent"""
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    @abstractmethod
    def select_action(self, state, test=False):
        """Select an action based on the current state"""
        pass
    
    def train(self):
        """Dummy training method (not used for baseline strategies)"""
        return None
    
    def store_transition(self, state, action, reward, next_state, done):
        """Dummy method for storing transitions"""
        pass
    
    def update_target(self):
        """Dummy method for updating target network"""
        pass
    
    def save_model(self, path="baseline_model.json"):
        """Save model parameters (if any)"""
        pass
    
    def load_model(self, path="baseline_model.json"):
        """Load model parameters (if any)"""
        pass


class MovingAverageCrossoverAgent(BaselineAgent):
    """
    Simple Moving Average Crossover strategy
    
    Buy when fast MA crosses above slow MA
    Sell when fast MA crosses below slow MA
    Hold otherwise
    """
    def __init__(self, state_dim, action_dim, fast_ma=10, slow_ma=50, **kwargs):
        """Initialize the Moving Average Crossover agent"""
        super().__init__(state_dim, action_dim)
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.price_history = []
        
    def select_action(self, state, test=False):
        """Select action based on moving average crossover strategy"""
        # Extract price from state
        price = state[1]
        
        # Update price history
        self.price_history.append(price)
        
        # Wait until we have enough data
        if len(self.price_history) <= self.slow_ma:
            return 1  # Hold until we have enough data
        
        # Calculate moving averages
        fast_ma_values = pd.Series(self.price_history).rolling(self.fast_ma).mean().values
        slow_ma_values = pd.Series(self.price_history).rolling(self.slow_ma).mean().values
        
        # Get current and previous values
        current_fast = fast_ma_values[-1]
        current_slow = slow_ma_values[-1]
        prev_fast = fast_ma_values[-2]
        prev_slow = slow_ma_values[-2]
        
        # Check for crossovers
        if np.isnan(current_fast) or np.isnan(current_slow) or np.isnan(prev_fast) or np.isnan(prev_slow):
            return 1  # Hold if we don't have valid values
        
        # Buy signal: Fast MA crosses above Slow MA
        if prev_fast <= prev_slow and current_fast > current_slow:
            return 2  # BUY
        
        # Sell signal: Fast MA crosses below Slow MA
        elif prev_fast >= prev_slow and current_fast < current_slow:
            return 0  # SELL
        
        # No crossover
        else:
            return 1  # HOLD


class RSIAgent(BaselineAgent):
    """
    Relative Strength Index (RSI) trading strategy
    
    Buy when RSI is below oversold_threshold (e.g., 30)
    Sell when RSI is above overbought_threshold (e.g., 70)
    Hold otherwise
    """
    def __init__(self, state_dim, action_dim, period=14, oversold_threshold=30, overbought_threshold=70, **kwargs):
        """Initialize the RSI agent"""
        super().__init__(state_dim, action_dim)
        self.period = period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.price_history = []
        
    def calculate_rsi(self, prices, period=14):
        """Calculate the Relative Strength Index (RSI)"""
        if len(prices) <= period:
            return 50  # Default to neutral RSI if not enough data
        
        # Convert to numpy array
        prices = np.array(prices)
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Calculate gains and losses
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        
        # Initial average gain and loss
        avg_gain = np.sum(gains[:period]) / period
        avg_loss = np.sum(losses[:period]) / period
        
        # Calculate average gain and loss for remaining periods
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        # Calculate RS and RSI
        if avg_loss == 0:
            return 100  # No losses, RSI is 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def select_action(self, state, test=False):
        """Select action based on RSI strategy"""
        # Extract price from state
        price = state[1]
        
        # Update price history
        self.price_history.append(price)
        
        # Wait until we have enough data
        if len(self.price_history) <= self.period:
            return 1  # Hold until we have enough data
        
        # Calculate RSI
        rsi = self.calculate_rsi(self.price_history, self.period)
        
        # Buy signal: RSI below oversold threshold
        if rsi < self.oversold_threshold:
            return 2  # BUY
        
        # Sell signal: RSI above overbought threshold
        elif rsi > self.overbought_threshold:
            return 0  # SELL
        
        # RSI in neutral zone
        else:
            return 1  # HOLD


class BuyAndHoldAgent(BaselineAgent):
    """
    Simple Buy and Hold strategy
    
    Buys at the first opportunity and holds
    """
    def __init__(self, state_dim, action_dim, **kwargs):
        """Initialize the Buy and Hold agent"""
        super().__init__(state_dim, action_dim)
        self.has_bought = False
        
    def select_action(self, state, test=False):
        """Select action based on buy and hold strategy"""
        # Get shares held from state
        shares_held = state[2]
        
        # If we haven't bought yet or don't have shares, buy
        if not self.has_bought or shares_held == 0:
            self.has_bought = True
            return 2  # BUY
        
        # Otherwise, hold
        return 1  # HOLD 