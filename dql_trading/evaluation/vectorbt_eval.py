import os
import sys

# Add the project root to the Python path
# Removed sys.path modification, '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Try to import vectorbt
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    print("Warning: vectorbt not installed. Advanced backtesting not available.")

def evaluate_with_vectorbt(trade_log, price_series):
    """
    Evaluate a trading strategy using vectorbt
    
    Parameters:
    -----------
    trade_log : list or pd.DataFrame
        List of trade dictionaries or DataFrame with trade data
    price_series : pd.Series
        Price data series
        
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    # Convert to DataFrame if it's a list
    if isinstance(trade_log, list):
        if not trade_log:
            # Return empty stats if no trades
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "num_trades": 0
            }
            
        trade_log = pd.DataFrame(trade_log)
    
    # Check if trade_log is empty or doesn't have the right columns
    if trade_log.empty:
        # Return empty stats
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "num_trades": 0
        }
    
    # Process trade log - ensure all required columns exist
    required_columns = ['timestamp', 'type', 'price', 'shares']
    
    # Check if we need to create a timestamp column
    if 'timestamp' not in trade_log.columns and 'step' in trade_log.columns:
        # Create timestamp from step if it exists
        if isinstance(price_series.index, pd.DatetimeIndex):
            # Map step to timestamp using price series index
            trade_log['timestamp'] = trade_log['step'].apply(
                lambda x: price_series.index[x] if x < len(price_series.index) else None
            )
        else:
            # If price series doesn't have datetime index, create numeric timestamps
            trade_log['timestamp'] = trade_log['step']
    
    # Drop rows with missing timestamps only if the column exists
    if 'timestamp' in trade_log.columns:
        trade_log = trade_log.dropna(subset=['timestamp'])
    
    # If still empty or missing key columns, return empty stats
    if trade_log.empty or not all(col in trade_log.columns for col in ['type', 'price']):
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "num_trades": 0
        }
    
    # Rest of the function...
    try:
        # Generate portfolio using vectorbt
        # More advanced analysis here
        
        # For now, return simple metrics from trade_log
        num_trades = len(trade_log)
        if num_trades == 0:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "num_trades": 0
            }
        
        # Calculate win rate if 'profit' or 'reward' column exists
        win_rate = 0.0
        if 'profit' in trade_log.columns:
            win_rate = (trade_log['profit'] > 0).mean()
        elif 'reward' in trade_log.columns:
            win_rate = (trade_log['reward'] > 0).mean()
        
        # Simple metrics
        return {
            "total_return": 0.0,  # Placeholder
            "sharpe_ratio": 0.0,  # Placeholder
            "sortino_ratio": 0.0,  # Placeholder
            "max_drawdown": 0.0,  # Placeholder
            "win_rate": win_rate,
            "profit_factor": 1.0,  # Placeholder
            "expectancy": 0.0,  # Placeholder
            "num_trades": num_trades
        }
        
    except Exception as e:
        print(f"Error in vectorbt evaluation: {e}")
        # Return dummy values in case of error
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "num_trades": 0,
            "error": str(e)
        } 