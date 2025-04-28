import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

def evaluate_with_vectorbt(trade_log, price_series: pd.Series):
    """
    Evaluate a trading strategy using vectorbt
    
    Parameters:
    -----------
    trade_log : list of dict
        List of trades with action and timestamp
    price_series : pandas.Series
        Price series with timestamps as index
        
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    if not VECTORBT_AVAILABLE:
        print("VectorBT not available. Returning empty metrics.")
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "drawdown": 0.0,
            "win_rate": 0.0
        }
    
    # Handle different trade log formats
    if isinstance(trade_log, list):
        # Convert list of dictionaries to DataFrame
        trade_log = pd.DataFrame(trade_log)
    
    # Check if we have 'type' field instead of 'side'
    if 'type' in trade_log.columns and 'side' not in trade_log.columns:
        trade_log['side'] = trade_log['type'].str.lower()
    
    # Create or convert timestamp column
    if 'timestamp' not in trade_log.columns and 'step' in trade_log.columns:
        # Map step to timestamp using price_series index
        trade_log['timestamp'] = trade_log['step'].apply(
            lambda x: price_series.index[x] if x < len(price_series) else None
        )
    
    # Drop rows with None timestamps
    trade_log = trade_log.dropna(subset=['timestamp'])
    
    # Create entry/exit signals - adjust for "BUY"/"SELL" instead of "buy"/"sell" 
    entries = trade_log[trade_log.side.str.lower() == "buy"].set_index("timestamp")
    entries = entries.reindex(price_series.index, fill_value=False)
    entries = entries['side'].astype(bool)
    
    exits = trade_log[trade_log.side.str.lower() == "sell"].set_index("timestamp") 
    exits = exits.reindex(price_series.index, fill_value=False)
    exits = exits['side'].astype(bool)

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    try:
        pf = vbt.Portfolio.from_signals(price_series, entries, exits, freq="1min")
        stats = pf.stats().to_dict()
        
        # Save plots
        fig = pf.plot()
        fig.write_image("results/equity_curve.png")
        return stats
    except Exception as e:
        print(f"Error in VectorBT evaluation: {e}")
        # Return dummy stats in case of error
        return {
            "total_return": 0.0,
            "total_trades": len(trade_log),
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "error_msg": str(e)
        } 