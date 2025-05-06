from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm as tqdm_console
import time
import random
import torch
import ta

class TrainingLogger:
    """
    A comprehensive logger for tracking and visualizing DQL agent training progress
    """
    def __init__(self, log_dir="logs", experiment_name=None, console_level=logging.INFO, 
                 file_level=logging.DEBUG, use_tqdm=True, live_plot=False):
        """
        Initialize the training logger
        
        Parameters:
        -----------
        log_dir : str
            Directory to save log files
        experiment_name : str, optional
            Name of the experiment (defaults to timestamp)
        console_level : int
            Logging level for console output
        file_level : int
            Logging level for file output
        use_tqdm : bool
            Whether to use tqdm progress bars
        live_plot : bool
            Whether to create live plots during training
        """
        # Create experiment name if not provided
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.use_tqdm = use_tqdm
        self.live_plot = live_plot
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Reset handlers to avoid duplication
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = os.path.join(self.log_dir, f"{experiment_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Metrics storage
        self.train_metrics = {
            'episode': [],
            'reward': [],
            'return_pct': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'win_rate': [],
            'trade_count': [],
            'timestamp': []
        }
        
        # Initialize progress bar
        self.pbar = None
        
        # Initialize live plot
        self.fig = None
        self.axes = None
        if self.live_plot:
            self.setup_live_plot()
    
    def log_hyperparameters(self, params):
        """Log hyperparameters at the start of training"""
        self.logger.info("Hyperparameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_system_info(self, env, agent):
        """Log environment and agent information"""
        self.logger.info(f"Environment: State dim={env.observation_space.shape[0]}, Action dim={env.action_space.n}")
        self.logger.info(f"Agent: {agent.__class__.__name__} with gamma={agent.gamma}, epsilon={agent.epsilon}")
    
    def start_training(self, num_episodes):
        """Start training and initialize progress tracking"""
        self.logger.info(f"Starting training for {num_episodes} episodes")
        self.start_time = time.time()
        
        if self.use_tqdm:
            try:
                # Try using notebook version first, fall back to console if it fails
                self.pbar = tqdm_notebook(total=num_episodes, desc="Training")
            except:
                self.pbar = tqdm_console(total=num_episodes, desc="Training")
    
    def log_episode(self, episode, metrics, trade_log=None):
        """
        Log metrics for a single episode
        
        Parameters:
        -----------
        episode : int
            Episode number
        metrics : dict
            Dictionary of metrics for the episode
        trade_log : list, optional
            List of trades made during the episode
        """
        # Log basic metrics
        msg = (f"Episode {episode}: " +
               f"Reward={metrics.get('reward', 0):.2f}, " +
               f"Return={metrics.get('total_return_pct', 0):.2f}%, " +
               f"Sharpe={metrics.get('sharpe_ratio', 0):.4f}, " +
               f"Trades={metrics.get('trade_count', 0)}")
        
        self.logger.info(msg)
        
        # Store metrics
        self.train_metrics['episode'].append(episode)
        self.train_metrics['reward'].append(metrics.get('reward', 0))
        self.train_metrics['return_pct'].append(metrics.get('total_return_pct', 0))
        self.train_metrics['sharpe_ratio'].append(metrics.get('sharpe_ratio', 0))
        self.train_metrics['max_drawdown'].append(metrics.get('max_drawdown', 0))
        self.train_metrics['win_rate'].append(metrics.get('win_rate', 0))
        self.train_metrics['trade_count'].append(metrics.get('trade_count', 0))
        self.train_metrics['timestamp'].append(time.time() - self.start_time)
        
        # Log detailed trade information
        if trade_log and len(trade_log) > 0:
            self.logger.debug(f"Made {len(trade_log)} trades in episode {episode}")
            for i, trade in enumerate(trade_log[-min(len(trade_log), 5):]):  # Log last 5 trades
                self.logger.debug(f"  Trade {i+1}: {trade['type']} at price {trade['price']:.5f}, shares: {trade['shares']}")
        
        # Update progress bar
        if self.pbar:
            self.pbar.update(1)
            self.pbar.set_postfix({
                'reward': f"{metrics.get('reward', 0):.2f}",
                'sharpe': f"{metrics.get('sharpe_ratio', 0):.2f}"
            })
        
        # Update live plot
        if self.live_plot and episode % 5 == 0:  # Update every 5 episodes
            self.update_live_plot()
    
    def log_training_summary(self):
        """Log summary statistics after training is complete"""
        if self.pbar:
            self.pbar.close()
        
        elapsed_time = time.time() - self.start_time
        num_episodes = len(self.train_metrics['episode'])
        
        self.logger.info(f"Training completed: {num_episodes} episodes in {elapsed_time:.2f} seconds")
        
        if num_episodes > 0:
            self.logger.info("Training Summary:")
            self.logger.info(f"  Final Sharpe Ratio: {self.train_metrics['sharpe_ratio'][-1]:.4f}")
            self.logger.info(f"  Best Sharpe Ratio: {max(self.train_metrics['sharpe_ratio']):.4f}")
            self.logger.info(f"  Average Reward: {np.mean(self.train_metrics['reward']):.2f}")
            self.logger.info(f"  Average Return: {np.mean(self.train_metrics['return_pct']):.2f}%")
            self.logger.info(f"  Average Trade Count: {np.mean(self.train_metrics['trade_count']):.1f}")
    
    def start_testing(self):
        """Initialize test logging"""
        self.logger.info("Starting backtest evaluation")
        self.test_start_time = time.time()
    
    def log_test_results(self, metrics, trade_log=None):
        """Log detailed test results"""
        self.logger.info("Test Results:")
        self.logger.info(f"  Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        self.logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        self.logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0) * 100:.2f}%")
        self.logger.info(f"  Win Rate: {metrics.get('win_rate', 0) * 100:.2f}%")
        self.logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
        
        # Log trade summary
        if trade_log:
            buy_trades = [t for t in trade_log if t['type'] == 'BUY']
            sell_trades = [t for t in trade_log if t['type'] == 'SELL']
            self.logger.info(f"  Trades Analysis: {len(buy_trades)} buys, {len(sell_trades)} sells")
            
            # Log profitable vs losing trades
            if 'reward' in trade_log[0]:
                profitable_trades = [t for t in trade_log if t.get('reward', 0) > 0]
                losing_trades = [t for t in trade_log if t.get('reward', 0) < 0]
                self.logger.info(f"  Trade Success: {len(profitable_trades)} profitable, {len(losing_trades)} losing")
    
    def save_metrics(self):
        """Save training metrics to CSV"""
        if len(self.train_metrics['episode']) > 0:
            metrics_df = pd.DataFrame(self.train_metrics)
            metrics_file = os.path.join(self.log_dir, "training_metrics.csv")
            metrics_df.to_csv(metrics_file, index=False)
            self.logger.info(f"Training metrics saved to {metrics_file}")
            return metrics_df
        return None
    
    def setup_live_plot(self):
        """Set up the live plot for training visualization"""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle(f"Training Progress - {self.experiment_name}", fontsize=16)
        
        # Initialize empty plots
        self.lines = {
            'reward': self.axes[0, 0].plot([], [], 'b-', label='Reward')[0],
            'sharpe': self.axes[0, 1].plot([], [], 'r-', label='Sharpe Ratio')[0],
            'return': self.axes[1, 0].plot([], [], 'g-', label='Return %')[0],
            'trades': self.axes[1, 1].plot([], [], 'k-', label='Trade Count')[0]
        }
        
        # Set up axes
        self.axes[0, 0].set_title('Reward')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].set_ylabel('Total Reward')
        self.axes[0, 0].grid(True)
        
        self.axes[0, 1].set_title('Sharpe Ratio')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Sharpe Ratio')
        self.axes[0, 1].grid(True)
        
        self.axes[1, 0].set_title('Return %')
        self.axes[1, 0].set_xlabel('Episode')
        self.axes[1, 0].set_ylabel('Return %')
        self.axes[1, 0].grid(True)
        
        self.axes[1, 1].set_title('Trading Activity')
        self.axes[1, 1].set_xlabel('Episode')
        self.axes[1, 1].set_ylabel('Number of Trades')
        self.axes[1, 1].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)
    
    def update_live_plot(self):
        """Update the live plot with current metrics"""
        if not self.fig or not plt.fignum_exists(self.fig.number):
            # Recreate plot if it was closed
            self.setup_live_plot()
        
        # Update the data
        episodes = self.train_metrics['episode']
        
        self.lines['reward'].set_data(episodes, self.train_metrics['reward'])
        self.lines['sharpe'].set_data(episodes, self.train_metrics['sharpe_ratio'])
        self.lines['return'].set_data(episodes, self.train_metrics['return_pct'])
        self.lines['trades'].set_data(episodes, self.train_metrics['trade_count'])
        
        # Adjust limits
        for ax in self.axes.flat:
            ax.relim()
            ax.autoscale_view()
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def load_data(data_path, start_date=None, end_date=None):
    """
    Load and preprocess the EUR/USD dataset
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file
    start_date : str, optional
        Start date in format 'YYYY-MM-DD'
    end_date : str, optional
        End date in format 'YYYY-MM-DD'
        
    Returns:
    --------
    df : pd.DataFrame
        Processed dataframe
    """
    # Load data
    df = pd.read_csv(
        data_path,
        parse_dates=["date"],
        usecols=["date", "open", "high", "low", "close", "volume", "tic"]
    )
    
    # Remove duplicates
    df.drop_duplicates(subset=['date'], inplace=True)
    
    # Filter by date if provided
    if start_date:
        df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] < end_date]
    
    # Sort and reset index
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"[✅] Loaded {len(df)} rows")
    print(f"Range: {df['date'].min()} → {df['date'].max()}")
    
    return df

def split_data(df, train_start=None, train_end=None, trade_start=None, trade_end=None):
    """
    Split the data into training and trading sets
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with all data
    train_start, train_end, trade_start, trade_end : str
        Date strings in format 'YYYY-MM-DD'
        
    Returns:
    --------
    train_df, trade_df : pd.DataFrame, pd.DataFrame
        Training and trading dataframes
    """
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    
    # Set default values if None is provided
    if train_start is None:
        train_start = df["date"].min()
    else:
        train_start = pd.to_datetime(train_start)
        
    if train_end is None:
        # Use 80% of data for training by default
        train_end_idx = int(len(df) * 0.8)
        train_end = df["date"].iloc[train_end_idx]
    else:
        train_end = pd.to_datetime(train_end)
        
    if trade_start is None:
        trade_start = train_end
    else:
        trade_start = pd.to_datetime(trade_start)
        
    if trade_end is None:
        trade_end = df["date"].max()
    else:
        trade_end = pd.to_datetime(trade_end)
    
    # Split
    train_df = df[(df["date"] >= train_start) & (df["date"] < train_end)].copy()
    trade_df = df[(df["date"] >= trade_start) & (df["date"] < trade_end)].copy()
    
    # Reset index
    train_df = train_df.reset_index(drop=True)
    trade_df = trade_df.reset_index(drop=True)
    
    print(f"Train rows: {len(train_df)}")
    print(f"Trade rows: {len(trade_df)}")
    
    return train_df, trade_df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data
        
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with added indicators
    """
    # Check column names and standardize if needed
    column_map = {
        'Close': 'close',
        'Open': 'open', 
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    }
    
    # Rename columns if needed (case-insensitive)
    df.columns = [col.lower() for col in df.columns]
    
    # Add EMA (Exponential Moving Average)
    df["ema"] = df["close"].ewm(span=10).mean()
    
    # Add RSI (Relative Strength Index)
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    
    # Add MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    
    # Add standard indicators (with standard names)
    df["SMA_14"] = df["close"].rolling(window=14, min_periods=1).mean()
    df["RSI_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    
    # Add Bollinger Bands
    bbands = ta.volatility.BollingerBands(df["close"])
    df["BB_upper"] = bbands.bollinger_hband()
    df["BB_middle"] = bbands.bollinger_mavg()
    df["BB_lower"] = bbands.bollinger_lband()
    
    # Fill NA values
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def visualize_trade_actions(trade_log, account_values, price_data, save_path=None):
    """
    Visualize price data with buy/sell actions
    
    Parameters:
    -----------
    trade_log : list
        List of trade dictionaries
    account_values : list
        List of account values
    price_data : pd.Series or pd.DataFrame
        Price data with timestamps as index
    save_path : str, optional
        Path to save visualization
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with visualization
    """
    if len(trade_log) == 0:
        print("No trades to visualize")
        return None
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price data
    if isinstance(price_data, pd.DataFrame) and 'close' in price_data.columns:
        ax1.plot(price_data.index, price_data['close'], 'b-', label='Close Price')
    else:
        ax1.plot(price_data.index, price_data, 'b-', label='Price')
    
    # Extract buy and sell points
    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []
    
    # Find time index for each trade
    for trade in trade_log:
        step = trade.get('step', 0)
        try:
            time_idx = price_data.index[step]
            price = trade.get('price', 0)
            
            if trade.get('type', '') == 'BUY':
                buy_times.append(time_idx)
                buy_prices.append(price)
            elif trade.get('type', '') == 'SELL':
                sell_times.append(time_idx)
                sell_prices.append(price)
        except IndexError:
            continue
    
    # Plot buy and sell points
    if buy_times:
        ax1.scatter(buy_times, buy_prices, color='green', marker='^', s=100, label='Buy')
    if sell_times:
        ax1.scatter(sell_times, sell_prices, color='red', marker='v', s=100, label='Sell')
    
    # Plot account value
    if account_values and len(account_values) == len(price_data):
        ax2.plot(price_data.index, account_values, 'g-', label='Account Value')
    
    # Set labels and title
    ax1.set_title('Trading Activity')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Account Value')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 

# === NEW FUNCTION ===========================================================
# Added for 65/20/15 (or any) chronological splits that need a dedicated
# validation window in addition to a hold-out test window.

def split_data_three(
    df: pd.DataFrame,
    train_frac: float = 0.65,
    val_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split *df* into train / validation / test.

    Parameters
    ----------
    df : pd.DataFrame
        Full, time-sorted OHLCV dataframe with a **date** column.
    train_frac : float, default 0.65
        Fraction of rows to allocate to the training set.
    val_frac : float, default 0.20
        Fraction of rows to allocate to the validation set.  The remainder
        (1-train_frac-val_frac) becomes the test set.

    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame)
        (train_df, val_df, test_df)
    """

    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be in (0, 1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")

    # Ensure chronological order
    df = df.sort_values("date").reset_index(drop=True)

    n_total = len(df)
    train_end_idx = int(n_total * train_frac)
    val_end_idx = int(n_total * (train_frac + val_frac))

    train_df = df.iloc[:train_end_idx].copy()
    val_df = df.iloc[train_end_idx:val_end_idx].copy()
    test_df = df.iloc[val_end_idx:].copy()

    print(
        f"[🔀] Split: train={len(train_df)} | val={len(val_df)} | test={len(test_df)} (total={n_total})"
    )

    # Reset indices for cleanliness
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

# ========================================================================== 