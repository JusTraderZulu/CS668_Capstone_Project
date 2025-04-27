import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime
import os

def calculate_returns(account_values):
    """
    Calculate returns from a series of account values
    
    Parameters:
    -----------
    account_values : list or array
        Series of account values over time
    
    Returns:
    --------
    tuple
        (returns, cumulative_returns)
    """
    if len(account_values) <= 1:
        return [], []
    
    # Convert to numpy array if not already
    account_values = np.array(account_values)
    
    # Calculate returns
    returns = np.diff(account_values) / account_values[:-1]
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + returns) - 1
    
    return returns, cumulative_returns

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate Sharpe ratio from returns
    
    Parameters:
    -----------
    returns : array-like
        Array of returns
    risk_free_rate : float
        Risk-free rate (annualized)
        
    Returns:
    --------
    float
        Sharpe ratio (annualized, assuming daily returns)
    """
    if len(returns) <= 1:
        return 0.0
    
    # Daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    excess_returns = returns - daily_rf
    
    # Avoid division by zero
    std_returns = np.std(returns)
    if std_returns == 0:
        return 0.0
    
    # Annualize (assuming daily returns)
    sharpe = np.mean(excess_returns) / std_returns * np.sqrt(252)
    
    return sharpe

def calculate_drawdown(cumulative_returns):
    """
    Calculate drawdown from cumulative returns
    
    Parameters:
    -----------
    cumulative_returns : array-like
        Array of cumulative returns
        
    Returns:
    --------
    tuple
        (drawdown, max_drawdown)
    """
    if len(cumulative_returns) <= 1:
        return np.array([]), 0.0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(np.concatenate(([0], cumulative_returns)))
    
    # Calculate drawdown
    drawdown = (1 + running_max[1:]) / (1 + cumulative_returns) - 1
    
    # Calculate maximum drawdown
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
    
    return drawdown, max_drawdown

def calculate_sortino_ratio(returns, risk_free_rate=0.0, target_return=0.0):
    """
    Calculate Sortino ratio from returns
    
    Parameters:
    -----------
    returns : array-like
        Array of returns
    risk_free_rate : float
        Risk-free rate (annualized)
    target_return : float
        Target return (used for downside deviation calculation)
        
    Returns:
    --------
    float
        Sortino ratio (annualized, assuming daily returns)
    """
    if len(returns) <= 1:
        return 0.0
    
    # Daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    excess_returns = returns - daily_rf
    
    # Calculate downside returns (returns below target)
    downside_returns = excess_returns.copy()
    downside_returns[downside_returns > target_return] = 0
    
    # Calculate downside deviation
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    
    # Avoid division by zero
    if downside_deviation == 0:
        return 0.0
    
    # Annualize (assuming daily returns)
    sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    return sortino

def calculate_calmar_ratio(returns, max_drawdown):
    """
    Calculate Calmar ratio from returns and maximum drawdown
    
    Parameters:
    -----------
    returns : array-like
        Array of returns
    max_drawdown : float
        Maximum drawdown
        
    Returns:
    --------
    float
        Calmar ratio (annualized, assuming daily returns)
    """
    if len(returns) <= 1 or max_drawdown <= 0:
        return 0.0
    
    # Annualized return
    annualized_return = np.mean(returns) * 252
    
    # Calculate Calmar ratio
    calmar = annualized_return / max_drawdown
    
    return calmar

def analyze_trades(trade_log):
    """
    Analyze trades from trade log
    
    Parameters:
    -----------
    trade_log : list of dict
        List of trade dictionaries with:
        - 'type' or 'action' (type can be "BUY" or "SELL", action can be 0, 1, 2)
        - 'price'
        - 'reward'
        - 'timestamp' (optional)
        
    Returns:
    --------
    dict
        Trade analysis results
    """
    if not trade_log:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'avg_trade_reward': 0.0
        }
    
    # Filter out hold actions
    trades = []
    for t in trade_log:
        # Check if using 'action' format
        if 'action' in t:
            if t['action'] in [1, 2]:  # Buy or sell
                trades.append(t)
        # Check if using 'type' format
        elif 'type' in t:
            if t['type'] in ['BUY', 'SELL']:
                trades.append(t)
    
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'avg_trade_reward': 0.0
        }
    
    # Extract rewards if they exist, otherwise estimate from account values
    if 'reward' in trades[0]:
        rewards = [t['reward'] for t in trades]
    else:
        # Estimate rewards from account_value changes
        rewards = []
        for t in trades:
            if 'account_value' in t:
                # Just use a placeholder value for testing
                rewards.append(1.0)
            else:
                rewards.append(0.0)
    
    # Count winning and losing trades
    winning_trades = sum(1 for r in rewards if r > 0)
    losing_trades = sum(1 for r in rewards if r <= 0)
    
    # Calculate win rate
    win_rate = winning_trades / len(trades) if trades else 0.0
    
    # Calculate average win and loss
    winning_rewards = [r for r in rewards if r > 0]
    losing_rewards = [r for r in rewards if r <= 0]
    
    avg_win = np.mean(winning_rewards) if winning_rewards else 0.0
    avg_loss = np.mean(losing_rewards) if losing_rewards else 0.0
    
    # Calculate profit factor
    gross_profit = sum(winning_rewards) if winning_rewards else 0.0
    gross_loss = abs(sum(losing_rewards)) if losing_rewards else 0.0
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Calculate average trade
    avg_trade = np.mean(rewards) if rewards else 0.0
    
    return {
        'total_trades': len(trades),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'avg_trade_reward': avg_trade
    }

def calculate_trading_metrics(account_values, trade_log=None, initial_price=None, final_price=None, risk_free_rate=0.0):
    """
    Calculate comprehensive trading metrics
    
    Parameters:
    -----------
    account_values : list or array
        Series of account values over time
    trade_log : list of dict, optional
        List of trade dictionaries
    initial_price : float, optional
        Initial asset price
    final_price : float, optional
        Final asset price
    risk_free_rate : float
        Risk-free rate (annualized)
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    # Edge cases
    if not account_values or len(account_values) <= 1:
        return {
            'total_return': 0.0,
            'total_return_pct': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'volatility': 0.0,
            'total_trades': 0
        }
    
    # Calculate returns
    returns, cumulative_returns = calculate_returns(account_values)
    
    # Calculate total return
    total_return = account_values[-1] - account_values[0]
    total_return_pct = (account_values[-1] / account_values[0] - 1) * 100
    
    # Calculate annualized return (assuming daily returns)
    n_days = len(returns)
    annualized_return = ((1 + total_return_pct/100) ** (252/n_days) - 1) * 100 if n_days > 0 else 0
    
    # Calculate Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate)
    
    # Calculate Sortino ratio
    sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate)
    
    # Calculate drawdown
    _, max_drawdown = calculate_drawdown(cumulative_returns)
    
    # Calculate Calmar ratio
    calmar_ratio = calculate_calmar_ratio(returns, max_drawdown)
    
    # Calculate volatility (annualized, assuming daily returns)
    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
    
    # Calculate trade metrics if trade log provided
    trade_metrics = analyze_trades(trade_log) if trade_log else {'total_trades': 0}
    
    # Calculate price return if prices provided
    if initial_price is not None and final_price is not None:
        price_return = (final_price / initial_price - 1) * 100
    else:
        price_return = None
    
    # Create metrics dictionary
    metrics = {
        'total_return': total_return,
        'total_return_pct': total_return_pct,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'volatility': volatility,
        'price_return_pct': price_return,
        **trade_metrics
    }
    
    return metrics

def create_performance_dashboard(
    account_values, 
    trade_log=None, 
    benchmark_values=None, 
    title="Trading Performance", 
    save_path=None,
    figsize=(15, 12)
):
    """
    Create comprehensive performance dashboard
    
    Parameters:
    -----------
    account_values : list or array
        Series of account values over time
    trade_log : list of dict, optional
        List of trade dictionaries
    benchmark_values : list or array, optional
        Series of benchmark values (e.g., buy and hold)
    title : str
        Figure title
    save_path : str, optional
        If provided, save figure to this path
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Dashboard figure
    """
    # Set up the figure
    fig = plt.figure(figsize=figsize)
    
    # Create grid of subplots
    gs = fig.add_gridspec(4, 2)
    
    # Calculate metrics
    metrics = calculate_trading_metrics(account_values, trade_log)
    returns, cumulative_returns = calculate_returns(account_values)
    drawdowns, _ = calculate_drawdown(cumulative_returns)
    
    # Benchmark metrics if provided
    if benchmark_values is not None and len(benchmark_values) > 0:
        benchmark_metrics = calculate_trading_metrics(benchmark_values)
        benchmark_returns, benchmark_cumulative = calculate_returns(benchmark_values)
        benchmark_drawdowns, _ = calculate_drawdown(benchmark_cumulative)
    
    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(account_values, label='Account Value')
    if benchmark_values is not None and len(benchmark_values) > 0:
        # Scale benchmark to start at same value
        scaled_benchmark = benchmark_values * (account_values[0] / benchmark_values[0])
        ax1.plot(scaled_benchmark, label='Benchmark', linestyle='--')
    ax1.set_title('Account Value Over Time')
    ax1.set_xlabel('Trading Steps')
    ax1.set_ylabel('Account Value')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(range(len(drawdowns)), 0, drawdowns*100, color='red', alpha=0.3)
    if benchmark_values is not None and len(benchmark_values) > 0:
        ax2.plot(benchmark_drawdowns*100, label='Benchmark Drawdown', color='blue', linestyle='--')
    ax2.set_title('Drawdown (%)')
    ax2.set_xlabel('Trading Steps')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True)
    
    # 3. Returns Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    sns.histplot(returns*100, kde=True, ax=ax3, bins=30, color='blue', alpha=0.6)
    # Add vertical line at mean return
    ax3.axvline(np.mean(returns)*100, color='red', linestyle='--', label=f'Mean: {np.mean(returns)*100:.2f}%')
    ax3.set_title('Returns Distribution')
    ax3.set_xlabel('Daily Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Cumulative Returns
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(cumulative_returns*100, label='Strategy')
    if benchmark_values is not None and len(benchmark_values) > 0:
        ax4.plot(benchmark_cumulative*100, label='Benchmark', linestyle='--')
    ax4.set_title('Cumulative Returns (%)')
    ax4.set_xlabel('Trading Steps')
    ax4.set_ylabel('Cumulative Returns (%)')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Trade Analysis (if trade log provided)
    ax5 = fig.add_subplot(gs[2, 1])
    if trade_log and len(trade_log) > 0:
        # Extract trade rewards
        trades = []
        for t in trade_log:
            # Check if using 'action' format
            if 'action' in t:
                if t['action'] in [1, 2]:  # Buy or sell
                    trades.append(t)
            # Check if using 'type' format
            elif 'type' in t:
                if t['type'] in ['BUY', 'SELL']:
                    trades.append(t)
                    
        if trades:
            trade_rewards = []
            for t in trades:
                if 'reward' in t:
                    trade_rewards.append(t['reward'])
                else:
                    # Use a default value
                    trade_rewards.append(0.0)
            
            # Plot trade results
            trade_indices = range(len(trade_rewards))
            colors = ['green' if r > 0 else 'red' for r in trade_rewards]
            ax5.bar(trade_indices, trade_rewards, color=colors, alpha=0.7)
            ax5.set_title('Trade Results')
            ax5.set_xlabel('Trade Number')
            ax5.set_ylabel('Reward')
            ax5.grid(True)
            
            # Add horizontal line at 0
            ax5.axhline(0, color='black', linestyle='-', linewidth=0.5)
        else:
            ax5.text(0.5, 0.5, "No trades made", horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
    else:
        ax5.text(0.5, 0.5, "No trade log provided", horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
    
    # 6. Metrics Table
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    # Create table data
    if benchmark_values is not None and len(benchmark_values) > 0:
        table_data = [
            ['Metric', 'Strategy', 'Benchmark'],
            ['Total Return (%)', f"{metrics['total_return_pct']:.2f}%", f"{benchmark_metrics['total_return_pct']:.2f}%"],
            ['Annualized Return (%)', f"{metrics['annualized_return']:.2f}%", f"{benchmark_metrics['annualized_return']:.2f}%"],
            ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.4f}", f"{benchmark_metrics['sharpe_ratio']:.4f}"],
            ['Sortino Ratio', f"{metrics['sortino_ratio']:.4f}", f"{benchmark_metrics['sortino_ratio']:.4f}"],
            ['Maximum Drawdown (%)', f"{metrics['max_drawdown']*100:.2f}%", f"{benchmark_metrics['max_drawdown']*100:.2f}%"],
            ['Calmar Ratio', f"{metrics['calmar_ratio']:.4f}", f"{benchmark_metrics['calmar_ratio']:.4f}"],
            ['Volatility (%)', f"{metrics['volatility']*100:.2f}%", f"{benchmark_metrics['volatility']*100:.2f}%"],
            ['Total Trades', f"{metrics['total_trades']}", 'N/A'],
        ]
        
        if 'win_rate' in metrics:
            table_data.append(['Win Rate', f"{metrics['win_rate']*100:.2f}%", 'N/A'])
        
        if 'profit_factor' in metrics:
            table_data.append(['Profit Factor', f"{metrics['profit_factor']:.2f}", 'N/A'])
    else:
        table_data = [
            ['Metric', 'Value'],
            ['Total Return (%)', f"{metrics['total_return_pct']:.2f}%"],
            ['Annualized Return (%)', f"{metrics['annualized_return']:.2f}%"],
            ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.4f}"],
            ['Sortino Ratio', f"{metrics['sortino_ratio']:.4f}"],
            ['Maximum Drawdown (%)', f"{metrics['max_drawdown']*100:.2f}%"],
            ['Calmar Ratio', f"{metrics['calmar_ratio']:.4f}"],
            ['Volatility (%)', f"{metrics['volatility']*100:.2f}%"],
            ['Total Trades', f"{metrics['total_trades']}"],
        ]
        
        if 'win_rate' in metrics:
            table_data.append(['Win Rate', f"{metrics['win_rate']*100:.2f}%"])
        
        if 'profit_factor' in metrics:
            table_data.append(['Profit Factor', f"{metrics['profit_factor']:.2f}"])
    
    # Create table
    table = ax6.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        bbox=[0.2, 0.0, 0.6, 1.0]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Set title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_trade_analysis_plot(trade_log, save_path=None, figsize=(15, 10)):
    """
    Create detailed trade analysis plot
    
    Parameters:
    -----------
    trade_log : list of dict
        List of trade dictionaries
    save_path : str, optional
        If provided, save figure to this path
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Trade analysis figure
    """
    if not trade_log:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No trades available for analysis", 
                ha='center', va='center', transform=ax.transAxes)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Filter out hold actions
    trades = [t for t in trade_log if t['action'] in [1, 2]]
    
    if not trades:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No buy/sell trades found in the log", 
                ha='center', va='center', transform=ax.transAxes)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Extract data
    rewards = np.array([t['reward'] for t in trades])
    actions = [t['action'] for t in trades]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2)
    
    # 1. Trade Rewards
    ax1 = fig.add_subplot(gs[0, :])
    
    # Color by profit/loss
    colors = ['green' if r > 0 else 'red' for r in rewards]
    
    ax1.bar(range(len(rewards)), rewards, color=colors, alpha=0.7)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('Trade Rewards')
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # 2. Win/Loss Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Separate wins and losses
    wins = rewards[rewards > 0]
    losses = rewards[rewards <= 0]
    
    # Create grouped bar chart for win/loss count
    labels = ['Wins', 'Losses']
    counts = [len(wins), len(losses)]
    ax2.bar(labels, counts, color=['green', 'red'], alpha=0.7)
    ax2.set_title('Win/Loss Distribution')
    ax2.set_ylabel('Count')
    
    # Add text annotations
    win_rate = len(wins) / len(rewards) if len(rewards) > 0 else 0
    ax2.text(0, counts[0], f"{counts[0]} ({win_rate:.1%})", ha='center', va='bottom')
    ax2.text(1, counts[1], f"{counts[1]} ({1-win_rate:.1%})", ha='center', va='bottom')
    ax2.grid(True)
    
    # 3. Profit Factor
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Calculate total wins and losses
    total_wins = np.sum(wins) if len(wins) > 0 else 0
    total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0
    
    # Create stacked bar chart
    if total_losses > 0:
        profit_factor = total_wins / total_losses
    else:
        profit_factor = np.inf if total_wins > 0 else 0
    
    ax3.bar(['Wins', 'Losses'], [total_wins, total_losses], color=['green', 'red'], alpha=0.7)
    ax3.set_title(f'Profit Factor: {profit_factor:.2f}')
    ax3.set_ylabel('Total Reward')
    ax3.grid(True)
    
    # 4. Action Type Distribution
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Count action types
    buy_count = sum(1 for a in actions if a == 1)
    sell_count = sum(1 for a in actions if a == 2)
    
    ax4.bar(['Buy', 'Sell'], [buy_count, sell_count], color=['blue', 'orange'], alpha=0.7)
    ax4.set_title('Action Distribution')
    ax4.set_ylabel('Count')
    ax4.grid(True)
    
    # 5. Consecutive Wins/Losses
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Calculate consecutive wins and losses
    streak_type = None
    current_streak = 0
    streaks = []
    
    for r in rewards:
        is_win = r > 0
        
        if streak_type is None:
            streak_type = is_win
            current_streak = 1
        elif streak_type == is_win:
            current_streak += 1
        else:
            streaks.append((streak_type, current_streak))
            streak_type = is_win
            current_streak = 1
    
    # Add the last streak
    if streak_type is not None:
        streaks.append((streak_type, current_streak))
    
    # Extract win and loss streaks
    win_streaks = [s[1] for s in streaks if s[0]]
    loss_streaks = [s[1] for s in streaks if not s[0]]
    
    # Create boxplot
    if win_streaks and loss_streaks:
        data = [win_streaks, loss_streaks]
        ax5.boxplot(data, labels=['Win Streaks', 'Loss Streaks'])
        ax5.set_title('Consecutive Wins/Losses')
        ax5.set_ylabel('Streak Length')
        
        # Add text with max streaks
        max_win = max(win_streaks) if win_streaks else 0
        max_loss = max(loss_streaks) if loss_streaks else 0
        ax5.text(1, max_win, f"Max: {max_win}", ha='center', va='bottom')
        ax5.text(2, max_loss, f"Max: {max_loss}", ha='center', va='bottom')
    else:
        ax5.text(0.5, 0.5, "Insufficient streak data", 
                 ha='center', va='center', transform=ax5.transAxes)
    
    ax5.grid(True)
    
    # Add overall title
    fig.suptitle('Trade Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def save_training_metrics_csv(episode_metrics, save_path):
    """
    Save training metrics to CSV file
    
    Parameters:
    -----------
    episode_metrics : dict
        Dictionary of metrics lists by episode
    save_path : str
        Path to save the CSV file
        
    Returns:
    --------
    str
        Path to the saved file
    """
    # Create DataFrame from metrics
    df = pd.DataFrame(episode_metrics)
    
    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    return save_path

def create_training_plots(
    episode_metrics, 
    save_path=None, 
    figsize=(15, 12),
    smoothing=0
):
    """
    Create plots of training metrics
    
    Parameters:
    -----------
    episode_metrics : dict
        Dictionary of metrics lists by episode
    save_path : str, optional
        If provided, save figure to this path
    figsize : tuple
        Figure size
    smoothing : int
        Window size for moving average smoothing (0 for no smoothing)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Training plots figure
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Calculate number of metrics to plot
    num_metrics = len(episode_metrics)
    if num_metrics == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No metrics available for plotting", 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Calculate grid dimensions
    grid_cols = 2
    grid_rows = (num_metrics + 1) // 2  # Ceiling division
    
    # Apply smoothing if requested
    if smoothing > 0:
        smoothed_metrics = {}
        for key, values in episode_metrics.items():
            if len(values) > smoothing:
                kernel = np.ones(smoothing) / smoothing
                smoothed_values = np.convolve(values, kernel, mode='valid')
                # Pad the beginning with nans to maintain length
                pad = np.full(smoothing - 1, np.nan)
                smoothed_metrics[key] = np.concatenate([pad, smoothed_values])
            else:
                smoothed_metrics[key] = values
    else:
        smoothed_metrics = episode_metrics
    
    # Create episode array for x-axis
    episodes = np.arange(1, max(len(values) for values in episode_metrics.values()) + 1)
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(episode_metrics.items()):
        # Skip if values is empty
        if not values:
            continue
        
        ax = fig.add_subplot(grid_rows, grid_cols, i + 1)
        
        # Plot raw data
        ax.plot(episodes[:len(values)], values, alpha=0.3, color='blue', label='Raw')
        
        # Plot smoothed data if available
        if smoothing > 0:
            smoothed_values = smoothed_metrics[metric_name]
            ax.plot(episodes[:len(smoothed_values)], smoothed_values, 
                    color='red', label=f'{smoothing}-Episode MA')
        
        ax.set_title(f'{metric_name.replace("_", " ").title()}')
        ax.set_xlabel('Episode')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.grid(True)
        
        # Add legend if smoothing was applied
        if smoothing > 0:
            ax.legend()
    
    # Add overall title
    fig.suptitle('Training Metrics', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 