import pkg_resources
import os
import sys

# Add the project root to the Python path
# Removed sys.path modification, '..')))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
import logging

# Try to import MLflow, but make it optional
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    print("MLflow not installed. Tracking features will be disabled.")
    MLFLOW_AVAILABLE = False

# Import our modules with new structure
from dql_trading.utils.data_processing import load_data, split_data, add_indicators, set_seed
from dql_trading.envs.trading_env import ForexTradingEnv, TraderConfig
from dql_trading.agents.dql_agent import DQLAgent
from dql_trading.baseline_strategies.baseline_agents import MovingAverageCrossoverAgent, RSIAgent, BuyAndHoldAgent
from dql_trading.utils.metrics import calculate_trading_metrics, create_performance_dashboard

def sanitize_string_for_mlflow(name):
    """
    Sanitize a string to be compatible with MLflow metric names
    Only allow alphanumerics, underscores, dashes, periods, spaces, colons, and slashes
    """
    # Replace parentheses and commas with underscores
    sanitized = re.sub(r'[^\w\-\.\s:\/]', '_', name)
    return sanitized

def compare_strategies(args):
    """
    Compare different trading strategies on the same dataset
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    # Create results directory
    experiment_name = f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = os.path.join('results', experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and prepare data
    print(f"\n📊 Loading data from {args.data_file}...")
    data_path = pkg_resources.resource_filename("dql_trading", f"data/{args.data_file}")
    df = load_data(data_path, start_date=args.start_date, end_date=args.end_date)
    
    # Add technical indicators
    # Define indicator columns based on what our add_indicators function creates
    indicator_columns = ["SMA_14", "RSI_14", "MACD", "MACD_signal", "BB_upper", "BB_middle", "BB_lower"]
    print(f"Adding technical indicators: {indicator_columns}")
    df = add_indicators(df)
    
    # Split data into train and test sets
    train_df, test_df = split_data(df, train_end=args.train_end, trade_start=args.trade_start, trade_end=args.trade_end)
    
    print(f"Training data: {len(train_df)} rows from {train_df.index.min()} to {train_df.index.max()}")
    print(f"Testing data: {len(test_df)} rows from {test_df.index.min()} to {test_df.index.max()}")
    
    # Create trader config
    trader_config = TraderConfig(
        name="Comparison Strategy",
        risk_tolerance=args.risk_tolerance,
        reward_goal=args.reward_goal,
        max_drawdown=args.max_drawdown,
        target_volatility=args.target_volatility,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        position_sizing=args.position_sizing,
        slippage=args.slippage
    )
    
    # Environment settings
    env_params = {
        "initial_amount": args.initial_amount,
        "transaction_cost_pct": args.transaction_cost,
        "reward_scaling": args.reward_scaling,
        "tech_indicator_list": ["close"] + indicator_columns
    }
    
    # Create the test environment
    test_env = ForexTradingEnv(
        df=test_df,
        trader_config=trader_config,
        **env_params
    )
    
    # Calculate state and action dimensions
    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.n
    
    # Create the strategies to compare
    strategies = {
        "Buy and Hold": BuyAndHoldAgent(state_dim=state_dim, action_dim=action_dim),
        "MA Crossover 10-50": MovingAverageCrossoverAgent(
            state_dim=state_dim, 
            action_dim=action_dim,
            fast_ma=10,
            slow_ma=50
        ),
        "MA Crossover 5-20": MovingAverageCrossoverAgent(
            state_dim=state_dim, 
            action_dim=action_dim,
            fast_ma=5,
            slow_ma=20
        ),
        "RSI 14-30-70": RSIAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            period=14,
            oversold_threshold=30,
            overbought_threshold=70
        )
    }
    
    # Add DQL agent if a model path is provided
    if args.dql_model:
        dql_agent = DQLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=args.gamma,
            lr=args.learning_rate,
            epsilon=0.0,  # No exploration in evaluation
            epsilon_min=0.0,
            epsilon_decay=1.0
        )
        
        # Load pre-trained model
        dql_agent.load_model(args.dql_model)
        strategies["DQL Agent"] = dql_agent
    
    # Start MLflow tracking if available
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(f"strategy_comparison")
        mlflow_run = mlflow.start_run(run_name=experiment_name)
        
        # Log parameters
        mlflow.log_params({
            "data_file": args.data_file,
            "start_date": args.start_date or "all",
            "end_date": args.end_date or "all",
            "initial_amount": args.initial_amount,
            "transaction_cost": args.transaction_cost,
            "indicators": str(indicator_columns)
        })
    
    # Results storage
    results = {}
    
    # Evaluate each strategy
    for name, agent in strategies.items():
        print(f"\nEvaluating {name}...")
        
        # Reset environment
        state = test_env.reset()
        if isinstance(state, np.ndarray) and state.ndim > 1:
            state = state.flatten()
        
        done = False
        
        # Testing loop
        while not done:
            # Select action
            action = agent.select_action(state, test=True)
            
            # Take action
            next_state, reward, done, _ = test_env.step(action)
            
            # Ensure next_state is properly shaped
            if isinstance(next_state, np.ndarray) and next_state.ndim > 1:
                next_state = next_state.flatten()
            
            # Update state
            state = next_state
        
        # Calculate metrics
        account_values = test_env.get_account_value_memory()
        trade_log = test_env.get_trade_log()
        
        metrics = calculate_trading_metrics(
            account_values=account_values,
            trade_log=trade_log
        )
        
        # Store results
        results[name] = {
            'account_values': account_values,
            'trade_log': trade_log,
            'metrics': metrics
        }
        
        # Create and save performance dashboard
        dashboard_path = os.path.join(results_dir, f"{name.replace(' ', '_').lower()}_dashboard.png")
        create_performance_dashboard(
            account_values=account_values,
            trade_log=trade_log,
            title=f"{name} Performance",
            save_path=dashboard_path
        )
        
        # Log metrics to MLflow if available
        if MLFLOW_AVAILABLE and mlflow_run:
            strategy_name = sanitize_string_for_mlflow(name.replace(' ', '_').lower())
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) or (hasattr(value, "item") and callable(getattr(value, "item"))):
                    mlflow_metric_name = f"{strategy_name}_{metric_name}"
                    mlflow.log_metric(mlflow_metric_name, float(value))
            
            # Log dashboard
            mlflow.log_artifact(dashboard_path)
        
        # Print key metrics
        print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.4f}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.4f}")
        print(f"  Total Trades: {metrics.get('total_trades', 0)}")
    
    # Create comparison chart
    fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
    
    for name, result in results.items():
        # Calculate returns from account values
        values = np.array(result['account_values'])
        returns = values / values[0] - 1
        ax.plot(returns * 100, label=name)  # Convert to percentage
    
    ax.set_title('Strategies Comparison')
    ax.set_xlabel('Trading Steps')
    ax.set_ylabel('Return (%)')
    ax.legend()
    ax.grid(True)
    
    # Save comparison chart
    comparison_path = os.path.join(results_dir, "strategies_comparison.png")
    plt.savefig(comparison_path)
    
    if MLFLOW_AVAILABLE and mlflow_run:
        mlflow.log_artifact(comparison_path)
    
    # Create metrics table
    metrics_data = []
    for name, result in results.items():
        metrics = result['metrics']
        metrics_data.append({
            'Strategy': name,
            'Total Return (%)': round(metrics['total_return_pct'], 2),
            'Sharpe Ratio': round(metrics['sharpe_ratio'], 4),
            'Max Drawdown (%)': round(metrics['max_drawdown'] * 100, 2),
            'Win Rate': round(metrics.get('win_rate', 0), 4),
            'Total Trades': metrics.get('total_trades', 0),
            'Profit Factor': round(metrics.get('profit_factor', 0), 4)
        })
    
    # Create DataFrame and save to CSV
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(results_dir, "performance_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    if MLFLOW_AVAILABLE and mlflow_run:
        mlflow.log_artifact(metrics_path)
        mlflow.end_run()
    
    # Print summary
    print("\n" + "=" * 50)
    print("STRATEGY COMPARISON SUMMARY:")
    print("=" * 50)
    print(metrics_df.to_string(index=False))
    print("=" * 50)
    print(f"Results saved to {results_dir}")
    
    return metrics_df

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Compare DQL agent with baseline strategies")
    
    # Data parameters
    parser.add_argument("--data_file", type=str, default="eurusd_all.csv", help="Data file name")
    parser.add_argument("--start_date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--train_end", type=str, default=None, help="End date for training data")
    parser.add_argument("--trade_start", type=str, default=None, help="Start date for testing data")
    parser.add_argument("--trade_end", type=str, default=None, help="End date for testing data")
    parser.add_argument("--indicators", type=str, nargs="+", default=None, help="Technical indicators to add")
    
    # Environment parameters
    parser.add_argument("--initial_amount", type=float, default=100000, help="Initial amount of money")
    parser.add_argument("--transaction_cost", type=float, default=0.0001, help="Transaction cost percentage")
    parser.add_argument("--reward_scaling", type=float, default=1e-4, help="Reward scaling factor")
    
    # Trader config
    parser.add_argument("--risk_tolerance", type=str, default="medium", help="Risk tolerance (low, medium, high)")
    parser.add_argument("--reward_goal", type=str, default="sharpe_ratio", help="Goal for reward calculation")
    parser.add_argument("--max_drawdown", type=float, default=0.1, help="Maximum drawdown tolerance")
    parser.add_argument("--target_volatility", type=float, default=0.02, help="Target volatility")
    parser.add_argument("--stop_loss", type=float, default=0.03, help="Stop loss percentage")
    parser.add_argument("--take_profit", type=float, default=0.05, help="Take profit percentage")
    parser.add_argument("--position_sizing", type=str, default="dynamic", help="Position sizing (fixed, dynamic)")
    parser.add_argument("--slippage", type=float, default=0.0002, help="Slippage")
    
    # DQL parameters
    parser.add_argument("--dql_model", type=str, default=None, help="Path to pretrained DQL model")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    
    return parser.parse_args()

def main(experiments=None, data_file=None, **kwargs):
    """
    Main function that can be imported and called from other modules
    
    Parameters:
    -----------
    experiments : list
        List of experiment names to compare
    data_file : str
        Name of the data file to use
    **kwargs : dict
        Additional arguments
        
    Returns:
    --------
    dict
        Dictionary containing comparison results
    """
    if not experiments:
        print("Error: No experiments provided for comparison")
        return None
        
    if not data_file:
        print("Error: No data file provided")
        return None
        
    # Create args similar to what would be parsed from command line
    class Args:
        pass
    
    args = Args()
    args.experiments = experiments if isinstance(experiments, list) else [experiments]
    args.data_file = data_file
    args.results_dir = kwargs.get('results_dir', 'results')
    args.output_dir = kwargs.get('output_dir', 'results/comparison')
    
    # Run the comparison
    return compare_strategies(args)

if __name__ == "__main__":
    args = parse_args()
    compare_strategies(args) 