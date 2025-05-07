import pkg_resources
import os
import sys

# Add the project root to the Python path
# Removed sys.path modification, '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# Import our modules with new structure
from dql_trading.utils.data_processing import load_data, split_data, visualize_trade_actions
from dql_trading.envs.trading_env import ForexTradingEnv, TraderConfig
from dql_trading.agents.dql_agent import DQLAgent
from dql_trading.utils.metrics import calculate_trading_metrics, create_performance_dashboard
# These need to be implemented in utils/metrics if they are needed
# create_trades_analysis, create_feature_importance_visualization

def evaluate_agent(agent, env, render=False):
    """
    Evaluate a trained agent in the given environment
    
    Parameters:
    -----------
    agent : DQLAgent
        Trained DQL agent
    env : ForexTradingEnv
        Environment to evaluate in
    render : bool
        Whether to render the environment during evaluation
        
    Returns:
    --------
    dict
        Dictionary with evaluation results
    """
    # Initialize metrics
    state = env.reset()
    done = False
    total_reward = 0
    step = 0
    price_history = []
    
    # Run one episode
    while not done:
        # Select action with no exploration (greedy)
        old_epsilon = agent.epsilon
        agent.epsilon = 0  # No exploration during evaluation
        action = agent.select_action(state, test=True)
        agent.epsilon = old_epsilon  # Restore exploration rate
        
        # Take the action
        next_state, reward, done, _ = env.step(action)
        
        # Render if requested
        if render:
            env.render()
        
        # Update state and metrics
        state = next_state
        total_reward += reward
        step += 1
        
        # Track price history
        price_history.append(state[1])  # Price is the second element in state
    
    # Get account value history and trade log
    account_values = env.get_account_value_memory()
    trade_log = env.get_trade_log()
    
    # Calculate comprehensive metrics
    metrics = calculate_trading_metrics(
        account_values=account_values,
        trade_log=trade_log,
        initial_price=price_history[0] if price_history else None,
        final_price=price_history[-1] if price_history else None
    )
    
    return {
        "total_reward": total_reward,
        "account_values": account_values,
        "trade_log": trade_log,
        "price_history": price_history,
        "metrics": metrics,
        "num_steps": step
    }

def create_evaluation_report(eval_results, agent, env, output_dir="results"):
    """
    Create a comprehensive evaluation report with performance metrics and visualizations
    
    Parameters:
    -----------
    eval_results : dict
        Results from evaluate_agent function
    agent : DQLAgent
        Trained agent
    env : ForexTradingEnv
        Environment used for evaluation
    output_dir : str
        Directory to save output files
        
    Returns:
    --------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Print summary metrics to console
    metrics = eval_results['metrics']
    print("\n📊 Evaluation Results:")
    print(f"Total Reward: {eval_results['total_reward']:.2f}")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    
    if 'sortino_ratio' in metrics:
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
    
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    
    if metrics.get('total_trades', 0) > 0 and 'win_rate' in metrics:
        print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        if 'profit_factor' in metrics:
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    if 'buy_hold_return_pct' in metrics:
        print(f"Buy & Hold Return: {metrics['buy_hold_return_pct']:.2f}%")
        print(f"Strategy vs B&H: {metrics['strategy_vs_buy_hold']:.2f}%")
    
    # 2. Create performance dashboard
    try:
        print("\n📈 Creating performance dashboard...")
        dashboard_fig = create_performance_dashboard(
            account_values=eval_results['account_values'],
            trade_log=eval_results['trade_log'],
            title="Performance Dashboard"
        )
        dashboard_path = os.path.join(output_dir, "performance_dashboard.png")
        dashboard_fig.savefig(dashboard_path)
        print(f"Performance dashboard saved to {dashboard_path}")
    except Exception as e:
        print(f"Could not create performance dashboard: {e}")
    
    # 3. Create trades analysis
    try:
        if eval_results.get('trade_log', []):
            print("\n🔍 Creating trades analysis...")
            from dql_trading.utils.metrics import create_trades_analysis
            trades_fig = create_trades_analysis(
                trade_log=eval_results['trade_log'],
                price_history=eval_results.get('price_history', [])
            )
            trades_path = os.path.join(output_dir, "trades_analysis.png")
            trades_fig.savefig(trades_path)
            print(f"Trades analysis saved to {trades_path}")
    except (ImportError, Exception) as e:
        print(f"Could not create trades analysis: {e}")
    
    # 4. Create feature importance visualization
    try:
        print("\n🧩 Creating feature importance visualization...")
        from dql_trading.envs.feature_tracking import create_feature_importance_visualization
        fi_fig = create_feature_importance_visualization(
            agent=agent,
            env=env,
            num_samples=min(1000, len(eval_results.get('price_history', [])))
        )
        fi_path = os.path.join(output_dir, "feature_importance.png")
        fi_fig.savefig(fi_path)
        print(f"Feature importance visualization saved to {fi_path}")
    except (ImportError, Exception) as e:
        print(f"Could not create feature importance visualization: {e}")
    
    # 5. Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(output_dir, "evaluation_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    
    # 6. Save trade log to CSV
    if eval_results.get('trade_log', []):
        trades_df = pd.DataFrame(eval_results['trade_log'])
        trades_path = os.path.join(output_dir, "trades.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"Trade log saved to {trades_path}")

def main(experiment=None, data_file=None):
    """
    Main function that loads a trained agent and evaluates it
    
    Parameters:
    -----------
    experiment : str
        Name of the experiment
    data_file : str
        Name of the data file
    """
    print("=" * 50)
    print(f"📊 DQL Trading Agent Evaluation: {experiment}")
    print("=" * 50)
    
    if not experiment:
        print("\n⚠️ No experiment name provided. Please specify an experiment name.")
        return
    
    if not data_file:
        print("\n⚠️ No data file provided. Please specify a data file.")
        return
    
    # Check if model exists
    model_path = os.path.join("results", experiment, "model.pth")
    if not os.path.exists(model_path):
        print(f"\n⚠️ Model not found at {model_path}. Please train the model first.")
        return
    
    # Load and prepare data
    print("\n📈 Loading and preparing data...")
    data_path = pkg_resources.resource_filename("dql_trading", f"data/{data_file}")
    # For debugging: use a smaller date range
    df = load_data(data_path)
    
    # Split data into train and trade sets
    _, trade_df = split_data(df)
    
    # Create a trading configuration
    trader_config = TraderConfig(
        name=f"{experiment} Evaluation",
        risk_tolerance="medium",
        reward_goal="sharpe_ratio",
        max_drawdown=0.08,
        target_volatility=0.015,
        stop_loss_pct=0.025,
        take_profit_pct=0.04,
        position_sizing="dynamic",
        slippage=0.0001,
        session_time=("00:00", "23:59")  # 24-hour forex trading
    )
    
    # Environment settings
    env_kwargs = {
        "initial_amount": 100000,
        "transaction_cost_pct": 0.0001,
        "reward_scaling": 1e-4,
        "tech_indicator_list": ["close"]
    }
    
    # Create the evaluation environment
    trade_env = ForexTradingEnv(
        df=trade_df,
        trader_config=trader_config,
        **env_kwargs
    )
    
    # Initialize the agent
    state_dim = trade_env.observation_space.shape[0]
    action_dim = trade_env.action_space.n
    
    agent = DQLAgent(
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    # Load trained weights
    print(f"\n🧠 Loading trained model from {model_path}...")
    agent.load_model(model_path)
    
    # Evaluate the agent
    print("\n🧪 Evaluating agent on test data...")
    eval_results = evaluate_agent(agent, trade_env, render=False)
    
    # Create output directory
    output_dir = os.path.join("results", experiment, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Generate comprehensive evaluation report 
        create_evaluation_report(eval_results, agent, trade_env, output_dir=output_dir)
    except Exception as e:
        print(f"\n⚠️ Error generating evaluation report: {e}")
        # Fallback to basic metrics
        print("\n📊 Evaluation Results:")
        metrics = eval_results['metrics']
        print(f"Total Reward: {eval_results['total_reward']:.2f}")
        print(f"Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        if 'win_rate' in metrics:
            print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
    
    return eval_results

def generate_performance_report(experiment_name, results_dir='results'):
    """
    Generate a detailed performance report for a trained agent.
    
    Parameters:
    -----------
    experiment_name : str
        Name of the experiment
    results_dir : str
        Directory containing results
        
    Returns:
    --------
    dict
        Dictionary of metrics and figures
    """
    experiment_dir = os.path.join(results_dir, experiment_name)
    
    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory '{experiment_dir}' not found.")
        return None
    
    # Load test metrics
    test_metrics_path = os.path.join(experiment_dir, 'test_metrics.csv')
    if os.path.exists(test_metrics_path):
        test_metrics = pd.read_csv(test_metrics_path)
        
        # Extract key metrics
        metrics = test_metrics.iloc[0].to_dict() if len(test_metrics) > 0 else {}
    else:
        print(f"Error: Test metrics file not found at {test_metrics_path}")
        metrics = {}
    
    # Load trade log
    trade_log_path = os.path.join(experiment_dir, 'test_trade_log.csv')
    trade_log = []
    
    if os.path.exists(trade_log_path):
        trade_log_df = pd.read_csv(trade_log_path)
        trade_log = trade_log_df.to_dict('records')
    
    # Generate figures
    figures = {}
    
    # 1. Performance Dashboard
    dashboard_path = os.path.join(experiment_dir, 'performance_dashboard.png')
    figures['dashboard'] = dashboard_path if os.path.exists(dashboard_path) else None
    
    # 2. Trade Visualization
    trades_viz_path = os.path.join(experiment_dir, 'test_trades.png')
    figures['trades'] = trades_viz_path if os.path.exists(trades_viz_path) else None
    
    # 3. Trades Analysis
    """
    trades_fig = create_trades_analysis(
        trade_log=trade_log,
        save_path=os.path.join(experiment_dir, 'trade_analysis.png')
    )
    figures['trade_analysis'] = os.path.join(experiment_dir, 'trade_analysis.png')
    """
    
    # 4. Feature Importance
    """
    fi_fig = create_feature_importance_visualization(
        experiment_name=experiment_name,
        save_path=os.path.join(experiment_dir, 'feature_importance.png')
    )
    figures['feature_importance'] = os.path.join(experiment_dir, 'feature_importance.png')
    """
    
    # 5. Comparison with baseline strategies
    baseline_comparison_path = os.path.join(experiment_dir, 'strategies_comparison.png')
    if os.path.exists(baseline_comparison_path):
        figures['baseline_comparison'] = baseline_comparison_path
    
    # Combine everything
    report = {
        'experiment_name': experiment_name,
        'metrics': metrics,
        'figures': figures,
        'trade_log': trade_log
    }
    
    return report

if __name__ == "__main__":
    main() 