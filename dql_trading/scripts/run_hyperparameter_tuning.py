import pkg_resources
#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for DQL Trading Agent

This script runs hyperparameter tuning for the DQL trading agent using
random search or grid search. It saves the optimal parameters to a JSON file.

Usage:
    python scripts/run_hyperparameter_tuning.py --data_file test_small.csv --search_type random --n_iter 20
"""
import os
import sys
import time

# Add the project root to the Python path
# Removed sys.path modification, '..')))

import argparse
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Import our modules
from dql_trading.utils.data_processing import load_data, split_data, add_indicators
from dql_trading.core.hyperparameter_tuning import HyperparameterTuner

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def run_tuning(args):
    """
    Run hyperparameter tuning with specified options
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    # Create directories if they don't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load and prepare data
    print(f"Loading data from {args.data_file}...")
    data_path = pkg_resources.resource_filename("dql_trading", f"data/{args.data_file}")
    df = load_data(data_path, start_date=args.start_date, end_date=args.end_date)
    
    # Subsample for memory saving if fraction < 1
    if 0 < args.tuning_fraction < 1.0:
        df = df.sample(frac=args.tuning_fraction, random_state=args.seed).sort_index()
        print(f"Subsampled data to {len(df)} rows (fraction={args.tuning_fraction}) for tuning")
    
    # Add indicators
    print("Adding technical indicators...")
    df = add_indicators(df)
    
    # Configure hyperparameter tuning
    print(f"Initializing hyperparameter tuner...")
    tuner = HyperparameterTuner(
        data=df,
        train_test_split=args.train_split,
        results_dir=args.results_dir,
        n_jobs=args.n_jobs,
        random_seed=args.seed,
        agent_type=args.agent_type  # Pass agent_type to tuner
    )
    
    # Define parameter distributions for search
    if args.search_type == 'grid':
        param_grid = {
            # Agent parameters
            'gamma': [0.9, 0.95, 0.99],
            'learning_rate': [0.0001, 0.0005, 0.001],
            'epsilon_decay': [0.99, 0.995, 0.999],
            'batch_size': [32, 64, 128],
            'buffer_size': [10000, 50000],
            
            # Environment parameters
            'env_reward_scaling': [1e-4, 1e-3],
            'env_transaction_cost_pct': [0.0001, 0.0005],
            
            # Training parameters
            'episodes': [args.episodes]
        }
        
        print(f"Running grid search with {len(param_grid)} parameter dimensions...")
        results = tuner.run_grid_search(param_grid, save_results=True)
        
    elif args.search_type == 'random':
        # Define parameter distributions
        param_distributions = {
            # Agent parameters
            'gamma': (0.85, 0.999),  # Range
            'learning_rate': (1e-5, 1e-3, 'log'),  # Log-uniform
            'epsilon_decay': (0.98, 0.999),  # Range
            'batch_size': [32, 64, 128, 256],  # Discrete choices
            'buffer_size': [10000, 20000, 50000, 100000],  # Discrete choices as integers
            
            # Environment parameters
            'env_reward_scaling': (1e-5, 1e-2, 'log'),  # Log-uniform
            'env_transaction_cost_pct': (0.00005, 0.001),  # Range
            
            # Training parameters
            'episodes': args.episodes  # Fixed value
        }
        
        print(f"Running random search with {args.n_iter} iterations...")
        
        # Fix parameter types for random search by preprocessing the param_list
        param_list = []
        for i in range(args.n_iter):
            params = {}
            for param_name, param_dist in param_distributions.items():
                if isinstance(param_dist, list):
                    # Categorical parameter - randomly select from list
                    val = np.random.choice(param_dist)
                    # Convert numpy types to Python types
                    if param_name in ['batch_size', 'buffer_size', 'episodes']:
                        params[param_name] = int(val)
                    else:
                        params[param_name] = float(val)
                        
                elif isinstance(param_dist, tuple) and len(param_dist) == 2:
                    # Uniform continuous distribution
                    low, high = param_dist
                    if param_name in ['batch_size', 'buffer_size', 'episodes']:
                        # Integer parameter
                        params[param_name] = int(np.random.randint(low, high+1))
                    else:
                        # Float parameter
                        params[param_name] = float(np.random.uniform(low, high))
                        
                elif isinstance(param_dist, tuple) and len(param_dist) == 3:
                    # Log-uniform distribution (min, max, 'log')
                    low, high, dist_type = param_dist
                    if dist_type == 'log':
                        # Sample log-uniformly
                        log_val = np.random.uniform(np.log10(low), np.log10(high))
                        params[param_name] = float(10 ** log_val)
                else:
                    # Use the value directly
                    if param_name in ['batch_size', 'buffer_size', 'episodes']:
                        params[param_name] = int(param_dist)
                    else:
                        params[param_name] = float(param_dist) if isinstance(param_dist, (int, float)) else param_dist
            
            # Double-check types for critical parameters
            for key in ['batch_size', 'buffer_size', 'episodes']:
                if key in params:
                    params[key] = int(params[key])
            
            for key in ['gamma', 'learning_rate', 'epsilon_decay', 'env_reward_scaling', 'env_transaction_cost_pct']:
                if key in params:
                    params[key] = float(params[key])
                    
            param_list.append(params)
        
        # Use the custom param_list with the run_random_search method
        print(f"Generated {len(param_list)} parameter combinations with correct types.")
        
        # Sequential evaluation
        results = []
        print(f"Starting hyperparameter evaluation with {len(param_list)} trials...")
        for i, params in enumerate(param_list):
            print(f"\n[Trial {i+1}/{len(param_list)}] Evaluating parameters: {params}")
            trial_start = time.time()
            
            # Pass the verbose flag to _evaluate_params by adding it to params
            if args.verbose:
                params['verbose'] = True
            
            result = tuner._evaluate_params(params)
            trial_time = time.time() - trial_start
            
            # Extract key metrics for summary
            sharpe = result.get('sharpe_ratio', 0) or 0
            returns = result.get('total_return_pct', 0) or 0
            trades = result.get('total_trades', 0) or 0
            
            print(f"[Trial {i+1}/{len(param_list)}] Completed in {trial_time:.1f}s - Results: Sharpe={sharpe:.4f}, Return={returns:.2f}%, Trades={trades}")
            
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(args.results_dir, f"random_search_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        
        print(f"Results saved to {results_path}")
    
    else:
        raise ValueError(f"Unknown search type: {args.search_type}")
    
    # Ensure results_path is defined even if grid search branch is used
    if 'results_path' not in locals():
        results_path = None

    # Visualize results
    print("Visualizing results...")
    try:
        tuner.visualize_results(results_df, top_n=args.top_n, save_plots=True)
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")
    
    # Get best parameters and save them
    best_params = tuner.get_best_params(results_df, metric=args.metric, min_trades=args.min_trades)
    
    # Clean up best parameters for saving
    clean_params = {k: v for k, v in best_params.items() if k != 'model_id'}
    
    # Convert numpy types to Python types
    for key in clean_params:
        if isinstance(clean_params[key], (np.integer, np.int64, np.int32)):
            clean_params[key] = int(clean_params[key])
        elif isinstance(clean_params[key], (np.floating, np.float64, np.float32)):
            clean_params[key] = float(clean_params[key])
    
    # Rename environment parameters
    env_params = {}
    for k in list(clean_params.keys()):
        if k.startswith('env_'):
            env_params[k[4:]] = clean_params.pop(k)
    
    # Create final parameters dictionary
    final_params = {
        'agent_params': clean_params,
        'env_params': env_params,
        'tuning_info': {
            'search_type': args.search_type,
            'metric_optimized': args.metric,
            'date_tuned': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_file': args.data_file,
            'train_split': args.train_split,
            'agent_type': args.agent_type  # Include agent type in the saved parameters
        }
    }
    
    # Save optimal parameters
    params_path = os.path.join(args.results_dir, 'optimal_parameters.json')
    with open(params_path, 'w') as f:
        json.dump(final_params, f, indent=4, cls=NumpyEncoder)
    
    print(f"Optimal parameters saved to {params_path}")
    print("\nBest parameters:")
    for k, v in final_params['agent_params'].items():
        print(f"  {k}: {v}")
    for k, v in final_params['env_params'].items():
        print(f"  {k}: {v}")
    
    # ------------------------------------------------------------------
    # Telegram notification with summary
    # ------------------------------------------------------------------
    if getattr(args, "notify", False):
        try:
            from dql_trading.utils.notifications import send_telegram_message, send_telegram_document

            # Determine the best metric value directly from results_df because
            # best_params intentionally excludes metric columns. Handle
            # max_drawdown (lower is better) separately.
            try:
                if args.metric == "max_drawdown":
                    best_val = results_df[args.metric].min()
                else:
                    best_val = results_df[args.metric].max()
                # Format to 4 decimals if numeric
                if isinstance(best_val, (int, float)):
                    best_val = f"{best_val:.4f}"
            except Exception:
                best_val = "n/a"

            msg = (
                f"✅ Hyperparameter tuning finished for *{args.data_file}*\n"
                f"*Best {args.metric}:* `{best_val}`"
            )
            send_telegram_message(msg)

            # Send results CSV if it exists
            if results_path and os.path.exists(results_path):
                send_telegram_document(results_path, caption="Tuning results CSV")
        except Exception as e:
            print(f"Telegram notification failed: {e}")
    
    return final_params

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for the DQL trading agent")
    
    # Data parameters
    parser.add_argument("--data_file", type=str, default="test_small.csv", help="Data file name")
    parser.add_argument("--start_date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train/test split ratio")
    
    # Tuning parameters
    parser.add_argument("--search_type", type=str, choices=['grid', 'random'], default='random',
                        help="Type of hyperparameter search to perform")
    parser.add_argument("--n_iter", type=int, default=20, help="Number of iterations for random search")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes for each evaluation")
    parser.add_argument("--tuning_fraction", type=float, default=1.0, help="Fraction of data to sample for tuning (0-1)")
    parser.add_argument("--metric", type=str, default="sharpe_ratio", 
                        choices=["sharpe_ratio", "total_return_pct", "sortino_ratio", "calmar_ratio"],
                        help="Metric to optimize")
    parser.add_argument("--min_trades", type=int, default=1,
                        help="Minimum number of trades a model must execute to be considered optimal")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top models to highlight")
    parser.add_argument("--notify", action="store_true", help="Send Telegram notification when completed")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress within each trial")
    
    # Agent parameters
    parser.add_argument("--agent_type", type=str, default="dql", help="Type of agent to use")
    
    # Output parameters
    parser.add_argument("--results_dir", type=str, default="results/hyperparameter_tuning",
                        help="Directory to save results")
    
    return parser.parse_args()

def main(data_file=None, search_type="random", n_iter=20, notify=False, agent_type="dql", **kwargs):
    """
    Main function that can be imported and called from other modules
    
    Parameters:
    -----------
    data_file : str
        Name of the data file to use
    search_type : str
        Type of search to perform ("random" or "grid")
    n_iter : int
        Number of iterations for random search
    notify : bool
        Send Telegram notification when completed
    agent_type : str
        Type of agent to use
    **kwargs : dict
        Additional arguments to pass to the hyperparameter tuning
        
    Returns:
    --------
    dict
        Dictionary containing the optimal parameters found
    """
    # Create args similar to what would be parsed from command line
    class Args:
        pass
    
    args = Args()
    args.data_file = data_file
    args.search_type = search_type
    args.n_iter = n_iter
    args.episodes = kwargs.get('episodes', 20)
    args.metric = kwargs.get('metric', 'sharpe_ratio')
    args.start_date = kwargs.get('start_date', None)
    args.end_date = kwargs.get('end_date', None)
    args.results_dir = kwargs.get('results_dir', 'results/hyperparameter_tuning')
    args.train_split = kwargs.get('train_split', 0.8)
    args.n_jobs = kwargs.get('n_jobs', 1)
    args.seed = kwargs.get('seed', 42)
    args.top_n = kwargs.get('top_n', 5)
    args.notify = notify
    args.verbose = kwargs.get('verbose', False)
    args.tuning_fraction = kwargs.get('tuning_fraction', 1.0)
    args.min_trades = kwargs.get('min_trades', 1)
    args.agent_type = agent_type  # Pass agent_type from arguments
    
    # Run hyperparameter tuning
    return run_tuning(args)

if __name__ == "__main__":
    args = parse_args()
    run_tuning(args) 