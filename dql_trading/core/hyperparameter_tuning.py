import pkg_resources
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
import time
import json
import uuid
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Import our modules
from dql_trading.utils.data_processing import load_data, split_data, add_indicators
from dql_trading.envs.trading_env import ForexTradingEnv, TraderConfig
from dql_trading.agents.dql_agent import DQLAgent
from dql_trading.utils.metrics import calculate_trading_metrics, create_performance_dashboard
from dql_trading.utils.visualization import (
    plot_hyperparameter_results, 
    plot_top_models_comparison,
    create_learning_curve_comparison,
    plot_parameter_importance
)

class HyperparameterTuner:
    """
    Class for hyperparameter tuning of DQN agents
    """
    def __init__(self, 
                 data,
                 train_test_split=0.8,
                 results_dir='results/hyperparameter_tuning',
                 n_jobs=None,
                 random_seed=42):
        """
        Initialize the hyperparameter tuner
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing OHLCV data
        train_test_split : float
            Ratio of data to use for training vs testing
        results_dir : str
            Directory to save results
        n_jobs : int, optional
            Number of parallel jobs to run. If None, uses all available CPUs - 1
        random_seed : int
            Random seed for reproducibility
        """
        self.data = data
        self.train_test_split = train_test_split
        self.results_dir = results_dir
        self.random_seed = random_seed
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Split data into train and test sets
        split_idx = int(len(data) * train_test_split)
        self.train_data = data.iloc[:split_idx].copy()
        self.test_data = data.iloc[split_idx:].copy()
        
        # Set number of parallel jobs
        if n_jobs is None:
            self.n_jobs = max(1, cpu_count() - 1)  # Leave one CPU free
        else:
            self.n_jobs = n_jobs
            
        # Initialize results storage
        self.results = []
        self.episode_data = {}
        
    def generate_parameter_grid(self, param_grid):
        """
        Generate all combinations of parameters from the grid
        
        Parameters:
        -----------
        param_grid : dict
            Dictionary of parameter names and their possible values
            
        Returns:
        --------
        list
            List of parameter dictionaries, one for each combination
        """
        keys = param_grid.keys()
        values = param_grid.values()
        
        # Generate all combinations
        combinations = list(itertools.product(*values))
        
        # Create a list of parameter dictionaries
        param_list = [dict(zip(keys, combo)) for combo in combinations]
        
        return param_list
    
    def _evaluate_params(self, params):
        """
        Evaluate a set of hyperparameters
        
        Parameters:
        -----------
        params : dict
            Dictionary of hyperparameters
            
        Returns:
        --------
        dict
            Dictionary containing evaluation results
        """
        # Generate a unique model ID
        model_id = str(uuid.uuid4())[:8]
        
        # Extract agent parameters and environment parameters
        agent_params = {k: v for k, v in params.items() if not k.startswith('env_') and k != 'episodes'}
        env_params = {k[4:]: v for k, v in params.items() if k.startswith('env_')}
        
        # Get number of episodes
        num_episodes = params.get('episodes', 100)
        
        # Handle specific parameter name conversions
        if 'learning_rate' in agent_params:
            agent_params['lr'] = agent_params.pop('learning_rate')
        if 'memory_size' in agent_params:
            agent_params['buffer_size'] = agent_params.pop('memory_size')
        
        # Set random seed
        np.random.seed(self.random_seed)
        
        # Create environment
        env = ForexTradingEnv(
            df=self.train_data,
            trader_config=TraderConfig(
                name="Default Trader",
                risk_tolerance="medium",
                reward_goal="sharpe_ratio",
                max_drawdown=0.1,
                target_volatility=0.02,
                stop_loss_pct=0.03,
                take_profit_pct=0.05,
                position_sizing="dynamic",
                slippage=0.0002
            ),
            **env_params
        )
        
        # Create agent
        agent = DQLAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            **agent_params
        )
        
        # Train agent
        episode_rewards = []
        episode_sharpe_ratios = []
        episode_returns = []
        episode_drawdowns = []
        
        for episode in range(num_episodes):
            state = env.reset()
            state = np.reshape(state, [1, env.observation_space.shape[0]])
            
            done = False
            total_reward = 0
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
                
                agent.store_transition(state, action, reward, next_state, done)
                agent.train()
                state = next_state
                total_reward += reward
                
                if done:
                    # Calculate metrics for this episode
                    account_values = env.get_account_value_memory()
                    trade_log = env.get_trade_log()
                    
                    # Safely extract price information
                    initial_price = None
                    final_price = None
                    
                    try:
                        if isinstance(state, np.ndarray) and state.size > 1:
                            if state.ndim > 1 and state.shape[1] > 1:
                                final_price = state[0, 1]
                            else:
                                final_price = state[1]
                        elif isinstance(state, list) and len(state) > 1:
                            final_price = state[1]
                    except (IndexError, TypeError):
                        final_price = None
                        
                    metrics = calculate_trading_metrics(
                        account_values=account_values,
                        trade_log=trade_log,
                        initial_price=None,
                        final_price=final_price
                    )
                    
                    # Store metrics
                    episode_rewards.append(total_reward)
                    episode_sharpe_ratios.append(metrics['sharpe_ratio'])
                    episode_returns.append(metrics['total_return_pct'])
                    episode_drawdowns.append(metrics['max_drawdown'])
                
            # Train the agent after each episode
            if len(agent.memory) > params.get('batch_size', 32):
                agent.update_target()
        
        # Store episode data
        self.episode_data[model_id] = {
            'episode_reward': episode_rewards,
            'sharpe_ratio': episode_sharpe_ratios,
            'total_return_pct': episode_returns,
            'max_drawdown': episode_drawdowns
        }
        
        # Evaluate on test data
        trade_env = ForexTradingEnv(
            df=self.test_data,
            trader_config=TraderConfig(
                name="Default Trader",
                risk_tolerance="medium",
                reward_goal="sharpe_ratio",
                max_drawdown=0.1,
                target_volatility=0.02,
                stop_loss_pct=0.03,
                take_profit_pct=0.05,
                position_sizing="dynamic",
                slippage=0.0002
            ),
            **env_params
        )
        
        state = trade_env.reset()
        state = np.reshape(state, [1, trade_env.observation_space.shape[0]])
        
        done = False
        
        while not done:
            action = agent.select_action(state, test=True)  # Use epsilon=0 for testing
            next_state, reward, done, _ = trade_env.step(action)
            state = np.reshape(next_state, [1, trade_env.observation_space.shape[0]])
        
        # Calculate final metrics
        account_values = trade_env.get_account_value_memory()
        trade_log = trade_env.get_trade_log()
        
        # Safely extract initial price and final price
        initial_price = None
        final_price = None
        
        # Try to extract price information safely
        try:
            if isinstance(state, np.ndarray) and state.ndim > 0 and state.size > 1:
                initial_price = state[1]
            elif isinstance(state, list) and len(state) > 1:
                initial_price = state[1]
        except (IndexError, TypeError):
            initial_price = None
            
        metrics = calculate_trading_metrics(
            account_values=account_values,
            trade_log=trade_log,
            initial_price=initial_price,
            final_price=final_price
        )
        
        # Add parameters to results
        result = {**params, **metrics, 'model_id': model_id}
        
        return result
    
    def run_grid_search(self, param_grid, save_results=True):
        """
        Run grid search over hyperparameter combinations
        
        Parameters:
        -----------
        param_grid : dict
            Dictionary of parameter names and their possible values
        save_results : bool
            Whether to save results to disk
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing results for all parameter combinations
        """
        # Generate parameter combinations
        param_list = self.generate_parameter_grid(param_grid)
        
        print(f"Running grid search with {len(param_list)} parameter combinations")
        print(f"Using {self.n_jobs} parallel workers")
        
        start_time = time.time()
        
        # Run in parallel if n_jobs > 1
        if self.n_jobs > 1 and len(param_list) > 1:
            with Pool(self.n_jobs) as pool:
                results = list(tqdm(pool.imap(self._evaluate_params, param_list), total=len(param_list)))
        else:
            # Run sequentially
            results = []
            for params in tqdm(param_list):
                results.append(self._evaluate_params(params))
        
        # Store results
        self.results = results
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.results_dir, f"grid_search_results_{timestamp}.csv")
            results_df.to_csv(results_path, index=False)
            
            # Save episode data
            episode_data_path = os.path.join(self.results_dir, f"episode_data_{timestamp}.json")
            with open(episode_data_path, 'w') as f:
                json.dump(self.episode_data, f)
            
            print(f"Results saved to {results_path}")
            print(f"Episode data saved to {episode_data_path}")
            
        elapsed_time = time.time() - start_time
        print(f"Grid search completed in {elapsed_time:.2f} seconds")
        
        return results_df
    
    def run_random_search(self, param_distributions, n_iter=10, save_results=True):
        """
        Run random search over hyperparameter space
        
        Parameters:
        -----------
        param_distributions : dict
            Dictionary of parameter names and their possible distributions
        n_iter : int
            Number of random combinations to try
        save_results : bool
            Whether to save results to disk
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing results for all sampled parameter combinations
        """
        # Set random seed
        np.random.seed(self.random_seed)
        
        # Generate random parameter combinations
        param_list = []
        
        for i in range(n_iter):
            params = {}
            for param_name, param_dist in param_distributions.items():
                if isinstance(param_dist, list):
                    # Categorical parameter - randomly select from list
                    params[param_name] = np.random.choice(param_dist)
                elif isinstance(param_dist, tuple) and len(param_dist) == 2:
                    # Uniform continuous distribution
                    low, high = param_dist
                    if isinstance(low, int) and isinstance(high, int):
                        # Integer parameter
                        params[param_name] = np.random.randint(low, high+1)
                    else:
                        # Float parameter
                        params[param_name] = np.random.uniform(low, high)
                elif isinstance(param_dist, tuple) and len(param_dist) == 3:
                    # Log-uniform distribution (min, max, 'log')
                    low, high, dist_type = param_dist
                    if dist_type == 'log':
                        # Sample log-uniformly
                        log_val = np.random.uniform(np.log10(low), np.log10(high))
                        params[param_name] = 10 ** log_val
                else:
                    # Use the value directly
                    params[param_name] = param_dist
            
            param_list.append(params)
        
        print(f"Running random search with {n_iter} parameter combinations")
        print(f"Using {self.n_jobs} parallel workers")
        
        start_time = time.time()
        
        # Run in parallel if n_jobs > 1
        if self.n_jobs > 1 and len(param_list) > 1:
            with Pool(self.n_jobs) as pool:
                results = list(tqdm(pool.imap(self._evaluate_params, param_list), total=len(param_list)))
        else:
            # Run sequentially
            results = []
            for params in tqdm(param_list):
                results.append(self._evaluate_params(params))
        
        # Store results
        self.results = results
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.results_dir, f"random_search_results_{timestamp}.csv")
            results_df.to_csv(results_path, index=False)
            
            # Save episode data
            episode_data_path = os.path.join(self.results_dir, f"episode_data_{timestamp}.json")
            with open(episode_data_path, 'w') as f:
                json.dump(self.episode_data, f)
            
            print(f"Results saved to {results_path}")
            print(f"Episode data saved to {episode_data_path}")
            
        elapsed_time = time.time() - start_time
        print(f"Random search completed in {elapsed_time:.2f} seconds")
        
        return results_df
    
    def visualize_results(self, results_df=None, top_n=5, save_plots=True):
        """
        Visualize hyperparameter tuning results
        
        Parameters:
        -----------
        results_df : pandas.DataFrame, optional
            DataFrame containing results. If None, uses stored results.
        top_n : int
            Number of top models to highlight
        save_plots : bool
            Whether to save plots to disk
            
        Returns:
        --------
        dict
            Dictionary containing plot figures
        """
        if results_df is None:
            if not self.results:
                raise ValueError("No results available. Run grid_search or random_search first.")
            results_df = pd.DataFrame(self.results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plots = {}
        
        # 1. Hyperparameter results overview
        save_path = os.path.join(plots_dir, f"hyperparameter_results_{timestamp}.png") if save_plots else None
        plots['overview'] = plot_hyperparameter_results(results_df, top_n=top_n, save_path=save_path)
        
        # 2. Top models comparison
        save_path = os.path.join(plots_dir, f"top_models_comparison_{timestamp}.png") if save_plots else None
        plots['top_models'] = plot_top_models_comparison(results_df, top_n=top_n, save_path=save_path)
        
        # 3. Parameter importance
        save_path = os.path.join(plots_dir, f"parameter_importance_{timestamp}.png") if save_plots else None
        plots['importance'] = plot_parameter_importance(results_df, save_path=save_path)
        
        # 4. Learning curves for top models
        if self.episode_data:
            # Get top model IDs
            top_models = results_df.sort_values('sharpe_ratio', ascending=False).head(top_n)
            model_ids = top_models['model_id'].tolist()
            
            # Get episode data for top models
            episode_data_list = [self.episode_data.get(model_id, {}) for model_id in model_ids]
            
            # Create learning curve comparison
            save_path = os.path.join(plots_dir, f"learning_curves_{timestamp}.png") if save_plots else None
            plots['learning_curves'] = create_learning_curve_comparison(
                episode_data_list, model_ids, 
                metrics=['episode_reward', 'sharpe_ratio', 'total_return_pct', 'max_drawdown'],
                save_path=save_path
            )
        
        return plots
    
    def get_best_params(self, results_df=None, metric='sharpe_ratio'):
        """
        Get the best hyperparameters based on a metric
        
        Parameters:
        -----------
        results_df : pandas.DataFrame, optional
            DataFrame containing results. If None, uses stored results.
        metric : str
            Metric to use for ranking models
            
        Returns:
        --------
        dict
            Dictionary containing best hyperparameters
        """
        if results_df is None:
            if not self.results:
                raise ValueError("No results available. Run grid_search or random_search first.")
            results_df = pd.DataFrame(self.results)
        
        # Find best model
        if metric == 'max_drawdown':
            # For drawdown, lower is better
            best_idx = results_df[metric].idxmin()
        else:
            # For other metrics, higher is better
            best_idx = results_df[metric].idxmax()
        
        best_model = results_df.iloc[best_idx]
        
        # Get parameter columns (exclude metrics)
        metric_cols = ['sharpe_ratio', 'total_return_pct', 'max_drawdown', 'win_rate', 
                      'sortino_ratio', 'calmar_ratio', 'volatility', 'model_id']
        param_cols = [col for col in results_df.columns if col not in metric_cols]
        
        # Extract best parameters
        best_params = {param: best_model[param] for param in param_cols}
        
        return best_params
    
    def save_best_model(self, results_df=None, metric='sharpe_ratio'):
        """
        Train and save the best model based on hyperparameter search
        
        Parameters:
        -----------
        results_df : pandas.DataFrame, optional
            DataFrame containing results. If None, uses stored results.
        metric : str
            Metric to use for ranking models
            
        Returns:
        --------
        tuple
            Tuple containing (best_params, model_path)
        """
        best_params = self.get_best_params(results_df, metric)
        
        # Extract agent parameters and environment parameters
        agent_params = {k: v for k, v in best_params.items() if not k.startswith('env_')}
        env_params = {k[4:]: v for k, v in best_params.items() if k.startswith('env_')}
        
        # Set random seed
        np.random.seed(self.random_seed)
        
        # Create environment with all data
        env = ForexTradingEnv(
            df=self.data,
            trader_config=TraderConfig(
                name="Default Trader",
                risk_tolerance="medium",
                reward_goal="sharpe_ratio",
                max_drawdown=0.1,
                target_volatility=0.02,
                stop_loss_pct=0.03,
                take_profit_pct=0.05,
                position_sizing="dynamic",
                slippage=0.0002
            ),
            **env_params
        )
        
        # Create agent
        agent = DQLAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            **agent_params
        )
        
        # Train agent
        episodes = best_params.get('episodes', 100)
        batch_size = best_params.get('batch_size', 32)
        
        print(f"Training best model for {episodes} episodes")
        
        for episode in tqdm(range(episodes)):
            state = env.reset()
            state = np.reshape(state, [1, env.observation_space.shape[0]])
            
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
                
                agent.store_transition(state, action, reward, next_state, done)
                agent.train()
                state = next_state
                
                if done:
                    break
                
            # Train the agent after each episode
            if len(agent.memory) > batch_size:
                agent.update_target()
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.results_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"best_model_{timestamp}")
        agent.save_model(model_path)
        
        # Save best parameters
        params_path = os.path.join(model_dir, f"best_params_{timestamp}.json")
        with open(params_path, 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            best_params_json = {}
            for k, v in best_params.items():
                if isinstance(v, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                    best_params_json[k] = int(v)
                elif isinstance(v, (np.float_, np.float16, np.float32, np.float64)):
                    best_params_json[k] = float(v)
                else:
                    best_params_json[k] = v
                    
            json.dump(best_params_json, f, indent=2)
        
        print(f"Best model saved to {model_path}")
        print(f"Best parameters saved to {params_path}")
        
        return best_params, model_path

def main():
    print("=" * 50)
    print("🧪 Hyperparameter Tuning for DQL Trading Agent")
    print("=" * 50)
    
    # Load and prepare data
    print("\n📊 Loading and preparing data...")
    data_path = pkg_resources.resource_filename("dql_trading", f"data/{args.data_file}")
    df = load_data(data_path, start_date="2023-01-01", end_date="2023-02-01")
    
    # Split data into train and trade sets
    train_df, trade_df = split_data(df, train_end="2023-01-21", trade_start="2023-01-21", trade_end="2023-02-01")
    
    # Create tuner
    tuner = HyperparameterTuner(df)
    
    # Define parameter grid for grid search
    param_grid = {
        'gamma': [0.9, 0.95, 0.99],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [32, 64],
        'memory_size': [10000],
        'epsilon_decay': [0.995, 0.999],
        'env_window_size': [10, 20],
        'episodes': [50]  # For demonstration purposes, use a small number
    }
    
    # Run grid search
    results = tuner.run_grid_search(param_grid)
    
    # Visualize results
    plots = tuner.visualize_results(results)
    
    # Get best parameters
    best_params = tuner.get_best_params(results)
    print("Best parameters:", best_params)
    
    # Save best model
    tuner.save_best_model(results)
    
    print("\n✅ Hyperparameter tuning completed!")
    
if __name__ == "__main__":
    main() 