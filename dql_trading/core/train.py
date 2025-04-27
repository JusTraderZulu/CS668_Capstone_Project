import pkg_resources
import os
import sys

# Add the project root to the Python path
# Removed sys.path modification, '..')))

# Now import the rest of the modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse, mlflow, json
from datetime import datetime

# Import our modules
from dql_trading.utils.data_processing import load_data, split_data, add_indicators, TrainingLogger, visualize_trade_actions, set_seed
from dql_trading.envs.trading_env import ForexTradingEnv, TraderConfig
from dql_trading.agents.dql_agent import DQLAgent
from dql_trading.utils.metrics import calculate_trading_metrics, create_performance_dashboard, create_training_plots
# Add import for baseline strategies
from dql_trading.baseline_strategies.baseline_agents import MovingAverageCrossoverAgent, RSIAgent, BuyAndHoldAgent
# Add import for reporting module
from dql_trading.reporting.reporting import TradingReport
# Import feature tracking wrapper
from dql_trading.envs.feature_tracking import FeatureTrackingWrapper, create_feature_importance_visualization

def train_agent(args, experiment_name: str):
    """
    Train a DQL agent on forex data with detailed logging and visualization
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    experiment_name : str
        Name of the MLflow experiment
    """
    # Initialize logger
    logger = TrainingLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        use_tqdm=args.progress_bar,
        live_plot=args.live_plot
    )
    
    # Create results directory
    results_dir = os.path.join('results', logger.experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Log start of experiment
    logger.logger.info("=" * 50)
    logger.logger.info(f"Starting DQL Trading Agent Training: {logger.experiment_name}")
    logger.logger.info("=" * 50)
    
    # Load and prepare data
    logger.logger.info("Loading and preparing data...")
    data_path = pkg_resources.resource_filename("dql_trading", f"data/{args.data_file}")
    df = load_data(data_path, start_date=args.start_date, end_date=args.end_date)
    
    # Add technical indicators if requested
    if args.indicators:
        logger.logger.info(f"Adding technical indicators: {args.indicators}")
        df = add_indicators(df)
    
    # Split data into train and test sets
    train_df, test_df = split_data(df, train_end=args.train_end, trade_start=args.trade_start, trade_end=args.trade_end)
    
    # Log data info
    logger.logger.info(f"Training data: {len(train_df)} rows from {train_df.index.min()} to {train_df.index.max()}")
    logger.logger.info(f"Testing data: {len(test_df)} rows from {test_df.index.min()} to {test_df.index.max()}")
    
    # Create trader config
    trader_config = TraderConfig(
        name=args.strategy_name,
        risk_tolerance=args.risk_tolerance,
        reward_goal=args.reward_goal,
        max_drawdown=args.max_drawdown,
        target_volatility=args.target_volatility,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        position_sizing=args.position_sizing,
        slippage=args.slippage
    )
    
    # Log trader config
    logger.logger.info("Trader configuration:")
    for key, value in trader_config.as_dict().items():
        logger.logger.info(f"  {key}: {value}")
    
    # Environment settings
    env_params = {
        "initial_amount": args.initial_amount,
        "transaction_cost_pct": args.transaction_cost,
        "reward_scaling": args.reward_scaling,
        "tech_indicator_list": ["close"] + args.indicators if args.indicators else ["close"]
    }
    
    # Create the training environment
    train_env = ForexTradingEnv(
        df=train_df,
        trader_config=trader_config,
        **env_params
    )
    
    # Wrap environment with feature tracking wrapper
    wrapped_train_env = FeatureTrackingWrapper(train_env)
    
    # Calculate state and action dimensions
    state_dim = wrapped_train_env.observation_space.shape[0]
    action_dim = wrapped_train_env.action_space.n
    
    # Create the agent using factory
    logger.logger.info(f"Creating {args.agent_type} agent...")
    from dql_trading.agents.agent_factory import create_agent
    agent = create_agent(
        agent_type=args.agent_type,
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=args.gamma,
        lr=args.learning_rate,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        # Custom agent parameters
        target_update_freq=args.target_update_freq
    )
    
    # Log agent parameters
    agent_params = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'gamma': args.gamma,
        'learning_rate': args.learning_rate,
        'epsilon': args.epsilon,
        'epsilon_min': args.epsilon_min,
        'epsilon_decay': args.epsilon_decay,
        'buffer_size': args.buffer_size,
        'batch_size': args.batch_size
    }
    logger.log_hyperparameters(agent_params)
    logger.log_system_info(wrapped_train_env, agent)
    
    # Start training
    logger.start_training(args.episodes)
    
    # Create learning metrics tracking
    learning_metrics = {
        'episode': [],
        'epsilon': [],
        'loss': [],
        'reward': [],
        'sharpe_ratio': [],
        'return_pct': [],
        'win_rate': []
    }
    
    # Training loop
    for episode in range(args.episodes):
        # Reset environment
        state = wrapped_train_env.reset()
        
        # Ensure the state is properly shaped
        if isinstance(state, np.ndarray) and state.ndim > 1:
            state = state.flatten()
        
        done = False
        total_reward = 0
        steps = 0
        episode_losses = []
        
        # Detailed episode start info
        logger.logger.debug(f"Episode {episode+1}/{args.episodes} - Starting with epsilon: {agent.epsilon:.4f}")
        
        # Episode loop
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action with detailed logging
            next_state, reward, done, _ = wrapped_train_env.step(action)
            steps += 1
            
            # Log step details at debug level
            action_names = ['SELL', 'HOLD', 'BUY']
            logger.logger.debug(
                f"  Step {steps}: Action={action_names[action]}, "
                f"Reward={reward:.4f}, Done={done}"
            )
            
            # Ensure next_state is properly shaped
            if isinstance(next_state, np.ndarray) and next_state.ndim > 1:
                next_state = next_state.flatten()
            
            # Store transition and train
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Train agent and capture loss
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)
        
        # Update target network
        agent.update_target()
        
        # Calculate episode metrics
        account_values = wrapped_train_env.get_account_value_memory()
        trade_log = wrapped_train_env.get_trade_log()
        
        metrics = calculate_trading_metrics(
            account_values=account_values,
            trade_log=trade_log
        )
        
        # Add reward to metrics
        metrics['reward'] = total_reward
        metrics['trade_count'] = len(trade_log)
        
        # Log episode results
        logger.log_episode(episode + 1, metrics, trade_log)
        
        # Store learning metrics
        learning_metrics['episode'].append(episode + 1)
        learning_metrics['epsilon'].append(agent.epsilon)
        learning_metrics['loss'].append(np.mean(episode_losses) if episode_losses else 0)
        learning_metrics['reward'].append(total_reward)
        learning_metrics['sharpe_ratio'].append(metrics.get('sharpe_ratio', 0))
        learning_metrics['return_pct'].append(metrics.get('total_return_pct', 0))
        learning_metrics['win_rate'].append(metrics.get('win_rate', 0))
        
        # Log to MLflow with train_ prefix
        mlflow.log_metric("train_epsilon", agent.epsilon, step=episode)
        mlflow.log_metric("train_reward", total_reward, step=episode)
        mlflow.log_metric("train_loss", np.mean(episode_losses) if episode_losses else 0, step=episode)
        mlflow.log_metric("train_sharpe", metrics.get('sharpe_ratio', 0), step=episode)
        mlflow.log_metric("train_return_pct", metrics.get('total_return_pct', 0), step=episode)
        mlflow.log_metric("train_win_rate", metrics.get('win_rate', 0), step=episode)
        mlflow.log_metric("train_trade_count", len(trade_log), step=episode)
        
        # Create and save trade visualization every n episodes
        if (episode + 1) % args.viz_interval == 0 or episode == args.episodes - 1:
            viz_path = os.path.join(results_dir, f"trades_episode_{episode+1}.png")
            visualize_trade_actions(
                trade_log=trade_log,
                account_values=account_values,
                price_data=train_df,
                save_path=viz_path
            )
            logger.logger.info(f"Trade visualization saved to {viz_path}")
    
    # Log training summary
    logger.log_training_summary()
    
    # Save training metrics
    metrics_df = logger.save_metrics()
    
    # Create and save learning curves
    learning_df = pd.DataFrame(learning_metrics)
    learning_curves_path = os.path.join(results_dir, "learning_curves.png")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot epsilon decay
    axes[0].plot(learning_metrics['episode'], learning_metrics['epsilon'])
    axes[0].set_title('Epsilon Decay')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Epsilon')
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(learning_metrics['episode'], learning_metrics['loss'])
    axes[1].set_title('Training Loss')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)
    
    # Plot reward
    axes[2].plot(learning_metrics['episode'], learning_metrics['reward'])
    axes[2].set_title('Episode Reward')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Reward')
    axes[2].grid(True)
    
    # Plot sharpe ratio
    axes[3].plot(learning_metrics['episode'], learning_metrics['sharpe_ratio'])
    axes[3].set_title('Sharpe Ratio')
    axes[3].set_xlabel('Episode')
    axes[3].set_ylabel('Sharpe Ratio')
    axes[3].grid(True)
    
    # Plot return percentage
    axes[4].plot(learning_metrics['episode'], learning_metrics['return_pct'])
    axes[4].set_title('Return %')
    axes[4].set_xlabel('Episode')
    axes[4].set_ylabel('Return %')
    axes[4].grid(True)
    
    # Plot win rate
    axes[5].plot(learning_metrics['episode'], learning_metrics['win_rate'])
    axes[5].set_title('Win Rate')
    axes[5].set_xlabel('Episode')
    axes[5].set_ylabel('Win Rate')
    axes[5].grid(True)
    
    plt.tight_layout()
    plt.savefig(learning_curves_path)
    plt.close()
    
    logger.logger.info(f"Learning curves saved to {learning_curves_path}")
    mlflow.log_artifact(learning_curves_path, "training_metrics")
    
    # Create and save training plots
    if metrics_df is not None:
        plots_path = os.path.join(results_dir, "training_plots.png")
        create_training_plots(
            episode_metrics={
                'reward': logger.train_metrics['reward'],
                'sharpe_ratio': logger.train_metrics['sharpe_ratio'],
                'total_return_pct': logger.train_metrics['return_pct'],
                'max_drawdown': logger.train_metrics['max_drawdown'],
                'win_rate': logger.train_metrics['win_rate'],
                'trade_count': logger.train_metrics['trade_count']
            },
            save_path=plots_path
        )
        logger.logger.info(f"Training plots saved to {plots_path}")
    
    # Save feature importance data and visualization
    feature_importance = wrapped_train_env.get_feature_importance()
    feature_importance_path = os.path.join(results_dir, "feature_importance.json")
    
    with open(feature_importance_path, 'w') as f:
        json.dump(feature_importance, f, indent=4)
    logger.logger.info(f"Feature importance data saved to {feature_importance_path}")
    
    # Create and save feature importance visualization
    fi_fig = create_feature_importance_visualization(wrapped_train_env)
    fi_path = os.path.join(results_dir, "feature_importance.png")
    fi_fig.savefig(fi_path)
    logger.logger.info(f"Feature importance visualization saved to {fi_path}")
    
    # Log feature importance metrics to MLflow
    for feature, importance in feature_importance.items():
        mlflow.log_metric(f"importance_{feature}", importance)
    
    # Log feature importance visualization to MLflow
    mlflow.log_artifact(fi_path, "feature_importance")
    
    # Save model
    model_path = os.path.join(results_dir, "model.pth")
    agent.save_model(model_path)
    logger.logger.info(f"Model saved to {model_path}")
    
    # Set a tag for metrics phase
    mlflow.set_tag("metrics_phase", "training")
    
    # Test metrics and baseline metrics data for the report
    test_metrics_data = None
    baseline_metrics = {}
    
    # Evaluate on test data
    if args.test:
        logger.start_testing()
        
        # Set a tag to indicate testing metrics
        mlflow.set_tag("metrics_phase", "testing")
        
        # Create test environment
        test_env = ForexTradingEnv(
            df=test_df,
            trader_config=trader_config,
            **env_params
        )
        
        # Wrap test environment with feature tracking
        wrapped_test_env = FeatureTrackingWrapper(test_env)
        
        # Create a dictionary to store all strategy results
        strategy_results = {}
        
        # First evaluate the DQL agent
        logger.logger.info("Evaluating DQL Agent...")
        
        # Reset environment
        state = wrapped_test_env.reset()
        if isinstance(state, np.ndarray) and state.ndim > 1:
            state = state.flatten()
        
        done = False
        total_reward = 0
        
        # Use deterministic policy for testing
        old_epsilon = agent.epsilon
        agent.epsilon = 0
        
        logger.logger.info("Running DQL agent backtest evaluation...")
        
        # Testing loop
        step = 0
        while not done:
            # Select action
            action = agent.select_action(state, test=True)
            
            # Take action
            next_state, reward, done, _ = wrapped_test_env.step(action)
            step += 1
            
            # Log every n steps
            if step % 100 == 0:
                logger.logger.debug(f"Test step {step}: Action={action_names[action]}, Reward={reward:.4f}")
            
            # Ensure next_state is properly shaped
            if isinstance(next_state, np.ndarray) and next_state.ndim > 1:
                next_state = next_state.flatten()
            
            # Update state and reward
            state = next_state
            total_reward += reward
        
        # Restore epsilon
        agent.epsilon = old_epsilon
        
        # Calculate test metrics
        account_values = wrapped_test_env.get_account_value_memory()
        trade_log = wrapped_test_env.get_trade_log()
        
        test_metrics = calculate_trading_metrics(
            account_values=account_values,
            trade_log=trade_log
        )
        
        # Store DQL agent results
        strategy_results["DQL Agent"] = {
            'account_values': account_values,
            'trade_log': trade_log,
            'metrics': test_metrics
        }
        
        # Log test results
        logger.log_test_results(test_metrics, trade_log)
        
        # Create and save performance dashboard
        dashboard_path = os.path.join(results_dir, "performance_dashboard.png")
        create_performance_dashboard(
            account_values=account_values,
            trade_log=trade_log,
            title=f"DQL Agent Backtest Performance: {args.strategy_name}",
            save_path=dashboard_path
        )
        logger.logger.info(f"Performance dashboard saved to {dashboard_path}")
        
        # Create trade visualization for test data
        viz_path = os.path.join(results_dir, "test_trades.png")
        visualize_trade_actions(
            trade_log=trade_log,
            account_values=account_values,
            price_data=test_df,
            save_path=viz_path
        )
        logger.logger.info(f"Test trade visualization saved to {viz_path}")
        
        # Save test metrics to CSV
        test_metrics_path = os.path.join(results_dir, "test_metrics.csv")
        pd.DataFrame([test_metrics]).to_csv(test_metrics_path, index=False)
        logger.logger.info(f"Test metrics saved to {test_metrics_path}")
        
        # Save test feature importance data and visualization
        test_feature_importance = wrapped_test_env.get_feature_importance()
        test_fi_path = os.path.join(results_dir, "test_feature_importance.json")
        
        with open(test_fi_path, 'w') as f:
            json.dump(test_feature_importance, f, indent=4)
        logger.logger.info(f"Test feature importance data saved to {test_fi_path}")
        
        # Create and save test feature importance visualization
        test_fi_fig = create_feature_importance_visualization(wrapped_test_env)
        test_fi_viz_path = os.path.join(results_dir, "test_feature_importance.png")
        test_fi_fig.savefig(test_fi_viz_path)
        logger.logger.info(f"Test feature importance visualization saved to {test_fi_viz_path}")
        
        # Add test_ prefix to test metrics
        test_metric_dict = {}
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)) or (hasattr(v, "item") and callable(getattr(v, "item"))):
                test_metric_dict[f"test_{k}"] = float(v)
        
        # Log test metrics to MLflow
        mlflow.log_metrics(test_metric_dict)
        
        # Log test feature importance metrics to MLflow
        for feature, importance in test_feature_importance.items():
            mlflow.log_metric(f"test_importance_{feature}", importance)
        
        # Log test feature importance visualization to MLflow
        mlflow.log_artifact(test_fi_viz_path, "test_feature_importance")
        
        # ====== BASELINE STRATEGIES COMPARISON =======
        logger.logger.info("\n===== Evaluating Baseline Strategies for Comparison =====")
        
        # Create baseline strategies
        state_dim = test_env.observation_space.shape[0]
        action_dim = test_env.action_space.n
        
        baseline_strategies = {
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
        
        # Evaluate each baseline strategy
        for name, strategy in baseline_strategies.items():
            logger.logger.info(f"\nEvaluating baseline: {name}...")
            
            # Create new environment for each strategy to avoid state contamination
            baseline_env = ForexTradingEnv(
                df=test_df,
                trader_config=trader_config,
                **env_params
            )
            
            # Wrap with feature tracking
            wrapped_baseline_env = FeatureTrackingWrapper(baseline_env)
            
            # Reset environment
            state = wrapped_baseline_env.reset()
            if isinstance(state, np.ndarray) and state.ndim > 1:
                state = state.flatten()
            
            done = False
            
            # Testing loop for baseline strategy
            while not done:
                # Select action
                action = strategy.select_action(state, test=True)
                
                # Take action
                next_state, reward, done, _ = wrapped_baseline_env.step(action)
                
                # Ensure next_state is properly shaped
                if isinstance(next_state, np.ndarray) and next_state.ndim > 1:
                    next_state = next_state.flatten()
                
                # Update state
                state = next_state
            
            # Calculate metrics
            account_values = wrapped_baseline_env.get_account_value_memory()
            trade_log = wrapped_baseline_env.get_trade_log()
            
            metrics = calculate_trading_metrics(
                account_values=account_values,
                trade_log=trade_log
            )
            
            # Store baseline results
            strategy_results[name] = {
                'account_values': account_values,
                'trade_log': trade_log,
                'metrics': metrics
            }
            
            # Log baseline metrics
            logger.logger.info(f"  Total Return: {metrics['total_return_pct']:.2f}%")
            logger.logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            logger.logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.4f}")
            logger.logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.4f}")
            logger.logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
            
            # Save baseline dashboard
            baseline_dashboard_path = os.path.join(results_dir, f"{name.replace(' ', '_').lower()}_dashboard.png")
            create_performance_dashboard(
                account_values=account_values,
                trade_log=trade_log,
                title=f"{name} Performance",
                save_path=baseline_dashboard_path
            )
            
            # Log to MLflow
            baseline_metric_dict = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)) or (hasattr(v, "item") and callable(getattr(v, "item"))):
                    metric_key = f"baseline_{name.replace(' ', '_').lower()}_{k}"
                    baseline_metric_dict[metric_key] = float(v)
            
            mlflow.log_metrics(baseline_metric_dict)
            mlflow.log_artifact(baseline_dashboard_path, "baseline_strategies")
        
        # Create comparison chart
        logger.logger.info("\nCreating strategy comparison chart...")
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        
        for name, result in strategy_results.items():
            # Calculate returns from account values
            values = np.array(result['account_values'])
            returns = values / values[0] - 1
            ax.plot(returns * 100, label=name)  # Convert to percentage
        
        ax.set_title('Strategy Comparison')
        ax.set_xlabel('Trading Steps')
        ax.set_ylabel('Return (%)')
        ax.legend()
        ax.grid(True)
        
        # Save comparison chart
        comparison_path = os.path.join(results_dir, "strategies_comparison.png")
        plt.savefig(comparison_path)
        mlflow.log_artifact(comparison_path, "baseline_strategies")
        
        # Create metrics comparison table
        metrics_data = []
        for name, result in strategy_results.items():
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
        metrics_path = os.path.join(results_dir, "strategy_comparison.csv")
        metrics_df.to_csv(metrics_path, index=False)
        mlflow.log_artifact(metrics_path, "baseline_strategies")
        
        # Print comparison summary
        logger.logger.info("\n" + "=" * 50)
        logger.logger.info("STRATEGY COMPARISON SUMMARY:")
        logger.logger.info("=" * 50)
        logger.logger.info("\n" + metrics_df.to_string(index=False))
        logger.logger.info("=" * 50)
        
        # Use VectorBT for advanced backtesting
        from dql_trading.evaluation.vectorbt_eval import evaluate_with_vectorbt
        vbt_stats = evaluate_with_vectorbt(test_env.trade_log, test_df["close"])
        
        # Filter numeric metrics for MLflow
        numeric_stats = {}
        for k, v in vbt_stats.items():
            if isinstance(v, (int, float)) or (hasattr(v, "item") and callable(getattr(v, "item"))):
                # Clean up metric name for MLflow (remove brackets, etc.)
                clean_key = f"test_vbt_{k.replace('[', '').replace(']', '').replace('%', 'pct')}"
                numeric_stats[clean_key] = float(v)
        
        mlflow.log_metrics(numeric_stats)
        
        # Log all test artifacts to a test folder
        for artifact_path in [dashboard_path, viz_path, test_metrics_path]:
            if os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path, "test_artifacts")
        
        # Store test metrics for report
        test_metrics_data = test_metrics
        
        # Store baseline metrics for report
        for name, result in strategy_results.items():
            baseline_metrics[name] = result['metrics']
    
    # Create performance report if reporting is enabled
    if args.generate_report:
        logger.logger.info("=" * 50)
        logger.logger.info(f"Generating performance report for: {logger.experiment_name}")
        logger.logger.info("=" * 50)
        
        # Initialize the report
        report = TradingReport(
            experiment_name=logger.experiment_name,
            output_dir=results_dir
        )
        
        # Add title page
        report.add_title_page(
            title=f"Trading Strategy Performance Report: {logger.experiment_name}",
            subtitle=f"DQL Agent with {args.strategy_name}",
            date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Add executive summary
        train_summary = {
            'total_return': f"{metrics.get('total_return_pct', 0):.2f}%",
            'sharpe_ratio': f"{metrics.get('sharpe_ratio', 0):.4f}",
            'max_drawdown': f"{metrics.get('max_drawdown', 0) * 100:.2f}%",
            'win_rate': f"{metrics.get('win_rate', 0) * 100:.2f}%",
            'total_trades': metrics.get('total_trades', 0)
        }
        
        test_summary = None
        if test_metrics_data:
            test_summary = {
                'total_return': f"{test_metrics_data.get('total_return_pct', 0):.2f}%",
                'sharpe_ratio': f"{test_metrics_data.get('sharpe_ratio', 0):.4f}",
                'max_drawdown': f"{test_metrics_data.get('max_drawdown', 0) * 100:.2f}%",
                'win_rate': f"{test_metrics_data.get('win_rate', 0) * 100:.2f}%",
                'total_trades': test_metrics_data.get('total_trades', 0)
            }
        
        report.add_executive_summary(
            train_summary=train_summary,
            test_summary=test_summary
        )
        
        # Add training metrics
        report.add_training_metrics(
            metrics=metrics,
            training_plots_path=os.path.join(results_dir, "training_plots.png"),
            learning_curves_path=os.path.join(results_dir, "learning_curves.png")
        )
        
        # Add testing metrics if available
        if test_metrics_data:
            report.add_testing_metrics(
                metrics=test_metrics_data,
                performance_dashboard_path=os.path.join(results_dir, "performance_dashboard.png"),
                trade_visualization_path=os.path.join(results_dir, "test_trades.png")
            )
            
            # Add baseline comparison if available
            if baseline_metrics:
                report.add_baseline_comparison(
                    dql_metrics=test_metrics_data,
                    baseline_metrics=baseline_metrics,
                    comparison_chart_path=os.path.join(results_dir, "strategies_comparison.png")
                )
        
        # Add hyperparameter analysis
        report.add_hyperparameter_analysis({
            'Agent Parameters': agent_params,
            'Environment Parameters': env_params,
            'Trader Configuration': trader_config.as_dict()
        })
        
        # Add conclusion
        report.add_conclusion(
            key_findings=[
                f"The DQL agent achieved a total return of {metrics.get('total_return_pct', 0):.2f}% on training data.",
                f"The agent performed with a Sharpe ratio of {metrics.get('sharpe_ratio', 0):.4f} during training.",
                f"Maximum drawdown during training was {metrics.get('max_drawdown', 0) * 100:.2f}%."
            ],
            recommendations=[
                "Consider running hyperparameter tuning to optimize agent performance.",
                "Test the agent on different market conditions and timeframes.",
                "Evaluate adding different technical indicators to the state representation."
            ]
        )
        
        # Generate the report
        report_path = report.generate()
        logger.logger.info(f"Performance report generated: {report_path}")
        
        # Log the report to MLflow
        mlflow.log_artifact(report_path, "reports")
    
    logger.logger.info("=" * 50)
    logger.logger.info(f"Training and evaluation completed: {logger.experiment_name}")
    logger.logger.info("=" * 50)
    
    # Create a summary dict for final metrics (training metrics)
    train_metrics = {"final": metrics}
    
    return train_metrics

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a DQL agent on forex data")
    
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
    parser.add_argument("--strategy_name", type=str, default="Default Strategy", help="Name of trading strategy")
    parser.add_argument("--risk_tolerance", type=str, default="medium", help="Risk tolerance (low, medium, high)")
    parser.add_argument("--reward_goal", type=str, default="sharpe_ratio", help="Goal for reward calculation")
    parser.add_argument("--max_drawdown", type=float, default=0.1, help="Maximum drawdown tolerance")
    parser.add_argument("--target_volatility", type=float, default=0.02, help="Target volatility")
    parser.add_argument("--stop_loss", type=float, default=0.03, help="Stop loss percentage")
    parser.add_argument("--take_profit", type=float, default=0.05, help="Take profit percentage")
    parser.add_argument("--position_sizing", type=str, default="dynamic", help="Position sizing (fixed, dynamic)")
    parser.add_argument("--slippage", type=float, default=0.0002, help="Slippage")
    
    # Agent parameters
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon_min", type=float, default=0.01, help="Minimum exploration rate")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Exploration rate decay")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--test", action="store_true", help="Test after training")
    
    # Logging and visualization
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar")
    parser.add_argument("--live_plot", action="store_true", help="Show live plot during training")
    parser.add_argument("--viz_interval", type=int, default=10, help="Visualization interval (episodes)")
    
    # Reporting
    parser.add_argument("--generate_report", action="store_true", help="Generate PDF performance report")
    
    # Optimal parameters loading
    parser.add_argument("--load_optimal_params", type=str, default=None, 
                        help="Path to optimal parameters JSON file from hyperparameter tuning")
    
    # New agent type parameter
    parser.add_argument("--agent_type", type=str, default="DQL", help="Type of agent to create")
    
    # New target update frequency parameter
    parser.add_argument("--target_update_freq", type=int, default=10, help="Target update frequency")
    
    return parser.parse_args()

def main(**kwargs):
    """Main function to run the training process"""
    # Parse arguments or use provided kwargs
    if not kwargs:
        args = parse_args()
    else:
        # Convert kwargs to an argparse.Namespace object
        args = argparse.Namespace(**kwargs)
    
    # Load optimal parameters if specified
    if hasattr(args, 'load_optimal_params') and args.load_optimal_params and os.path.exists(args.load_optimal_params):
        print(f"Loading optimal parameters from {args.load_optimal_params}")
        with open(args.load_optimal_params, 'r') as f:
            optimal_params = json.load(f)
        
        # Update agent parameters
        if 'agent_params' in optimal_params:
            agent_params = optimal_params['agent_params']
            for param, value in agent_params.items():
                if hasattr(args, param):
                    setattr(args, param, value)
                    print(f"  Set {param} = {value}")
        
        # Update environment parameters
        if 'env_params' in optimal_params:
            env_params = optimal_params['env_params']
            for param, value in env_params.items():
                if hasattr(args, param):
                    setattr(args, param, value)
                    print(f"  Set {param} = {value}")
        
        print("Optimal parameters loaded successfully.")
    
    # Set random seeds for reproducibility
    set_seed(42)
    
    # Set up MLflow
    os.makedirs("mlruns", exist_ok=True)  # Ensure directory exists
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name=args.experiment_name) as run:
        # Log experiment parameters
        params = vars(args)
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                mlflow.log_param(key, value)
            
        train_metrics = train_agent(args, experiment_name=args.experiment_name)
        
        # Convert metrics to float and clean names
        final_metrics = {}
        for k, v in train_metrics["final"].items():
            if isinstance(v, (int, float)) or (hasattr(v, "item") and callable(getattr(v, "item"))):
                clean_key = f"train_final_{k.replace('[', '').replace(']', '').replace('%', 'pct')}"
                final_metrics[clean_key] = float(v)
        
        mlflow.log_metrics(final_metrics)
        mlflow.log_artifacts("results", "training_artifacts")
        print(f"Run logged to {mlflow.get_tracking_uri()}  ➜  id: {run.info.run_id}")
        print(f"View this run at http://localhost:5000/experiments/{mlflow.active_run().info.experiment_id}/runs/{run.info.run_id}")

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args)) 