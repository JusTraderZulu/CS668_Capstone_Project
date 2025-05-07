#!/usr/bin/env python3
"""
DQL Trading Framework - Top-level Entry Point

This script provides a unified command-line interface to the DQL Trading Framework.
It allows users to run various workflows, train models, and generate reports.

Usage:
    dql-trading <command> [options]

Commands:
    train           - Train a new DQL agent
    initial-workflow - Run the initial workflow (training + hyperparameter tuning)
    full-workflow   - Run the full workflow with optimal parameters
    tune            - Run hyperparameter tuning
    report          - Generate a report for an experiment
    evaluate        - Evaluate a trained model
    compare         - Compare different strategies

Examples:
    dql-trading train --data_file test_small.csv --experiment_name my_experiment
    dql-trading initial-workflow --data_file test_small.csv --experiment_name initial_run
    dql-trading report --experiment my_experiment
"""
import sys
import os
import argparse
import subprocess
import time
import importlib.util
import logging
from pathlib import Path
from importlib import import_module

def check_dependency(module_name):
    """Check if a module is available"""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DQL Trading Framework")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new DQL agent")
    train_parser.add_argument("--data_file", type=str, required=True, help="Data file name")
    train_parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name")
    train_parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    train_parser.add_argument("--agent_type", type=str, default="dql", choices=["dql", "custom", "memory"], 
                               help="Type of RL agent to use")
    train_parser.add_argument("--test", action="store_true", help="Run testing after training")
    train_parser.add_argument("--load_optimal_params", type=str, default=None, 
                             help="Path to optimal parameters JSON file from hyperparameter tuning")
    train_parser.add_argument("--train_frac", type=float, default=0.65,
                             help="Fraction of data used for training (chronological order)")
    train_parser.add_argument("--val_frac", type=float, default=0.20,
                             help="Fraction of data used for validation / tuning")
    train_parser.add_argument("--target_update_freq", type=int, default=10,
                              help="How often (episodes) to update the target network")
    train_parser.add_argument("--notify", action="store_true",
                               help="Send Telegram message when training completes (requires env vars)")
    train_parser.add_argument("--generate_report", dest="generate_report", action="store_true", default=True,
                              help="Generate a PDF performance report after training (default: on)")
    train_parser.add_argument("--no_report", dest="generate_report", action="store_false",
                              help="Disable report generation after training")
    
    # Initial workflow command
    init_parser = subparsers.add_parser(
        "initial-workflow",
        help="Run the initial workflow (training + tuning)"
    )
    init_parser.add_argument("--data_file", type=str, required=True, help="Data file name")
    init_parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name")
    init_parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    init_parser.add_argument(
        "--n_iter",
        type=int,
        default=20,
        help="Number of iterations for hyperparameter search",
    )
    # NEW: allow agent type selection identical to train command so workflows can run memory/other agents
    init_parser.add_argument(
        "--agent_type",
        type=str,
        default="dql",
        choices=["dql", "custom", "memory"],
        help="Type of agent to create for the baseline run",
    )
    # NEW: optional Telegram notification flag so users can be alerted when workflow completes
    init_parser.add_argument(
        "--notify",
        action="store_true",
        help="Send Telegram notification when workflow completes (requires env vars)",
    )
    
    # Full workflow command
    full_parser = subparsers.add_parser("full-workflow", 
                                        help="Run the full workflow with optimal parameters")
    full_parser.add_argument("--data_file", type=str, required=True, help="Data file name")
    full_parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name")
    full_parser.add_argument("--skip_tuning", action="store_true", 
                              help="Skip hyperparameter tuning")
    full_parser.add_argument("--agent_type", type=str, default="dql", choices=["dql", "custom", "memory"], 
                               help="Type of agent to create")
    full_parser.add_argument("--episodes", type=int, default=100, help="Number of episodes for full workflow")
    full_parser.add_argument("--tuning_episodes", type=int, default=20, help="Number of episodes for tuning")
    full_parser.add_argument("--n_iter", type=int, default=20, help="Number of iterations for hyperparameter search")
    full_parser.add_argument("--optimization_metric", type=str, default="sharpe_ratio", help="Optimization metric")
    full_parser.add_argument("--target_update_freq", type=int, default=10, help="Target update frequency")
    full_parser.add_argument("--tuning_fraction", type=float, default=1.0, help="Fraction of data used during tuning")
    full_parser.add_argument("--reward_scaling", type=float, default=1e-4, help="Reward scaling factor")
    full_parser.add_argument("--transaction_cost", type=float, default=0.0001, help="Transaction cost percentage")
    # Trader-config overrides
    full_parser.add_argument("--risk_tolerance", type=str, default="medium", choices=["low","medium","high"], help="Trader risk tolerance")
    full_parser.add_argument("--reward_goal", type=str, default="sharpe_ratio", choices=["sharpe_ratio","profit","sortino","calmar"], help="Reward goal used inside env")
    full_parser.add_argument("--max_drawdown", type=float, default=0.1, help="Maximum acceptable drawdown")
    full_parser.add_argument("--target_volatility", type=float, default=0.02, help="Target volatility")
    full_parser.add_argument("--stop_loss", type=float, default=0.03, help="Stop-loss percentage")
    full_parser.add_argument("--take_profit", type=float, default=0.05, help="Take-profit percentage")
    full_parser.add_argument("--position_sizing", type=str, default="dynamic", choices=["fixed","dynamic"], help="Position sizing method")
    full_parser.add_argument("--slippage", type=float, default=0.0002, help="Slippage percentage")
    full_parser.add_argument("--buy_pct", type=float, default=0.10, help="Fraction of equity allocated when buying")
    # Optional Telegram notification flag for full workflow
    full_parser.add_argument(
        "--notify",
        action="store_true",
        help="Send Telegram notification when workflow completes (requires env vars)",
    )
    
    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Run hyperparameter tuning")
    tune_parser.add_argument("--data_file", type=str, required=True, help="Data file name")
    tune_parser.add_argument(
        "--search_type",
        type=str,
        default="random",
        choices=["random", "grid"],
        help="Type of search",
    )
    tune_parser.add_argument(
        "--n_iter",
        type=int,
        default=20,
        help="Number of iterations for random search",
    )
    tune_parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Train/test split ratio",
    )
    tune_parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes for each evaluation",
    )
    tune_parser.add_argument("--tuning_fraction", type=float, default=1.0, help="Fraction of data used during tuning")
    # NEW: allow tuning different agent types
    tune_parser.add_argument(
        "--agent_type",
        type=str,
        default="dql",
        choices=["dql", "custom", "memory"],
        help="Type of agent to tune",
    )
    # Optional notification flag
    tune_parser.add_argument(
        "--notify",
        action="store_true",
        help="Send Telegram notification when tuning completes (requires env vars)",
    )
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate a report for an experiment")
    report_parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    report_parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    eval_parser.add_argument("--data_file", type=str, required=True, help="Data file name")
    eval_parser.add_argument(
        "--agent_type",
        type=str,
        default="dql",
        choices=["dql", "custom", "memory"],
        help="Type of agent to load for evaluation",
    )
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare different strategies")
    compare_parser.add_argument("--experiments", type=str, nargs="+", required=True, 
                                 help="List of experiment names to compare")
    compare_parser.add_argument("--data_file", type=str, required=True, help="Data file name")
    
    # Add a check dependencies command for convenience
    subparsers.add_parser("check-dependencies", help="Check if all dependencies are installed")
    
    return parser.parse_args()

def check_dependencies():
    """Check that all key dependencies are installed"""
    dependencies = {
        'numpy': 'Core numerical operations',
        'pandas': 'Data handling',
        'torch': 'Deep learning framework',
        'matplotlib': 'Plotting and visualization',
        'gym': 'Reinforcement learning environments',
        'reportlab': 'PDF report generation',
        'mlflow': 'Experiment tracking',
    }
    
    missing = []
    for dep, description in dependencies.items():
        if not check_dependency(dep):
            missing.append(f"{dep} ({description})")
    
    if not missing:
        print("✅ All dependencies are installed!")
        return True
    else:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nTo install dependencies, run:")
        print("pip install -r requirements.txt")
        return False

def report(args):
    """Generate a report for a DQL trading experiment.
    
    Args:
        args: argparse.Namespace with the CLI arguments
    """
    logger = logging.getLogger('dql_trading.cli')
    
    try:
        # Check if the experiment name is provided
        if not args.experiment:
            logger.error("Error: Experiment name is required for report generation")
            return 1
            
        # Import the generate_report module and use its main function
        try:
            from dql_trading.scripts.generate_report import main as generate_report_main
            
            # Call the main function directly
            success = generate_report_main(
                experiment=args.experiment,
                results_dir=args.results_dir
            )
            
            return 0 if success else 1
            
        except ImportError as e:
            logger.error(f"Error importing report generation module: {str(e)}")
            return 1
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            logger.error(f"Details: {type(e).__name__}: {str(e)}")
            return 1
    except Exception as e:
        logger.error(f"Unexpected error in report command: {str(e)}")
        return 1

def add_report_parser(subparsers):
    """Add the report command parser to the subparsers object."""
    parser = subparsers.add_parser(
        'report', 
        help='Generate a report for a DQL trading experiment'
    )
    parser.add_argument(
        '--experiment', 
        type=str, 
        required=True,
        help='Name of the experiment to generate a report for'
    )
    parser.add_argument(
        '--results_dir', 
        type=str, 
        default='results',
        help='Directory containing results'
    )
    parser.set_defaults(func=report)

def main():
    """Main function that handles command-line interface"""
    start_time = time.time()
    args = parse_args()
    
    if args.command is None:
        print("Error: No command specified. Use --help for available commands.")
        sys.exit(1)
    
    # Set up Python path to include the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Special case for checking dependencies
    if args.command == "check-dependencies":
        success = check_dependencies()
        return 0 if success else 1
    
    # Check dependencies before running commands
    if not check_dependency('torch') or not check_dependency('numpy'):
        print("\nError: Missing essential dependencies.")
        print("Please install the required dependencies first:")
        print("pip install -r requirements.txt")
        return 1
    
    # Set up environment for subprocess calls
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    
    try:
        if args.command == "train":
            # Using direct imports for training
            from dql_trading.core.train import main as train_main
            # Create a dictionary with all args and with defaults for any missing args
            train_args = {
                'data_file': args.data_file,
                'experiment_name': args.experiment_name,
                'episodes': args.episodes,
                'agent_type': args.agent_type,
                'test': args.test,
                'load_optimal_params': args.load_optimal_params,
                'log_dir': 'logs',                     # default log_dir
                'progress_bar': False,                 # default progress_bar
                'live_plot': False,                    # default live_plot
                'generate_report': args.generate_report,
                'viz_interval': 10,                    # default viz_interval
                'risk_tolerance': 'medium',            # default risk_tolerance
                'reward_goal': 'sharpe_ratio',         # default reward_goal
                'initial_amount': 100000,              # default initial_amount
                'transaction_cost': 0.0001,            # default transaction_cost
                'reward_scaling': 1e-4,                # default reward_scaling
                'indicators': None,                    # default indicators
                
                # Date range parameters
                'start_date': None,                    # default start_date
                'end_date': None,                      # default end_date
                'train_end': None,                     # default train_end
                'trade_start': None,                   # default trade_start
                'trade_end': None,                     # default trade_end
                
                # Trading strategy parameters
                'strategy_name': 'Default Strategy',   # default strategy_name
                'max_drawdown': 0.1,                   # default max_drawdown
                'target_volatility': 0.02,             # default target_volatility
                'stop_loss': 0.03,                     # default stop_loss
                'take_profit': 0.05,                   # default take_profit
                'position_sizing': 'dynamic',          # default position_sizing
                'slippage': 0.0002,                    # default slippage
                
                # Agent parameters
                'gamma': 0.99,                         # default gamma
                'learning_rate': 1e-4,                 # default learning_rate
                'epsilon': 1.0,                        # default epsilon
                'epsilon_min': 0.01,                   # default epsilon_min
                'epsilon_decay': 0.995,                # default epsilon_decay
                'buffer_size': 10000,                  # default buffer_size
                'batch_size': 64,                      # default batch_size
                'target_update_freq': args.target_update_freq,
                'notify': getattr(args, 'notify', False),

                # New split fractions
                'train_frac': args.train_frac,
                'val_frac': args.val_frac,
            }
            train_main(**train_args)
            
        elif args.command == "initial-workflow":
            from dql_trading.scripts.run_initial_workflow import main as workflow_main
            workflow_main(
                data_file=args.data_file,
                experiment_name=args.experiment_name,
                episodes=args.episodes,
                n_iter=args.n_iter,
                agent_type=getattr(args, "agent_type", "dql"),
                notify=getattr(args, "notify", False),
            )
            
        elif args.command == "full-workflow":
            from dql_trading.scripts.run_full_workflow import main as workflow_main
            workflow_main(
                data_file=args.data_file,
                experiment_name=args.experiment_name,
                skip_tuning=args.skip_tuning,
                agent_type=args.agent_type,
                episodes=getattr(args, "episodes", 100),
                tuning_episodes=getattr(args, "tuning_episodes", 20),
                n_iter=getattr(args, "n_iter", 20),
                optimization_metric=getattr(args, "optimization_metric", "sharpe_ratio"),
                target_update_freq=getattr(args, "target_update_freq", 10),
                tuning_fraction=getattr(args, "tuning_fraction", 1.0),
                reward_scaling=getattr(args, "reward_scaling", 1e-4),
                transaction_cost=getattr(args, "transaction_cost", 0.0001),
                risk_tolerance=getattr(args, "risk_tolerance", "medium"),
                reward_goal=getattr(args, "reward_goal", "sharpe_ratio"),
                max_drawdown=getattr(args, "max_drawdown", 0.1),
                target_volatility=getattr(args, "target_volatility", 0.02),
                stop_loss=getattr(args, "stop_loss", 0.03),
                take_profit=getattr(args, "take_profit", 0.05),
                position_sizing=getattr(args, "position_sizing", "dynamic"),
                slippage=getattr(args, "slippage", 0.0002),
                buy_pct=getattr(args, "buy_pct", 0.10),
                notify=getattr(args, "notify", False),
            )
            
        elif args.command == "tune":
            from dql_trading.scripts.run_hyperparameter_tuning import main as tune_main
            tune_main(
                data_file=args.data_file,
                search_type=args.search_type,
                n_iter=args.n_iter,
                train_split=args.train_split,
                episodes=args.episodes,
                agent_type=getattr(args, "agent_type", "dql"),
                notify=getattr(args, "notify", False),
                tuning_fraction=getattr(args, "tuning_fraction", 1.0),
            )
            
        elif args.command == "report":
            report(args)
            
        elif args.command == "evaluate":
            from dql_trading.evaluation.evaluate import main as eval_main
            eval_main(
                experiment=args.experiment,
                data_file=args.data_file,
                agent_type=getattr(args, "agent_type", "dql"),
            )
            
        elif args.command == "compare":
            from dql_trading.evaluation.compare_strategies import main as compare_main
            compare_main(
                experiments=args.experiments,
                data_file=args.data_file
            )
            
    except ImportError as e:
        print(f"\nError: Missing dependency: {str(e)}")
        print("Please install the required dependencies first:")
        print("pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        return 1
    
    elapsed_time = time.time() - start_time
    print(f"\nCommand completed successfully in {elapsed_time:.2f} seconds.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 