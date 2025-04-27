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
    train_parser.add_argument("--agent_type", type=str, default="dql", choices=["dql", "custom"], 
                               help="Type of agent to create")
    train_parser.add_argument("--test", action="store_true", help="Run testing after training")
    
    # Initial workflow command
    init_parser = subparsers.add_parser("initial-workflow", 
                                         help="Run the initial workflow (training + tuning)")
    init_parser.add_argument("--data_file", type=str, required=True, help="Data file name")
    init_parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name")
    init_parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    init_parser.add_argument("--n_iter", type=int, default=20, 
                               help="Number of iterations for hyperparameter search")
    
    # Full workflow command
    full_parser = subparsers.add_parser("full-workflow", 
                                        help="Run the full workflow with optimal parameters")
    full_parser.add_argument("--data_file", type=str, required=True, help="Data file name")
    full_parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name")
    full_parser.add_argument("--skip_tuning", action="store_true", 
                              help="Skip hyperparameter tuning")
    full_parser.add_argument("--agent_type", type=str, default="dql", choices=["dql", "custom"], 
                               help="Type of agent to create")
    
    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Run hyperparameter tuning")
    tune_parser.add_argument("--data_file", type=str, required=True, help="Data file name")
    tune_parser.add_argument("--search_type", type=str, default="random", 
                             choices=["random", "grid"], help="Type of search")
    tune_parser.add_argument("--n_iter", type=int, default=20, 
                               help="Number of iterations for random search")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate a report for an experiment")
    report_parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    report_parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    eval_parser.add_argument("--data_file", type=str, required=True, help="Data file name")
    
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
            train_main(
                data_file=args.data_file,
                experiment_name=args.experiment_name,
                episodes=args.episodes,
                agent_type=args.agent_type,
                test=args.test
            )
            
        elif args.command == "initial-workflow":
            from dql_trading.scripts.run_initial_workflow import main as workflow_main
            workflow_main(
                data_file=args.data_file,
                experiment_name=args.experiment_name,
                episodes=args.episodes,
                n_iter=args.n_iter
            )
            
        elif args.command == "full-workflow":
            from dql_trading.scripts.run_full_workflow import main as workflow_main
            workflow_main(
                data_file=args.data_file,
                experiment_name=args.experiment_name,
                skip_tuning=args.skip_tuning,
                agent_type=args.agent_type
            )
            
        elif args.command == "tune":
            from dql_trading.scripts.run_hyperparameter_tuning import main as tune_main
            tune_main(
                data_file=args.data_file,
                search_type=args.search_type,
                n_iter=args.n_iter
            )
            
        elif args.command == "report":
            from dql_trading.scripts.generate_report import main as report_main
            report_main(
                experiment=args.experiment,
                results_dir=args.results_dir
            )
            
        elif args.command == "evaluate":
            from dql_trading.evaluation.evaluate import main as eval_main
            eval_main(
                experiment=args.experiment,
                data_file=args.data_file
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