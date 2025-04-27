#!/usr/bin/env python3
"""
Initial Workflow Script for DQL Trading Agent

This script implements the initial workflow process:
1. Train a new model with default parameters
2. Test the model
3. Run hyperparameter tuning to find optimal parameters

This is meant to be run when first starting with a new model/dataset, 
before running the optimized workflow (run_full_workflow.py).

Usage:
    python scripts/run_initial_workflow.py --data_file test_small.csv --experiment_name initial_run
"""
import os
import sys
import argparse
import subprocess
import logging
import json
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("initial_workflow.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("initial_workflow")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run initial DQL trading agent workflow")
    
    # Data parameters
    parser.add_argument("--data_file", type=str, default="test_small.csv", 
                        help="Data file name (in data directory)")
    parser.add_argument("--start_date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=None, help="End date (YYYY-MM-DD)")
    
    # Experiment parameters
    parser.add_argument("--experiment_name", type=str, required=True, 
                        help="Name for this experiment run")
    parser.add_argument("--episodes", type=int, default=100, 
                        help="Number of episodes for training")
    parser.add_argument("--tuning_episodes", type=int, default=20, 
                        help="Number of episodes for each hyperparameter evaluation")
    parser.add_argument("--n_iter", type=int, default=20, 
                        help="Number of iterations for random hyperparameter search")
    parser.add_argument("--optimization_metric", type=str, default="sharpe_ratio", 
                        choices=["sharpe_ratio", "total_return_pct", "sortino_ratio", "calmar_ratio"],
                        help="Metric to optimize during hyperparameter tuning")
    
    # Learning parameters (for initial model)
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate for initial model")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for initial model")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Initial exploration rate")
    
    # Agent parameters
    parser.add_argument("--agent_type", type=str, default="dql",
                        choices=["dql", "custom"],
                        help="Type of agent to create")
    parser.add_argument("--target_update_freq", type=int, default=10,
                        help="Target network update frequency (for custom agent)")
    
    # Directory parameters
    parser.add_argument("--results_dir", type=str, default="results", 
                        help="Directory to save results")
    parser.add_argument("--tuning_dir", type=str, default="results/hyperparameter_tuning", 
                        help="Directory for hyperparameter tuning results")
    
    return parser.parse_args()

def train_initial_model(args):
    """Train a model with default/specified parameters"""
    logger.info("Starting initial model training...")
    
    # Create results directory
    os.makedirs(os.path.join(args.results_dir, args.experiment_name), exist_ok=True)
    
    # Build command for training
    cmd = [
        "python", "dql_trading/core/train.py",
        "--data_file", args.data_file,
        "--experiment_name", args.experiment_name,
        "--episodes", str(args.episodes),
        "--learning_rate", str(args.learning_rate),
        "--gamma", str(args.gamma),
        "--epsilon", str(args.epsilon),
        "--agent_type", args.agent_type,
        "--target_update_freq", str(args.target_update_freq),
        "--test",  # Include testing phase
        "--progress_bar"  # Show progress bar
    ]
    
    # Add optional parameters if provided
    if args.start_date:
        cmd.extend(["--start_date", args.start_date])
    if args.end_date:
        cmd.extend(["--end_date", args.end_date])
    
    # Run the training process
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Initial model training completed successfully")
        logger.debug(process.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Initial model training failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False

def run_hyperparameter_tuning(args):
    """Run hyperparameter tuning process"""
    logger.info("Starting hyperparameter tuning...")
    
    # Create tuning directory if it doesn't exist
    os.makedirs(args.tuning_dir, exist_ok=True)
    
    # Build command for hyperparameter tuning
    cmd = [
        "python", "dql_trading/scripts/run_hyperparameter_tuning.py",
        "--data_file", args.data_file,
        "--search_type", "random",
        "--n_iter", str(args.n_iter),
        "--episodes", str(args.tuning_episodes),
        "--metric", args.optimization_metric,
        "--results_dir", args.tuning_dir
    ]
    
    # Add optional parameters if provided
    if args.start_date:
        cmd.extend(["--start_date", args.start_date])
    if args.end_date:
        cmd.extend(["--end_date", args.end_date])
    
    # Run the hyperparameter tuning process
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Hyperparameter tuning completed successfully")
        logger.debug(process.stdout)
        
        # Check if optimal parameters file was created
        optimal_params_path = os.path.join(args.tuning_dir, "optimal_parameters.json")
        if not os.path.exists(optimal_params_path):
            logger.error(f"Optimal parameters file not found at {optimal_params_path}")
            return False
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Hyperparameter tuning failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False

def generate_basic_report(args):
    """Generate a basic report for the initial model"""
    logger.info("Generating basic report for initial model...")
    
    # Build command for report generation
    cmd = [
        "python", "dql_trading/scripts/generate_report.py",
        "--experiment", args.experiment_name,
        "--results_dir", args.results_dir
    ]
    
    # Run the report generation process
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Basic report generation completed")
        
        # Try to extract the report path from the output
        report_path = None
        for line in process.stdout.split('\n'):
            if "Report generated:" in line:
                report_path = line.split("Report generated:")[-1].strip()
        
        if report_path and os.path.exists(report_path):
            logger.info(f"Basic report generated at: {report_path}")
        else:
            logger.warning("Report was generated but couldn't determine the path")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Report generation failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False

def print_next_steps(args):
    """Print instructions for next steps"""
    optimal_params_path = os.path.join(args.tuning_dir, "optimal_parameters.json")
    
    print("\n" + "="*80)
    print("INITIAL WORKFLOW COMPLETED SUCCESSFULLY".center(80))
    print("="*80)
    print("\nYou've completed the initial workflow:")
    print(f"1. Trained an initial model: {args.experiment_name}")
    print(f"2. Tested the model")
    print(f"3. Ran hyperparameter tuning to find optimal parameters")
    
    print("\nThe optimal parameters have been saved to:")
    print(f"  {optimal_params_path}")
    
    print("\nNEXT STEPS:")
    print("To train an optimized model using these parameters, run:")
    print(f"  python dql_trading.py full-workflow --experiment_name optimized_{args.experiment_name} --data_file {args.data_file} --skip_tuning")
    print("\nOr to start fresh with hyperparameter tuning:")
    print(f"  python dql_trading.py full-workflow --experiment_name optimized_{args.experiment_name} --data_file {args.data_file}")
    print("="*80 + "\n")

def run_workflow(args):
    """Execute the initial workflow pipeline"""
    start_time = time.time()
    logger.info(f"Starting initial workflow for experiment: {args.experiment_name}")
    
    # Step 1: Train the initial model (includes testing)
    training_success = train_initial_model(args)
    if not training_success:
        logger.error("Initial model training failed, stopping workflow")
        return False
    
    # Step 2: Run hyperparameter tuning
    tuning_success = run_hyperparameter_tuning(args)
    if not tuning_success:
        logger.error("Hyperparameter tuning failed, stopping workflow")
        return False
    
    # Step 3: Generate a basic report (optional)
    generate_basic_report(args)
    
    # Calculate total runtime
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Initial workflow completed successfully in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Results stored in {os.path.join(args.results_dir, args.experiment_name)}")
    logger.info(f"Optimal parameters stored in {args.tuning_dir}")
    
    # Print next steps for the user
    print_next_steps(args)
    
    return True

def main(data_file=None, experiment_name=None, episodes=100, n_iter=20, **kwargs):
    """
    Main function that can be imported and called from other modules
    
    Parameters:
    -----------
    data_file : str
        Name of the data file to use
    experiment_name : str
        Name of the experiment
    episodes : int
        Number of episodes for training
    n_iter : int
        Number of iterations for hyperparameter search
    **kwargs : dict
        Additional arguments to pass to the workflow
        
    Returns:
    --------
    bool
        True if the workflow completed successfully, False otherwise
    """
    # Create args similar to what would be parsed from command line
    class Args:
        pass
    
    args = Args()
    args.data_file = data_file
    args.experiment_name = experiment_name
    args.episodes = episodes
    args.n_iter = n_iter
    args.start_date = kwargs.get('start_date', None)
    args.end_date = kwargs.get('end_date', None)
    args.results_dir = kwargs.get('results_dir', 'results')
    args.tuning_dir = kwargs.get('tuning_dir', 'results/hyperparameter_tuning')
    args.tuning_episodes = kwargs.get('tuning_episodes', 20)
    args.optimization_metric = kwargs.get('optimization_metric', 'sharpe_ratio')
    args.learning_rate = kwargs.get('learning_rate', 0.0001)
    args.gamma = kwargs.get('gamma', 0.99)
    args.epsilon = kwargs.get('epsilon', 1.0)
    args.agent_type = kwargs.get('agent_type', 'dql')
    args.target_update_freq = kwargs.get('target_update_freq', 10)
    
    return run_workflow(args)

if __name__ == "__main__":
    args = parse_args()
    success = run_workflow(args)
    sys.exit(0 if success else 1) 