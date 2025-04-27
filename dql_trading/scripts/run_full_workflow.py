#!/usr/bin/env python3
"""
Full Workflow Script for DQL Trading Agent

This script combines hyperparameter tuning, model training, and report generation
into a single, unified workflow. It can be used to:
1. Run hyperparameter optimization
2. Train a model with the optimal parameters
3. Generate a comprehensive PDF report

Usage:
    python scripts/run_full_workflow.py --data_file test_small.csv --experiment_name my_experiment
"""
import os
import sys

# Add the project root to the Python path
# Removed sys.path modification, '..')))

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
        logging.FileHandler("workflow.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("workflow")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run full DQL trading agent workflow")
    
    # Data parameters
    parser.add_argument("--data_file", type=str, default="test_small.csv", 
                        help="Data file name (in data directory)")
    parser.add_argument("--start_date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=None, help="End date (YYYY-MM-DD)")
    
    # Experiment parameters
    parser.add_argument("--experiment_name", type=str, required=True, 
                        help="Name for this experiment run")
    parser.add_argument("--skip_tuning", action="store_true", 
                        help="Skip hyperparameter tuning and use existing optimal_parameters.json")
    parser.add_argument("--episodes", type=int, default=100, 
                        help="Number of episodes for training")
    parser.add_argument("--tuning_episodes", type=int, default=20, 
                        help="Number of episodes for each hyperparameter evaluation")
    parser.add_argument("--n_iter", type=int, default=20, 
                        help="Number of iterations for random hyperparameter search")
    parser.add_argument("--optimization_metric", type=str, default="sharpe_ratio", 
                        choices=["sharpe_ratio", "total_return_pct", "sortino_ratio", "calmar_ratio"],
                        help="Metric to optimize during hyperparameter tuning")
    
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

def run_hyperparameter_tuning(args):
    """Run hyperparameter tuning process"""
    logger.info("Starting hyperparameter tuning...")
    
    # Create tuning directory if it doesn't exist
    os.makedirs(args.tuning_dir, exist_ok=True)
    
    # Build command for hyperparameter tuning
    cmd = [
        "python", "scripts/run_hyperparameter_tuning.py",
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

def train_model_with_optimal_params(args):
    """Train model using optimal parameters from hyperparameter tuning"""
    logger.info("Starting model training with optimal parameters...")
    
    # Path to optimal parameters
    optimal_params_path = os.path.join(args.tuning_dir, "optimal_parameters.json")
    
    # Verify file exists
    if not os.path.exists(optimal_params_path):
        logger.error(f"Optimal parameters file not found at {optimal_params_path}")
        return False
    
    # Build command for training
    cmd = [
        "python", "core/train.py",
        "--data_file", args.data_file,
        "--load_optimal_params", optimal_params_path,
        "--experiment_name", args.experiment_name,
        "--episodes", str(args.episodes),
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
        logger.info("Model training completed successfully")
        logger.debug(process.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Model training failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False

def generate_report(args):
    """Generate PDF report for the trained model"""
    logger.info("Generating final performance report...")
    
    # Build command for report generation
    cmd = [
        "python", "scripts/generate_report.py",
        "--experiment", args.experiment_name,
        "--results_dir", args.results_dir
    ]
    
    # Run the report generation process
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Report generation completed successfully")
        logger.debug(process.stdout)
        
        # Try to extract the report path from the output
        report_path = None
        for line in process.stdout.split('\n'):
            if "Report generated:" in line:
                report_path = line.split("Report generated:")[-1].strip()
        
        if report_path and os.path.exists(report_path):
            logger.info(f"Report successfully generated at: {report_path}")
        else:
            logger.warning("Report was generated but couldn't determine the path")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Report generation failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False

def run_workflow(args):
    """Execute the full workflow pipeline"""
    start_time = time.time()
    logger.info(f"Starting full workflow for experiment: {args.experiment_name}")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Step 1: Hyperparameter Tuning (unless skipped)
    if not args.skip_tuning:
        tuning_success = run_hyperparameter_tuning(args)
        if not tuning_success:
            logger.error("Hyperparameter tuning failed, stopping workflow")
            return False
    else:
        logger.info("Skipping hyperparameter tuning as requested")
        # Verify optimal parameters file exists
        optimal_params_path = os.path.join(args.tuning_dir, "optimal_parameters.json")
        if not os.path.exists(optimal_params_path):
            logger.error(f"Skipped tuning but optimal parameters file not found at {optimal_params_path}")
            return False
    
    # Step 2: Train Model with Optimal Parameters
    training_success = train_model_with_optimal_params(args)
    if not training_success:
        logger.error("Model training failed, stopping workflow")
        return False
    
    # Step 3: Generate Report
    report_success = generate_report(args)
    if not report_success:
        logger.error("Report generation failed")
        return False
    
    # Calculate total runtime
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Full workflow completed successfully in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Results stored in {os.path.join(args.results_dir, args.experiment_name)}")
    
    return True

if __name__ == "__main__":
    args = parse_args()
    success = run_workflow(args)
    sys.exit(0 if success else 1) 