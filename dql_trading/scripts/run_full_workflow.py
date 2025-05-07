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
import shutil  # NEW: for copying files

# -----------------------------------------------------------------------------
# Ensure the project root (two levels up) is on sys.path so that `import
# dql_trading.*` works even when this script is executed directly with a relative
# path like `python dql_trading/scripts/run_full_workflow.py ...`. Without this
# adjustment, the import will fail and Telegram notifications cannot be sent.
# -----------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    parser.add_argument("--tuning_fraction", type=float, default=1.0,
                        help="Fraction of data to use during hyperparameter tuning (to reduce memory)")
    parser.add_argument("--optimization_metric", type=str, default="sharpe_ratio", 
                        choices=["sharpe_ratio", "total_return_pct", "sortino_ratio", "calmar_ratio"],
                        help="Metric to optimize during hyperparameter tuning")
    
    # Reward goal for TraderConfig
    parser.add_argument("--reward_goal", type=str, default="sharpe_ratio",
                        choices=["sharpe_ratio", "profit", "sortino"],
                        help="Reward function goal used inside the environment")
    
    # Agent parameters
    parser.add_argument("--agent_type", type=str, default="dql",
                        choices=["dql", "custom", "memory"],
                        help="Type of agent to create")
    parser.add_argument("--target_update_freq", type=int, default=10,
                        help="Target network update frequency (for custom agent)")
    
    # Directory parameters
    parser.add_argument("--results_dir", type=str, default="results", 
                        help="Directory to save results")
    parser.add_argument("--tuning_dir", type=str, default="results/hyperparameter_tuning", 
                        help="Directory for hyperparameter tuning results")
    
    # Notification flag (optional)
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send Telegram notification with summary & PDF report at the end (requires env vars)",
    )
    
    return parser.parse_args()

def run_hyperparameter_tuning(args):
    """Run hyperparameter tuning process"""
    logger.info("Starting hyperparameter tuning...")
    
    # Create tuning directory if it doesn't exist
    os.makedirs(args.tuning_dir, exist_ok=True)
    
    # Build command for hyperparameter tuning
    cmd = [
        sys.executable, "-m", "dql_trading.scripts.run_hyperparameter_tuning",
        "--data_file", args.data_file,
        "--search_type", "random",
        "--n_iter", str(args.n_iter),
        "--episodes", str(args.tuning_episodes),
        "--tuning_fraction", str(args.tuning_fraction),
        "--metric", args.optimization_metric,
        "--results_dir", args.tuning_dir,
        "--agent_type", args.agent_type,  # Pass agent_type to tuning script
        "--reward_goal", args.reward_goal  # Pass reward_goal
    ]
    
    # Add optional parameters if provided
    if args.start_date:
        cmd.extend(["--start_date", args.start_date])
    if args.end_date:
        cmd.extend(["--end_date", args.end_date])
    
    # Add notify flag if provided
    if getattr(args, "notify", False):
        cmd.append("--notify")
    
    # Launch tuner with unbuffered output so print statements appear immediately
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    # Relay tuner stdout to our logger in real time
    for line in process.stdout:
        logger.info(line.rstrip())

    exit_code = process.wait()
    if exit_code != 0:
        logger.error(f"Hyperparameter tuning failed with exit code {exit_code}")
        return False

    logger.info("Hyperparameter tuning completed successfully")
    
    # Check if optimal parameters file was created
    optimal_params_path = os.path.join(args.tuning_dir, "optimal_parameters.json")
    if not os.path.exists(optimal_params_path):
        logger.error(f"Optimal parameters file not found at {optimal_params_path}")
        return False
    
    # NEW: copy optimal parameters to experiment directory so reports can find them
    exp_dir = os.path.join(args.results_dir, args.experiment_name)
    try:
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copy2(optimal_params_path, os.path.join(exp_dir, "hyperparameters.json"))
        logger.info("Copied optimal parameters to experiment directory")
    except Exception as copy_err:
        logger.warning(f"Failed to copy optimal parameters to experiment dir: {copy_err}")
    
    return True

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
        sys.executable, "dql_trading/core/train.py",
        "--data_file", args.data_file,
        "--load_optimal_params", optimal_params_path,
        "--experiment_name", args.experiment_name,
        "--episodes", str(args.episodes),
        "--agent_type", args.agent_type,
        "--target_update_freq", str(args.target_update_freq),
        "--test",  # Include testing phase
        "--reward_goal", args.reward_goal,
        "--progress_bar"  # Show progress bar
    ]
    
    # Pass on notify flag so train.py can send its own summary & PDF (if asked)
    if getattr(args, "notify", False):
        cmd.append("--notify")
    
    # Add optional parameters if provided
    if args.start_date:
        cmd.extend(["--start_date", args.start_date])
    if args.end_date:
        cmd.extend(["--end_date", args.end_date])
    
    # Run the training process
    logger.info(f"Running command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    for line in process.stdout:
        logger.info(line.rstrip())

    exit_code = process.wait()
    if exit_code != 0:
        logger.error(f"Model training failed with exit code {exit_code}")
        return False

    logger.info("Model training completed successfully")
    return True

def generate_report(args):
    """Generate PDF report for the trained model"""
    logger.info("Generating final performance report...")
    
    # Build command for report generation
    cmd = [
        sys.executable, "dql_trading/scripts/generate_report.py",
        "--experiment", args.experiment_name,
        "--results_dir", args.results_dir
    ]
    
    # Run the report generation process
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Ensure the subprocess sees the project root on PYTHONPATH (same as other steps)
    env = os.environ.copy()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(cmd, check=True, text=True, env=env)
        logger.info("Report generation completed successfully")

        # Compute expected path directly
        report_path = os.path.join(
            args.results_dir,
            args.experiment_name,
            f"{args.experiment_name}_report.pdf",
        )

        if os.path.exists(report_path):
            logger.info(f"Report successfully generated at: {report_path}")
        else:
            logger.warning(
                f"Report generation finished but '{report_path}' not found."
            )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Report generation failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False

def run_workflow(args):
    """Execute the full workflow pipeline"""
    start_time = time.time()
    logger.info(f"Starting full workflow for experiment: {args.experiment_name}")
    
    # Attach a per-experiment file handler so logs go to logs/<experiment>.log
    try:
        os.makedirs("logs", exist_ok=True)
        exp_log_path = os.path.join("logs", f"{args.experiment_name}.log")
        exp_handler = logging.FileHandler(exp_log_path)
        exp_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        # Avoid adding duplicate handlers if run_workflow is called multiple times
        if exp_handler not in logger.handlers:
            logger.addHandler(exp_handler)
            logger.info(f"Attached experiment log handler at {exp_log_path}")
    except Exception as log_err:
        logger.warning(f"Failed to attach experiment log handler: {log_err}")
    
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
    
    # --------------------------------------------------------------
    # Optional Telegram notification with report
    # --------------------------------------------------------------
    if getattr(args, "notify", False):
        try:
            from dql_trading.utils.notifications import (
                send_telegram_message,
                send_telegram_document,
            )

            summary_msg = (
                f"✅ Full workflow finished for *{args.experiment_name}*\n"
                f"Runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s"
            )
            send_telegram_message(summary_msg)

            pdf_path = os.path.join(
                args.results_dir, args.experiment_name, f"{args.experiment_name}_report.pdf"
            )
            if os.path.exists(pdf_path):
                send_telegram_document(pdf_path, caption=f"Report for *{args.experiment_name}*")
        except Exception as notify_err:
            logger.warning(f"Telegram notification failed: {notify_err}")
    
    return True

if __name__ == "__main__":
    args = parse_args()
    success = run_workflow(args)
    sys.exit(0 if success else 1)

def main(data_file=None, experiment_name=None, skip_tuning=False, agent_type="dql", **kwargs):
    """
    Main function that can be imported and called from other modules
    
    Parameters:
    -----------
    data_file : str
        Name of the data file to use
    experiment_name : str
        Name of the experiment
    skip_tuning : bool
        Whether to skip the hyperparameter tuning step
    agent_type : str
        Type of agent to create ("dql", "custom", or "memory")
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
    args.skip_tuning = skip_tuning
    args.agent_type = agent_type
    args.episodes = kwargs.get('episodes', 100)
    args.tuning_episodes = kwargs.get('tuning_episodes', 20)
    args.n_iter = kwargs.get('n_iter', 20)
    args.tuning_fraction = kwargs.get('tuning_fraction', 1.0)
    args.optimization_metric = kwargs.get('optimization_metric', 'sharpe_ratio')
    args.target_update_freq = kwargs.get('target_update_freq', 10)
    args.results_dir = kwargs.get('results_dir', 'results')
    args.notify = kwargs.get('notify', False)
    # Place hyperparameter tuning results inside the experiment folder by default so
    # each experiment keeps its own set of optimal parameters and tuning history.
    # This prevents collisions if multiple experiments are run back-to-back.
    default_tuning_dir = os.path.join(args.results_dir, args.experiment_name, 'hyperparameter_tuning')
    args.tuning_dir = kwargs.get('tuning_dir', default_tuning_dir)
    args.start_date = kwargs.get('start_date', None)
    args.end_date = kwargs.get('end_date', None)
    args.reward_goal = kwargs.get('reward_goal', 'sharpe_ratio')
    
    return run_workflow(args) 