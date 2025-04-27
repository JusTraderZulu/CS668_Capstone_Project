#!/usr/bin/env python3
"""
Complete Experiment Runner with Feature Importance Fix

This script runs a complete experiment and then automatically fixes the
feature importance section in the generated report.

Usage:
    python complete_experiment.py --data_file test_small.csv --experiment_name my_test --episodes 3
"""
import os
import sys
import argparse
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("experiment_runner")

def run_experiment(args):
    """Run the main experiment"""
    cmd = ["python", "run_complete_experiment.py"]
    
    # Add all arguments
    if args.data_file:
        cmd.extend(["--data_file", args.data_file])
    if args.experiment_name:
        cmd.extend(["--experiment_name", args.experiment_name])
    if args.episodes:
        cmd.extend(["--episodes", str(args.episodes)])
    
    # Run the experiment with real-time output
    logger.info(f"Running experiment with command: {' '.join(cmd)}")
    
    # Use Popen to get output in real-time while also capturing it
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Capture output for parsing while showing it in real-time
    all_output = []
    for line in iter(process.stdout.readline, ''):
        print(line, end='')  # Print in real-time
        all_output.append(line)
    
    # Wait for process to complete
    exit_code = process.wait()
    
    if exit_code != 0:
        logger.error(f"Experiment failed with exit code {exit_code}")
        return False, None
    
    # Extract actual experiment name from output
    actual_experiment_name = None
    for line in all_output:
        if line.startswith("Starting complete experiment:"):
            actual_experiment_name = line.split(":", 1)[1].strip()
            logger.info(f"Detected actual experiment name: {actual_experiment_name}")
            break
    
    return True, actual_experiment_name

def fix_report(experiment_name):
    """Fix the feature importance section in the report"""
    cmd = ["python", "merge_report_with_features.py", experiment_name]
    
    logger.info(f"Fixing feature importance section with command: {' '.join(cmd)}")
    
    # Run with real-time output
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Show output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    # Wait for process to complete
    exit_code = process.wait()
    
    if exit_code != 0:
        logger.error(f"Report fix failed with exit code {exit_code}")
        return False
    
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a complete experiment with feature importance fix")
    parser.add_argument("--data_file", type=str, help="Data file to use")
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("--episodes", type=int, help="Number of episodes to run")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Run the experiment
    logger.info("STEP 1: Running the experiment")
    success, actual_experiment_name = run_experiment(args)
    if not success:
        logger.error("Experiment failed, stopping.")
        sys.exit(1)
    
    if not actual_experiment_name:
        logger.error("Could not determine actual experiment name, using provided name")
        actual_experiment_name = args.experiment_name
    
    # Fix the report
    logger.info(f"STEP 2: Fixing feature importance section in the report for {actual_experiment_name}")
    if not fix_report(actual_experiment_name):
        logger.error("Report fix failed, but experiment completed successfully.")
        sys.exit(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"Complete experiment with feature importance fix completed in {total_time:.2f} seconds")
    logger.info(f"Final report is in results/{actual_experiment_name}/{actual_experiment_name}_report.pdf")
    
    # Success!
    sys.exit(0)

if __name__ == "__main__":
    main() 