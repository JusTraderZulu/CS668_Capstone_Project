#!/usr/bin/env python3
"""
Run a complete experiment with training, testing, and report generation
to verify feature importance integration in the workflow.
"""
import os
import subprocess
import time
from datetime import datetime

# Create a unique experiment name based on current time
experiment_name = f"feature_test_{datetime.now().strftime('%m%d_%H%M')}"

def run_command(cmd, description):
    """Run a command and print output"""
    print(f"\n{'='*80}\n{description}\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Stream output
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        return False
    return True

def run_experiment():
    """Run the full experiment workflow"""
    # Make sure output directory exists
    os.makedirs("results", exist_ok=True)
    
    print(f"Starting complete experiment: {experiment_name}")
    
    # ===== TRAINING PHASE =====
    # Run training for a small number of episodes with feature importance tracking
    training_cmd = [
        "python", "-m", "dql_trading.core.train",
        "--experiment_name", experiment_name,
        "--episodes", "5",  # Small number for quick testing
        "--data_file", "test_small.csv",  # Use small dataset
        "--indicators", "ema", "rsi", "macd",  # Include indicators
        "--progress_bar",  # Show progress
        "--strategy_name", "Feature_Test_Strategy",
        "--test",  # Also run testing phase
        "--generate_report"  # Generate a report
    ]
    
    if not run_command(training_cmd, "TRAINING AND TESTING PHASE"):
        print("Training failed, exiting.")
        return False
    
    # Check if feature importance files were generated
    results_dir = os.path.join("results", experiment_name)
    
    files_to_check = [
        "feature_importance.json",
        "feature_importance.png",
        "test_feature_importance.json",
        "test_feature_importance.png",
        f"{experiment_name}_report.pdf"
    ]
    
    print("\nChecking for generated files:")
    all_files_exist = True
    for file in files_to_check:
        file_path = os.path.join(results_dir, file)
        if os.path.exists(file_path):
            print(f"✓ {file} - FOUND")
        else:
            print(f"✗ {file} - MISSING")
            all_files_exist = False
    
    if all_files_exist:
        print("\nSUCCESS! All feature importance files were generated and included in the report.")
        print(f"Experiment results are in: {results_dir}")
        print(f"Report path: {os.path.join(results_dir, f'{experiment_name}_report.pdf')}")
        return True
    else:
        print("\nSome files are missing, feature importance integration may not be complete.")
        return False

if __name__ == "__main__":
    start_time = time.time()
    success = run_experiment()
    end_time = time.time()
    
    print(f"\nExperiment {'completed successfully' if success else 'failed'}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    if success:
        print(f"\nTo view the report, open: results/{experiment_name}/{experiment_name}_report.pdf") 