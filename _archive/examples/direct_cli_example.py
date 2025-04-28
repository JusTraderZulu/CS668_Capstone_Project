#!/usr/bin/env python3
"""
Example script showing how to use the DQL Trading framework's CLI directly
after cloning (without installation).

Usage:
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Run this script

This script demonstrates how to run CLI commands programmatically.
"""
import os
import sys
import subprocess
import time

# Add the repository root to the Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_root)

def run_command(cmd):
    """Run a command and print its output"""
    print(f"\n$ {cmd}")
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for process to complete
    process.wait()
    
    elapsed = time.time() - start_time
    print(f"\nCommand completed in {elapsed:.2f} seconds with exit code: {process.returncode}")
    return process.returncode

def main():
    """Run example CLI commands"""
    print("DQL Trading CLI Examples")
    print("=======================")
    
    # 1. Simple help command
    print("\nExample 1: Getting help")
    run_command(f"python {repo_root}/dql_trading.py --help")
    
    # 2. Show train command help
    print("\nExample 2: Getting help for 'train' command")
    run_command(f"python {repo_root}/dql_trading.py train --help")
    
    # Note: The following commands would require proper data, etc.
    # They are just provided as examples of what a user would run
    
    print("\nExample Commands (Not Executed):")
    print("--------------------------------")
    print("# Running the full workflow")
    print(f"python {repo_root}/dql_trading.py full-workflow \\\n  --data_file=your_data.csv \\\n  --experiment_name=my_experiment")
    
    print("\n# Training a model")
    print(f"python {repo_root}/dql_trading.py train \\\n  --data_file=your_data.csv \\\n  --experiment_name=training_run \\\n  --episodes=100")
    
    print("\n# Generating a report")
    print(f"python {repo_root}/dql_trading.py report \\\n  --experiment=my_experiment")
    
    print("\nNote: After installation, these commands would be available as 'dql-trading <command>'")
    
if __name__ == "__main__":
    main() 