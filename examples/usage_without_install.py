#!/usr/bin/env python3
"""
Example script showing how to use the DQL Trading package without installation.

This demonstrates:
1. How to import the package from any location
2. How to directly use the various components

Note: This script will check for dependencies and demonstrate the basic usage,
      but won't run actual functionality without the dependencies installed.
"""
import os
import sys
import importlib.util

# Helper function to check if dependencies are available
def check_dependency(module_name):
    """Check if a module is available"""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

# Add the repository root to the Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

# Check dependencies before proceeding
essential_deps = ['numpy', 'pandas', 'torch', 'matplotlib']
missing_deps = [dep for dep in essential_deps if not check_dependency(dep)]

if missing_deps:
    print("\n⚠️ Some dependencies are missing. This example will only show pseudocode.")
    print("To run with full functionality, install dependencies:")
    print("pip install -r requirements.txt\n")
    
    # Now we'll show the example code without actually running it
    print("Example 1: Basic Imports and Objects")
    print("===================================")
    print("# Import components (not actually executed)")
    print("from dql_trading import DQLAgent, ForexTradingEnv, TraderConfig")
    print("from dql_trading.reporting import TradingReport")
    print("")
    print("# Create a trader configuration")
    print("config = TraderConfig(")
    print("    name='Example Strategy',")
    print("    risk_tolerance=0.2,")
    print("    reward_goal=0.1,")
    print("    max_drawdown=0.05")
    print(")")
    
    print("\nExample 2: Command Line Interface")
    print("===============================")
    print("# Run commands directly using:")
    print("python dql_trading.py train --data_file your_data.csv --experiment_name test_run")
    
    print("\nExample 3: Creating Reports")
    print("=========================")
    print("# Create a report programmatically:")
    print("report = TradingReport('My Experiment')")
    print("report.add_title_page('DQL Trading Results')")
    print("report.generate_pdf('my_report.pdf')")
    
    print("\nDone! Install dependencies to use the actual functionality.")
    sys.exit(0)

# Now we can import from the package (only if dependencies are installed)
try:
    from dql_trading import DQLAgent, ForexTradingEnv, TraderConfig
    from dql_trading.reporting.reporting import TradingReport
    
    # ------------------------------------------------------------
    # Example 1: Basic imports and object creation
    # ------------------------------------------------------------
    print("Example 1: Basic Imports and Objects")
    print("===================================")
    
    # Create a trader configuration
    config = TraderConfig(
        name="Example Strategy",
        risk_tolerance=0.2,
        reward_goal=0.1,
        max_drawdown=0.05
    )
    print(f"Created TraderConfig: {config.name}")
    
    # Create a simple environment (this won't run without actual data)
    print("Creating ForexTradingEnv (would require actual data)...")
    
    # Create an agent
    print("Creating DQLAgent...")
    agent = DQLAgent(
        state_dim=10,  # Example value
        action_dim=3,  # Example value
        gamma=0.99,
        lr=0.001
    )
    
    print(f"Agent created with buffer size: {agent.buffer_size}")
    print("")
    
    # ------------------------------------------------------------
    # Example 2: Running commands via the CLI
    # ------------------------------------------------------------
    print("Example 2: Command Line Interface")
    print("===============================")
    print("You can run commands directly using:")
    print("python dql_trading.py train --data_file your_data.csv --experiment_name test_run")
    print("")
    
    # ------------------------------------------------------------
    # Example 3: Creating a report
    # ------------------------------------------------------------
    print("Example 3: Creating Reports")
    print("=========================")
    print("You can create a report programmatically:")
    print("report = TradingReport('My Experiment')")
    print("report.add_title_page('DQL Trading Results')")
    print("report.generate_pdf('my_report.pdf')")
    print("")
    
    print("Done! You've successfully imported and used the dql_trading package without installation.")
    
except ImportError as e:
    print(f"Error: {e}")
    print("This is likely due to missing dependencies.")
    print("Run 'python dql_trading.py check-dependencies' to check which ones are missing.") 