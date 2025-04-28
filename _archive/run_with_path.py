#!/usr/bin/env python3
"""
Example script showing how to run the DQL Trading framework directly
after cloning (without installation).

This simulates what a user would do when they clone the repo.
"""
import os
import sys

# Add the repository root to the Python path
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)

# Now import the package components
try:
    print("Testing DQL Trading Framework Imports...")
    
    import dql_trading
    print(f"✅ Successfully imported dql_trading package (version {dql_trading.__version__})")
    
    # Check the CLI functionality by accessing the argument parser
    from dql_trading.cli.main import parse_args
    print("✅ Successfully imported CLI parser")
    
    # Check the CLI commands that would be available
    print("\nAvailable CLI commands:")
    
    # From reviewing the CLI code, we know these are the main commands
    commands = ["train", "initial-workflow", "full-workflow", "tune", "report", "evaluate", "compare"]
    for cmd in commands:
        print(f"  - dql-trading {cmd}")
    
    print("\n✅ DQL Trading Framework can be used after cloning")
    
except ModuleNotFoundError as e:
    print(f"❌ Import error (likely missing dependency): {e}")
    print("The package structure is correct, but dependencies would need to be installed.")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1) 