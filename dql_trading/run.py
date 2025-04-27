#!/usr/bin/env python3
"""
DQL Trading Framework Runner

A standalone script that can be executed directly from the package directory:
    cd dql_trading
    python run.py <command> [options]

This provides a convenient way to run the framework without requiring 
installation or special Python path handling.
"""
import os
import sys

# Ensure the package's parent directory is in the Python path
# This is only needed when running this script directly
package_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(package_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the main function from the CLI module
from dql_trading.cli.main import main

if __name__ == "__main__":
    sys.exit(main()) 