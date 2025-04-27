#!/usr/bin/env python3
"""
DQL Trading Framework - Main Entry Point

This module allows the package to be run as a command:
    python -m dql_trading <command> [options]

It serves as the main entry point for command-line usage.
"""
import sys
from dql_trading.cli.main import main

if __name__ == "__main__":
    sys.exit(main()) 