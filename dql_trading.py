#!/usr/bin/env python3
"""
DQL Trading Framework - Command Line Launcher

This script serves as a convenient entry point to the DQL Trading framework
without requiring installation.

Usage:
    python dql_trading.py <command> [options]

Commands:
    train           - Train a new DQL agent
    initial-workflow - Run the initial workflow (training + hyperparameter tuning)
    full-workflow   - Run the full workflow with optimal parameters
    tune            - Run hyperparameter tuning
    report          - Generate a report for an experiment
    evaluate        - Evaluate a trained model
    compare         - Compare different strategies
    check-dependencies - Check if all required dependencies are installed

Examples:
    python dql_trading.py train --data_file test_small.csv --experiment_name my_experiment
    python dql_trading.py report --experiment my_experiment
    python dql_trading.py check-dependencies
"""
import os
import sys
import importlib
import importlib.util as _ilu

# Add the current directory to the Python path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# -----------------------------------------------------------------------------
# Prevent module shadowing:
# When this wrapper is executed, the filename `dql_trading.py` would normally
# mask the *package* directory `dql_trading/`, causing `import dql_trading.*`
# (e.g. `dql_trading.utils.notifications`) to fail with the error
# "dql_trading is not a package".  We proactively import the actual package
# from the sibling directory and register it under the correct key in
# `sys.modules`, replacing any accidental module entry that points to this
# wrapper file.
# -----------------------------------------------------------------------------
pkg_init = os.path.join(REPO_ROOT, "dql_trading", "__init__.py")
if os.path.exists(pkg_init):
    # Remove wrong entry if present
    mod = sys.modules.get("dql_trading")
    if mod is None or not hasattr(mod, "__path__"):
        sys.modules.pop("dql_trading", None)
        spec = _ilu.spec_from_file_location("dql_trading", pkg_init)
        pkg = _ilu.module_from_spec(spec)
        spec.loader.exec_module(pkg)  # type: ignore[attr-defined]
        sys.modules["dql_trading"] = pkg
        # Also expose submodule search locations
        if REPO_ROOT not in getattr(pkg, "__path__", []):
            pkg.__path__.append(os.path.join(REPO_ROOT, "dql_trading"))

def check_dependency(module_name):
    """Check if a module is available without importing it"""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def main():
    """Main entry point"""
    # First check if we're just checking dependencies
    if len(sys.argv) > 1 and sys.argv[1] == "check-dependencies":
        # Try to import the check_dependencies function directly
        try:
            from dql_trading.cli.main import check_dependencies
            return 0 if check_dependencies() else 1
        except ImportError:
            # Fall back to our own implementation if that fails
            basic_dependencies = ['numpy', 'pandas', 'torch', 'matplotlib', 'gym', 'reportlab']
            missing = [dep for dep in basic_dependencies if not check_dependency(dep)]
            
            if not missing:
                print("✅ All basic dependencies are installed!")
                return 0
            else:
                print("❌ Missing dependencies:")
                for dep in missing:
                    print(f"  - {dep}")
                print("\nTo install dependencies, run:")
                print("pip install -r requirements.txt")
                return 1
    
    # For all other commands, try to import the main function
    try:
        from dql_trading.cli.main import main as cli_main
        return cli_main()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nIt seems some dependencies are missing. Check dependencies with:")
        print("python dql_trading.py check-dependencies")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 