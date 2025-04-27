# DQL Trading Framework

A comprehensive Deep Q-Learning framework for algorithmic trading.

## Overview

This framework provides a complete pipeline for training, evaluating, and deploying Deep Q-Learning agents for trading in financial markets. The DQL Trading Framework is structured as a Python package (`dql_trading`) that can be used directly after cloning, without requiring installation.

## Features

- **Multiple Agent Types**: DQL Agent and Custom Agent implementations
- **Modular Design**: Easily extend with new agents, environments, and strategies
- **Comprehensive Evaluation**: Detailed metrics and performance analysis
- **Professional Reporting**: Generate PDF reports with visualizations and analysis
- **Hyperparameter Tuning**: Automated tuning to find optimal parameters
- **Baseline Comparison**: Compare DQL performance against traditional strategies
- **MLflow Integration**: Track experiments and metrics
- **Docker Support**: Containerized deployment

## Getting Started

### Install via pip

The easiest way to use the DQL Trading Framework is to install it via pip:

```bash
pip install dql-trading
```

After installation, you can:

```python
# Import components
from dql_trading import DQLAgent, ForexTradingEnv, TraderConfig

# Create and use objects
config = TraderConfig(name="My Strategy", risk_tolerance=0.2)
agent = DQLAgent(state_dim=10, action_dim=3, gamma=0.99)
```

You can also use the command-line interface:

```bash
# Train a model
dql-trading train --data_file=your_data.csv --experiment_name=my_experiment
```

### Clone and Run

Alternatively, you can clone the repository and use it directly:

```bash
# Clone the repository
git clone https://github.com/yourusername/DQL_agent.git
cd DQL_agent

# Install dependencies
pip install -r requirements.txt

# Run a command
python dql_trading.py train --data_file=your_data.csv --experiment_name=my_experiment
```

### Using as a Python Package

You can import and use the components directly in your Python code:

```python
import os
import sys
# Add the repository to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components
from dql_trading import DQLAgent, ForexTradingEnv, TraderConfig

# Create and use objects
config = TraderConfig(name="My Strategy", risk_tolerance=0.2)
agent = DQLAgent(state_dim=10, action_dim=3, gamma=0.99)
```

See the `examples/` directory for more examples.

## Command Line Interface

The framework provides a convenient CLI for common operations:

```bash
# Train a model
python dql_trading.py train --data_file=your_data.csv --experiment_name=my_experiment

# Run the full workflow (hyperparameter tuning + training + evaluation)
python dql_trading.py full-workflow --data_file=your_data.csv --experiment_name=my_experiment

# Generate a report for an experiment
python dql_trading.py report --experiment=my_experiment
```

## Feature Importance Tracking

The framework includes automatic feature importance tracking, allowing you to understand which features have the most significant impact on your agent's decision-making.

### Complete Experiment Script

The simplest way to run an experiment with feature importance tracking is to use the complete experiment script:

```bash
# Run a complete experiment with feature importance integration
python complete_experiment.py --data_file=your_data.csv --experiment_name=my_experiment --episodes=100
```

This script:
1. Runs the experiment (training and optional testing)
2. Automatically captures feature importance data
3. Generates a complete report with the feature importance section included

### Manual Report Fix

If you've already run experiments using the standard commands, you can add feature importance analysis to existing reports:

```bash
# Fix an existing report to include feature importance
python merge_report_with_features.py <experiment_name>
```

The feature importance tracking works with all experiment types:
- Training runs
- Testing runs
- Hyperparameter tuning
- Strategy comparison

As long as your experiment follows the project's standard file structure, the feature importance data will be captured automatically.

## Project Structure

The framework is organized into modules:

```
dql_trading/
├── agents/            # DQL and custom agent implementations
├── baseline_strategies/ # Traditional trading strategies for comparison
├── cli/               # Command-line interface
├── core/              # Core training and testing functionality
├── data/              # Data handling utilities and sample data
├── envs/              # Trading environments (Forex, stocks, etc.)
├── evaluation/        # Performance evaluation tools
├── reporting/         # PDF report generation
├── scripts/           # Workflow scripts
└── utils/             # Utility functions
```

## Customization

You can extend the framework by:

1. **Creating custom agents**: Inherit from the base `Agent` class
2. **Adding new environments**: Implement new trading environments for different assets
3. **Implementing new strategies**: Add baseline strategies for comparison
4. **Extending the workflows**: Modify the workflow scripts for your research

## Documentation

For detailed documentation, see the [docs](./docs) directory.

## License

MIT License

## Citation

If you use this framework in your research, please cite:

```
@software{dql_trading_framework,
  author = {Your Name},
  title = {DQL Trading Framework},
  year = {2023},
  url = {https://github.com/yourusername/DQL_agent}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 