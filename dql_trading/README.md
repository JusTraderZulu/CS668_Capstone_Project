# DQL Trading Framework

A comprehensive Deep Q-Learning framework for algorithmic trading.

## Overview

This framework provides a complete pipeline for training, evaluating, and deploying Deep Q-Learning agents for trading in financial markets. DQL Trading is structured as a standalone Python package that can be used directly after cloning, without requiring installation.

## Features

- **Multiple Agent Types**: DQL Agent and Custom Agent implementations
- **Modular Design**: Easily extend with new agents, environments, and strategies
- **Comprehensive Evaluation**: Detailed metrics and performance analysis
- **Professional Reporting**: Generate PDF reports with visualizations and analysis
- **Hyperparameter Tuning**: Automated tuning to find optimal parameters
- **Baseline Comparison**: Compare DQL performance against traditional strategies
- **MLflow Integration**: Track experiments and metrics

## Getting Started

### Installation

No installation is required. Simply clone the repository and use it directly:

```bash
# Clone the repository
git clone https://github.com/yourusername/dql_trading.git
cd dql_trading

# Install dependencies
pip install -r requirements.txt
```

### Running Commands

You can use the framework in three ways:

1. **Direct Module Execution**:
   ```bash
   python -m dql_trading train --data_file=your_data.csv --experiment_name=my_experiment
   ```

2. **Using the Run Script**:
   ```bash
   python run.py train --data_file=your_data.csv --experiment_name=my_experiment
   ```

3. **Importing in Python Code**:
   ```python
   from dql_trading import DQLAgent, ForexTradingEnv, TraderConfig
   
   # Create and use objects
   config = TraderConfig(name="My Strategy", risk_tolerance=0.2)
   agent = DQLAgent(state_dim=10, action_dim=3, gamma=0.99)
   ```

## Command Reference

```bash
# Check dependencies
python -m dql_trading check-dependencies

# Train a model
python -m dql_trading train --data_file=your_data.csv --experiment_name=my_experiment

# Run the full workflow (hyperparameter tuning + training + evaluation)
python -m dql_trading full-workflow --data_file=your_data.csv --experiment_name=my_experiment

# Generate a report for an experiment
python -m dql_trading report --experiment=my_experiment
```

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

## License

MIT License

## Citation

If you use this framework in your research, please cite:

```
@software{dql_trading_framework,
  author = {Your Name},
  title = {DQL Trading Framework},
  year = {2023},
  url = {https://github.com/yourusername/dql_trading}
}
``` 