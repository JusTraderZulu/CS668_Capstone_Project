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
git clone https://github.com/justinborneo/DQL_agent.git
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
# Check dependencies
python dql_trading.py check-dependencies

# Train a model with testing
python dql_trading.py train --data_file test_small.csv --experiment_name my_experiment --episodes 100 --test

# Run hyperparameter tuning
python dql_trading.py tune --data_file test_small.csv --search_type random --n_iter 20

# Initial workflow (training + tuning)
python dql_trading.py initial-workflow --data_file test_small.csv --experiment_name initial_run --episodes 100 --n_iter 20

# Full workflow (hyperparameter tuning + training + evaluation)
python dql_trading.py full-workflow --data_file test_small.csv --experiment_name my_experiment

# Skip tuning and use existing parameters
python dql_trading.py full-workflow --data_file test_small.csv --experiment_name my_experiment --skip_tuning

# Generate a report for an experiment
python dql_trading.py report --experiment my_experiment

# Evaluate a trained model on new data
python dql_trading.py evaluate --experiment my_experiment --data_file new_data.csv

# Compare multiple strategies
python dql_trading.py compare --experiments model1 model2 model3 --data_file test_data.csv
```

## Feature Importance Analysis

The framework automatically tracks feature importance during training and testing, helping you understand which features have the most significant impact on your agent's decision-making.

### How Feature Importance Works

Feature importance is calculated and integrated into the workflow:

1. **Automatic Tracking**: Feature importance is calculated during training and testing
2. **Visualization**: Visual representations are saved as PNG files
3. **Data Storage**: Raw importance values are saved as JSON
4. **Report Integration**: A dedicated section in reports shows feature importance analysis

### Accessing Feature Importance Data

Feature importance data is available through multiple pathways:

1. **PDF Reports**: Every report contains a "Feature Importance Analysis" section
2. **JSON Files**: Raw data is stored in `results/experiment_name/feature_importance.json`
3. **Visualizations**: PNG files in `results/experiment_name/feature_importance.png`

Even when feature importance data isn't available, the system will generate a fallback analysis in reports to ensure completeness.

### Example Results

The feature importance section in reports includes:
- Ranking of features by importance
- Visualization of importance distributions
- Analysis of which features drive agent decisions
- Recommendations based on feature importance

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
  author = {Justin Borneo},
  title = {DQL Trading Framework},
  year = {2023},
  url = {https://github.com/justinborneo/DQL_agent}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 