"""
DQL Trading Framework
====================

A comprehensive system for training, evaluating, and deploying
Deep Q-Learning agents for financial trading tasks.

Key Components
-------------
* Agents: DQLAgent, CustomAgent
* Environments: ForexTradingEnv
* Training: Core training and hyperparameter tuning
* Evaluation: Performance metrics and strategy comparison
* Reporting: PDF report generation

Usage
-----
To use this package after cloning the repository:

```python
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dql_trading import DQLAgent, ForexTradingEnv, TraderConfig
```

Or use the CLI interface:
```bash
python dql_trading.py train --data_file=your_data.csv --experiment_name=my_experiment
```
"""

__version__ = "0.1.0"

# Import key components to make them available at package level
try:
    from dql_trading.agents.dql_agent import DQLAgent
    from dql_trading.agents.custom_agent import CustomAgent
    from dql_trading.agents.agent_factory import create_agent
    from dql_trading.envs.trading_env import ForexTradingEnv, TraderConfig

    # Additional useful imports
    from dql_trading.reporting import TradingReport
    from dql_trading.evaluation.evaluate import evaluate_agent

    __all__ = [
        'DQLAgent',
        'CustomAgent',
        'create_agent',
        'ForexTradingEnv',
        'TraderConfig',
        'TradingReport',
        'evaluate_agent',
    ]
except ImportError as e:
    # In case dependencies aren't installed
    missing_module = str(e).split("'")[1] if "'" in str(e) else str(e)
    print(f"Warning: Missing dependency: {missing_module}")
    print("To use this package, install required dependencies: pip install -r requirements.txt")
    
    # Still define __all__ for documentation purposes
    __all__ = [
        'DQLAgent',
        'CustomAgent',
        'create_agent',
        'ForexTradingEnv',
        'TraderConfig',
        'TradingReport',
        'evaluate_agent',
    ]
