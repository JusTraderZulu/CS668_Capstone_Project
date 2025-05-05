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

## Quick Start (CLI-only)

All functionality is exposed through **one entry point**:

```bash
python dql_trading.py <command> [options]
```

First install the dependencies (if you just cloned the repo):

```bash
pip install -r requirements.txt
```

### Typical Researcher Workflow

| Step | Command | Why you might run it |
|------|---------|----------------------|
| 0. Verify environment | `python dql_trading.py check-dependencies` | Confirms Python packages (numpy, torch, etc.) are installed. |
| 1. Quick train & test | `python dql_trading.py train --data_file test_small.csv --experiment_name exp1 --episodes 50 --test` | Fast iteration: trains for *N* episodes, auto-generates a PDF report (enabled by default). |
| 2. Hyper-parameter search | `python dql_trading.py tune --data_file test_small.csv --search_type random --n_iter 30` | Finds better agent parameters via random/grid search. Stores `optimal_parameters.json`. |
| 3. Initial workflow | `python dql_trading.py initial-workflow --data_file test_small.csv --experiment_name exp2 --episodes 100 --n_iter 30` | Runs tuning *then* trains a model with the best parameters. Good starting point for a new dataset. |
| 4. Full workflow | `python dql_trading.py full-workflow --data_file test_small.csv --experiment_name exp3` | Hyper-parameter tuning → training → evaluation → PDF report, fully automated. |
| 5. Skip tuning reuse | `python dql_trading.py full-workflow --data_file test_small.csv --experiment_name exp3 --skip_tuning` | Retrain/evaluate using previously discovered optimal parameters. |
| 6. Generate / rebuild report | `python dql_trading.py report --experiment exp1` | Re-creates `results/exp1/exp1_report.pdf` without retraining. Handy after manually modifying figures. |
| 7. Evaluate on fresh data | `python dql_trading.py evaluate --experiment exp1 --data_file new_data.csv` | Tests a saved model on unseen market data. |
| 8. Compare strategies | `python dql_trading.py compare --experiments exp1 exp2 exp3 --data_file test_small.csv` | Produces side-by-side performance charts & metrics. |
| 9. Cloud run with Telegram ping | `python dql_trading.py train … --notify` | Sends a summary + PDF to Telegram when finished. |

Each command writes all artifacts to `results/<experiment_name>/`, including:

* `*_report.pdf` – PDF with dynamic Table-of-Contents, Feature-Importance, etc.
* `feature_importance.(json|png)` – raw scores & visualization
* `training_plots.png`, `learning_curves.png`, dashboards, etc.

> Tip: run small episode counts first to validate pipeline, then scale up.

See `python dql_trading.py --help` for full argument lists.

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
  year = {2025},
  url = {https://github.com/justinborneo/DQL_agent}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Telegram Notifications

You can receive a text summary **and the PDF report** directly in Telegram.

1. Create a bot with **@BotFather** → note the token.
2. Obtain your chat-ID via **@userinfobot**.
3. Export the credentials in your shell:

```bash
export TELEGRAM_BOT_TOKEN="<token>"
export TELEGRAM_CHAT_ID="<chat_id>"
```

4. Add `--notify` to any `train` command.  Example:

```bash
python dql_trading.py train \
    --data_file eurusd_all.csv \
    --experiment_name mem_cloud \
    --episodes 300 \
    --agent_type memory \
    --test \
    --notify
```

When training ends you'll receive a message like:

```
✅ Training finished for mem_cloud
Return: 12.34%  |  Sharpe: 0.95  |  Drawdown: 4.20%
```
…followed by the attached `mem_cloud_report.pdf`.

## Telegram Notifications (NEW)

The framework can send a Telegram message (and attach the PDF report) when a training or workflow command finishes.

1. Create a Telegram bot via [BotFather](https://core.telegram.org/bots#botfather) and obtain its *token*.
2. Send a message to your bot from your chat, then grab your *chat-id* (e.g. via https://api.telegram.org/bot<token>/getUpdates).
3. Provide the two credentials **at runtime** as environment variables:

```bash
export TELEGRAM_BOT_TOKEN="<your-bot-token>"
export TELEGRAM_CHAT_ID="<your-chat-id>"
```

⚠️  *Do **not** bake these secrets into the Docker image or commit them to Git.*  Passing them as env-vars keeps the image generic and your credentials safe.

### Example

```bash
docker run --platform=linux/arm64 \
  -e TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN \
  -e TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID \
  -v $HOME/dql_data:/app/data \
  -v $HOME/dql_results:/app/results \
  iad.ocir.io/<your-namespace>/dql-trading:latest \
  initial-workflow \
    --data_file eurusd_1y.csv \
    --experiment_name mem_init \
    --agent_type memory \
    --episodes 50 --n_iter 25 --notify
```

Any command that supports `--notify` (`train`, `initial-workflow`, `tune`) will now deliver a completion message and attach the generated PDF report (if applicable). 