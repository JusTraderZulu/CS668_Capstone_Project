# DQL Trading Agent Reporting System

## Overview

The DQL Trading Agent Reporting System provides a comprehensive solution for generating professional PDF reports for trading agents. This system analyzes experiment results and automatically creates detailed performance reports with visualizations, metrics, and analysis.

## Features

- Generate professional PDF reports with consistent formatting
- Include title pages, headers, and footers
- Automatically extract and format metrics from experiment results
- Include visualizations: training curves, performance charts, and trade visualizations
- Compare agent performance against baseline strategies
- Generate data-driven conclusions and recommendations
- Compatible with the DQL agent training pipeline

## Installation Requirements

The reporting system requires the following dependencies:

```
reportlab
numpy
pandas
PIL (Pillow)
matplotlib
fpdf
```

These can be installed via pip:

```bash
pip install reportlab numpy pandas pillow matplotlib fpdf
```

## Usage

### Generating Reports from Experiment Results

The easiest way to generate a report is to use the `generate_report.py` script, which automatically analyzes your experiment results:

```bash
python generate_report.py --experiment <experiment_name>
```

Where `<experiment_name>` is the name of the experiment folder in your results directory.

Additional options:
- `--results_dir` or `-r`: Specify a custom results directory (default: "results")
- `--output_dir` or `-o`: Specify a custom output directory for the report

Example:
```bash
python generate_report.py --experiment optimized_dql_test --output_dir reports
```

### Integrating Reporting into Your Training Pipeline

You can also integrate report generation directly into your training and evaluation scripts:

```python
from reporting import TradingReport

# Initialize report
report = TradingReport(
    experiment_name="my_experiment",
    output_dir="results/my_experiment"
)

# Add report components
report.add_title_page(
    title="DQL Trading Agent",
    subtitle="Performance Analysis",
    date="2023-04-24"
)

report.add_executive_summary(
    train_summary={
        'total_return': "14.3%",
        'sharpe_ratio': "1.42",
        'win_rate': "62%",
        'total_trades': 186
    },
    test_summary={
        'total_return': "15.7%",
        'sharpe_ratio': "1.65",
        'win_rate': "68%",
        'total_trades': 42
    }
)

# Add training metrics
report.add_training_metrics(
    metrics=training_metrics,
    training_plots_path="path/to/training_plots.png",
    learning_curves_path="path/to/learning_curves.png"
)

# Generate the PDF
report_path = report.generate()
print(f"Report generated at: {report_path}")
```

## Expected Report Structure

The generated PDF report includes:

1. **Title Page**: With experiment name, date, and subtitle
2. **Executive Summary**: Key performance metrics at a glance
3. **Training Metrics**: Detailed analysis of the training process
4. **Testing Results**: Performance on unseen data
5. **Baseline Comparison**: How the agent compares to traditional strategies
6. **Hyperparameter Analysis**: Documentation of the agent's configuration
7. **Conclusion**: Key findings and recommendations

## File Structure Requirements

For best results, your experiment directory should include:

- `test_metrics.csv`: Testing metrics data
- `training_plots.png` or similar: Visualization of training performance
- `learning_curves.png` or similar: Visualization of learning progress
- `performance_dashboard.png` or similar: Test performance visualization
- `test_trades.png` or similar: Visualization of trade actions
- `strategies_comparison.png` or similar: Comparison with baseline strategies
- `strategy_comparison.csv`: CSV file with baseline comparison data

The report generator will automatically detect these files and include them in the report. 