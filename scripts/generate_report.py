#!/usr/bin/env python3
"""
Report Generation Script for DQL Trading Agent

This script generates a comprehensive PDF report for a trained DQL trading agent.
It includes performance metrics, visualizations, and comparisons with baseline strategies.

Usage:
    python scripts/generate_report.py --experiment my_experiment
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import traceback
from reporting.reporting import TradingReport
from utils.metrics import calculate_trading_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("report_generation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("report_generator")

def load_metrics(results_dir, experiment_name):
    """
    Load saved metrics from training and testing runs.
    
    Parameters:
    -----------
    results_dir : str
        Directory where results are stored
    experiment_name : str
        Name of the experiment
        
    Returns:
    --------
    dict
        Dictionary containing training and testing metrics
    """
    experiment_dir = os.path.join(results_dir, experiment_name)
    
    # Validate experiment directory exists
    if not os.path.exists(experiment_dir):
        logger.error(f"Experiment directory not found: {experiment_dir}")
        raise FileNotFoundError(f"No experiment directory found at {experiment_dir}")
    
    # Initialize metrics dictionaries
    train_metrics = {}
    test_metrics = None
    hyperparams = {}
    
    logger.info(f"Loading metrics from {experiment_dir}")
    
    # Load training metrics if available
    train_metrics_path = os.path.join(experiment_dir, "train_metrics.json")
    if os.path.exists(train_metrics_path):
        try:
            with open(train_metrics_path, 'r') as f:
                train_metrics = json.load(f)
            logger.info(f"Loaded training metrics from {train_metrics_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing train_metrics.json: {e}")
            logger.warning("Will attempt to use alternative sources for metrics")
    else:
        train_metrics_path = os.path.join(experiment_dir, "train_metrics.csv")
        if os.path.exists(train_metrics_path):
            try:
                train_metrics_df = pd.read_csv(train_metrics_path)
                # Get the last row for final metrics
                train_metrics = train_metrics_df.iloc[-1].to_dict()
                logger.info(f"Loaded training metrics from CSV: {train_metrics_path}")
            except Exception as e:
                logger.error(f"Error loading train_metrics.csv: {e}")
                logger.warning("Will use minimal placeholder metrics")
    
    # If no training metrics, try to extract some info from other files
    if not train_metrics and os.path.exists(os.path.join(experiment_dir, "training_plots.png")):
        # At least we have training plots, so let's create a minimal train_metrics
        train_metrics = {
            "total_return_pct": 0.0,  # Placeholder
            "sharpe_ratio": 0.0,      # Placeholder
            "max_drawdown": 0.0,      # Placeholder
            "win_rate": 0.0,          # Placeholder
            "training_completed": True
        }
        logger.warning("No training metrics file found. Using minimal placeholder metrics.")
    
    # Load test metrics
    test_metrics_path = os.path.join(experiment_dir, "test_metrics.json")
    if os.path.exists(test_metrics_path):
        try:
            with open(test_metrics_path, 'r') as f:
                test_metrics = json.load(f)
            logger.info(f"Loaded test metrics from {test_metrics_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing test_metrics.json: {e}")
    else:
        test_metrics_path = os.path.join(experiment_dir, "test_metrics.csv")
        if os.path.exists(test_metrics_path):
            try:
                test_metrics_df = pd.read_csv(test_metrics_path)
                # Get the last row for final metrics
                test_metrics = test_metrics_df.iloc[-1].to_dict()
                logger.info(f"Loaded test metrics from CSV: {test_metrics_path}")
            except Exception as e:
                logger.error(f"Error parsing test metrics CSV: {e}")
                logger.warning("Attempting manual parsing of CSV file")
                # Try to parse the file manually
                try:
                    with open(test_metrics_path, 'r') as f:
                        lines = f.readlines()
                        if len(lines) >= 2:
                            headers = lines[0].strip().split(',')
                            values = lines[1].strip().split(',')
                            if len(headers) == len(values):
                                test_metrics = {headers[i]: values[i] for i in range(len(headers))}
                                # Convert numeric values
                                for k, v in test_metrics.items():
                                    try:
                                        test_metrics[k] = float(v) if v else 0
                                    except ValueError:
                                        pass
                                logger.info("Successfully parsed test metrics manually")
                except Exception as e:
                    logger.error(f"Manual parsing also failed: {e}")
    
    # Load strategy comparison if available
    strategy_path = os.path.join(experiment_dir, "strategy_comparison.csv")
    if os.path.exists(strategy_path):
        try:
            strategy_df = pd.read_csv(strategy_path)
            # Add strategy comparison data to test metrics
            if test_metrics is None:
                test_metrics = {}
            
            # Find DQL Agent row
            dql_row = strategy_df[strategy_df['Strategy'] == 'DQL Agent']
            if not dql_row.empty:
                for col in strategy_df.columns:
                    if col != 'Strategy':
                        # Clean column name
                        clean_col = col.lower().replace(' ', '_').replace('(%)', '')
                        test_metrics[clean_col] = dql_row[col].values[0]
                logger.info("Added strategy comparison metrics to test metrics")
        except Exception as e:
            logger.error(f"Could not parse strategy comparison CSV: {e}")
    
    # Try to extract hyperparameters from model config or optimal parameters
    model_path = os.path.join(experiment_dir, "model.pth")
    if os.path.exists(model_path):
        hyperparams["model_available"] = True
        logger.info("Model file found for this experiment")
    
    # Check for hyperparameter file
    hyper_path = os.path.join(experiment_dir, "hyperparameters.json")
    if os.path.exists(hyper_path):
        try:
            with open(hyper_path, 'r') as f:
                hyperparams.update(json.load(f))
            logger.info(f"Loaded hyperparameters from {hyper_path}")
        except Exception as e:
            logger.error(f"Error loading hyperparameters.json: {e}")
    
    # If test metrics exist but training metrics don't, copy some metrics from test to train
    if test_metrics and not train_metrics:
        logger.warning("No training metrics found, using test metrics as proxy")
        train_metrics = {
            "total_return_pct": test_metrics.get("total_return_pct", 0.0),
            "sharpe_ratio": test_metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": test_metrics.get("max_drawdown", 0.0),
            "win_rate": test_metrics.get("win_rate", 0.0),
            "training_completed": True
        }
    
    # Summary of what was loaded
    metrics_summary = {
        "has_train_metrics": bool(train_metrics),
        "has_test_metrics": bool(test_metrics),
        "has_hyperparams": bool(hyperparams)
    }
    logger.info(f"Metrics loading summary: {metrics_summary}")
    
    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "hyperparams": hyperparams
    }

def find_visualization_files(results_dir, experiment_name):
    """
    Find visualization files generated during training and testing.
    
    Parameters:
    -----------
    results_dir : str
        Directory where results are stored
    experiment_name : str
        Name of the experiment
        
    Returns:
    --------
    dict
        Dictionary containing paths to visualization files
    """
    experiment_dir = os.path.join(results_dir, experiment_name)
    
    # Validate experiment directory exists
    if not os.path.exists(experiment_dir):
        logger.error(f"Experiment directory not found: {experiment_dir}")
        raise FileNotFoundError(f"No experiment directory found at {experiment_dir}")
    
    # Common visualization filenames
    viz_files = {
        "training_performance": None,
        "learning_curves": None,
        "test_performance": None,
        "trade_visualization": None,
        "strategy_comparison": None
    }
    
    logger.info(f"Searching for visualization files in {experiment_dir}")
    
    # Check if directory is empty
    if not os.listdir(experiment_dir):
        logger.error(f"Experiment directory is empty: {experiment_dir}")
        raise ValueError(f"Experiment directory is empty: {experiment_dir}")
    
    # Look for training visualizations
    for filename in os.listdir(experiment_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            filepath = os.path.join(experiment_dir, filename)
            
            # Training performance plots
            if "training_plots" in filename.lower():
                viz_files["training_performance"] = filepath
                logger.info(f"Found training performance plot: {filename}")
            elif "training_performance" in filename.lower():
                viz_files["training_performance"] = filepath
                logger.info(f"Found training performance plot: {filename}")
                
            # Learning curves
            if "learning_curves" in filename.lower():
                viz_files["learning_curves"] = filepath
                logger.info(f"Found learning curves plot: {filename}")
                
            # Test performance  
            if "performance_dashboard" in filename.lower():
                viz_files["test_performance"] = filepath
                logger.info(f"Found test performance plot: {filename}")
            elif "test_performance" in filename.lower():
                viz_files["test_performance"] = filepath
                logger.info(f"Found test performance plot: {filename}")
                
            # Trade visualization
            if "test_trades" in filename.lower():
                viz_files["trade_visualization"] = filepath
                logger.info(f"Found trade visualization plot: {filename}")
            elif "trade_visualization" in filename.lower():
                viz_files["trade_visualization"] = filepath
                logger.info(f"Found trade visualization plot: {filename}")
                
            # Strategy comparison
            if "strategies_comparison" in filename.lower():
                viz_files["strategy_comparison"] = filepath
                logger.info(f"Found strategy comparison plot: {filename}")
            elif "strategy_comparison" in filename.lower() and filename.endswith(".png"):
                viz_files["strategy_comparison"] = filepath
                logger.info(f"Found strategy comparison plot: {filename}")
    
    # If training performance is still missing, look for episode trades
    if not viz_files["training_performance"]:
        for filename in os.listdir(experiment_dir):
            if filename.startswith("trades_episode_") and filename.endswith(".png"):
                viz_files["training_performance"] = os.path.join(experiment_dir, filename)
                logger.info(f"Using {filename} as training performance visualization")
                break
    
    # Check if we need fallbacks for missing visualizations
    if not viz_files["test_performance"] and viz_files["trade_visualization"]:
        viz_files["test_performance"] = viz_files["trade_visualization"]
        logger.info("Using trade visualization as fallback for test performance")
    
    if not viz_files["trade_visualization"] and viz_files["test_performance"]:
        viz_files["trade_visualization"] = viz_files["test_performance"]
        logger.info("Using test performance as fallback for trade visualization")
    
    # Check if we found the required visualizations
    missing_viz = [k for k, v in viz_files.items() if v is None]
    if missing_viz:
        logger.warning(f"Could not find the following visualizations: {', '.join(missing_viz)}")
    
    return viz_files

def load_baseline_comparison(results_dir, experiment_name):
    """
    Load baseline comparison metrics if available.
    
    Parameters:
    -----------
    results_dir : str
        Directory where results are stored
    experiment_name : str
        Name of the experiment
        
    Returns:
    --------
    dict or None
        Dictionary containing baseline metrics or None if not available
    """
    experiment_dir = os.path.join(results_dir, experiment_name)
    
    # Check for baseline comparison CSV
    baseline_path = os.path.join(experiment_dir, "strategy_comparison.csv")
    if os.path.exists(baseline_path):
        try:
            baseline_df = pd.read_csv(baseline_path)
            
            # Convert to dict format required by TradingReport
            baseline_metrics = {}
            for _, row in baseline_df.iterrows():
                strategy_name = row["Strategy"]
                if strategy_name != "DQL Agent":  # Skip the agent itself
                    baseline_metrics[strategy_name] = {}
                    for col in baseline_df.columns:
                        if col != "Strategy":
                            try:
                                # Try to convert to float
                                value = float(row[col])
                                
                                # Clean column name and add to metrics
                                clean_col = col.lower().replace(' ', '_').replace('(%)', '')
                                baseline_metrics[strategy_name][clean_col] = value
                            except (ValueError, TypeError):
                                # If conversion fails, just use the string value
                                baseline_metrics[strategy_name][col] = row[col]
            
            return baseline_metrics
        except Exception as e:
            print(f"Warning: Could not parse baseline comparison: {e}")
    
    return None

def format_summary_metrics(metrics):
    """Format metrics for the executive summary section"""
    summary = {}
    
    # Check for common metrics
    if metrics:
        if "total_return_pct" in metrics:
            try:
                summary["total_return"] = f"{float(metrics['total_return_pct']):.2f}%"
            except (ValueError, TypeError):
                summary["total_return"] = metrics['total_return_pct']
        elif "return_pct" in metrics:
            try:
                summary["total_return"] = f"{float(metrics['return_pct']):.2f}%"
            except (ValueError, TypeError):
                summary["total_return"] = metrics['return_pct']
            
        if "sharpe_ratio" in metrics:
            try:
                summary["sharpe_ratio"] = f"{float(metrics['sharpe_ratio']):.4f}"
            except (ValueError, TypeError):
                summary["sharpe_ratio"] = metrics['sharpe_ratio']
            
        if "max_drawdown" in metrics:
            try:
                value = float(metrics['max_drawdown'])
                # Check if already in percentage format
                if abs(value) <= 1.0:
                    summary["max_drawdown"] = f"{value * 100:.2f}%"
                else:
                    summary["max_drawdown"] = f"{value:.2f}%"
            except (ValueError, TypeError):
                summary["max_drawdown"] = metrics['max_drawdown']
                
        if "win_rate" in metrics:
            try:
                value = float(metrics['win_rate'])
                # Check if already in percentage format
                if value <= 1.0:
                    summary["win_rate"] = f"{value * 100:.2f}%"
                else:
                    summary["win_rate"] = f"{value:.2f}%"
            except (ValueError, TypeError):
                summary["win_rate"] = metrics['win_rate']
                
        if "total_trades" in metrics:
            summary["total_trades"] = metrics["total_trades"]
    
    return summary

def generate_key_findings(train_metrics, test_metrics, baseline_metrics=None):
    """Generate key findings based on metrics"""
    findings = []
    
    # Test performance
    if test_metrics:
        try:
            if "total_return_pct" in test_metrics:
                findings.append(f"The DQL agent achieved a {float(test_metrics['total_return_pct']):.2f}% return on the test set.")
            elif "return_pct" in test_metrics:
                findings.append(f"The DQL agent achieved a {float(test_metrics['return_pct']):.2f}% return on the test set.")
        except (ValueError, TypeError):
            findings.append("The DQL agent was tested on unseen data.")
            
        try:
            if "win_rate" in test_metrics:
                win_rate = float(test_metrics["win_rate"])
                win_rate = win_rate * 100 if win_rate <= 1.0 else win_rate
                findings.append(f"The agent demonstrated a win rate of {win_rate:.2f}%.")
        except (ValueError, TypeError):
            pass
            
        try:
            if "sharpe_ratio" in test_metrics:
                findings.append(f"Risk-adjusted performance (Sharpe ratio) was {float(test_metrics['sharpe_ratio']):.4f}.")
        except (ValueError, TypeError):
            pass
    
    # Compare with baseline
    if baseline_metrics and test_metrics:
        try:
            # Find best baseline
            best_baseline = None
            best_return = -float('inf')
            
            for name, metrics in baseline_metrics.items():
                try:
                    if "total_return_pct" in metrics and float(metrics["total_return_pct"]) > best_return:
                        best_return = float(metrics["total_return_pct"])
                        best_baseline = name
                    elif "return_pct" in metrics and float(metrics["return_pct"]) > best_return:
                        best_return = float(metrics["return_pct"])
                        best_baseline = name
                except (ValueError, TypeError):
                    continue
            
            if best_baseline:
                try:
                    agent_return = float(test_metrics.get("total_return_pct", test_metrics.get("return_pct", 0)))
                    
                    if agent_return > best_return:
                        findings.append(f"The DQL agent outperformed the best baseline strategy ({best_baseline}) by {agent_return - best_return:.2f}%.")
                    else:
                        findings.append(f"The best baseline strategy ({best_baseline}) outperformed the DQL agent by {best_return - agent_return:.2f}%.")
                except (ValueError, TypeError):
                    findings.append(f"The DQL agent was compared against several baseline strategies including {best_baseline}.")
        except Exception as e:
            print(f"Warning: Could not generate baseline comparison finding: {e}")
    
    # Training insights
    if train_metrics and test_metrics:
        try:
            train_return = float(train_metrics.get("total_return_pct", train_metrics.get("return_pct", 0)))
            test_return = float(test_metrics.get("total_return_pct", test_metrics.get("return_pct", 0)))
            
            if abs(train_return - test_return) > 5.0:
                findings.append(f"There is a {abs(train_return - test_return):.2f}% gap between training and testing returns, suggesting potential overfitting.")
        except (ValueError, TypeError):
            pass
    
    # If no findings were generated, add some generic ones
    if not findings:
        findings.append("The experiment tested a Deep Q-Learning agent for trading.")
        findings.append("The agent was trained and tested on market data to evaluate its performance.")
    
    return findings

def generate_recommendations(metrics):
    """Generate recommendations based on performance metrics"""
    recommendations = []
    
    # Default recommendations
    recommendations.append("Further tune hyperparameters to optimize agent performance.")
    recommendations.append("Test the agent on different market conditions and timeframes.")
    
    # Add specific recommendations based on metrics
    if metrics.get("test_metrics"):
        test_metrics = metrics["test_metrics"]
        
        # Sharpe ratio recommendations
        try:
            if "sharpe_ratio" in test_metrics and float(test_metrics["sharpe_ratio"]) < 1.0:
                recommendations.append("Focus on improving risk-adjusted returns by adjusting the reward function to penalize volatility.")
        except (ValueError, TypeError):
            pass
        
        # Win rate recommendations
        try:
            if "win_rate" in test_metrics:
                win_rate = float(test_metrics["win_rate"])
                win_rate = win_rate * 100 if win_rate <= 1.0 else win_rate
                if win_rate < 50:
                    recommendations.append("Consider implementing a more robust entry/exit strategy to improve win rate.")
        except (ValueError, TypeError):
            pass
        
        # Return recommendations
        try:
            return_key = "total_return_pct" if "total_return_pct" in test_metrics else "return_pct"
            if return_key in test_metrics and float(test_metrics[return_key]) < 0:
                recommendations.append("Investigate alternative feature engineering and state representations to capture more market signals.")
        except (ValueError, TypeError):
            pass
    
    return recommendations

def generate_report(experiment_name, results_dir="results", output_dir=None):
    """
    Generate a comprehensive report for the specified experiment.
    
    Parameters:
    -----------
    experiment_name : str
        Name of the experiment
    results_dir : str
        Directory where results are stored
    output_dir : str, optional
        Directory where to save the report (defaults to results_dir/experiment_name)
        
    Returns:
    --------
    str
        Path to the generated report
    """
    logger.info(f"Generating report for experiment: {experiment_name}")
    
    # Parameter validation
    if not experiment_name:
        logger.error("Experiment name cannot be empty")
        raise ValueError("Experiment name cannot be empty")
    
    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(results_dir, experiment_name)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory set to: {output_dir}")
        
        # Load metrics and visualizations
        try:
            metrics = load_metrics(results_dir, experiment_name)
            viz_files = find_visualization_files(results_dir, experiment_name)
            baseline_metrics = load_baseline_comparison(results_dir, experiment_name)
        except Exception as e:
            logger.error(f"Error loading experiment data: {e}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to load experiment data: {str(e)}")
        
        # Initialize report
        try:
            report = TradingReport(
                experiment_name=experiment_name,
                output_dir=output_dir
            )
            logger.info("Initialized TradingReport object")
        except Exception as e:
            logger.error(f"Failed to initialize report: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize report: {str(e)}")
        
        # Add title page
        try:
            report.add_title_page(
                title=f"DQL Trading Agent: {experiment_name}",
                subtitle="Performance Analysis Report",
                date=datetime.now().strftime("%Y-%m-%d")
            )
            logger.info("Added title page to report")
        except Exception as e:
            logger.error(f"Error adding title page: {e}")
            logger.warning("Will continue without title page")
        
        # Format metrics for executive summary
        train_summary = format_summary_metrics(metrics["train_metrics"])
        test_summary = format_summary_metrics(metrics["test_metrics"]) if metrics["test_metrics"] else None
        
        # Add executive summary
        try:
            report.add_executive_summary(
                train_summary=train_summary,
                test_summary=test_summary
            )
            logger.info("Added executive summary to report")
        except Exception as e:
            logger.error(f"Error adding executive summary: {e}")
            logger.warning("Will continue without executive summary")
        
        # Add training metrics
        if metrics["train_metrics"] and viz_files["training_performance"]:
            try:
                report.add_training_metrics(
                    metrics=metrics["train_metrics"],
                    training_plots_path=viz_files["training_performance"],
                    learning_curves_path=viz_files["learning_curves"]
                )
                logger.info("Added training metrics to report")
            except Exception as e:
                logger.error(f"Error adding training metrics: {e}")
                logger.warning("Will continue without training metrics section")
        
        # Add testing metrics
        if metrics["test_metrics"] and viz_files["test_performance"]:
            try:
                report.add_testing_metrics(
                    metrics=metrics["test_metrics"],
                    performance_dashboard_path=viz_files["test_performance"],
                    trade_visualization_path=viz_files["trade_visualization"] or viz_files["test_performance"]
                )
                logger.info("Added testing metrics to report")
            except Exception as e:
                logger.error(f"Error adding testing metrics: {e}")
                logger.warning("Will continue without testing metrics section")
        
        # Add baseline comparison if available
        if baseline_metrics and metrics["test_metrics"] and viz_files["strategy_comparison"]:
            try:
                report.add_baseline_comparison(
                    dql_metrics=metrics["test_metrics"],
                    baseline_metrics=baseline_metrics,
                    comparison_chart_path=viz_files["strategy_comparison"]
                )
                logger.info("Added baseline comparison to report")
            except Exception as e:
                logger.error(f"Error adding baseline comparison: {e}")
                logger.warning("Will continue without baseline comparison section")
        
        # Add hyperparameter analysis
        if metrics["hyperparams"]:
            try:
                # Format hyperparameters for the report
                formatted_hyperparams = {}
                
                # Common hyperparameter groupings
                if "agent_params" in metrics["hyperparams"]:
                    formatted_hyperparams["Agent Parameters"] = metrics["hyperparams"]["agent_params"]
                elif any(k in metrics["hyperparams"] for k in ["learning_rate", "gamma", "epsilon"]):
                    # Extract agent parameters
                    agent_params = {}
                    for k in ["learning_rate", "gamma", "epsilon", "epsilon_min", "epsilon_decay", 
                            "batch_size", "buffer_size", "memory_size", "hidden_layers", "update_target_every"]:
                        if k in metrics["hyperparams"]:
                            agent_params[k] = metrics["hyperparams"][k]
                    if agent_params:
                        formatted_hyperparams["Agent Parameters"] = agent_params
                
                if "env_params" in metrics["hyperparams"]:
                    formatted_hyperparams["Environment Parameters"] = metrics["hyperparams"]["env_params"]
                elif any(k in metrics["hyperparams"] for k in ["initial_amount", "transaction_cost", "reward_scaling"]):
                    # Extract environment parameters
                    env_params = {}
                    for k in ["initial_amount", "transaction_cost", "reward_scaling", "window_size"]:
                        if k in metrics["hyperparams"]:
                            env_params[k] = metrics["hyperparams"][k]
                    if env_params:
                        formatted_hyperparams["Environment Parameters"] = env_params
                
                # Add trader config if available
                if "trader_config" in metrics["hyperparams"]:
                    formatted_hyperparams["Trader Configuration"] = metrics["hyperparams"]["trader_config"]
                
                # If no grouping was possible, use raw hyperparameters
                if not formatted_hyperparams and metrics["hyperparams"]:
                    formatted_hyperparams["Model Configuration"] = metrics["hyperparams"]
                
                # If we have no hyperparameters but we know the model exists
                if not formatted_hyperparams and metrics["hyperparams"].get("model_available"):
                    formatted_hyperparams["Model Configuration"] = {
                        "model_file": "model.pth",
                        "model_saved": True
                    }
                
                if formatted_hyperparams:
                    report.add_hyperparameter_analysis(formatted_hyperparams)
                    logger.info("Added hyperparameter analysis to report")
            except Exception as e:
                logger.error(f"Error adding hyperparameter analysis: {e}")
                logger.warning("Will continue without hyperparameter analysis section")
        
        # Generate key findings and recommendations
        try:
            key_findings = generate_key_findings(
                metrics["train_metrics"], 
                metrics["test_metrics"],
                baseline_metrics
            )
            
            recommendations = generate_recommendations(metrics)
            
            # Add conclusion
            report.add_conclusion(
                key_findings=key_findings,
                recommendations=recommendations
            )
            logger.info("Added conclusion to report")
        except Exception as e:
            logger.error(f"Error adding conclusion: {e}")
            logger.warning("Will continue without conclusion section")
        
        # Generate the report
        try:
            report_path = report.generate()
            logger.info(f"Report generated successfully: {report_path}")
            
            # Verify the report file was created
            if not os.path.exists(report_path):
                logger.warning(f"Report file not found at expected path: {report_path}")
            else:
                logger.info(f"Report file size: {os.path.getsize(report_path)} bytes")
                
            return report_path
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to generate report: {str(e)}")
            
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        logger.error(traceback.format_exc())
        raise

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate performance report for DQL trading agent")
    parser.add_argument("--experiment", "-e", type=str, required=True, 
                        help="Name of the experiment to generate a report for")
    parser.add_argument("--results_dir", "-r", type=str, default="results",
                        help="Directory containing experiment results")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Directory to save the report (defaults to results_dir/experiment_name)")
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        default="INFO", help="Set logging level")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        report_path = generate_report(
            experiment_name=args.experiment,
            results_dir=args.results_dir,
            output_dir=args.output_dir
        )
        logger.info(f"Report successfully generated at: {report_path}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 