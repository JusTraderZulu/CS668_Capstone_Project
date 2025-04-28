import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from fpdf import FPDF
import json
import numpy as np
from typing import Dict, List, Optional, Any, Union
from PIL import Image
import textwrap

class TradingReportPDF(FPDF):
    """Custom PDF class for trading reports with headers and footers"""
    
    def __init__(self, experiment_name: str):
        super().__init__()
        self.experiment_name = experiment_name
        # Set up document properties
        self.set_author("DQL Trading Agent")
        self.set_creator("DQL Trading Agent")
        self.set_title(f"Trading Report: {experiment_name}")
        # Page setup
        self.set_margins(15, 15, 15)
        self.set_auto_page_break(True, margin=15)
        # Set colors
        self.primary_color = (0, 32, 96)  # Dark blue
        self.secondary_color = (0, 112, 192)  # Light blue
        self.accent_color = (208, 206, 206)  # Light gray
        
    def header(self):
        """Add header to each page"""
        self.set_font("Arial", "B", 10)
        self.set_text_color(*self.primary_color)
        # Header line
        self.cell(0, 10, f"DQL Trading Agent - {self.experiment_name}", 0, 1, "R")
        self.line(15, 15, 195, 15)
        # Reset position
        self.ln(5)
        
    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(*self.primary_color)
        # Footer line
        self.line(15, 282, 195, 282)
        # Page number
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")
        # Date
        self.cell(0, 10, datetime.now().strftime("%Y-%m-%d"), 0, 0, "R")
        
    def chapter_title(self, title: str):
        """Add a chapter title to the document"""
        self.set_font("Arial", "B", 16)
        self.set_text_color(*self.primary_color)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(4)
        
    def section_title(self, title: str):
        """Add a section title to the document"""
        self.set_font("Arial", "B", 12)
        self.set_text_color(*self.secondary_color)
        self.cell(0, 6, title, 0, 1, "L")
        self.ln(2)
        
    def text(self, text: str):
        """Add paragraph text to the document"""
        self.set_font("Arial", "", 10)
        self.set_text_color(0, 0, 0)  # Black
        self.multi_cell(0, 5, text)
        self.ln(2)
        
    def add_metric_table(self, metrics: Dict[str, Any], title: str = None):
        """Add a metric table to the document"""
        if title:
            self.section_title(title)
            
        # Setup table
        self.set_font("Arial", "B", 10)
        self.set_text_color(0, 0, 0)  # Black
        self.set_fill_color(*self.accent_color)
        
        # Table headers
        col_width = 90
        row_height = 7
        self.cell(col_width, row_height, "Metric", 1, 0, "C", True)
        self.cell(col_width, row_height, "Value", 1, 1, "C", True)
        
        # Table rows
        self.set_font("Arial", "", 10)
        # Ensure consistent order
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[0])
        
        # Alternate row colors
        fill = False
        for metric, value in sorted_metrics:
            # Format the metric name nicely
            metric_name = " ".join(word.capitalize() for word in metric.split("_"))
            
            # Format the value based on type
            if isinstance(value, float):
                # Display percentages nicely
                if "percent" in metric.lower() or "rate" in metric.lower() or "ratio" in metric.lower():
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
                
            # Add the row
            self.cell(col_width, row_height, metric_name, 1, 0, "L", fill)
            self.cell(col_width, row_height, formatted_value, 1, 1, "R", fill)
            
            # Toggle fill color
            fill = not fill
            
        self.ln(5)
        
    def add_image(self, img_path: str, caption: str = None, w: int = 180, h: int = 90):
        """Add an image with optional caption to the document"""
        # Check if image exists
        if not os.path.exists(img_path):
            self.text(f"Image not found: {img_path}")
            return
            
        # Add image
        self.image(img_path, x=15, y=None, w=w, h=h)
        self.ln(2)
        
        # Add caption if provided
        if caption:
            self.set_font("Arial", "I", 9)
            self.set_text_color(0, 0, 0)  # Black
            self.cell(0, 5, caption, 0, 1, "C")
            
        self.ln(5)
        
    def comparison_table(self, strategies: Dict[str, Dict[str, Any]], metrics: List[str], title: str = None):
        """Add a comparison table for multiple strategies"""
        if title:
            self.section_title(title)
            
        # Calculate column width
        num_strategies = len(strategies)
        metric_col_width = 60
        strategy_col_width = int((180 - metric_col_width) / num_strategies)
        row_height = 7
        
        # Setup table
        self.set_font("Arial", "B", 10)
        self.set_text_color(0, 0, 0)  # Black
        self.set_fill_color(*self.accent_color)
        
        # Table headers
        self.cell(metric_col_width, row_height, "Metric", 1, 0, "C", True)
        for strategy_name in strategies.keys():
            self.cell(strategy_col_width, row_height, strategy_name, 1, 0, "C", True)
        self.ln()
        
        # Table rows
        self.set_font("Arial", "", 10)
        fill = False
        
        for metric in metrics:
            # Format the metric name nicely
            metric_name = " ".join(word.capitalize() for word in metric.split("_"))
            self.cell(metric_col_width, row_height, metric_name, 1, 0, "L", fill)
            
            for strategy_name, strategy_metrics in strategies.items():
                value = strategy_metrics.get(metric, "N/A")
                
                # Format the value based on type
                if isinstance(value, float):
                    # Display percentages nicely
                    if "percent" in metric.lower() or "rate" in metric.lower() or "ratio" in metric.lower():
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                    
                self.cell(strategy_col_width, row_height, formatted_value, 1, 0, "R", fill)
                
            self.ln()
            fill = not fill
            
        self.ln(5)


class TradingReport:
    """Class for generating comprehensive trading strategy reports in PDF format"""
    
    def __init__(self, experiment_name: str, output_dir: str = "results"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.pdf = TradingReportPDF(experiment_name)
        self.report_path = os.path.join(output_dir, f"{experiment_name}_report.pdf")
        
        # --- Tracking for dynamic table of contents --------------------
        # key -> starting page number where section begins
        self._section_pages: Dict[str, int] = {}
        # page number where the TOC placeholder lives (inserted right after
        #   the title page).  We fill it at generate() time.
        self._toc_page_no: Optional[int] = None
        
    def add_title_page(self, title: str, subtitle: str = None, date: str = None):
        """Add a title page to the report

        Immediately after creating the title page we insert *another* blank
        page that will hold the Table-of-Contents.  We record its page number
        so that we can come back and fill it in when `generate()` is called.
        """

        # --- Title Page (page 1) --------------------------------------
        self.pdf.add_page()
        
        # Add logo if available
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
        if os.path.exists(logo_path):
            self.pdf.image(logo_path, x=80, y=40, w=50)
            y_pos = 100
        else:
            y_pos = 80
        
        # Add title
        self.pdf.set_y(y_pos)
        self.pdf.set_font("Arial", "B", 24)
        self.pdf.set_text_color(*self.pdf.primary_color)
        self.pdf.cell(0, 10, title, 0, 1, "C")
        
        # Add subtitle if provided
        if subtitle:
            self.pdf.set_font("Arial", "B", 16)
            self.pdf.set_text_color(*self.pdf.secondary_color)
            self.pdf.ln(5)
            self.pdf.cell(0, 10, subtitle, 0, 1, "C")
        
        # Add date if provided
        if date:
            self.pdf.set_font("Arial", "", 12)
            self.pdf.set_text_color(0, 0, 0)  # Black
            self.pdf.ln(10)
            self.pdf.cell(0, 10, f"Generated on: {date}", 0, 1, "C")
        
        # --- Insert TOC placeholder (page 2) ---------------------------
        self.pdf.add_page()
        self._toc_page_no = self.pdf.page_no()
        
    def add_executive_summary(self, train_summary: Dict[str, Any], test_summary: Optional[Dict[str, Any]] = None):
        """Add an executive summary section to the report"""
        self.pdf.add_page()
        # Record starting page for TOC
        self._section_pages["Executive Summary"] = self.pdf.page_no()

        self.pdf.chapter_title("Executive Summary")
        
        # Introduction text
        self.pdf.text(
            "This report presents the performance analysis of a Deep Q-Learning (DQL) agent "
            f"trained to trade in forex markets. The agent was trained as part of experiment '{self.experiment_name}'."
        )
        
        # Training performance summary
        self.pdf.section_title("Training Performance Summary")
        self.pdf.text("The agent achieved the following results during the training phase:")
        
        # Format training metrics as a simple table
        col_width = 90
        row_height = 7
        self.pdf.set_font("Arial", "B", 10)
        self.pdf.set_fill_color(*self.pdf.accent_color)
        
        for metric, value in train_summary.items():
            metric_name = " ".join(word.capitalize() for word in metric.split("_"))
            self.pdf.cell(col_width, row_height, metric_name, 1, 0, "L")
            self.pdf.cell(col_width, row_height, str(value), 1, 1, "R")
        
        self.pdf.ln(5)
        
        # Test performance summary if available
        if test_summary:
            self.pdf.section_title("Test Performance Summary")
            self.pdf.text("The agent achieved the following results on unseen test data:")
            
            # Format test metrics as a simple table
            for metric, value in test_summary.items():
                metric_name = " ".join(word.capitalize() for word in metric.split("_"))
                self.pdf.cell(col_width, row_height, metric_name, 1, 0, "L")
                self.pdf.cell(col_width, row_height, str(value), 1, 1, "R")
                
            self.pdf.ln(5)
        
        # Summary conclusion
        self.pdf.text(
            "The detailed analysis in the following sections provides a comprehensive breakdown of the "
            "agent's performance, training process, and comparison with baseline strategies."
        )
        
    def add_training_metrics(self, metrics: Dict[str, Any], training_plots_path: str, learning_curves_path: str):
        """Add training metrics and visualizations to the report"""
        self.pdf.add_page()
        self._section_pages["Training Performance Analysis"] = self.pdf.page_no()
        self.pdf.chapter_title("Training Performance Analysis")
        
        # Overview text
        self.pdf.text(
            "This section presents the performance metrics and learning progress of the agent during the training phase. "
            "The metrics provide insight into the agent's trading effectiveness, risk management, and learning stability."
        )
        
        # Add detailed metrics table
        self.pdf.add_metric_table(metrics, "Training Performance Metrics")
        
        # Add training plots
        if os.path.exists(training_plots_path):
            self.pdf.add_image(
                training_plots_path,
                caption="Figure 1: Training Performance - Cumulative Returns and Trade Actions",
                h=100
            )
        
        # Add learning curves
        if os.path.exists(learning_curves_path):
            self.pdf.add_image(
                learning_curves_path,
                caption="Figure 2: Learning Curves - Rewards, Loss, and Epsilon Decay",
                h=100
            )
            
        # Analysis text
        if metrics.get('total_return_pct', 0) > 0:
            performance_analysis = "The agent demonstrated positive returns during training, indicating successful learning of profitable trading patterns."
        else:
            performance_analysis = "The agent did not achieve positive returns during training, suggesting further optimization may be needed."
            
        if metrics.get('sharpe_ratio', 0) > 1:
            risk_analysis = "The Sharpe ratio indicates acceptable risk-adjusted returns."
        else:
            risk_analysis = "The low Sharpe ratio suggests the strategy may not provide adequate risk-adjusted returns."
            
        self.pdf.section_title("Analysis")
        self.pdf.text(performance_analysis + " " + risk_analysis)
        
    def add_testing_metrics(self, metrics: Dict[str, Any], performance_dashboard_path: str, trade_visualization_path: str):
        """Add testing metrics and visualizations to the report"""
        self.pdf.add_page()
        self._section_pages["Out-of-Sample Testing Results"] = self.pdf.page_no()
        self.pdf.chapter_title("Out-of-Sample Testing Results")
        
        # Overview text
        self.pdf.text(
            "This section evaluates the agent's performance on previously unseen test data. "
            "Out-of-sample testing provides insight into the agent's ability to generalize its trading strategy "
            "beyond the data it was trained on."
        )
        
        # Add detailed metrics table
        self.pdf.add_metric_table(metrics, "Test Performance Metrics")
        
        # Add performance dashboard
        if os.path.exists(performance_dashboard_path):
            self.pdf.add_image(
                performance_dashboard_path,
                caption="Figure 3: Test Performance Dashboard",
                h=120
            )
        
        # Add trade visualization
        if os.path.exists(trade_visualization_path):
            self.pdf.add_image(
                trade_visualization_path,
                caption="Figure 4: Test Data Trade Visualization",
                h=100
            )
            
        # Analysis text
        if metrics.get('total_return_pct', 0) > 0:
            performance_analysis = "The agent successfully generalized to unseen data, achieving positive returns."
        else:
            performance_analysis = "The agent struggled to generalize to unseen data, suggesting potential overfitting to the training set."
            
        self.pdf.section_title("Analysis")
        self.pdf.text(performance_analysis)
        
    def add_baseline_comparison(self, dql_metrics: Dict[str, Any], baseline_metrics: Dict[str, Dict[str, Any]], comparison_chart_path: str):
        """Add baseline comparison to the report"""
        self.pdf.add_page()
        self._section_pages["Comparison with Baseline Strategies"] = self.pdf.page_no()
        self.pdf.chapter_title("Comparison with Baseline Strategies")
        
        # Overview text
        self.pdf.text(
            "This section compares the DQL agent's performance against traditional trading strategies. "
            "This comparison helps evaluate whether the deep reinforcement learning approach provides advantages "
            "over simpler, rule-based strategies."
        )
        
        # Create comparison dictionary
        comparison = {"DQL Agent": dql_metrics}
        comparison.update(baseline_metrics)
        
        # Key metrics to compare
        key_metrics = [
            "total_return_pct",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "total_trades"
        ]
        
        # Add comparison table
        self.pdf.comparison_table(comparison, key_metrics, "Strategy Comparison")
        
        # Add comparison chart
        if os.path.exists(comparison_chart_path):
            self.pdf.add_image(
                comparison_chart_path,
                caption="Figure 5: Cumulative Returns Comparison",
                h=100
            )
            
        # Analysis
        self.pdf.section_title("Analysis")
        
        # Compare DQL to best baseline
        best_baseline = max(baseline_metrics.items(), key=lambda x: x[1].get('total_return_pct', -float('inf')))
        best_name, best_metrics = best_baseline
        
        if dql_metrics.get('total_return_pct', 0) > best_metrics.get('total_return_pct', 0):
            comparison_text = (
                f"The DQL agent outperformed the best baseline strategy ({best_name}) in terms of total returns. "
                "This suggests that the deep reinforcement learning approach successfully captured complex patterns "
                "that rule-based strategies could not exploit."
            )
        else:
            comparison_text = (
                f"The DQL agent was outperformed by the {best_name} strategy in terms of total returns. "
                "This suggests that further optimization of the DQL approach is needed, or that the current market "
                "conditions may be particularly favorable to this specific rule-based strategy."
            )
            
        self.pdf.text(comparison_text)
        
    def add_hyperparameter_analysis(self, parameters: Dict[str, Dict[str, Any]]):
        """Add hyperparameter analysis to the report"""
        self.pdf.add_page()
        self._section_pages["Model Configuration and Hyperparameters"] = self.pdf.page_no()
        self.pdf.chapter_title("Model Configuration and Hyperparameters")
        
        # Overview text
        self.pdf.text(
            "This section documents the configuration and hyperparameters used for the DQL agent, "
            "trading environment, and overall system setup. These parameters significantly impact "
            "the agent's learning and trading performance."
        )
        
        # Add parameter sections
        for section_name, params in parameters.items():
            self.pdf.section_title(section_name)
            
            # Format as table
            col_width = 90
            row_height = 7
            self.pdf.set_font("Arial", "B", 10)
            self.pdf.set_fill_color(*self.pdf.accent_color)
            self.pdf.cell(col_width, row_height, "Parameter", 1, 0, "C", True)
            self.pdf.cell(col_width, row_height, "Value", 1, 1, "C", True)
            
            self.pdf.set_font("Arial", "", 10)
            fill = False
            
            # Sort parameters for consistent presentation
            sorted_params = sorted(params.items())
            
            for param, value in sorted_params:
                # Format parameter name
                param_name = " ".join(word.capitalize() for word in param.split("_"))
                
                # Format value
                if isinstance(value, (dict, list)):
                    formatted_value = json.dumps(value, sort_keys=True)[:40] + "..."
                elif isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                    
                self.pdf.cell(col_width, row_height, param_name, 1, 0, "L", fill)
                self.pdf.cell(col_width, row_height, formatted_value, 1, 1, "R", fill)
                fill = not fill
                
            self.pdf.ln(5)
            
        # Parameter impact analysis
        self.pdf.section_title("Parameter Impact Analysis")
        self.pdf.text(
            "The agent's behavior and performance are significantly influenced by these parameters. "
            "Key learning parameters like learning rate, gamma, and epsilon decay control how quickly "
            "the agent adapts and balances exploration versus exploitation. "
            "Environment parameters like window size and reward formula determine what information "
            "the agent sees and what goals it optimizes for."
        )
        
    def add_feature_importance(self, feature_importance_data=None, feature_importance_path=None, 
                           test_feature_importance_data=None, test_feature_importance_path=None):
        """Add feature importance analysis to the report"""
        print("DEBUG: Starting feature importance section generation")
        print(f"DEBUG: feature_importance_data: {feature_importance_data}")
        print(f"DEBUG: feature_importance_path: {feature_importance_path}")
        print(f"DEBUG: test_feature_importance_data: {test_feature_importance_data}")
        print(f"DEBUG: test_feature_importance_path: {test_feature_importance_path}")
        
        self.pdf.add_page()
        self._section_pages["Feature Importance Analysis"] = self.pdf.page_no()
        self.pdf.chapter_title("Feature Importance Analysis")
        
        # Overview text
        self.pdf.text(
            "This section analyzes which features have the most significant impact on the agent's decision-making. "
            "Understanding feature importance helps identify which market signals the agent relies on most heavily "
            "and can guide future model improvements."
        )
        
        # Add training feature importance visualization
        if feature_importance_path and os.path.exists(feature_importance_path):
            print(f"DEBUG: Adding feature importance image from {feature_importance_path}")
            try:
                self.pdf.add_image(
                    feature_importance_path,
                    caption="Figure 6: Training Feature Importance Analysis - Correlation with Reward",
                    h=110
                )
                print("DEBUG: Successfully added feature importance image")
            except Exception as e:
                print(f"DEBUG: Error adding feature importance image: {e}")
        else:
            print(f"DEBUG: Feature importance path not found or invalid: {feature_importance_path}")
        
        # Add test feature importance visualization if available
        if test_feature_importance_path and os.path.exists(test_feature_importance_path):
            print(f"DEBUG: Adding test feature importance image from {test_feature_importance_path}")
            try:
                self.pdf.add_image(
                    test_feature_importance_path,
                    caption="Figure 7: Testing Feature Importance Analysis - Correlation with Reward",
                    h=110
                )
                print("DEBUG: Successfully added test feature importance image")
            except Exception as e:
                print(f"DEBUG: Error adding test feature importance image: {e}")
        else:
            print(f"DEBUG: Test feature importance path not found or invalid: {test_feature_importance_path}")
        
        # Add feature importance data if available
        if feature_importance_data and isinstance(feature_importance_data, dict):
            print(f"DEBUG: Adding feature importance data table with {len(feature_importance_data)} features")
            try:
                # Format as table
                self.pdf.section_title("Training Feature Importance Metrics")
                
                # Create a sorted list of features by importance
                sorted_features = sorted(feature_importance_data.items(), key=lambda x: x[1], reverse=True)
                
                col_width = 90
                row_height = 7
                self.pdf.set_font("Arial", "B", 10)
                self.pdf.set_fill_color(*self.pdf.accent_color)
                self.pdf.cell(col_width, row_height, "Feature", 1, 0, "C", True)
                self.pdf.cell(col_width, row_height, "Importance Score", 1, 1, "C", True)
                
                self.pdf.set_font("Arial", "", 10)
                fill = False
                
                for feature, importance in sorted_features:
                    self.pdf.cell(col_width, row_height, feature, 1, 0, "L", fill)
                    self.pdf.cell(col_width, row_height, f"{importance:.4f}", 1, 1, "R", fill)
                    fill = not fill
                    
                self.pdf.ln(5)
                print("DEBUG: Successfully added feature importance table")
                
                # Add test feature importance data if available
                if test_feature_importance_data and isinstance(test_feature_importance_data, dict):
                    print(f"DEBUG: Adding test feature importance data table with {len(test_feature_importance_data)} features")
                    try:
                        self.pdf.section_title("Testing Feature Importance Metrics")
                        
                        # Create a sorted list of features by importance
                        sorted_test_features = sorted(test_feature_importance_data.items(), key=lambda x: x[1], reverse=True)
                        
                        col_width = 90
                        row_height = 7
                        self.pdf.set_font("Arial", "B", 10)
                        self.pdf.set_fill_color(*self.pdf.accent_color)
                        self.pdf.cell(col_width, row_height, "Feature", 1, 0, "C", True)
                        self.pdf.cell(col_width, row_height, "Importance Score", 1, 1, "C", True)
                        
                        self.pdf.set_font("Arial", "", 10)
                        fill = False
                        
                        for feature, importance in sorted_test_features:
                            self.pdf.cell(col_width, row_height, feature, 1, 0, "L", fill)
                            self.pdf.cell(col_width, row_height, f"{importance:.4f}", 1, 1, "R", fill)
                            fill = not fill
                            
                        self.pdf.ln(5)
                        print("DEBUG: Successfully added test feature importance table")
                    except Exception as e:
                        print(f"DEBUG: Error adding test feature importance table: {e}")
                else:
                    print(f"DEBUG: No valid test feature importance data available: {test_feature_importance_data}")
                    
                # Add analysis of top features
                print("DEBUG: Adding feature importance analysis section")
                try:
                    self.pdf.section_title("Analysis")
                    
                    # Compare training and testing feature importance
                    if sorted_features and test_feature_importance_data:
                        top_train_features = sorted_features[:3] if len(sorted_features) >= 3 else sorted_features
                        top_train_names = [f[0] for f in top_train_features]
                        
                        sorted_test_features = sorted(test_feature_importance_data.items(), key=lambda x: x[1], reverse=True)
                        top_test_features = sorted_test_features[:3] if len(sorted_test_features) >= 3 else sorted_test_features
                        top_test_names = [f[0] for f in top_test_features]
                        
                        # Check if there's overlap between top features
                        common_features = set(top_train_names).intersection(set(top_test_names))
                        
                        if common_features:
                            analysis_text = (
                                f"The features {', '.join(common_features)} are consistently important in both "
                                f"training and testing, indicating they are reliable signals for the agent's decision-making. "
                                f"This consistency suggests that the agent's feature utilization is stable across different market conditions."
                            )
                        else:
                            analysis_text = (
                                f"During training, the most influential features were {', '.join(top_train_names)}, "
                                f"while in testing the most important were {', '.join(top_test_names)}. "
                                f"This discrepancy suggests that the agent may be adapting to different market conditions, "
                                f"or that there could be some overfitting to specific features in the training data."
                            )
                        
                        self.pdf.text(analysis_text)
                        
                    elif sorted_features:
                        # Just analyze training features
                        top_features = sorted_features[:3] if len(sorted_features) >= 3 else sorted_features
                        top_names = [f[0] for f in top_features]
                        
                        analysis_text = (
                            f"The most influential features for the agent's decision-making are "
                            f"{', '.join(top_names[:-1])} and {top_names[-1]}. "
                            f"This suggests that the agent is particularly attentive to these market signals "
                            f"when determining its trading actions. Future model improvements could focus on "
                            f"refining how these key features are processed or adding complementary features."
                        )
                        
                        self.pdf.text(analysis_text)
                        
                        # Add recommendation based on lowest importance features
                        bottom_features = sorted_features[-2:] if len(sorted_features) >= 2 else []
                        if bottom_features:
                            bottom_names = [f[0] for f in bottom_features]
                            self.pdf.text(
                                f"The features with the least influence are {' and '.join(bottom_names)}. "
                                f"If computational efficiency is a concern, these features could potentially "
                                f"be removed without significantly impacting the agent's performance."
                            )
                    print("DEBUG: Successfully added feature importance analysis text")
                except Exception as e:
                    print(f"DEBUG: Error adding feature importance analysis: {e}")
            except Exception as e:
                print(f"DEBUG: Error processing feature importance data: {e}")
        else:
            print(f"DEBUG: No valid feature importance data available: {feature_importance_data}")
            self.pdf.text(
                "No detailed feature importance data is available for this experiment. "
                "The visualization above shows the relative importance of different input features "
                "based on their correlation with the agent's reward."
            )
        
        print("DEBUG: Feature importance section completed")
        
    def add_conclusion(self, key_findings: List[str], recommendations: List[str]):
        """Add conclusion and recommendations to the report"""
        self.pdf.add_page()
        self._section_pages["Conclusion and Recommendations"] = self.pdf.page_no()
        self.pdf.chapter_title("Conclusion and Recommendations")
        
        # Key findings
        self.pdf.section_title("Key Findings")
        for i, finding in enumerate(key_findings, 1):
            self.pdf.text(f"{i}. {finding}")
        self.pdf.ln(5)
        
        # Recommendations
        self.pdf.section_title("Recommendations")
        for i, recommendation in enumerate(recommendations, 1):
            self.pdf.text(f"{i}. {recommendation}")
        self.pdf.ln(5)
        
        # Next steps
        self.pdf.section_title("Next Steps")
        self.pdf.text(
            "1. Conduct additional testing across different market conditions and timeframes.\n"
            "2. Explore alternative reward functions to better align agent behavior with trading objectives.\n"
            "3. Consider ensemble approaches combining the DQL agent with rule-based strategies.\n"
            "4. Implement real-time monitoring for live trading deployment."
        )
        
    def generate(self) -> str:
        """Generate the final PDF report and return the file path"""
        # Ensure output folder exists
        os.makedirs(self.output_dir, exist_ok=True)

        # --------------------------------------------------------------
        # Populate the Table-of-Contents placeholder with real numbers
        # --------------------------------------------------------------
        if self._toc_page_no is not None:
            current_page = self.pdf.page_no()
            # Jump to the TOC page we inserted right after the title page
            self.pdf.page = self._toc_page_no

            self.pdf.chapter_title("Table of Contents")

            ordered_keys = [
                "Executive Summary",
                "Training Performance Analysis",
                "Out-of-Sample Testing Results",
                "Comparison with Baseline Strategies",
                "Model Configuration and Hyperparameters",
                "Feature Importance Analysis",
                "Conclusion and Recommendations",
            ]
            for key in ordered_keys:
                page_no = self._section_pages.get(key, "-")
                self.pdf.cell(0, 10, key, 0, 0)
                self.pdf.cell(0, 10, str(page_no), 0, 1, "R")

            # Return to the last page so any further modifications are safe
            self.pdf.page = current_page

        # Output the PDF
        self.pdf.output(self.report_path)
        return self.report_path 