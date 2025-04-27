import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_hyperparameter_results(results_df, top_n=5, save_path=None, figsize=(15, 15)):
    """
    Create visualizations of hyperparameter tuning results
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing hyperparameter tuning results
    top_n : int
        Number of top models to highlight
    save_path : str, optional
        Path to save the plot
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plots
    """
    # Check if dataframe is empty or too small
    if results_df.empty or len(results_df) < 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Not enough data for meaningful visualization.\nNeed at least 2 parameter combinations.",
               ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Get parameter columns (exclude metrics)
    metric_cols = ['sharpe_ratio', 'total_return_pct', 'max_drawdown', 'win_rate', 
                  'sortino_ratio', 'calmar_ratio', 'volatility', 'model_id']
    param_cols = [col for col in results_df.columns if col not in metric_cols]
    
    # Create a grid layout
    n_param_plots = len(param_cols)
    n_metric_plots = min(3, len(results_df.columns) - len(param_cols))  # Top metrics to visualize
    n_correlation_plots = 1 if len(results_df) >= 3 else 0  # Only show correlation with enough data
    n_pairplot = 1
    
    total_plots = n_param_plots + n_metric_plots + n_correlation_plots + n_pairplot
    n_rows = max(1, (total_plots + 1) // 2)  # Ceiling division for rows, min 1 row
    
    # Sort by Sharpe ratio for best models
    sorted_df = results_df.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)
    top_models = sorted_df.head(top_n)
    
    # 1. Parameter distribution plots
    plot_idx = 1
    for i, param in enumerate(param_cols):
        if param not in sorted_df.columns:
            continue
            
        ax = fig.add_subplot(n_rows, 2, plot_idx)
        plot_idx += 1
        
        # Try to determine if parameter is categorical or numeric
        if sorted_df[param].dtype in [np.float64, np.int64] and len(sorted_df) >= 3:
            # Numeric parameter with enough data points
            try:
                # Joint plot of parameter vs sharpe ratio
                sns.scatterplot(x=param, y='sharpe_ratio', data=sorted_df, ax=ax, alpha=0.6)
                
                # Highlight top models
                sns.scatterplot(x=param, y='sharpe_ratio', data=top_models, 
                               ax=ax, color='red', s=100, label=f'Top {min(top_n, len(top_models))}')
                
                # Add trendline if enough data points
                if len(sorted_df) > 2:
                    sns.regplot(x=param, y='sharpe_ratio', data=sorted_df, 
                               ax=ax, scatter=False, color='blue', line_kws={'linestyle':'--'})
            except Exception as e:
                # Fallback to a simple bar plot
                ax.bar(sorted_df[param].values, sorted_df['sharpe_ratio'].values)
                ax.set_xlabel(param)
                ax.set_ylabel('Sharpe Ratio')
        else:
            # Categorical parameter or too few points
            try:
                # Simple bar plot
                param_values = sorted_df[param].values
                sharpe_values = sorted_df['sharpe_ratio'].values
                
                x_pos = np.arange(len(param_values))
                ax.bar(x_pos, sharpe_values)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(param_values)
            except Exception as e:
                ax.text(0.5, 0.5, f"Cannot plot parameter: {param}\n{str(e)}", 
                        ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(f'{param} vs Sharpe Ratio')
        ax.grid(True, alpha=0.3)
    
    # 2. Top metrics distribution
    metrics_to_plot = ['sharpe_ratio', 'total_return_pct', 'max_drawdown']
    metrics_to_plot = metrics_to_plot[:min(len(metrics_to_plot), n_metric_plots)]
    
    for i, metric in enumerate(metrics_to_plot):
        if plot_idx > n_rows * 2:
            break  # Don't exceed the grid size
            
        ax = fig.add_subplot(n_rows, 2, plot_idx)
        plot_idx += 1
        
        # Simple bar plot for small datasets
        if len(sorted_df) < 5:
            x_pos = np.arange(len(sorted_df))
            ax.bar(x_pos, sorted_df[metric].values)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"Model {i+1}" for i in range(len(sorted_df))])
        else:
            # Plot histogram with KDE
            sns.histplot(sorted_df[metric], kde=True, ax=ax)
            
            # Add vertical lines for top models
            for j, (_, row) in enumerate(top_models.iterrows()):
                ax.axvline(x=row[metric], color='red', linestyle='--', alpha=0.7,
                          label=f'Top {j+1}' if j == 0 else None)
        
        ax.set_title(f'Distribution of {metric}')
        ax.grid(True, alpha=0.3)
        if i == 0:  # Only add legend for first plot
            ax.legend()
    
    # 3. Correlation heatmap
    if n_correlation_plots > 0 and len(results_df) >= 3:
        ax = fig.add_subplot(n_rows, 2, plot_idx)
        plot_idx += 1
        
        # Select relevant columns for correlation
        corr_cols = param_cols + ['sharpe_ratio', 'total_return_pct', 'max_drawdown']
        
        # Filter to only include numeric columns
        numeric_cols = [col for col in corr_cols if sorted_df[col].dtype in [np.float64, np.int64]]
        
        if numeric_cols and len(numeric_cols) >= 2:
            # Compute correlation matrix
            corr_matrix = sorted_df[numeric_cols].corr()
            
            # Plot heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=ax, fmt=".2f", linewidths=.5)
            ax.set_title('Parameter-Metric Correlations')
        else:
            ax.text(0.5, 0.5, "No numeric parameters for correlation analysis", 
                    ha='center', va='center')
    
    # 4. Metrics pairplot (in last position)
    if plot_idx <= n_rows * 2:
        ax = fig.add_subplot(n_rows, 2, plot_idx)
        
        # Create a scatter plot of Sharpe vs Return
        sns.scatterplot(x='total_return_pct', y='sharpe_ratio', 
                       data=sorted_df, ax=ax, alpha=0.6)
        
        # Highlight top models
        sns.scatterplot(x='total_return_pct', y='sharpe_ratio', 
                       data=top_models, ax=ax, color='red', s=100)
        
        # Add model_id labels to top models
        for _, row in top_models.iterrows():
            ax.annotate(f"ID: {row['model_id']}", 
                       (row['total_return_pct'], row['sharpe_ratio']),
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_title('Sharpe Ratio vs Total Return')
        ax.grid(True, alpha=0.3)
    
    # Adjust layout and add title
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.suptitle('Hyperparameter Tuning Results', fontsize=16, y=0.99)
    
    # Save if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_top_models_comparison(results_df, top_n=5, save_path=None, figsize=(15, 10)):
    """
    Create a detailed comparison of top models from hyperparameter tuning
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing hyperparameter tuning results
    top_n : int
        Number of top models to compare
    save_path : str, optional
        Path to save the plot
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the comparison plots
    """
    # Check if dataframe is empty or too small
    if results_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available for visualization.",
               ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Sort by Sharpe ratio for best models
    sorted_df = results_df.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)
    top_models = sorted_df.head(min(top_n, len(sorted_df)))
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # 1. Radar chart of metrics (left side)
    ax1 = fig.add_subplot(1, 2, 1, polar=True)
    
    # Select metrics for radar chart (normalize them)
    # Use only metrics that exist in the dataframe
    available_metrics = ['sharpe_ratio', 'total_return_pct', 'sortino_ratio', 'calmar_ratio', 'win_rate']
    metrics = [m for m in available_metrics if m in results_df.columns]
    
    if len(metrics) >= 3 and len(top_models) > 0:
        # Prepare data
        radar_data = top_models[metrics].copy()
        
        # Replace inf with max finite value
        for col in radar_data.columns:
            finite_values = radar_data[col][~np.isinf(radar_data[col])]
            if len(finite_values) > 0:
                max_finite = finite_values.max()
                radar_data[col] = radar_data[col].replace([np.inf, -np.inf], max_finite)
        
        # Normalize to 0-1 range for each metric
        for col in radar_data.columns:
            if radar_data[col].min() < 0:
                # For metrics that can be negative
                radar_data[col] = (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min())
            else:
                # For non-negative metrics
                if radar_data[col].max() != 0:
                    radar_data[col] = radar_data[col] / radar_data[col].max()
        
        # Number of metrics
        N = len(metrics)
        
        # Angles for the radar chart
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Plot each model
        for i, (idx, model) in enumerate(top_models.iterrows()):
            values = radar_data.iloc[i].tolist()
            values += values[:1]  # Close the loop
            
            # Plot the values
            ax1.plot(angles, values, linewidth=2, label=f"Model {model['model_id']}")
            ax1.fill(angles, values, alpha=0.1)
        
        # Set labels
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metrics)
        
        # Add legend
        ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        ax1.set_title('Normalized Performance Metrics')
    else:
        # Not enough data for radar chart
        ax1.text(0, 0, "Not enough data for radar chart\n(need 3+ metrics and 1+ model)",
                ha='center', va='center', fontsize=12)
        ax1.axis('off')
    
    # 2. Parameter table (right side)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')
    
    # Get parameter columns (exclude metrics)
    metric_cols = ['sharpe_ratio', 'total_return_pct', 'max_drawdown', 'win_rate', 
                  'sortino_ratio', 'calmar_ratio', 'volatility', 'model_id']
    param_cols = [col for col in results_df.columns if col not in metric_cols]
    
    if len(top_models) > 0:
        # Prepare data for table
        table_data = []
        
        # Add header
        headers = ['Metric'] + [f"Model {m['model_id']}" for _, m in top_models.iterrows()]
        table_data.append(headers)
        
        # Add parameters
        for param in param_cols:
            row = [param]
            for _, model in top_models.iterrows():
                row.append(f"{model[param]}")
            table_data.append(row)
        
        # Add key metrics
        for metric in [m for m in ['sharpe_ratio', 'total_return_pct', 'max_drawdown', 'win_rate'] if m in top_models.columns]:
            row = [metric]
            for _, model in top_models.iterrows():
                # Format based on metric type
                if metric == 'sharpe_ratio' or metric == 'sortino_ratio' or metric == 'calmar_ratio':
                    row.append(f"{model[metric]:.4f}")
                elif metric == 'total_return_pct' or metric == 'win_rate':
                    row.append(f"{model[metric]:.2f}%")
                elif metric == 'max_drawdown':
                    row.append(f"{model[metric]:.2f}%")
                else:
                    row.append(f"{model[metric]}")
            table_data.append(row)
        
        # Create table
        table = ax2.table(
            cellText=table_data[1:],  # Data
            colLabels=table_data[0],  # Headers
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
    else:
        ax2.text(0.5, 0.5, "No models available for comparison",
                ha='center', va='center', fontsize=14)
    
    # Add title
    plt.suptitle(f'Top {min(top_n, len(top_models))} Models Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_parameter_importance(results_df, save_path=None, figsize=(12, 10)):
    """
    Visualize the relative importance of hyperparameters
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing hyperparameter tuning results
    save_path : str, optional
        Path to save the plot
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the parameter importance plot
    """
    # Check if dataframe is empty or too small
    if results_df.empty or len(results_df) < 3:  # Need at least 3 points for correlation
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Not enough data for parameter importance analysis.\nNeed at least 3 parameter combinations.",
               ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Get parameter columns (exclude metrics)
    metric_cols = ['sharpe_ratio', 'total_return_pct', 'max_drawdown', 'win_rate', 
                  'sortino_ratio', 'calmar_ratio', 'volatility', 'model_id']
    param_cols = [col for col in results_df.columns if col not in metric_cols]
    
    # Filter to only numeric parameters
    numeric_params = [col for col in param_cols if results_df[col].dtype in [np.float64, np.int64]]
    
    # If no numeric parameters, return empty figure
    if not numeric_params:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No numeric parameters for importance analysis",
                ha='center', va='center')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # 1. Correlation based importance
    # Calculate correlation with Sharpe ratio
    correlations = []
    for param in numeric_params:
        corr = results_df[param].corr(results_df['sharpe_ratio'])
        correlations.append((param, abs(corr)))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # Plot
    params, corrs = zip(*correlations)
    axes[0].barh(params, corrs, color='skyblue')
    axes[0].set_title('Parameter Importance (Correlation with Sharpe Ratio)')
    axes[0].set_xlabel('Absolute Correlation')
    axes[0].grid(True, alpha=0.3)
    
    # Add values
    for i, v in enumerate(corrs):
        axes[0].text(v + 0.01, i, f"{v:.4f}", va='center')
    
    # 2. Decision tree based importance
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data
        X = results_df[numeric_params].copy()
        y = results_df['sharpe_ratio']
        
        # Handle inf values
        for col in X.columns:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = X[col].fillna(X[col].mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train random forest
        rf = RandomForestRegressor(n_estimators=min(50, max(10, len(results_df))), random_state=42)
        rf.fit(X_scaled, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Plot
        axes[1].barh([numeric_params[i] for i in indices], importances[indices], color='lightgreen')
        axes[1].set_title('Parameter Importance (Random Forest Feature Importance)')
        axes[1].set_xlabel('Importance')
        axes[1].grid(True, alpha=0.3)
        
        # Add values
        for i, v in enumerate(importances[indices]):
            axes[1].text(v + 0.01, i, f"{v:.4f}", va='center')
            
    except ImportError:
        axes[1].text(0.5, 0.5, "scikit-learn required for this analysis",
                   ha='center', va='center', transform=axes[1].transAxes)
    except Exception as e:
        axes[1].text(0.5, 0.5, f"Error in analysis: {str(e)}",
                   ha='center', va='center', transform=axes[1].transAxes)
    
    # Add title
    plt.suptitle('Hyperparameter Importance Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_learning_curve_comparison(episode_data_list, model_ids, metrics=['episode_reward', 'sharpe_ratio'], 
                                    smoothing=10, save_path=None, figsize=(15, 10)):
    """
    Create comparative learning curves for multiple models
    
    Parameters:
    -----------
    episode_data_list : list of dict
        List of dictionaries containing episode metrics for each model
    model_ids : list
        List of model IDs corresponding to each element in episode_data_list
    metrics : list
        List of metrics to plot
    smoothing : int
        Window size for moving average smoothing
    save_path : str, optional
        Path to save the plot
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the learning curves
    """
    # Check if we have valid data
    if not episode_data_list or not model_ids:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No episode data available for learning curve visualization.",
               ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Filter valid episode data
    valid_data = []
    valid_ids = []
    for data, model_id in zip(episode_data_list, model_ids):
        if data and any(metric in data for metric in metrics):
            valid_data.append(data)
            valid_ids.append(model_id)
    
    if not valid_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No valid episode data containing the requested metrics.",
               ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Filter to only include metrics that exist in the data
    available_metrics = []
    for metric in metrics:
        if any(metric in data for data in valid_data):
            available_metrics.append(metric)
    
    if not available_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"None of the requested metrics {metrics} found in episode data.",
               ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Create figure
    fig, axes = plt.subplots(len(available_metrics), 1, figsize=figsize, sharex=True)
    
    # Single metric case
    if len(available_metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        
        # Plot each model's learning curve
        for j, (episode_data, model_id) in enumerate(zip(valid_data, valid_ids)):
            if metric not in episode_data:
                continue
                
            values = episode_data[metric]
            episodes = np.arange(1, len(values) + 1)
            
            # Plot raw data with low alpha
            ax.plot(episodes, values, alpha=0.2, label=f"Model {model_id} (raw)")
            
            # Apply smoothing if enough data points
            if smoothing > 0 and len(values) > smoothing:
                kernel = np.ones(smoothing) / smoothing
                smoothed_values = np.convolve(values, kernel, mode='valid')
                
                # Plot smoothed data
                smooth_episodes = episodes[smoothing-1:]
                ax.plot(smooth_episodes, smoothed_values, linewidth=2, 
                       label=f"Model {model_id} (MA-{smoothing})")
            else:
                # Not enough data for smoothing
                ax.plot(episodes, values, linewidth=2, label=f"Model {model_id}")
        
        ax.set_title(f'{metric.replace("_", " ").title()} Learning Curve')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        # Only add legend to first plot
        if i == 0:
            ax.legend()
    
    # Set x-label on bottom plot
    axes[-1].set_xlabel('Episode')
    
    # Add title
    plt.suptitle('Learning Curve Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 