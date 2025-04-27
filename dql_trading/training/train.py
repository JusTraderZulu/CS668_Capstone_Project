from dql_trading.envs.feature_tracking import FeatureTrackingWrapper, create_feature_importance_visualization
import json
import os

def train_agent(env, agent, config, experiment_name=None, results_dir='results'):
    """
    Train a DQL agent with the given environment and configuration
    
    Parameters:
    -----------
    env : gym.Env
        Training environment
    agent : DQLAgent
        Agent to train
    config : dict
        Training configuration
    experiment_name : str, optional
        Experiment name for logging
    results_dir : str
        Directory to save results
        
    Returns:
    --------
    dict
        Training results
    """
    # Wrap the environment for feature tracking
    tracking_env = FeatureTrackingWrapper(env)
    
    # Set up MLflow if available
    use_mlflow = False
    try:
        import mlflow
        use_mlflow = True
        if experiment_name:
            mlflow.set_experiment(experiment_name)
    except ImportError:
        print("MLflow not available, continuing without it")
    
    # MLflow tracking
    if use_mlflow:
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(config)
            
            # Training loop (use the existing code)
            # ... training loop ...
            
            # After training, log feature importance
            importance = tracking_env.get_feature_importance()
            
            # Log as individual metrics
            for feature, importance_val in importance.items():
                mlflow.log_metric(f"importance_{feature}", importance_val)
            
            # Create and log visualization
            fi_fig = create_feature_importance_visualization(tracking_env)
            fi_path = os.path.join(results_dir, experiment_name, "feature_importance.png")
            fi_fig.savefig(fi_path)
            mlflow.log_artifact(fi_path)
            
            # Log as JSON for programmatic access
            importance_json_path = os.path.join(results_dir, experiment_name, "feature_importance.json")
            with open(importance_json_path, "w") as f:
                json.dump(importance, f)
            mlflow.log_artifact(importance_json_path)
    else:
        # Regular training without MLflow
        # ... existing training loop ...
        
        # After training, save feature importance
        importance = tracking_env.get_feature_importance()
        fi_fig = create_feature_importance_visualization(tracking_env)
        
        # Ensure directories exist
        os.makedirs(os.path.join(results_dir, experiment_name), exist_ok=True)
        
        # Save visualization
        fi_path = os.path.join(results_dir, experiment_name, "feature_importance.png")
        fi_fig.savefig(fi_path)
        
        # Save JSON
        importance_json_path = os.path.join(results_dir, experiment_name, "feature_importance.json")
        with open(importance_json_path, "w") as f:
            json.dump(importance, f)
    
    # ... rest of function ... 