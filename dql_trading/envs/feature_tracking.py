import numpy as np
import gym
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class CorrelationTracker:
    """Tracks correlation between features and rewards"""
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.reset()
        
    def reset(self):
        self.states = []
        self.rewards = []
        
    def update(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        
    def get_importance(self):
        """Calculate feature importance based on correlation with reward"""
        if len(self.states) < 3:  # Need at least a few data points
            return np.ones(len(self.feature_names)) / len(self.feature_names)
            
        states_array = np.array(self.states)
        rewards_array = np.array(self.rewards)
        
        # Calculate absolute correlation between each feature and reward
        importance = np.zeros(states_array.shape[1])
        
        for i in range(states_array.shape[1]):
            # Handle constant features
            if np.std(states_array[:, i]) > 0:
                corr = np.corrcoef(states_array[:, i], rewards_array)[0, 1]
                importance[i] = abs(corr) if not np.isnan(corr) else 0
            else:
                importance[i] = 0
                
        # Normalize
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        else:
            importance = np.ones_like(importance) / len(importance)
            
        return importance


class FeatureTrackingWrapper(gym.Wrapper):
    """Wrapper that tracks feature importance during training"""
    def __init__(self, env):
        super().__init__(env)
        self.feature_names = self.env.get_feature_names() if hasattr(self.env, 'get_feature_names') else [f"Feature_{i}" for i in range(self.observation_space.shape[0])]
        self.feature_importance = {name: [] for name in self.feature_names}
        self.episode_states = []
        self.episode_rewards = []
        self.correlation_tracker = CorrelationTracker(self.feature_names)
        self.importance_history = []
        
    def reset(self):
        state = self.env.reset()
        self.episode_states = []
        self.episode_rewards = []
        self.correlation_tracker.reset()
        return state
        
    def step(self, action):
        # Store current state
        current_state = self.env.unwrapped._get_state() if hasattr(self.env.unwrapped, '_get_state') else self.env.unwrapped.state
        self.episode_states.append(current_state)
        
        # Execute action
        next_state, reward, done, info = self.env.step(action)
        
        # Store reward
        self.episode_rewards.append(reward)
        
        # Update correlation tracker
        self.correlation_tracker.update(current_state, action, reward)
        
        if done:
            # Calculate feature importance for this episode
            importance = self.correlation_tracker.get_importance()
            self.importance_history.append(importance)
            
            for i, name in enumerate(self.feature_names):
                self.feature_importance[name].append(importance[i])
                info['feature_importance'] = dict(zip(self.feature_names, importance))
                
        return next_state, reward, done, info
    
    def get_feature_importance(self):
        """Return average feature importance across episodes"""
        if not any(self.feature_importance.values()):
            return {name: 0.0 for name in self.feature_names}
        return {name: np.mean(values) if values else 0.0 for name, values in self.feature_importance.items()}
    
    def get_importance_history(self):
        """Return the history of importance values for plotting evolution over time"""
        return self.importance_history

def create_feature_importance_visualization(env_wrapper, save_path=None):
    """Create a visualization of feature importance from a FeatureTrackingWrapper"""
    if not isinstance(env_wrapper, FeatureTrackingWrapper):
        raise TypeError("This function requires a FeatureTrackingWrapper instance")
        
    importance = env_wrapper.get_feature_importance()
    history = env_wrapper.get_importance_history()
    
    # ------------------------------------------------------------------
    # Dynamically size the figure based on how many features we need to
    # display.  Each horizontal bar needs some vertical space; otherwise
    # long feature lists get squashed or clipped when the image is scaled
    # down to fit in the PDF.
    # ------------------------------------------------------------------

    n_features = len(importance)
    # Allocate ~0.4 inch per feature for the bar plot plus 4 inches for the
    # evolution plot and margins.  Minimum height 8 inches, maximum 18.
    fig_height = min(18, max(8, 0.4 * n_features + 4))

    fig, axes = plt.subplots(2, 1, figsize=(12, fig_height))
    
    # 1. Current importance
    feature_names = list(importance.keys())
    values = list(importance.values())
    
    # Sort for better visualization
    sorted_idx = np.argsort(values)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_values = [values[i] for i in sorted_idx]
    
    axes[0].barh(sorted_features, sorted_values, color='skyblue')
    axes[0].set_title('Feature Importance (Correlation with Reward)')
    axes[0].set_xlabel('Importance Score')
    axes[0].grid(True, alpha=0.3)
    
    # Add values to bars
    for i, v in enumerate(sorted_values):
        axes[0].text(v + 0.01, i, f"{v:.4f}", va='center')
    
    # 2. Importance evolution over time (if we have history)
    if history and len(history) > 1:
        history_array = np.array(history)
        
        # Get top 5 features for clarity
        if len(feature_names) > 5:
            avg_importance = np.mean(history_array, axis=0)
            top_indices = np.argsort(avg_importance)[-5:]  # Top 5 features
            
            for idx in top_indices:
                axes[1].plot(history_array[:, idx], label=feature_names[idx])
        else:
            for i, name in enumerate(feature_names):
                axes[1].plot(history_array[:, i], label=name)
                
        axes[1].set_title('Feature Importance Evolution')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Importance Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Not enough history for evolution plot",
                   ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig 