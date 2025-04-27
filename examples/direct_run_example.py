#!/usr/bin/env python3
"""
Example script showing how to use the DQL Trading framework directly
after cloning (without installation).

Usage:
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Run this script

This script demonstrates how to use the framework by importing it directly.
"""
import os
import sys

# Add the repository root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import the package components
from dql_trading import DQLAgent, ForexTradingEnv, TraderConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

# Example code for loading data and creating environment
def simulate_data():
    """Create some simple simulated data for demonstration"""
    # Generate sample price data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='H')
    prices = np.random.randn(1000).cumsum() + 100
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + 0.01 * np.random.rand(1000)),
        'low': prices * (1 - 0.01 * np.random.rand(1000)),
        'close': prices * (1 + 0.005 * (np.random.rand(1000) - 0.5)),
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    return df

def main():
    """Run a simple example training process"""
    print("DQL Trading Example")
    print("-------------------")
    
    # Create sample data (in a real case, you would load from a CSV file)
    df = simulate_data()
    print(f"Generated sample data with {len(df)} rows")
    
    # Split into train/test
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Create trader configuration
    trader_config = TraderConfig(
        name="ExampleStrategy",
        risk_tolerance=0.2,
        reward_goal=0.1,
        max_drawdown=0.05,
        target_volatility=0.01,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        position_sizing="dynamic",
        slippage=0.0001
    )
    
    # Create environment
    env_params = {
        "initial_amount": 10000,
        "transaction_cost_pct": 0.001,
        "reward_scaling": 1.0,
        "tech_indicator_list": ["close"]
    }
    
    env = ForexTradingEnv(
        df=train_df,
        trader_config=trader_config,
        **env_params
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        lr=0.001,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64
    )
    
    print("Training agent for 5 episodes...")
    
    # Run a few training episodes
    for episode in range(5):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition and train
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Train agent
            agent.train()
        
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")
    
    print("\nExample completed successfully!")
    
if __name__ == "__main__":
    main() 