#!/usr/bin/env python3
"""
Simple example of training a DQL agent using the dql_trading package.
"""
import os
import matplotlib.pyplot as plt

from dql_trading import DQLAgent, ForexTradingEnv, TraderConfig
from dql_trading.utils.data_processing import load_data, split_data, add_indicators, visualize_trade_actions
from dql_trading.utils.metrics import calculate_trading_metrics, create_performance_dashboard

# Load and prepare data
data_file = "test_small.csv"
# Use the path relative to the project root
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dql_trading", "data", data_file)
df = load_data(data_path)
df = add_indicators(df)

# Split data into train and test sets
train_df, test_df = split_data(df)

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

# Environment settings
env_params = {
    "initial_amount": 10000,
    "transaction_cost_pct": 0.001,
    "reward_scaling": 1.0,
    "tech_indicator_list": ["close"]  # Using only 'close' as the indicator
}

# Create the training environment
train_env = ForexTradingEnv(
    df=train_df,
    trader_config=trader_config,
    **env_params
)

# Create the agent
state_dim = train_env.observation_space.shape[0]
action_dim = train_env.action_space.n

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

# Training loop
episodes = 10
for episode in range(episodes):
    state = train_env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = train_env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        agent.train()
    
    agent.update_target()
    
    # Calculate episode metrics
    account_values = train_env.get_account_value_memory()
    trade_log = train_env.get_trade_log()
    metrics = calculate_trading_metrics(account_values=account_values, trade_log=trade_log)
    
    print(f"Episode {episode+1}/{episodes}")
    print(f"  Total Reward: {total_reward:.4f}")
    print(f"  Return: {metrics['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print(f"  Trades: {len(trade_log)}")

# Test the agent
print("\nEvaluating agent...")
test_env = ForexTradingEnv(
    df=test_df,
    trader_config=trader_config,
    **env_params
)

state = test_env.reset()
done = False

# Use deterministic policy for testing
old_epsilon = agent.epsilon
agent.epsilon = 0

while not done:
    action = agent.select_action(state, test=True)
    next_state, reward, done, _ = test_env.step(action)
    state = next_state

# Restore epsilon
agent.epsilon = old_epsilon

# Calculate test metrics
account_values = test_env.get_account_value_memory()
trade_log = test_env.get_trade_log()
metrics = calculate_trading_metrics(account_values=account_values, trade_log=trade_log)

print("Test Results:")
print(f"  Return: {metrics['total_return_pct']:.2f}%")
print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
print(f"  Win Rate: {metrics['win_rate']:.2f}%")
print(f"  Trades: {len(trade_log)}")

# Create and display performance dashboard
plt.figure(figsize=(15, 10))
create_performance_dashboard(
    account_values=account_values,
    trade_log=trade_log,
    title="DQL Agent Backtest Performance"
)
plt.tight_layout()
plt.savefig("dql_performance.png")
print(f"Performance dashboard saved to dql_performance.png")

# Visualize trades
visualize_trade_actions(
    trade_log=trade_log,
    account_values=account_values,
    price_data=test_df,
    save_path="dql_trades.png"
)
print(f"Trade visualization saved to dql_trades.png") 