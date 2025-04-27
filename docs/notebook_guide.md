# DQL Trading System Notebook Guide

This document provides a comprehensive guide for creating a Jupyter notebook that demonstrates the DQL Trading System. It includes all sections, markdown content, and code snippets.

## Table of Contents

1. [Introduction](#introduction)
2. [Flexible Deployment Options](#flexible-deployment-options)
3. [Environment Setup](#environment-setup)
4. [Creating and Configuring Trading Agents](#creating-and-configuring-trading-agents)
5. [Training Process](#training-process)
6. [Agent Evaluation](#agent-evaluation)
7. [Visualizing Results](#visualizing-results)
8. [Report Generation](#report-generation)
9. [Custom Agent Implementation](#custom-agent-implementation)
10. [Conclusion](#conclusion)

## Introduction

### Markdown Cell

```markdown
# DQL Trading System Tutorial

This notebook demonstrates how to use the Deep Q-Learning Trading System for algorithmic trading in forex markets. The system provides a comprehensive framework for developing, training, and evaluating reinforcement learning agents for trading.

## Key Features

- **Modular Architecture**: Cleanly separated components for easy extension
- **Multiple Agent Types**: Standard DQL and customizable agent architectures
- **Comprehensive Evaluation**: Compare against baseline strategies
- **Professional Reporting**: Generate detailed PDF reports with visualizations
- **Docker Support**: Containerized deployment for consistent environments

## Table of Contents

1. Environment Setup and Configuration
2. Creating and Configuring Trading Agents
3. Training Process
4. Agent Evaluation
5. Visualizing Results
6. Report Generation
7. Deployment Options (Local & Docker)
```

## Flexible Deployment Options

### Markdown Cell

```markdown
## Flexible Deployment Options

This DQL Trading System is designed with flexibility in mind. **You can run it in two ways**:

### 1. Direct Local Installation

If you prefer to run directly on your machine:

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run an experiment
python dql_trading.py full-workflow --data_file test_small.csv --experiment_name my_experiment
```

### 2. Docker Containerization

For consistent environments and easier deployment:

```bash
# Build the Docker image
docker-compose build

# Run an experiment
./docker-run.sh --agent_type dql --data_file test_small.csv --episodes 100
```

**Important**: The code works identically in both environments. All scripts, file paths, and functionality remain consistent whether running locally or in Docker. The system will automatically detect its environment and adapt accordingly.

Choose the option that best fits your workflow:
- **Local installation**: Better for development, debugging, and interactive work
- **Docker**: Better for production, cloud deployment, and consistent environments
```

## Environment Setup

### Markdown Cell

```markdown
## 1. Environment Setup and Configuration

First, let's import the necessary modules and set up our environment. We'll check if we're running in Docker or locally, and configure our experiment parameters.
```

### Code Cell

```python
# Import system modules
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path if not already there
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# Check if we're running in a Docker container
in_docker = os.path.exists('/.dockerenv')
print(f"Running in Docker: {in_docker}")

# Set up visualization style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

# Display available data files
data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
print(f"\nAvailable data files:\n{data_files}")
```

### Markdown Cell

```markdown
### Experiment Configuration

Now let's set up the configuration for our experiment. This defines parameters like which data file to use, agent type, and hyperparameters.
```

### Code Cell

```python
# Configuration for our experiment
config = {
    "data_file": "test_small.csv",  # Sample data file
    "experiment_name": "notebook_example",
    "agent_type": "dql",  # Options: dql, custom
    "episodes": 50,
    "batch_size": 32,
    "memory_size": 1000,
    "target_update_freq": 10,
    "learning_rate": 0.001,
    "gamma": 0.99,  # Discount factor
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995
}

# Create results directory if it doesn't exist
os.makedirs(f"results/{config['experiment_name']}", exist_ok=True)

# Save configuration
with open(f"results/{config['experiment_name']}/config.json", 'w') as f:
    json.dump(config, f, indent=4)

print(f"Configuration saved to results/{config['experiment_name']}/config.json")
```

## Creating and Configuring Trading Agents

### Markdown Cell

```markdown
## 2. Creating and Configuring Trading Agents

Next, we'll load our trading data and create both the trading environment and the agent.
```

### Code Cell

```python
# Import our agent and environment modules
from agents.agent_factory import create_agent
from envs.trading_env import TradingEnvironment

# Load sample data
data_path = os.path.join('data', config['data_file'])
data = pd.read_csv(data_path)
print(f"Loaded data with {len(data)} rows and {len(data.columns)} columns")
print(f"Columns: {data.columns.tolist()}")
data.head()
```

### Markdown Cell

```markdown
### Creating the Trading Environment

The `TradingEnvironment` is a custom Gym environment that simulates trading on our data. It handles the state representation, action execution, and reward calculation.
```

### Code Cell

```python
# Create the trading environment
env = TradingEnvironment(data)
print(f"Environment observation space: {env.observation_space.shape}")
print(f"Environment action space: {env.action_space}")
```

### Markdown Cell

```markdown
### Creating the Agent

We use a factory pattern to create different types of agents. The `agent_factory` module creates either a standard DQL agent or a custom agent based on the configuration.
```

### Code Cell

```python
# Create the agent
agent = create_agent(
    agent_type=config['agent_type'],
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n,
    batch_size=config['batch_size'],
    memory_size=config['memory_size'],
    learning_rate=config['learning_rate'],
    gamma=config['gamma'],
    epsilon_start=config['epsilon_start'],
    epsilon_end=config['epsilon_end'],
    epsilon_decay=config['epsilon_decay'],
    target_update_freq=config['target_update_freq']
)

print(f"Created {config['agent_type']} agent")
```

## Training Process

### Markdown Cell

```markdown
## 3. Training Process

Now we'll train our agent. The training process involves:
1. The agent selecting actions based on the current state
2. The environment executing those actions and returning rewards 
3. The agent learning from these experiences through reinforcement learning

We'll also track and visualize the agent's performance during training.
```

### Code Cell

```python
# Define a simplified training function for the notebook
def train_notebook_agent(agent, env, episodes=50, verbose=True):
    rewards = []
    portfolio_values = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
            
        # Update target network periodically
        if (episode + 1) % config['target_update_freq'] == 0:
            agent.update_target_network()
        
        rewards.append(total_reward)
        portfolio_values.append(env.portfolio_value)
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes} - Reward: {total_reward:.2f} - Portfolio: {env.portfolio_value:.2f}")
    
    return rewards, portfolio_values

# Train the agent
rewards, portfolio_values = train_notebook_agent(agent, env, episodes=config['episodes'])
```

### Markdown Cell

```markdown
### Visualizing Training Results

Let's visualize how our agent performed during training. We'll plot both the rewards and portfolio values over time.
```

### Code Cell

```python
# Plot training results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.title('Training Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(1, 2, 2)
plt.plot(portfolio_values)
plt.title('Portfolio Value')
plt.xlabel('Episode')
plt.ylabel('Value')

plt.tight_layout()
plt.savefig(f"results/{config['experiment_name']}/training_results.png")
plt.show()

# Save the agent model
agent.save(f"results/{config['experiment_name']}/model.pth")
print(f"Agent model saved to results/{config['experiment_name']}/model.pth")
```

## Agent Evaluation

### Markdown Cell

```markdown
## 4. Agent Evaluation

After training, we need to evaluate our agent's performance. This involves testing it on the dataset with exploration turned off (no random actions) and calculating key performance metrics.
```

### Code Cell

```python
# Evaluation function (simplified for notebook)
def evaluate_notebook_agent(agent, env, episodes=1):
    # Set agent to evaluation mode (no exploration)
    agent.epsilon = 0
    
    rewards = []
    trades = []
    portfolio_history = []
    actions = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        portfolio_history.append(env.portfolio_value)
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            total_reward += reward
            
            actions.append(action)
            portfolio_history.append(env.portfolio_value)
            
            if info.get('trade'):
                trades.append(info['trade'])
        
        rewards.append(total_reward)
    
    # Calculate metrics
    final_portfolio = portfolio_history[-1]
    roi = (final_portfolio / portfolio_history[0] - 1) * 100
    
    # Count trade types
    buys = sum(1 for t in trades if t.get('action') == 'buy')
    sells = sum(1 for t in trades if t.get('action') == 'sell')
    holds = sum(1 for a in actions if a == 0)  # Assuming 0 is hold
    
    # Calculate profit factor if there are trades
    winning_trades = [t for t in trades if t.get('profit', 0) > 0]
    losing_trades = [t for t in trades if t.get('profit', 0) <= 0]
    
    profit_factor = 0
    if losing_trades and sum(abs(t.get('profit', 0)) for t in losing_trades) > 0:
        profit_factor = sum(t.get('profit', 0) for t in winning_trades) / sum(abs(t.get('profit', 0)) for t in losing_trades)
    
    win_rate = len(winning_trades) / len(trades) if trades else 0
    
    return {
        'final_portfolio': final_portfolio,
        'roi': roi,
        'total_trades': len(trades),
        'buys': buys,
        'sells': sells,
        'holds': holds,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'portfolio_history': portfolio_history,
        'actions': actions
    }

# Run evaluation
eval_results = evaluate_notebook_agent(agent, env)

# Display evaluation results
print(f"Evaluation Results:")
print(f"  Final Portfolio Value: ${eval_results['final_portfolio']:.2f}")
print(f"  ROI: {eval_results['roi']:.2f}%")
print(f"  Total Trades: {eval_results['total_trades']}")
print(f"  Win Rate: {eval_results['win_rate']*100:.2f}%")
print(f"  Profit Factor: {eval_results['profit_factor']:.2f}")
print(f"  Action Distribution: Buys: {eval_results['buys']}, Sells: {eval_results['sells']}, Holds: {eval_results['holds']}")
```

## Visualizing Results

### Markdown Cell

```markdown
## 5. Visualizing Results

Now let's create some visualizations to better understand our agent's performance.
```

### Code Cell

```python
# Plot portfolio value over time
plt.figure()
plt.plot(eval_results['portfolio_history'])
plt.title('Portfolio Value Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.savefig(f"results/{config['experiment_name']}/portfolio_value.png")
plt.show()

# Plot action distribution
action_labels = ['Hold', 'Buy', 'Sell']
action_counts = [
    eval_results['holds'], 
    eval_results['buys'], 
    eval_results['sells']
]

plt.figure()
plt.bar(action_labels, action_counts)
plt.title('Action Distribution')
plt.ylabel('Count')
plt.grid(True, axis='y')
plt.savefig(f"results/{config['experiment_name']}/action_distribution.png")
plt.show()

# Save evaluation results to a JSON file
with open(f"results/{config['experiment_name']}/evaluation_results.json", 'w') as f:
    # Convert non-serializable numpy arrays to lists
    eval_results_json = eval_results.copy()
    eval_results_json['portfolio_history'] = [float(v) for v in eval_results['portfolio_history']]
    eval_results_json['actions'] = [int(a) for a in eval_results['actions']]
    json.dump(eval_results_json, f, indent=4)

print(f"Evaluation results saved to results/{config['experiment_name']}/evaluation_results.json")
```

## Report Generation

### Markdown Cell

```markdown
## 6. Report Generation

The DQL Trading System can generate professional PDF reports with all the visualizations and metrics we've calculated. Here's how to create one:
```

### Code Cell

```python
# This would typically run the report generation script
# For the notebook, we'll just show the command

report_command = f"python scripts/generate_report.py --experiment {config['experiment_name']} --results_dir results"
print(f"To generate a report, execute this command:")
print(report_command)

print("\nThe report will include:")
print("- Title page and executive summary")
print("- Training metrics and visualizations")
print("- Testing results and performance metrics")
print("- Baseline strategy comparison")
print("- Hyperparameter analysis")
print("- Conclusion and trading insights")
```

## Custom Agent Implementation

### Markdown Cell

```markdown
## 7. Using a Custom Agent

The system supports creating custom agent implementations with enhanced architectures. Here's how to use the custom agent:
```

### Code Cell

```python
# Create a custom agent example
custom_config = config.copy()
custom_config['agent_type'] = 'custom'
custom_config['experiment_name'] = 'notebook_custom_agent'

# Create results directory
os.makedirs(f"results/{custom_config['experiment_name']}", exist_ok=True)

print(f"To train a custom agent, you would run:")
print(f"python dql_trading.py train --agent_type custom --data_file {custom_config['data_file']} \\\n      --experiment_name {custom_config['experiment_name']} --episodes {custom_config['episodes']}")

# Display the agent class structure
print("\nCustom Agent Structure:")
print("- Enhanced network architecture with dueling networks")
print("- Double Q-Learning implementation")
print("- Prioritized experience replay")
print("- More efficient target network updates")
```

## Conclusion

### Markdown Cell

```markdown
## Conclusion

In this notebook, we've explored the key components of the DQL Trading System:

1. Setting up and configuring the system
2. Creating and configuring trading agents
3. Training an agent using reinforcement learning
4. Evaluating agent performance with key metrics
5. Visualizing training and testing results
6. Generating professional reports
7. Using custom agent implementations

The system provides a comprehensive framework for developing, testing, and deploying reinforcement learning agents for algorithmic trading, with the flexibility to run either locally or in Docker containers.

For more details, see:
- The main README.md file
- The reporting/README_reporting.md file for report generation details
- The docs/ directory for deployment guides
``` 