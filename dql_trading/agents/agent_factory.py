"""
Agent Factory Module

Provides a factory function to create different types of agents.
"""
import os
import sys

# Add the project root to the Python path
# Removed sys.path modification, '..')))

from dql_trading.agents.dql_agent import DQLAgent
from dql_trading.agents.custom_agent import CustomAgent
from dql_trading.agents.memory_agent import MemoryAgent

def create_agent(agent_type, state_dim, action_dim, **params):
    """
    Factory function to create an agent of the specified type.
    
    Parameters:
    -----------
    agent_type : str
        Type of agent to create ('dql', 'custom', 'memory')
    state_dim : int
        Dimension of the state space
    action_dim : int
        Dimension of the action space
    **params : dict
        Additional parameters to pass to the agent constructor
        
    Returns:
    --------
    Agent object
        An instance of the specified agent type
        
    Raises:
    -------
    ValueError
        If the agent_type is not supported
    """
    # Make a copy of params to avoid modifying the original
    agent_params = params.copy()
    
    if agent_type.lower() == 'dql':
        # Remove custom agent specific parameters
        if 'target_update_freq' in agent_params:
            agent_params.pop('target_update_freq')
        return DQLAgent(state_dim=state_dim, action_dim=action_dim, **agent_params)
    
    elif agent_type.lower() == 'custom':
        # Additional custom agent parameters can be extracted here
        target_update_freq = agent_params.pop('target_update_freq', 10)
        return CustomAgent(
            state_dim=state_dim, 
            action_dim=action_dim, 
            target_update_freq=target_update_freq,
            **agent_params
        )
    elif agent_type.lower() == 'memory':
        # Remove unsupported param for MemoryAgent
        agent_params.pop('target_update_freq', None)
        # Trading incentive parameters with defaults
        force_trade_prob = agent_params.pop('force_trade_prob', 0.15)
        trade_reward_bonus = agent_params.pop('trade_reward_bonus', 0.05)
        return MemoryAgent(
            state_dim=state_dim, 
            action_dim=action_dim,
            force_trade_prob=force_trade_prob, 
            trade_reward_bonus=trade_reward_bonus,
            **agent_params
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}. Use 'dql', 'custom', or 'memory'.") 