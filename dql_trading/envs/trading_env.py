import gym
import numpy as np
import pandas as pd
from gym import spaces
import datetime

class TraderConfig:
    """
    Configuration class for the forex trader
    """
    def __init__(
        self,
        name="Default Trader",
        risk_tolerance="medium",              # "low", "medium", "high"
        reward_goal="sharpe_ratio",           # "profit", "sortino", etc.
        max_drawdown=0.1,
        target_volatility=0.02,
        stop_loss_pct=0.03,
        take_profit_pct=0.05,
        position_sizing="dynamic",            # or "fixed"
        slippage=0.0002,
        session_time=("00:00", "23:59")       # 24-hour forex trading
    ):
        self.name = name
        self.risk_tolerance = risk_tolerance
        self.reward_goal = reward_goal
        self.max_drawdown = max_drawdown
        self.target_volatility = target_volatility
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_sizing = position_sizing
        self.slippage = slippage
        self.session_time = session_time
        
    def as_dict(self):
        return self.__dict__


class ForexTradingEnv(gym.Env):
    """
    Custom Forex Trading Environment that follows gym interface
    
    Actions:
    0: SELL
    1: HOLD
    2: BUY
    
    State:
    [cash, current_price, shares_held, price_feature1, ...price_featureN]
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, trader_config=None, **kwargs):
        """
        Initialize the trading environment
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        trader_config : TraderConfig
            Configuration object
        """
        super(ForexTradingEnv, self).__init__()
        
        # Settings
        self.trader_config = trader_config or TraderConfig()
        self.initial_amount = kwargs.get('initial_amount', 100000)
        self.transaction_cost_pct = kwargs.get('transaction_cost_pct', 0.0001)
        self.reward_scaling = kwargs.get('reward_scaling', 1e-4)
        self.tech_indicators = kwargs.get('tech_indicator_list', ['close'])
        
        # Initialize
        self.df = df.copy()
        self.action_space = spaces.Discrete(3)  # SELL, HOLD, BUY
        
        # State space: [cash, price, shares_held, technical_indicators...]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(3 + len(self.tech_indicators),),
            dtype=np.float32
        )
        
        # Episode tracking
        self.terminal = False
        self.day = 0
        self.account_memory = []
        self.trade_log = []
        
    def reset(self):
        """
        Reset the environment for new episode
        """
        self.day = 0
        self.terminal = False
        self.account_memory = []
        self.trade_log = []
        
        # Initial state
        cash = self.initial_amount
        self.data = self.df.iloc[self.day]
        price = self.data['close']
        shares_held = 0
        
        # State includes cash, price, shares_held, and technical indicators
        tech_indicators = [self.data[tech] for tech in self.tech_indicators]
        self.state = [cash, price, shares_held] + tech_indicators
        
        self.account_memory.append(cash)
        
        return self.state
    
    def step(self, action):
        """
        Take action in the environment
        
        Parameters:
        -----------
        action : int
            0: SELL
            1: HOLD
            2: BUY
        """
        self.terminal = self.day >= len(self.df.index) - 1
        
        if self.terminal:
            # If terminal, just return current state
            account_value = self.state[0] + self.state[1] * self.state[2]
            self.account_memory.append(account_value)
            return self.state, 0, True, {}
        
        # Get current values from state
        cash = self.state[0]
        current_price = self.state[1]
        shares_held = self.state[2]
        
        # Update the day
        self.day += 1
        self.data = self.df.iloc[self.day]
        next_price = self.data['close']
        
        # Calculate account value before action
        prev_account_value = cash + shares_held * current_price
        
        # Execute the action
        if action == 0 and shares_held > 0:  # SELL
            # Apply transaction cost
            cash += shares_held * current_price * (1 - self.transaction_cost_pct)
            # Log the trade
            self.trade_log.append({
                "type": "SELL",
                "price": current_price,
                "shares": shares_held,
                "account_value": prev_account_value,
                "step": self.day,
                "reward": 0.0  # Will be updated after calculating reward
            })
            shares_held = 0
            
        elif action == 2:  # BUY
            # Calculate the position size (10% of portfolio)
            account_value = cash + shares_held * current_price
            buy_budget = account_value * 0.10
            
            # Calculate how many shares we can buy
            max_shares = buy_budget // (current_price * (1 + self.transaction_cost_pct))
            
            if max_shares > 0:
                # Execute the buy
                cost = max_shares * current_price * (1 + self.transaction_cost_pct)
                cash -= cost
                shares_held += max_shares
                # Log the trade
                self.trade_log.append({
                    "type": "BUY",
                    "price": current_price,
                    "shares": max_shares,
                    "account_value": prev_account_value,
                    "step": self.day,
                    "reward": 0.0  # Will be updated after calculating reward
                })
        
        # Update state with new values
        tech_indicators = [self.data[tech] for tech in self.tech_indicators]
        self.state = [cash, next_price, shares_held] + tech_indicators
        
        # Calculate reward based on change in account value
        new_account_value = cash + shares_held * next_price
        self.account_memory.append(new_account_value)
        reward = self._calculate_reward()
        
        # Update the most recent trade's reward if one was made this step
        if self.trade_log and self.trade_log[-1]["step"] == self.day:
            self.trade_log[-1]["reward"] = reward
        
        return self.state, reward, self.terminal, {}
    
    def _calculate_reward(self):
        """
        Calculate reward based on trader configuration
        """
        # Default to change in account value
        current_account_value = self.account_memory[-1]
        
        if len(self.account_memory) < 2:
            return 0
        
        prev_value = self.account_memory[-2]
        reward = current_account_value - prev_value
        
        # Scale reward
        reward *= self.reward_scaling
        
        # Apply more sophisticated reward calculations based on trader_config
        if self.trader_config.reward_goal == "sharpe_ratio" and len(self.account_memory) > 2:
            returns = pd.Series(self.account_memory).pct_change().dropna()
            if len(returns) > 1:
                reward = (returns.mean() / (returns.std() + 1e-6)) * (252**0.5)
                
        elif self.trader_config.reward_goal == "sortino" and len(self.account_memory) > 2:
            returns = pd.Series(self.account_memory).pct_change().dropna()
            downside = returns[returns < 0]
            if len(downside) > 0:
                reward = (returns.mean() / (downside.std() + 1e-6)) * (252**0.5)
                
        # Penalize for exceeding maximum drawdown
        if len(self.account_memory) > 1:
            max_account_value = max(self.account_memory)
            drawdown = (max_account_value - current_account_value) / max_account_value
            if drawdown > self.trader_config.max_drawdown:
                reward -= 10  # Heavy penalty for exceeding max drawdown
                
        return reward
                
    def render(self, mode='human'):
        """
        Render the environment
        """
        current_price = self.state[1]
        cash = self.state[0]
        shares_held = self.state[2]
        account_value = cash + shares_held * current_price
        
        print(f"Day: {self.day}")
        print(f"Price: {current_price:.2f}")
        print(f"Cash: {cash:.2f}")
        print(f"Shares Held: {shares_held}")
        print(f"Account Value: {account_value:.2f}")
        print(f"Trades Made: {len(self.trade_log)}")
        print("-" * 30)
        
    def get_account_value_memory(self):
        """
        Return the account value history
        """
        return self.account_memory
    
    def get_trade_log(self):
        """
        Return the trade log
        """
        return self.trade_log
    
    def get_feature_names(self):
        """Return names of features in the state space"""
        # Base features
        feature_names = ['Price', 'Account_Balance', 'Position']
        
        # Add technical indicators if used
        if hasattr(self, 'tech_indicators') and self.tech_indicators:
            feature_names.extend(self.tech_indicators)
        
        # Add custom features if any
        if hasattr(self, 'custom_features') and self.custom_features:
            feature_names.extend(self.custom_features)
            
        return feature_names 