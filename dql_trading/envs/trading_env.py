import gym
import numpy as np
import pandas as pd
from gym import spaces
import datetime
from collections import deque
import ta  # Technical Analysis library used for indicators

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
    0: SELL / reduce or close existing long (no new shorts)
    1: HOLD
    2: BUY / increase long position
    3: CLOSE position (flat)
    
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
        
        # ----------------------
        # Settings
        # ----------------------
        self.trader_config = trader_config or TraderConfig()
        self.initial_amount = kwargs.get('initial_amount', 100000)
        self.transaction_cost_pct = kwargs.get('transaction_cost_pct', 0.0001)
        self.reward_scaling = kwargs.get('reward_scaling', 1e-4)

        # Look-back window for recurrent agents (e.g. DRQN)
        self.lookback_window = int(kwargs.get('lookback_window', 1))

        # Fraction of account value to use when adding to a long position.
        # Default 0.10 (10%) replicates previous behaviour.
        self.buy_pct = float(kwargs.get('buy_pct', 0.10))

        # Which indicators / features to include.  We will make sure they
        # exist in the dataframe later.
        self.tech_indicators = kwargs.get(
            'tech_indicator_list',
            [
                # Basic OHLCV columns will be added automatically; here we
                # specify additional indicators.
                'atr',
                'BB_upper', 'BB_middle', 'BB_lower',
                'bb_z',                 # Price location inside Bollinger bands
                'macd', 'macd_signal',  # Momentum
                'macd_hist',            # MACD histogram
                'rsi',
                'support_dist', 'resistance_dist',
                # Candlestick anatomy (normalised by ATR)
                'body_size', 'upper_wick', 'lower_wick',
                # Volume (normalised)
                'vol_norm',
                'time_of_day',
            ],
        )

        # Prepare dataframe (add indicators, S/R, etc.)
        self.df = df.copy()
        self._prepare_dataframe()
        
        # ----------------------
        # Action space
        # 0: SELL, 1: HOLD, 2: BUY, 3: CLOSE
        # ----------------------
        self.action_space = spaces.Discrete(4)
        
        # ------------------------------------------------------------------
        # Observation space
        # ------------------------------------------------------------------
        # Base feature count (price, cash, position) plus tech indicators
        self._feature_len = 3 + len(self.tech_indicators)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window * self._feature_len,),
            dtype=np.float32,
        )
        
        # Episode tracking
        self.terminal = False
        self.day = 0
        self.account_memory = []
        self.trade_log = []
        
        # Maintain a deque to store recent states for look-back windows
        self._state_queue: deque = deque(maxlen=self.lookback_window)
        
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
        self._state_queue.clear()
        self._append_state(cash, price, shares_held)
        self.state = self._get_state()
        
        self.account_memory.append(cash)
        
        return self.state
    
    def step(self, action):
        """
        Take action in the environment
        
        Parameters:
        -----------
        action : int
            0: SELL / reduce or close existing long (no new shorts)
            1: HOLD
            2: BUY / increase long position
            3: CLOSE position (flat)
        """
        self.terminal = self.day >= len(self.df.index) - 1
        
        if self.terminal:
            # If terminal, just return current state
            account_value = self.state[0] + self.state[1] * self.state[2]
            self.account_memory.append(account_value)
            return self.state, 0, True, {}
        
        # Get current values from state (they are the last entry in the queue)
        cash = self._state_queue[-1][0]
        current_price = self._state_queue[-1][1]
        # The deque stores position flag, not raw shares; keep separate record
        # of shares_held in an attribute for simplicity.
        if not hasattr(self, '_shares_held'):
            self._shares_held = 0
        shares_held = self._shares_held
        
        # Update the day
        self.day += 1
        self.data = self.df.iloc[self.day]
        next_price = self.data['close']
        
        # Calculate account value before action
        prev_account_value = cash + shares_held * current_price
        
        # Execute the action -------------------------------------------------
        # Allocate at most *buy_pct* fraction of account value for new buys.
        account_value = cash + shares_held * current_price
        trade_budget = account_value * self.buy_pct

        if action == 2:  # BUY  ------------------------------------------------
            max_shares = trade_budget // (current_price * (1 + self.transaction_cost_pct))
            if max_shares > 0:
                cost = max_shares * current_price * (1 + self.transaction_cost_pct)
                cash -= cost
                shares_held += max_shares
                self.trade_log.append({
                    "type": "BUY",
                    "price": current_price,
                    "shares": max_shares,
                    "account_value": prev_account_value,
                    "step": self.day,
                    "reward": 0.0,
                })

        elif action == 0:  # SELL  -------------------------------------------
            # In long-only mode, SELL means reduce or fully close an existing
            # long position.  We *never* create a short position.
            if shares_held > 0:
                proceeds = shares_held * current_price * (1 - self.transaction_cost_pct)
                cash += proceeds
                self.trade_log.append({
                    "type": "SELL",
                    "price": current_price,
                    "shares": shares_held,
                    "account_value": prev_account_value,
                    "step": self.day,
                    "reward": 0.0,
                })
                shares_held = 0

        elif action == 3:  # CLOSE -------------------------------------------
            # Duplicate of SELL semantics – close any open long position.
            if shares_held > 0:
                proceeds = shares_held * current_price * (1 - self.transaction_cost_pct)
                cash += proceeds
                self.trade_log.append({
                    "type": "CLOSE",
                    "price": current_price,
                    "shares": shares_held,
                    "account_value": prev_account_value,
                    "step": self.day,
                    "reward": 0.0,
                })
                shares_held = 0
        
        # Append new state to queue and get flattened
        self._append_state(cash, next_price, shares_held)
        self.state = self._get_state()

        # Update shares_held attribute
        self._shares_held = shares_held
        
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
        """Render the environment."""
        if not self._state_queue:
            print("Environment not initialised.")
            return
        last_state = self._state_queue[-1]
        cash = last_state[0]
        price = last_state[1]
        position_flag = last_state[2]
        account_value = cash + self._shares_held * price

        print(f"Day: {self.day}")
        print(f"Price: {price:.4f}")
        print(f"Cash: {cash:.2f}")
        print(f"Position: {position_flag} ({self._shares_held} shares)")
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_dataframe(self):
        """Ensure all required indicator columns exist in *self.df*.

        We calculate missing indicators lazily here so that the environment
        does not depend on the caller having run `add_indicators`.
        """
        df = self.df

        # Make sure index is datetime for time features
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df.index = pd.to_datetime(df['date'])
            else:
                # Fallback: create a synthetic datetime index
                df.index = pd.date_range(start=datetime.datetime.utcnow(), periods=len(df), freq='1min')

        # Average True Range (ATR)
        if 'atr' not in df.columns:
            atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'])
            df['atr'] = atr_indicator.average_true_range()

        # Bollinger Bands
        for col in ['BB_upper', 'BB_middle', 'BB_lower']:
            if col not in df.columns:
                bbands = ta.volatility.BollingerBands(df['close'])
                df['BB_upper'] = bbands.bollinger_hband()
                df['BB_middle'] = bbands.bollinger_mavg()
                df['BB_lower'] = bbands.bollinger_lband()
                break  # done

        # MACD + signal
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()

        # RSI
        if 'rsi' not in df.columns:
            rsi = ta.momentum.RSIIndicator(df['close'], window=14)
            df['rsi'] = rsi.rsi()

        # Support & Resistance levels (distance to rolling max/min of last 20 bars)
        window_sr = 20
        df['rolling_max'] = df['close'].rolling(window_sr, min_periods=1).max()
        df['rolling_min'] = df['close'].rolling(window_sr, min_periods=1).min()
        df['resistance_dist'] = df['rolling_max'] - df['close']
        df['support_dist'] = df['close'] - df['rolling_min']

        # Time of day feature – minutes past midnight scaled 0-1
        df['time_of_day'] = (
            df.index.hour * 60 + df.index.minute
        ) / 1440.0

        # Forward/Backward fill any NA introduced
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)

        # ------------------------------------------------------------------
        # Derived / custom features
        # ------------------------------------------------------------------

        # MACD histogram (difference between macd and signal)
        if 'macd_hist' not in df.columns and 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger band z-score: (price - middle) / bandwidth
        if 'bb_z' not in df.columns and {'BB_upper', 'BB_lower', 'BB_middle'}.issubset(df.columns):
            band_width = df['BB_upper'] - df['BB_lower']
            df['bb_z'] = (df['close'] - df['BB_middle']) / (band_width.replace(0, 1e-6))

        # Volume (normalised by 20-period rolling mean).  If volume missing,
        # we still create the column filled with zeros so the state vector
        # length is consistent.
        if 'vol_norm' not in df.columns:
            if 'volume' in df.columns:
                df['vol_norm'] = df['volume'] / df['volume'].rolling(20, min_periods=1).mean()
            else:
                df['vol_norm'] = 0.0

        # Candlestick anatomy (normalised by ATR)
        if {'body_size', 'upper_wick', 'lower_wick'}.isdisjoint(df.columns):
            atr_nonzero = df['atr'].replace(0, 1e-6)
            df['body_size'] = (df['close'] - df['open']).abs() / atr_nonzero
            df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / atr_nonzero
            df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / atr_nonzero

        self.df = df

    def _append_state(self, cash, price, shares_held):
        """Append current features to the deque for look-back construction."""
        tech_values = [self.data[tech] for tech in self.tech_indicators]

        # Position flag: -1 short, 0 flat, 1 long
        position_flag = 1 if shares_held > 0 else (-1 if shares_held < 0 else 0)

        state_vec = [cash, price, position_flag] + tech_values
        self._state_queue.append(state_vec)

    def _get_state(self):
        """Return flattened state vector covering *lookback_window* steps."""
        # If we have fewer than lookback_window states (at episode start),
        # left-pad with the oldest available state.
        if len(self._state_queue) < self.lookback_window:
            pad_vec = self._state_queue[0] if self._state_queue else [0] * self._feature_len
            while len(self._state_queue) < self.lookback_window:
                self._state_queue.appendleft(pad_vec)

        # Flatten
        flat_state = [elem for state in self._state_queue for elem in state]
        return flat_state

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def action_to_string(self, action: int) -> str:
        mapping = {0: 'SELL', 1: 'HOLD', 2: 'BUY', 3: 'CLOSE'}
        return mapping.get(action, 'UNKNOWN')

    # For wrappers that rely on raw state
    def _get_raw_state(self):
        return self.state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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