"""
Trading Environment Module for the Autonomous Trading Bot

This module implements the trading environment for reinforcement learning,
providing state representation, reward calculation, and trading simulation.

Improvements:
- Risk-adjusted reward function
- Enhanced state representation with technical indicators
- Better position management
- Improved reward scaling
- More realistic trading simulation
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
import logging
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Trading environment for reinforcement learning.
    
    Improvements:
    - Risk-adjusted rewards instead of raw returns
    - Enhanced state representation with technical indicators
    - Better position management with risk constraints
    - More realistic trading simulation with fees
    - Improved reward scaling for more stable learning
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.0004,  # 0.04% fee
        window_size: int = 20,
        max_position_size: float = 0.1,
        max_drawdown: float = 0.15,
        use_risk_adjusted_rewards: bool = True,
        use_position_features: bool = True,
        use_technical_indicators: bool = True
    ):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with OHLCV data
            initial_balance: Initial account balance
            transaction_fee: Transaction fee as a fraction of trade value
            window_size: Window size for state representation
            max_position_size: Maximum position size as a fraction of balance
            max_drawdown: Maximum allowed drawdown as a fraction of peak balance
            use_risk_adjusted_rewards: Whether to use risk-adjusted rewards
            use_position_features: Whether to include position features in state
            use_technical_indicators: Whether to use technical indicators in state
        """
        super(TradingEnvironment, self).__init__()
        
        # Store parameters
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.use_risk_adjusted_rewards = use_risk_adjusted_rewards
        self.use_position_features = use_position_features
        self.use_technical_indicators = use_technical_indicators
        
        # Add technical indicators if not already present
        if use_technical_indicators:
            self._add_technical_indicators()
        
        # Set up action and observation spaces
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Calculate observation space dimension
        self.n_features = 5  # OHLCV
        
        if use_technical_indicators:
            # Add technical indicators to feature count
            self.n_features += 10  # SMA, EMA, MACD, RSI, BB, etc.
        
        if use_position_features:
            # Add position features to feature count
            self.n_features += 3  # Position, Profit/Loss, Duration
        
        # State dimension: window_size * n_features
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size * self.n_features,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()
        
        logger.info(f"Trading Environment initialized with {len(df)} data points and {self.n_features} features")
    
    def _add_technical_indicators(self):
        """Add technical indicators to the dataframe if not already present."""
        # Check if indicators already exist
        if 'sma_20' in self.df.columns:
            return
        
        # SMA
        self.df['sma_20'] = self.df['close'].rolling(20).mean()
        self.df['sma_50'] = self.df['close'].rolling(50).mean()
        
        # EMA
        self.df['ema_12'] = self.df['close'].ewm(span=12).mean()
        self.df['ema_26'] = self.df['close'].ewm(span=26).mean()
        
        # MACD
        self.df['macd'] = self.df['ema_12'] - self.df['ema_26']
        self.df['macd_signal'] = self.df['macd'].ewm(span=9).mean()
        
        # RSI
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        # Avoid division by zero
        loss = loss.replace(0, 1e-10)
        rs = gain / loss
        self.df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        self.df['bb_middle'] = self.df['close'].rolling(20).mean()
        self.df['bb_std'] = self.df['close'].rolling(20).std()
        self.df['bb_upper'] = self.df['bb_middle'] + 2 * self.df['bb_std']
        self.df['bb_lower'] = self.df['bb_middle'] - 2 * self.df['bb_std']
        
        # Fill NaN values
        self.df.fillna(method='bfill', inplace=True)
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(0, inplace=True)
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state observation
        """
        # Reset state variables
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.position_history = []
        self.trade_history = []
        self.portfolio_values = [self.initial_balance]
        self.peak_balance = self.initial_balance
        self.current_drawdown = 0.0
        self.trade_count = 0
        self.position_duration = 0
        
        # Get initial state
        state = self._get_state()
        
        return state
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0 = Hold, 1 = Buy, 2 = Sell)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Get current price
        current_price = self.df.iloc[self.current_step]['close']
        
        # Initialize info dictionary
        info = {
            'current_step': self.current_step,
            'current_price': current_price,
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self.balance + self.position_value,
            'trade_made': False,
            'trade_type': None,
            'trade_profit': 0.0
        }
        
        # Initialize reward
        reward = 0.0
        
        # Process action
        if action == 1:  # Buy
            if self.position <= 0:  # Only buy if not already long
                # Close short position if exists
                if self.position < 0:
                    trade_profit = self._close_position(current_price)
                    info['trade_made'] = True
                    info['trade_type'] = 'close_short'
                    info['trade_profit'] = trade_profit
                    reward += self._calculate_reward(trade_profit, current_price)
                
                # Open long position
                self._open_position(current_price, 'long')
                info['trade_made'] = True
                info['trade_type'] = 'open_long'
        
        elif action == 2:  # Sell
            if self.position >= 0:  # Only sell if not already short
                # Close long position if exists
                if self.position > 0:
                    trade_profit = self._close_position(current_price)
                    info['trade_made'] = True
                    info['trade_type'] = 'close_long'
                    info['trade_profit'] = trade_profit
                    reward += self._calculate_reward(trade_profit, current_price)
                
                # Open short position
                self._open_position(current_price, 'short')
                info['trade_made'] = True
                info['trade_type'] = 'open_short'
        
        # Update position value
        if self.position > 0:
            self.position_value = self.position * current_price
            self.position_duration += 1
        elif self.position < 0:
            self.position_value = self.position * current_price
            self.position_duration += 1
        else:
            self.position_value = 0.0
            self.position_duration = 0
        
        # Calculate portfolio value
        portfolio_value = self.balance + self.position_value
        self.portfolio_values.append(portfolio_value)
        
        # Update peak balance and drawdown
        self.peak_balance = max(self.peak_balance, portfolio_value)
        self.current_drawdown = 1.0 - (portfolio_value / self.peak_balance) if self.peak_balance > 0 else 0.0
        
        # Check for max drawdown
        if self.current_drawdown >= self.max_drawdown:
            logger.warning(f"Maximum drawdown reached: {self.current_drawdown:.2%}")
            # Close any open position
            if self.position != 0:
                trade_profit = self._close_position(current_price)
                info['trade_made'] = True
                info['trade_type'] = 'close_max_drawdown'
                info['trade_profit'] = trade_profit
                reward += self._calculate_reward(trade_profit, current_price)
        
        # Update info with portfolio value
        info['portfolio_value'] = portfolio_value
        info['drawdown'] = self.current_drawdown
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        
        # If done, close any open position
        if done and self.position != 0:
            trade_profit = self._close_position(current_price)
            info['trade_made'] = True
            info['trade_type'] = 'close_final'
            info['trade_profit'] = trade_profit
            reward += self._calculate_reward(trade_profit, current_price)
        
        # Get next state
        next_state = self._get_state()
        
        # Add small reward for holding a profitable position
        if not info['trade_made'] and self.position != 0:
            # Calculate unrealized profit/loss
            if self.position > 0:
                unrealized_pl = (current_price - self.entry_price) * self.position
            else:
                unrealized_pl = (self.entry_price - current_price) * abs(self.position)
            
            # Add small reward based on unrealized profit/loss
            if unrealized_pl > 0:
                reward += 0.01 * (unrealized_pl / (self.entry_price * abs(self.position)))
            else:
                reward -= 0.01 * abs(unrealized_pl / (self.entry_price * abs(self.position)))
        
        # Penalize for excessive drawdown
        if self.current_drawdown > 0.1:  # 10% drawdown
            reward -= 0.1 * (self.current_drawdown - 0.1)
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """
        Get current state observation.
        
        Returns:
            State observation vector
        """
        # Get window of data
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        window = self.df.iloc[start_idx:end_idx].copy()
        
        # Pad window if needed
        if len(window) < self.window_size:
            padding = self.window_size - len(window)
            padding_df = self.df.iloc[0:padding].copy()
            window = pd.concat([padding_df, window], ignore_index=True)
        
        # Extract features
        features = []
        
        # OHLCV data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            # Normalize price data by the first close price in the window
            if col != 'volume':
                features.append(window[col].values / window['close'].values[0])
            else:
                # Normalize volume by its mean
                mean_volume = window['volume'].mean()
                if mean_volume > 0:
                    features.append(window['volume'].values / mean_volume)
                else:
                    features.append(window['volume'].values)
        
        # Technical indicators
        if self.use_technical_indicators:
            # SMA
            features.append(window['sma_20'].values / window['close'].values)
            features.append(window['sma_50'].values / window['close'].values)
            
            # EMA
            features.append(window['ema_12'].values / window['close'].values)
            features.append(window['ema_26'].values / window['close'].values)
            
            # MACD (already normalized)
            features.append(window['macd'].values)
            features.append(window['macd_signal'].values)
            
            # RSI (already normalized)
            features.append(window['rsi_14'].values / 100.0)
            
            # Bollinger Bands
            features.append((window['close'].values - window['bb_lower'].values) / 
                           (window['bb_upper'].values - window['bb_lower'].values))
            
            # Volatility (20-day standard deviation / close)
            volatility = window['close'].rolling(20).std().fillna(0).values / window['close'].values
            features.append(volatility)
            
            # Price momentum (close / close 5 days ago - 1)
            momentum = window['close'].values / window['close'].shift(5).fillna(window['close'].iloc[0]).values - 1
            features.append(momentum)
        
        # Position features
        if self.use_position_features:
            # Current position normalized by max position size
            position_feature = np.ones(self.window_size) * (self.position / (self.initial_balance * self.max_position_size))
            features.append(position_feature)
            
            # Unrealized profit/loss normalized by position value
            if self.position != 0 and self.entry_price > 0:
                current_price = window['close'].values[-1]
                if self.position > 0:
                    unrealized_pl = (current_price - self.entry_price) / self.entry_price
                else:
                    unrealized_pl = (self.entry_price - current_price) / self.entry_price
                unrealized_pl_feature = np.ones(self.window_size) * unrealized_pl
            else:
                unrealized_pl_feature = np.zeros(self.window_size)
            features.append(unrealized_pl_feature)
            
            # Position duration normalized by window size
            duration_feature = np.ones(self.window_size) * (self.position_duration / self.window_size)
            features.append(duration_feature)
        
        # Flatten and concatenate features
        state = np.concatenate(features)
        
        # Replace NaN and inf values
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return state
    
    def _open_position(self, price: float, direction: str):
        """
        Open a new position.
        
        Args:
            price: Entry price
            direction: Position direction ('long' or 'short')
        """
        # Calculate position size
        position_size = self.balance * self.max_position_size / price
        
        # Apply transaction fee
        fee = position_size * price * self.transaction_fee
        self.balance -= fee
        
        # Set position
        if direction == 'long':
            self.position = position_size
        else:  # short
            self.position = -position_size
        
        # Set entry price
        self.entry_price = price
        
        # Calculate position value
        self.position_value = self.position * price
        
        # Reset position duration
        self.position_duration = 0
        
        # Record position
        self.position_history.append({
            'step': self.current_step,
            'type': 'open',
            'direction': direction,
            'price': price,
            'position': self.position,
            'fee': fee,
            'balance': self.balance
        })
        
        logger.debug(f"Opened {direction} position: {abs(self.position):.6f} units at {price:.2f}")
    
    def _close_position(self, price: float) -> float:
        """
        Close current position.
        
        Args:
            price: Exit price
            
        Returns:
            Profit/loss from the trade
        """
        if self.position == 0:
            return 0.0
        
        # Calculate profit/loss
        if self.position > 0:  # Long position
            profit_loss = (price - self.entry_price) * self.position
        else:  # Short position
            profit_loss = (self.entry_price - price) * abs(self.position)
        
        # Apply transaction fee
        fee = abs(self.position) * price * self.transaction_fee
        profit_loss -= fee
        
        # Update balance
        self.balance += self.position_value + profit_loss
        
        # Record trade
        trade = {
            'entry_step': self.position_history[-1]['step'],
            'exit_step': self.current_step,
            'direction': 'long' if self.position > 0 else 'short',
            'entry_price': self.entry_price,
            'exit_price': price,
            'position_size': abs(self.position),
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss / (self.entry_price * abs(self.position)),
            'fee': fee,
            'duration': self.position_duration
        }
        self.trade_history.append(trade)
        
        # Record position
        self.position_history.append({
            'step': self.current_step,
            'type': 'close',
            'direction': 'long' if self.position > 0 else 'short',
            'price': price,
            'position': self.position,
            'profit_loss': profit_loss,
            'fee': fee,
            'balance': self.balance
        })
        
        # Reset position
        self.position = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.position_duration = 0
        self.trade_count += 1
        
        logger.debug(f"Closed position with P/L: {profit_loss:.2f}")
        
        return profit_loss
    
    def _calculate_reward(self, profit_loss: float, current_price: float) -> float:
        """
        Calculate reward based on profit/loss.
        
        Args:
            profit_loss: Profit/loss from the trade
            current_price: Current price
            
        Returns:
            Calculated reward
        """
        if self.use_risk_adjusted_rewards:
            # Calculate risk-adjusted reward
            if profit_loss == 0:
                return 0.0
            
            # Get recent volatility
            start_idx = max(0, self.current_step - 20)
            recent_prices = self.df.iloc[start_idx:self.current_step+1]['close'].values
            if len(recent_prices) > 1:
                returns = np.diff(recent_prices) / recent_prices[:-1]
                volatility = np.std(returns)
            else:
                volatility = 0.01  # Default volatility
            
            # Avoid division by zero
            if volatility < 0.0001:
                volatility = 0.0001
            
            # Calculate Sharpe-like ratio for the trade
            if self.entry_price > 0:
                profit_loss_pct = profit_loss / (self.entry_price * abs(self.position))
                risk_adjusted_reward = profit_loss_pct / (volatility * np.sqrt(self.position_duration))
            else:
                risk_adjusted_reward = profit_loss / (self.initial_balance * volatility)
            
            # Scale reward
            reward = np.clip(risk_adjusted_reward, -10.0, 10.0)
        else:
            # Simple reward based on profit/loss percentage
            if self.entry_price > 0:
                profit_loss_pct = profit_loss / (self.entry_price * abs(self.position))
                reward = np.clip(profit_loss_pct * 10.0, -10.0, 10.0)
            else:
                reward = np.clip(profit_loss / self.initial_balance * 10.0, -10.0, 10.0)
        
        return reward
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary of performance statistics
        """
        if not self.portfolio_values:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'trade_count': 0
            }
        
        # Calculate returns
        initial_value = self.portfolio_values[0]
        final_value = self.portfolio_values[-1]
        total_return = (final_value / initial_value) - 1.0
        
        # Calculate Sharpe ratio
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(self.portfolio_values)
        drawdown = 1.0 - np.array(self.portfolio_values) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Calculate win rate and profit factor
        if self.trade_history:
            winning_trades = [t for t in self.trade_history if t['profit_loss'] > 0]
            losing_trades = [t for t in self.trade_history if t['profit_loss'] <= 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            win_rate = win_count / len(self.trade_history) if self.trade_history else 0.0
            
            total_profit = sum(t['profit_loss'] for t in winning_trades)
            total_loss = abs(sum(t['profit_loss'] for t in losing_trades))
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        else:
            win_rate = 0.0
            profit_factor = 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trade_count': len(self.trade_history)
        }
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            current_price = self.df.iloc[self.current_step]['close']
            portfolio_value = self.balance + self.position_value
            
            print(f"Step: {self.current_step}")
            print(f"Price: {current_price:.2f}")
            print(f"Balance: {self.balance:.2f}")
            print(f"Position: {self.position:.6f}")
            print(f"Position Value: {self.position_value:.2f}")
            print(f"Portfolio Value: {portfolio_value:.2f}")
            print(f"Drawdown: {self.current_drawdown:.2%}")
            print(f"Trade Count: {self.trade_count}")
            print("-------------------")
    
    def close(self):
        """Close the environment."""
        pass
