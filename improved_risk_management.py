"""
Risk Management Module for the Autonomous Trading Bot

This module implements comprehensive risk management strategies
to protect capital and ensure sustainable trading performance.

Improvements:
- Dynamic position sizing based on volatility
- Maximum drawdown protection
- Daily risk limits
- Trailing stop-loss mechanisms
- Risk-adjusted position sizing
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class RiskManager:
    """
    Risk Manager for controlling trading risk exposure.
    
    Implements various risk management strategies including:
    - Position size limits
    - Maximum risk per trade
    - Maximum daily risk
    - Maximum drawdown protection
    - Dynamic position sizing based on volatility
    """
    
    def __init__(
        self,
        max_position_size: float = 0.1,
        max_risk_per_trade: float = 0.01,
        max_daily_risk: float = 0.03,
        max_drawdown: float = 0.15,
        risk_free_rate: float = 0.02,
        use_dynamic_position_sizing: bool = True,
        volatility_lookback: int = 20,
        volatility_scaling_factor: float = 1.0
    ):
        """
        Initialize the risk manager.
        
        Args:
            max_position_size: Maximum position size as a fraction of total capital
            max_risk_per_trade: Maximum risk per trade as a fraction of total capital
            max_daily_risk: Maximum daily risk as a fraction of total capital
            max_drawdown: Maximum allowed drawdown as a fraction of peak capital
            risk_free_rate: Annual risk-free rate for risk-adjusted calculations
            use_dynamic_position_sizing: Whether to use dynamic position sizing
            volatility_lookback: Lookback period for volatility calculation
            volatility_scaling_factor: Scaling factor for volatility-based position sizing
        """
        self.max_position_size = max_position_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_risk = max_daily_risk
        self.max_drawdown = max_drawdown
        self.risk_free_rate = risk_free_rate
        self.use_dynamic_position_sizing = use_dynamic_position_sizing
        self.volatility_lookback = volatility_lookback
        self.volatility_scaling_factor = volatility_scaling_factor
        
        # Initialize tracking variables
        self.peak_capital = 0.0
        self.current_drawdown = 0.0
        self.daily_risk_used = 0.0
        self.last_risk_reset = datetime.now()
        self.trade_history = []
        
        logger.info("Risk Manager initialized")
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        stop_loss: float,
        balance: float,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            price: Current price
            stop_loss: Stop loss price
            balance: Current account balance
            volatility: Price volatility (if None, will use fixed risk)
            
        Returns:
            Position size in base currency units
        """
        # Update peak capital if needed
        self.peak_capital = max(self.peak_capital, balance)
        
        # Calculate current drawdown
        self.current_drawdown = 1.0 - (balance / self.peak_capital) if self.peak_capital > 0 else 0.0
        
        # Check if we're in maximum drawdown
        if self.current_drawdown >= self.max_drawdown:
            logger.warning(f"Maximum drawdown reached: {self.current_drawdown:.2%}. Reducing position size.")
            return 0.0  # No trading when max drawdown is reached
        
        # Reset daily risk if a day has passed
        current_time = datetime.now()
        if (current_time - self.last_risk_reset).days > 0:
            self.daily_risk_used = 0.0
            self.last_risk_reset = current_time
        
        # Check if we've used up our daily risk budget
        if self.daily_risk_used >= self.max_daily_risk:
            logger.warning(f"Daily risk limit reached: {self.daily_risk_used:.2%}. No more trades allowed today.")
            return 0.0
        
        # Calculate risk amount
        risk_amount = balance * self.max_risk_per_trade
        
        # Calculate remaining daily risk
        remaining_daily_risk = self.max_daily_risk - self.daily_risk_used
        max_risk_amount = balance * remaining_daily_risk
        risk_amount = min(risk_amount, max_risk_amount)
        
        # Calculate position size based on stop loss distance
        stop_distance_pct = abs(price - stop_loss) / price
        if stop_distance_pct <= 0.0001:  # Prevent division by zero or tiny stops
            logger.warning(f"Stop distance too small: {stop_distance_pct:.6f}. Using minimum distance.")
            stop_distance_pct = 0.0001
        
        # Base position size calculation
        position_size = risk_amount / (price * stop_distance_pct)
        
        # Apply dynamic position sizing if enabled and volatility is provided
        if self.use_dynamic_position_sizing and volatility is not None:
            # Scale position size inversely with volatility
            volatility_factor = 1.0 / (volatility * self.volatility_scaling_factor)
            position_size = position_size * volatility_factor
            logger.debug(f"Applied volatility adjustment: {volatility_factor:.4f}")
        
        # Apply maximum position size limit
        max_allowed_position = balance * self.max_position_size / price
        position_size = min(position_size, max_allowed_position)
        
        # Scale down position size based on drawdown
        drawdown_factor = 1.0 - (self.current_drawdown / self.max_drawdown)
        position_size = position_size * drawdown_factor
        
        logger.info(f"Calculated position size: {position_size:.6f} units (${position_size * price:.2f})")
        return position_size
    
    def update_daily_risk(self, risk_amount: float, balance: float):
        """
        Update the daily risk tracker.
        
        Args:
            risk_amount: Risk amount in currency units
            balance: Current account balance
        """
        risk_pct = risk_amount / balance if balance > 0 else 0.0
        self.daily_risk_used += risk_pct
        logger.debug(f"Updated daily risk: {self.daily_risk_used:.2%}")
    
    def calculate_risk_adjusted_reward(
        self,
        returns: float,
        volatility: float,
        trade_duration: float = 1.0  # in days
    ) -> float:
        """
        Calculate risk-adjusted reward using Sharpe ratio.
        
        Args:
            returns: Trade returns
            volatility: Trade volatility
            trade_duration: Duration of trade in days
            
        Returns:
            Risk-adjusted reward
        """
        # Annualize returns and volatility
        annual_factor = 365.0 / trade_duration
        annualized_returns = (1.0 + returns) ** annual_factor - 1.0
        annualized_volatility = volatility * np.sqrt(annual_factor)
        
        # Avoid division by zero
        if annualized_volatility < 0.0001:
            annualized_volatility = 0.0001
        
        # Calculate Sharpe ratio
        sharpe_ratio = (annualized_returns - self.risk_free_rate) / annualized_volatility
        
        return sharpe_ratio
    
    def check_drawdown_limit(self, balance: float) -> bool:
        """
        Check if current drawdown exceeds the maximum allowed drawdown.
        
        Args:
            balance: Current account balance
            
        Returns:
            True if drawdown limit is exceeded, False otherwise
        """
        # Update peak capital if needed
        self.peak_capital = max(self.peak_capital, balance)
        
        # Calculate current drawdown
        self.current_drawdown = 1.0 - (balance / self.peak_capital) if self.peak_capital > 0 else 0.0
        
        # Check if drawdown limit is exceeded
        if self.current_drawdown >= self.max_drawdown:
            logger.warning(f"Maximum drawdown limit exceeded: {self.current_drawdown:.2%}")
            return True
        
        return False
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        initial_stop: float,
        atr: Optional[float] = None,
        multiplier: float = 2.0
    ) -> float:
        """
        Calculate trailing stop loss level.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            initial_stop: Initial stop loss level
            atr: Average True Range (if None, will use fixed percentage)
            multiplier: ATR multiplier
            
        Returns:
            Updated stop loss level
        """
        # For long positions
        if entry_price < current_price:
            # Use ATR-based trailing stop if available
            if atr is not None:
                atr_stop = current_price - (atr * multiplier)
                return max(atr_stop, initial_stop)
            else:
                # Use percentage-based trailing stop
                price_move = current_price - entry_price
                trail_amount = price_move * 0.5  # Trail by 50% of the move
                return max(entry_price + trail_amount, initial_stop)
        
        # For short positions
        elif entry_price > current_price:
            # Use ATR-based trailing stop if available
            if atr is not None:
                atr_stop = current_price + (atr * multiplier)
                return min(atr_stop, initial_stop)
            else:
                # Use percentage-based trailing stop
                price_move = entry_price - current_price
                trail_amount = price_move * 0.5  # Trail by 50% of the move
                return min(entry_price - trail_amount, initial_stop)
        
        # If entry price equals current price
        return initial_stop
    
    def get_risk_metrics(self, balance: float) -> Dict[str, float]:
        """
        Get current risk metrics.
        
        Args:
            balance: Current account balance
            
        Returns:
            Dictionary of risk metrics
        """
        return {
            'current_drawdown': self.current_drawdown,
            'daily_risk_used': self.daily_risk_used,
            'peak_capital': self.peak_capital,
            'max_position_size': self.max_position_size * balance,
            'max_risk_per_trade': self.max_risk_per_trade * balance,
            'remaining_daily_risk': (self.max_daily_risk - self.daily_risk_used) * balance
        }

class PositionManager:
    """
    Position Manager for tracking and managing trading positions.
    """
    
    def __init__(self, risk_manager: RiskManager):
        """
        Initialize the position manager.
        
        Args:
            risk_manager: Risk manager instance
        """
        self.risk_manager = risk_manager
        self.positions = {}  # Symbol -> position details
        self.closed_positions = []
        
        logger.info("Position Manager initialized")
    
    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        position_size: float,
        stop_loss: float,
        take_profit: Optional[float] = None,
        balance: float = 0.0
    ) -> Dict[str, Any]:
        """
        Open a new trading position.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ('long' or 'short')
            entry_price: Entry price
            position_size: Position size in base currency units
            stop_loss: Stop loss price
            take_profit: Take profit price (optional)
            balance: Current account balance
            
        Returns:
            Position details dictionary
        """
        # Calculate risk amount
        risk_amount = position_size * abs(entry_price - stop_loss)
        
        # Update daily risk tracker
        if balance > 0:
            self.risk_manager.update_daily_risk(risk_amount, balance)
        
        # Create position details
        position = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now(),
            'risk_amount': risk_amount,
            'max_price': entry_price if direction == 'long' else float('inf'),
            'min_price': entry_price if direction == 'short' else 0.0
        }
        
        # Store position
        self.positions[symbol] = position
        
        logger.info(f"Opened {direction} position for {symbol}: {position_size:.6f} units at {entry_price:.2f}")
        return position
    
    def update_position(
        self,
        symbol: str,
        current_price: float,
        atr: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Update position details with current price.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            atr: Average True Range (optional)
            
        Returns:
            Updated position details or empty dict if position doesn't exist
        """
        # Check if position exists
        if symbol not in self.positions:
            return {}
        
        position = self.positions[symbol]
        
        # Update max/min price
        if position['direction'] == 'long':
            position['max_price'] = max(position['max_price'], current_price)
        else:
            position['min_price'] = min(position['min_price'], current_price)
        
        # Calculate unrealized profit/loss
        if position['direction'] == 'long':
            position['unrealized_pl'] = (current_price - position['entry_price']) * position['position_size']
        else:
            position['unrealized_pl'] = (position['entry_price'] - current_price) * position['position_size']
        
        # Calculate unrealized profit/loss percentage
        position['unrealized_pl_pct'] = position['unrealized_pl'] / (position['entry_price'] * position['position_size'])
        
        # Update position duration
        position['duration'] = (datetime.now() - position['entry_time']).total_seconds() / 3600  # in hours
        
        return position
    
    def update_stop_loss(
        self,
        symbol: str,
        new_stop_loss: float,
        reason: str = 'manual'
    ) -> bool:
        """
        Update stop loss for a position.
        
        Args:
            symbol: Trading symbol
            new_stop_loss: New stop loss price
            reason: Reason for updating stop loss
            
        Returns:
            True if successful, False otherwise
        """
        # Check if position exists
        if symbol not in self.positions:
            logger.warning(f"Cannot update stop loss: No position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        old_stop = position['stop_loss']
        
        # Validate new stop loss
        if position['direction'] == 'long' and new_stop_loss > position['entry_price']:
            logger.warning(f"Invalid stop loss for long position: {new_stop_loss} > {position['entry_price']}")
            return False
        elif position['direction'] == 'short' and new_stop_loss < position['entry_price']:
            logger.warning(f"Invalid stop loss for short position: {new_stop_loss} < {position['entry_price']}")
            return False
        
        # Update stop loss
        position['stop_loss'] = new_stop_loss
        position['stop_updated_time'] = datetime.now()
        position['stop_update_reason'] = reason
        
        logger.info(f"Updated stop loss for {symbol} from {old_stop:.2f} to {new_stop_loss:.2f} ({reason})")
        return True
    
    def update_take_profit(
        self,
        symbol: str,
        new_take_profit: float
    ) -> bool:
        """
        Update take profit for a position.
        
        Args:
            symbol: Trading symbol
            new_take_profit: New take profit price
            
        Returns:
            True if successful, False otherwise
        """
        # Check if position exists
        if symbol not in self.positions:
            logger.warning(f"Cannot update take profit: No position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        old_tp = position.get('take_profit', None)
        
        # Validate new take profit
        if position['direction'] == 'long' and new_take_profit < position['entry_price']:
            logger.warning(f"Invalid take profit for long position: {new_take_profit} < {position['entry_price']}")
            return False
        elif position['direction'] == 'short' and new_take_profit > position['entry_price']:
            logger.warning(f"Invalid take profit for short position: {new_take_profit} > {position['entry_price']}")
            return False
        
        # Update take profit
        position['take_profit'] = new_take_profit
        position['tp_updated_time'] = datetime.now()
        
        logger.info(f"Updated take profit for {symbol} from {old_tp:.2f if old_tp else 'None'} to {new_take_profit:.2f}")
        return True
    
    def check_positions(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Check all positions against current prices for stop loss/take profit hits.
        
        Args:
            current_prices: Dictionary of current prices (symbol -> price)
            
        Returns:
            List of closed positions
        """
        closed = []
        
        for symbol, position in list(self.positions.items()):
            # Skip if price not available
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            # Update position with current price
            self.update_position(symbol, current_price)
            
            # Check for stop loss hit
            if position['direction'] == 'long' and current_price <= position['stop_loss']:
                position['exit_price'] = position['stop_loss']
                position['exit_reason'] = 'stop_loss'
                closed.append(self.close_position(symbol, position['stop_loss'], 'stop_loss'))
            
            elif position['direction'] == 'short' and current_price >= position['stop_loss']:
                position['exit_price'] = position['stop_loss']
                position['exit_reason'] = 'stop_loss'
                closed.append(self.close_position(symbol, position['stop_loss'], 'stop_loss'))
            
            # Check for take profit hit
            elif position.get('take_profit') is not None:
                if position['direction'] == 'long' and current_price >= position['take_profit']:
                    position['exit_price'] = position['take_profit']
                    position['exit_reason'] = 'take_profit'
                    closed.append(self.close_position(symbol, position['take_profit'], 'take_profit'))
                
                elif position['direction'] == 'short' and current_price <= position['take_profit']:
                    position['exit_price'] = position['take_profit']
                    position['exit_reason'] = 'take_profit'
                    closed.append(self.close_position(symbol, position['take_profit'], 'take_profit'))
        
        return closed
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str = 'manual'
    ) -> Dict[str, Any]:
        """
        Close a trading position.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            exit_reason: Reason for closing position
            
        Returns:
            Closed position details or empty dict if position doesn't exist
        """
        # Check if position exists
        if symbol not in self.positions:
            logger.warning(f"Cannot close position: No position found for {symbol}")
            return {}
        
        position = self.positions[symbol]
        
        # Calculate profit/loss
        if position['direction'] == 'long':
            profit_loss = (exit_price - position['entry_price']) * position['position_size']
        else:
            profit_loss = (position['entry_price'] - exit_price) * position['position_size']
        
        # Calculate profit/loss percentage
        profit_loss_pct = profit_loss / (position['entry_price'] * position['position_size'])
        
        # Update position with exit details
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now()
        position['exit_reason'] = exit_reason
        position['profit_loss'] = profit_loss
        position['profit_loss_pct'] = profit_loss_pct
        position['duration'] = (position['exit_time'] - position['entry_time']).total_seconds() / 3600  # in hours
        
        # Store closed position
        self.closed_positions.append(position)
        
        # Remove from active positions
        del self.positions[symbol]
        
        logger.info(f"Closed {position['direction']} position for {symbol}: P/L {profit_loss:.2f} ({profit_loss_pct:.2%})")
        return position
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get position details for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position details or empty dict if position doesn't exist
        """
        return self.positions.get(symbol, {})
    
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active positions.
        
        Returns:
            Dictionary of all positions (symbol -> position details)
        """
        return self.positions
    
    def get_position_exposure(self) -> float:
        """
        Get total position exposure.
        
        Returns:
            Total position exposure in currency units
        """
        total_exposure = 0.0
        
        for position in self.positions.values():
            total_exposure += position['entry_price'] * position['position_size']
        
        return total_exposure
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for closed positions.
        
        Returns:
            Dictionary of performance statistics
        """
        if not self.closed_positions:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'avg_duration': 0.0
            }
        
        # Calculate statistics
        total_trades = len(self.closed_positions)
        winning_trades = [p for p in self.closed_positions if p['profit_loss'] > 0]
        losing_trades = [p for p in self.closed_positions if p['profit_loss'] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        
        avg_profit = sum(p['profit_loss'] for p in winning_trades) / win_count if win_count > 0 else 0.0
        avg_loss = sum(p['profit_loss'] for p in losing_trades) / loss_count if loss_count > 0 else 0.0
        
        total_profit = sum(p['profit_loss'] for p in winning_trades)
        total_loss = abs(sum(p['profit_loss'] for p in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_duration = sum(p['duration'] for p in self.closed_positions) / total_trades
        
        return {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': profit_factor,
            'avg_duration': avg_duration
        }
