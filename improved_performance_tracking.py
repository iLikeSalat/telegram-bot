"""
Performance Tracking Module for the Autonomous Trading Bot

This module implements comprehensive performance tracking and analysis
for monitoring trading performance and early stopping.

Features:
- Detailed performance metrics calculation
- Early stopping based on performance
- Performance visualization
- Trade statistics analysis
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Performance Tracker for monitoring trading performance.
    
    Features:
    - Tracks portfolio value over time
    - Calculates performance metrics (returns, Sharpe ratio, drawdown)
    - Implements early stopping based on performance
    - Generates performance visualizations
    - Analyzes trade statistics
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        data_dir: str = "performance_data",
        early_stopping_patience: int = 15,
        min_improvement: float = 0.001
    ):
        """
        Initialize the performance tracker.
        
        Args:
            initial_balance: Initial account balance
            data_dir: Directory to save performance data
            early_stopping_patience: Number of episodes without improvement before stopping
            min_improvement: Minimum improvement to reset patience counter
        """
        self.initial_balance = initial_balance
        self.data_dir = data_dir
        self.early_stopping_patience = early_stopping_patience
        self.min_improvement = min_improvement
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.portfolio_values = []
        self.rewards = []
        self.trades = []
        
        # Initialize episode tracking
        self.episode_portfolio_values = []
        self.episode_rewards = []
        self.episode_trades = []
        
        # Initialize performance history
        self.performance_history = []
        
        # Initialize early stopping variables
        self.best_performance = -float('inf')
        self.patience_counter = 0
        
        logger.info("Performance Tracker initialized")
    
    def start_episode(self):
        """Start tracking a new episode."""
        self.episode_portfolio_values = [self.initial_balance]
        self.episode_rewards = []
        self.episode_trades = []
        
        logger.debug("Started tracking new episode")
    
    def update_portfolio_value(self, value: float):
        """
        Update portfolio value.
        
        Args:
            value: Current portfolio value
        """
        self.episode_portfolio_values.append(value)
    
    def update_reward(self, reward: float):
        """
        Update reward.
        
        Args:
            reward: Current reward
        """
        self.episode_rewards.append(reward)
    
    def add_trade(self, trade: Dict[str, Any]):
        """
        Add a trade to the tracker.
        
        Args:
            trade: Trade details dictionary
        """
        self.episode_trades.append(trade)
    
    def end_episode(self) -> Dict[str, Any]:
        """
        End episode tracking and calculate metrics.
        
        Returns:
            Dictionary of episode metrics
        """
        # Calculate episode metrics
        metrics = self._calculate_metrics(self.episode_portfolio_values, self.episode_rewards, self.episode_trades)
        
        # Add to performance history
        self.performance_history.append(metrics)
        
        # Update overall tracking
        self.portfolio_values.extend(self.episode_portfolio_values)
        self.rewards.extend(self.episode_rewards)
        self.trades.extend(self.episode_trades)
        
        # Check for early stopping
        metrics['should_stop'] = self._check_early_stopping(metrics['performance_score'])
        
        # Save episode data
        self._save_episode_data(len(self.performance_history), metrics)
        
        logger.info(f"Episode ended with return: {metrics['total_return']:.4f}, "
                   f"Sharpe: {metrics['sharpe_ratio']:.4f}, "
                   f"Max DD: {metrics['max_drawdown']:.4f}")
        
        return metrics
    
    def _calculate_metrics(
        self,
        portfolio_values: List[float],
        rewards: List[float],
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            portfolio_values: List of portfolio values
            rewards: List of rewards
            trades: List of trade dictionaries
            
        Returns:
            Dictionary of performance metrics
        """
        if len(portfolio_values) < 2:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'trade_count': 0,
                'performance_score': -float('inf')
            }
        
        # Calculate returns
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value / initial_value) - 1.0
        
        # Calculate returns series
        returns_series = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        
        # Calculate Sharpe ratio (annualized)
        if len(returns_series) > 1 and np.std(returns_series) > 0:
            sharpe_ratio = np.mean(returns_series) / np.std(returns_series) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = 1.0 - np.array(portfolio_values) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Calculate trade statistics
        if trades:
            # Extract profit/loss from trades
            trade_pls = [t.get('profit_loss', 0) for t in trades if 'profit_loss' in t]
            
            if trade_pls:
                # Calculate win rate
                winning_trades = [pl for pl in trade_pls if pl > 0]
                win_rate = len(winning_trades) / len(trade_pls) if trade_pls else 0.0
                
                # Calculate profit factor
                total_profit = sum(pl for pl in trade_pls if pl > 0)
                total_loss = abs(sum(pl for pl in trade_pls if pl <= 0))
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            else:
                win_rate = 0.0
                profit_factor = 0.0
            
            trade_count = len(trades)
        else:
            win_rate = 0.0
            profit_factor = 0.0
            trade_count = 0
        
        # Calculate average reward
        avg_reward = np.mean(rewards) if rewards else 0.0
        
        # Calculate performance score (combined metric)
        # This balances return, risk, and consistency
        if max_drawdown > 0:
            calmar_ratio = total_return / max_drawdown
        else:
            calmar_ratio = total_return * 10.0  # Arbitrary high value if no drawdown
        
        # Performance score combines multiple metrics
        performance_score = (
            0.4 * total_return +  # 40% weight on total return
            0.3 * sharpe_ratio / 10.0 +  # 30% weight on risk-adjusted return (scaled)
            0.2 * calmar_ratio +  # 20% weight on drawdown-adjusted return
            0.1 * win_rate  # 10% weight on win rate
        )
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trade_count': trade_count,
            'avg_reward': avg_reward,
            'calmar_ratio': calmar_ratio,
            'performance_score': performance_score
        }
    
    def _check_early_stopping(self, performance_score: float) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            performance_score: Current performance score
            
        Returns:
            True if early stopping should be triggered, False otherwise
        """
        # Check if performance improved
        if performance_score > self.best_performance + self.min_improvement:
            # Reset patience counter
            self.patience_counter = 0
            self.best_performance = performance_score
            logger.debug(f"New best performance: {performance_score:.4f}")
            return False
        else:
            # Increment patience counter
            self.patience_counter += 1
            logger.debug(f"No improvement for {self.patience_counter} episodes. "
                        f"Best: {self.best_performance:.4f}, Current: {performance_score:.4f}")
            
            # Check if patience exceeded
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {self.patience_counter} episodes without improvement")
                return True
            
            return False
    
    def _save_episode_data(self, episode: int, metrics: Dict[str, Any]):
        """
        Save episode data to disk.
        
        Args:
            episode: Episode number
            metrics: Episode metrics
        """
        # Create episode directory
        episode_dir = os.path.join(self.data_dir, f"episode_{episode}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # Save portfolio values
        pd.DataFrame({
            'step': range(len(self.episode_portfolio_values)),
            'portfolio_value': self.episode_portfolio_values
        }).to_csv(os.path.join(episode_dir, 'portfolio_values.csv'), index=False)
        
        # Save rewards
        if self.episode_rewards:
            pd.DataFrame({
                'step': range(len(self.episode_rewards)),
                'reward': self.episode_rewards
            }).to_csv(os.path.join(episode_dir, 'rewards.csv'), index=False)
        
        # Save trades
        if self.episode_trades:
            pd.DataFrame(self.episode_trades).to_csv(
                os.path.join(episode_dir, 'trades.csv'), index=False)
        
        # Save metrics
        pd.DataFrame([metrics]).to_csv(os.path.join(episode_dir, 'metrics.csv'), index=False)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary across all episodes.
        
        Returns:
            Dictionary of performance summary
        """
        if not self.performance_history:
            return {
                'total_episodes': 0,
                'best_episode': 0,
                'best_performance': 0.0,
                'best_return': 0.0,
                'best_sharpe': 0.0,
                'average_return': 0.0,
                'average_sharpe': 0.0,
                'average_max_drawdown': 0.0,
                'average_win_rate': 0.0,
                'total_trades': 0
            }
        
        # Find best episode
        best_idx = np.argmax([p['performance_score'] for p in self.performance_history])
        best_episode = best_idx + 1
        best_performance = self.performance_history[best_idx]
        
        # Calculate averages
        avg_return = np.mean([p['total_return'] for p in self.performance_history])
        avg_sharpe = np.mean([p['sharpe_ratio'] for p in self.performance_history])
        avg_max_dd = np.mean([p['max_drawdown'] for p in self.performance_history])
        avg_win_rate = np.mean([p['win_rate'] for p in self.performance_history])
        total_trades = sum([p['trade_count'] for p in self.performance_history])
        
        return {
            'total_episodes': len(self.performance_history),
            'best_episode': best_episode,
            'best_performance': best_performance['performance_score'],
            'best_return': best_performance['total_return'],
            'best_sharpe': best_performance['sharpe_ratio'],
            'average_return': avg_return,
            'average_sharpe': avg_sharpe,
            'average_max_drawdown': avg_max_dd,
            'average_win_rate': avg_win_rate,
            'total_trades': total_trades
        }
    
    def generate_performance_plots(self):
        """Generate performance plots and save to disk."""
        if not self.performance_history:
            logger.warning("No performance history to plot")
            return
        
        # Create figures directory
        fig_dir = os.path.join(self.data_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        # Plot portfolio value
        if self.portfolio_values:
            plt.figure(figsize=(12, 6))
            plt.plot(self.portfolio_values)
            plt.xlabel('Step')
            plt.ylabel('Portfolio Value')
            plt.title('Portfolio Value Over Time')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(fig_dir, 'portfolio_value.png'))
            plt.close()
        
        # Plot episode returns
        returns = [p['total_return'] for p in self.performance_history]
        plt.figure(figsize=(12, 6))
        plt.plot(returns)
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('Episode Returns')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(fig_dir, 'episode_returns.png'))
        plt.close()
        
        # Plot episode Sharpe ratios
        sharpes = [p['sharpe_ratio'] for p in self.performance_history]
        plt.figure(figsize=(12, 6))
        plt.plot(sharpes)
        plt.xlabel('Episode')
        plt.ylabel('Sharpe Ratio')
        plt.title('Episode Sharpe Ratios')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(fig_dir, 'episode_sharpes.png'))
        plt.close()
        
        # Plot episode max drawdowns
        drawdowns = [p['max_drawdown'] for p in self.performance_history]
        plt.figure(figsize=(12, 6))
        plt.plot(drawdowns)
        plt.xlabel('Episode')
        plt.ylabel('Max Drawdown')
        plt.title('Episode Max Drawdowns')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(fig_dir, 'episode_drawdowns.png'))
        plt.close()
        
        # Plot episode win rates
        win_rates = [p['win_rate'] for p in self.performance_history]
        plt.figure(figsize=(12, 6))
        plt.plot(win_rates)
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.title('Episode Win Rates')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(fig_dir, 'episode_win_rates.png'))
        plt.close()
        
        # Plot episode performance scores
        scores = [p['performance_score'] for p in self.performance_history]
        plt.figure(figsize=(12, 6))
        plt.plot(scores)
        plt.xlabel('Episode')
        plt.ylabel('Performance Score')
        plt.title('Episode Performance Scores')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(fig_dir, 'episode_performance_scores.png'))
        plt.close()
        
        logger.info(f"Performance plots saved to {fig_dir}")
    
    def calculate_drawdown_series(self, portfolio_values: List[float]) -> List[float]:
        """
        Calculate drawdown series from portfolio values.
        
        Args:
            portfolio_values: List of portfolio values
            
        Returns:
            List of drawdown values
        """
        if not portfolio_values:
            return []
        
        # Calculate drawdown series
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = 1.0 - np.array(portfolio_values) / peak
        
        return drawdown.tolist()
    
    def calculate_underwater_periods(self, portfolio_values: List[float]) -> List[Dict[str, Any]]:
        """
        Calculate underwater periods (periods of consecutive drawdown).
        
        Args:
            portfolio_values: List of portfolio values
            
        Returns:
            List of underwater period dictionaries
        """
        if not portfolio_values:
            return []
        
        # Calculate drawdown series
        drawdown_series = self.calculate_drawdown_series(portfolio_values)
        
        # Find underwater periods
        underwater_periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdown_series):
            if not in_drawdown and dd > 0:
                # Start of underwater period
                in_drawdown = True
                start_idx = i
            elif in_drawdown and dd == 0:
                # End of underwater period
                underwater_periods.append({
                    'start': start_idx,
                    'end': i,
                    'duration': i - start_idx,
                    'max_drawdown': max(drawdown_series[start_idx:i])
                })
                in_drawdown = False
        
        # Add last period if still underwater
        if in_drawdown:
            underwater_periods.append({
                'start': start_idx,
                'end': len(drawdown_series) - 1,
                'duration': len(drawdown_series) - 1 - start_idx,
                'max_drawdown': max(drawdown_series[start_idx:])
            })
        
        return underwater_periods
    
    def calculate_trade_statistics(self) -> Dict[str, Any]:
        """
        Calculate detailed trade statistics.
        
        Returns:
            Dictionary of trade statistics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'avg_trade': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0
            }
        
        # Extract profit/loss from trades
        trade_pls = [t.get('profit_loss', 0) for t in self.trades if 'profit_loss' in t]
        
        if not trade_pls:
            return {
                'total_trades': len(self.trades),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'avg_trade': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0
            }
        
        # Calculate statistics
        winning_trades = [pl for pl in trade_pls if pl > 0]
        losing_trades = [pl for pl in trade_pls if pl <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / len(trade_pls) if trade_pls else 0.0
        
        avg_profit = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        
        total_profit = sum(winning_trades)
        total_loss = abs(sum(losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_trade = np.mean(trade_pls)
        max_profit = max(trade_pls) if trade_pls else 0.0
        max_loss = min(trade_pls) if trade_pls else 0.0
        
        return {
            'total_trades': len(trade_pls),
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'total_profit': total_profit,
            'total_loss': total_loss
        }
