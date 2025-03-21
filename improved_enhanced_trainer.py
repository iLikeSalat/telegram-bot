"""
Enhanced Trainer Module for the Autonomous Trading Bot

This module provides an improved training approach with better stability,
risk management, and performance tracking.

Improvements:
- Better feature normalization
- Enhanced error handling
- Early stopping based on performance
- Improved hyperparameter settings
- Risk-adjusted training process
"""

import os
import logging
import argparse
import traceback
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import json

# Import custom modules
from rl_model import PPOAgent
from trading_environment import TradingEnvironment
from risk_management import RiskManager, PositionManager
from performance_tracking import PerformanceTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_detailed.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add a separate debug logger for detailed debugging
debug_logger = logging.getLogger('debug')
debug_handler = logging.FileHandler("training_debug.log")
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(debug_formatter)
debug_logger.addHandler(debug_handler)
debug_logger.setLevel(logging.DEBUG)

class EnhancedTrainer:
    """
    Enhanced trainer class for the autonomous trading bot.
    
    Improvements:
    - Better feature normalization for more stable training
    - Enhanced error handling and debugging
    - Early stopping based on performance metrics
    - Improved hyperparameter settings
    - Risk-adjusted training process
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
        """
        # Default configuration
        self.config = {
            # Training parameters
            "symbols": ["BTCUSDT"],
            "initial_balance": 10000.0,
            "max_position_size": 0.05,  # Reduced position size for better risk management
            "transaction_fee": 0.0004,  # 0.04% fee
            
            # RL model parameters
            "state_dim": 30,
            "action_dim": 3,
            "hidden_dims": [512, 256, 128],  # Increased network capacity
            "learning_rate": 3e-5,  # Reduced learning rate for more stable learning
            "gamma": 0.99,  # Discount factor
            "gae_lambda": 0.95,  # GAE lambda parameter
            "clip_ratio": 0.2,  # PPO clip ratio
            "value_coef": 0.5,  # Value loss coefficient
            "entropy_coef": 0.01,  # Entropy bonus coefficient
            
            # Risk management parameters
            "max_risk_per_trade": 0.01,  # 1% risk per trade
            "max_daily_risk": 0.03,  # 3% daily risk
            "max_drawdown": 0.15,  # 15% maximum drawdown
            
            # Training control parameters
            "episodes": 100,
            "save_interval": 10,
            "eval_interval": 5,
            "early_stopping_patience": 15,
            "use_early_stopping": True,
            "use_dynamic_position_sizing": True,
            "use_risk_adjusted_rewards": True,
            
            # Feature engineering parameters
            "window_size": 20,
            "use_technical_indicators": True,
            "use_position_features": True,
            "normalize_features": True,
            
            # System parameters
            "data_dir": "data",
            "models_dir": "models",
            "performance_dir": "performance_data",
            "random_seed": 42,
            "device": None  # Auto-detect
        }
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        # Create directories
        os.makedirs(self.config["data_dir"], exist_ok=True)
        os.makedirs(self.config["models_dir"], exist_ok=True)
        os.makedirs(self.config["performance_dir"], exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(self.config["random_seed"])
        torch.manual_seed(self.config["random_seed"])
        
        # Initialize components
        self.rl_agent = None
        self.risk_manager = None
        self.position_manager = None
        self.performance_tracker = None
        self.env = None
        
        # Feature normalization parameters
        self.feature_means = None
        self.feature_stds = None
        
        logger.info("Enhanced Trainer initialized")
    
    def _load_config(self, config_path):
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Update configuration
            self.config.update(loaded_config)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    def initialize(self):
        """
        Initialize all components.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize risk manager
            self.risk_manager = RiskManager(
                max_position_size=self.config["max_position_size"],
                max_risk_per_trade=self.config["max_risk_per_trade"],
                max_daily_risk=self.config["max_daily_risk"],
                max_drawdown=self.config["max_drawdown"],
                use_dynamic_position_sizing=self.config["use_dynamic_position_sizing"]
            )
            
            # Initialize position manager
            self.position_manager = PositionManager(self.risk_manager)
            
            # Initialize performance tracker
            self.performance_tracker = PerformanceTracker(
                initial_balance=self.config["initial_balance"],
                data_dir=self.config["performance_dir"],
                early_stopping_patience=self.config["early_stopping_patience"]
            )
            
            # Initialize RL agent
            self.rl_agent = PPOAgent(
                state_dim=self.config["state_dim"],
                action_dim=self.config["action_dim"],
                hidden_dims=self.config["hidden_dims"],
                lr_policy=self.config["learning_rate"],
                gamma=self.config["gamma"],
                gae_lambda=self.config["gae_lambda"],
                clip_ratio=self.config["clip_ratio"],
                value_coef=self.config["value_coef"],
                entropy_coef=self.config["entropy_coef"],
                device=self.config["device"],
                use_early_stopping=self.config["use_early_stopping"]
            )
            
            # Load model if exists
            model_path = os.path.join(self.config["models_dir"], "ppo_agent.pt")
            if os.path.exists(model_path):
                self.rl_agent.load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
            
            logger.info("All components initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            debug_logger.error(f"Initialization error details: {traceback.format_exc()}")
            return False
    
    def load_historical_data(self, data_file):
        """
        Load historical data from a file.
        
        Args:
            data_file: Path to historical data file
            
        Returns:
            DataFrame with historical data
        """
        try:
            if not os.path.exists(data_file):
                logger.error(f"Data file not found: {data_file}")
                return None
            
            # Load data from CSV
            df = pd.read_csv(data_file)
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.debug(f"Converted timestamp column to datetime: {df['timestamp'].dtype}")
            
            logger.info(f"Loaded historical data from {data_file}")
            logger.debug(f"Historical data shape: {df.shape}")
            debug_logger.debug(f"Historical data columns: {df.columns.tolist()}")
            debug_logger.debug(f"Historical data types: {df.dtypes}")
            debug_logger.debug(f"First few rows: {df.head(3).to_dict('records')}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            debug_logger.error(f"Data loading error details: {traceback.format_exc()}")
            return None
    
    def preprocess_data(self, df):
        """
        Preprocess historical data for training.
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            DataFrame with preprocessed data
        """
        try:
            # Ensure all required columns are present
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Required column missing: {col}")
                    return None
            
            # Convert numeric columns to float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = df[col].astype(float)
            
            # Log data statistics before preprocessing
            debug_logger.debug("Data statistics before preprocessing:")
            debug_logger.debug(f"Price range: {df['close'].min()} to {df['close'].max()}")
            debug_logger.debug(f"Volume range: {df['volume'].min()} to {df['volume'].max()}")
            
            # Calculate technical indicators if enabled
            if self.config["use_technical_indicators"]:
                logger.debug("Calculating technical indicators...")
                df = self._add_technical_indicators(df)
            
            # Drop NaN values
            original_len = len(df)
            df = df.dropna()
            logger.debug(f"Dropped {original_len - len(df)} rows with NaN values")
            
            # Check for infinite values and replace with large numbers
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].isin([np.inf, -np.inf]).any():
                    count = df[col].isin([np.inf, -np.inf]).sum()
                    logger.warning(f"Found {count} infinite values in column {col}, replacing with large numbers")
                    df[col] = df[col].replace([np.inf, -np.inf], [1e9, -1e9])
            
            # Normalize features if enabled
            if self.config["normalize_features"]:
                df = self._normalize_features(df)
            
            logger.info(f"Preprocessed data shape: {df.shape}")
            return df
        
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            debug_logger.error(f"Preprocessing error details: {traceback.format_exc()}")
            return None
    
    def _add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # SMA
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        
        # EMA
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        # Avoid division by zero
        loss = loss.replace(0, 1e-10)
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        # Avoid division by zero
        df['bb_middle'] = df['bb_middle'].replace(0, 1e-10)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr_14'] = true_range.rolling(14).mean()
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        # Avoid division by zero
        denom = high_14 - low_14
        denom = denom.replace(0, 1e-10)
        df['stoch_k'] = 100 * ((df['close'] - low_14) / denom)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Price momentum
        df['momentum_14'] = df['close'] / df['close'].shift(14) - 1
        
        # Rate of Change
        df['roc_14'] = (df['close'] / df['close'].shift(14) - 1) * 100
        
        # Price volatility
        df['volatility_14'] = df['close'].rolling(14).std() / df['close'].rolling(14).mean()
        
        return df
    
    def _normalize_features(self, df):
        """
        Normalize features for better training stability.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with normalized features
        """
        # Columns to normalize (exclude timestamp and categorical features)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate means and standard deviations if not already calculated
        if self.feature_means is None or self.feature_stds is None:
            self.feature_means = df[numeric_cols].mean()
            self.feature_stds = df[numeric_cols].std()
            
            # Save normalization parameters
            norm_params = {
                'means': self.feature_means.to_dict(),
                'stds': self.feature_stds.to_dict()
            }
            with open(os.path.join(self.config["data_dir"], 'normalization_params.json'), 'w') as f:
                json.dump(norm_params, f)
        
        # Normalize features
        for col in numeric_cols:
            # Avoid division by zero
            std = self.feature_stds[col] if self.feature_stds[col] > 0 else 1.0
            df[col] = (df[col] - self.feature_means[col]) / std
        
        return df
    
    def create_environment(self, df):
        """
        Create a trading environment with the preprocessed data.
        
        Args:
            df: DataFrame with preprocessed data
            
        Returns:
            TradingEnvironment instance
        """
        try:
            # Create environment
            env = TradingEnvironment(
                df=df,
                initial_balance=self.config["initial_balance"],
                transaction_fee=self.config["transaction_fee"],
                window_size=self.config["window_size"],
                max_position_size=self.config["max_position_size"],
                max_drawdown=self.config["max_drawdown"],
                use_risk_adjusted_rewards=self.config["use_risk_adjusted_rewards"],
                use_position_features=self.config["use_position_features"],
                use_technical_indicators=self.config["use_technical_indicators"]
            )
            
            logger.info("Trading environment created")
            return env
        
        except Exception as e:
            logger.error(f"Error creating environment: {str(e)}")
            debug_logger.error(f"Environment creation error details: {traceback.format_exc()}")
            return None
    
    def train(self, data_file, episodes=None, save_interval=None):
        """
        Train the agent on historical data.
        
        Args:
            data_file: Path to historical data file
            episodes: Number of episodes to train (overrides config)
            save_interval: Interval for saving model (overrides config)
            
        Returns:
            Dictionary with training results
        """
        # Override config if parameters provided
        if episodes is not None:
            self.config["episodes"] = episodes
        if save_interval is not None:
            self.config["save_interval"] = save_interval
        
        # Load and preprocess data
        df = self.load_historical_data(data_file)
        if df is None:
            return {"success": False, "error": "Failed to load data"}
        
        df = self.preprocess_data(df)
        if df is None:
            return {"success": False, "error": "Failed to preprocess data"}
        
        # Initialize components
        if not self.initialize():
            return {"success": False, "error": "Failed to initialize components"}
        
        # Create environment
        self.env = self.create_environment(df)
        if self.env is None:
            return {"success": False, "error": "Failed to create environment"}
        
        # Training loop
        logger.info(f"Starting training for {self.config['episodes']} episodes")
        
        best_performance = -float('inf')
        best_episode = 0
        
        for episode in range(1, self.config["episodes"] + 1):
            # Start tracking episode
            self.performance_tracker.start_episode()
            
            # Reset environment
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            # Episode loop
            while not done:
                # Select action
                action, action_prob, value = self.rl_agent.select_action(state)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.rl_agent.store_transition(state, action, action_prob, reward, value, done)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                
                # Update performance tracker
                self.performance_tracker.update_portfolio_value(info["portfolio_value"])
                self.performance_tracker.update_reward(reward)
                
                # Add trade to performance tracker if made
                if info["trade_made"]:
                    self.performance_tracker.add_trade({
                        "step": self.env.current_step,
                        "action": action,
                        "price": info["current_price"],
                        "profit_loss": info.get("trade_profit", 0),
                        "portfolio_value": info["portfolio_value"]
                    })
            
            # End episode and get metrics
            metrics = self.performance_tracker.end_episode()
            
            # Update agent
            if len(self.rl_agent.states) > 0:
                # Get value estimate of final state
                _, _, next_value = self.rl_agent.select_action(state)
                
                # Update policy and value networks
                update_metrics = self.rl_agent.update(next_value)
                
                # Log update metrics
                logger.info(f"Episode {episode} update metrics: {update_metrics}")
            
            # Log episode results
            logger.info(f"Episode {episode}/{self.config['episodes']} - "
                       f"Return: {metrics['total_return']:.4f}, "
                       f"Reward: {episode_reward:.4f}, "
                       f"Trades: {metrics['trade_count']}")
            
            # Save model periodically
            if episode % self.config["save_interval"] == 0:
                model_path = os.path.join(self.config["models_dir"], "ppo_agent.pt")
                self.rl_agent.save_model(model_path)
                logger.info(f"Model saved to {model_path}")
            
            # Save best model
            if metrics['performance_score'] > best_performance:
                best_performance = metrics['performance_score']
                best_episode = episode
                best_model_path = os.path.join(self.config["models_dir"], "ppo_agent_best.pt")
                self.rl_agent.save_model(best_model_path)
                logger.info(f"New best model saved with performance score: {best_performance:.4f}")
            
            # Check for early stopping
            if metrics['should_stop'] and self.config["use_early_stopping"]:
                logger.info(f"Early stopping triggered after {episode} episodes")
                break
        
        # Save final model
        final_model_path = os.path.join(self.config["models_dir"], "ppo_agent_final.pt")
        self.rl_agent.save_model(final_model_path)
        
        # Get performance summary
        performance_summary = self.performance_tracker.get_performance_summary()
        
        # Generate final performance plots
        self.performance_tracker.generate_performance_plots()
        
        # Return results
        results = {
            "success": True,
            "episodes_completed": episode,
            "best_episode": best_episode,
            "best_performance": best_performance,
            "performance_summary": performance_summary
        }
        
        logger.info(f"Training completed. Best performance: {best_performance:.4f} at episode {best_episode}")
        return results
    
    def evaluate(self, data_file, model_path=None):
        """
        Evaluate the agent on historical data.
        
        Args:
            data_file: Path to historical data file
            model_path: Path to model file (uses latest if None)
            
        Returns:
            Dictionary with evaluation results
        """
        # Load and preprocess data
        df = self.load_historical_data(data_file)
        if df is None:
            return {"success": False, "error": "Failed to load data"}
        
        df = self.preprocess_data(df)
        if df is None:
            return {"success": False, "error": "Failed to preprocess data"}
        
        # Initialize components
        if not self.initialize():
            return {"success": False, "error": "Failed to initialize components"}
        
        # Load specific model if provided
        if model_path and os.path.exists(model_path):
            self.rl_agent.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        
        # Create environment
        self.env = self.create_environment(df)
        if self.env is None:
            return {"success": False, "error": "Failed to create environment"}
        
        # Start tracking episode
        self.performance_tracker.start_episode()
        
        # Reset environment
        state = self.env.reset()
        done = False
        
        # Evaluation loop (no training)
        while not done:
            # Select action (no exploration)
            action, _, _ = self.rl_agent.select_action(state, training=False)
            
            # Take action
            next_state, reward, done, info = self.env.step(action)
            
            # Update state
            state = next_state
            
            # Update performance tracker
            self.performance_tracker.update_portfolio_value(info["portfolio_value"])
            self.performance_tracker.update_reward(reward)
            
            # Add trade to performance tracker if made
            if info["trade_made"]:
                self.performance_tracker.add_trade({
                    "step": self.env.current_step,
                    "action": action,
                    "price": info["current_price"],
                    "profit_loss": info.get("trade_profit", 0),
                    "portfolio_value": info["portfolio_value"]
                })
        
        # End episode and get metrics
        metrics = self.performance_tracker.end_episode()
        
        # Generate performance plots
        self.performance_tracker.generate_performance_plots()
        
        # Get trade statistics
        trade_stats = self.performance_tracker.calculate_trade_statistics()
        
        # Return results
        results = {
            "success": True,
            "total_return": metrics["total_return"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
            "trade_count": metrics["trade_count"],
            "performance_score": metrics["performance_score"],
            "trade_statistics": trade_stats
        }
        
        logger.info(f"Evaluation completed. Return: {metrics['total_return']:.4f}, "
                   f"Sharpe: {metrics['sharpe_ratio']:.4f}, "
                   f"Max DD: {metrics['max_drawdown']:.4f}")
        return results
    
    def backtest(self, data_file, model_path=None, plot_results=True):
        """
        Backtest the agent on historical data with detailed analysis.
        
        Args:
            data_file: Path to historical data file
            model_path: Path to model file (uses latest if None)
            plot_results: Whether to generate plots
            
        Returns:
            Dictionary with backtest results
        """
        # Run evaluation
        eval_results = self.evaluate(data_file, model_path)
        if not eval_results["success"]:
            return eval_results
        
        # Get additional backtest metrics
        if self.env is not None:
            # Get portfolio values
            portfolio_values = self.env.portfolio_values
            
            # Calculate drawdown series
            drawdown_series = self.performance_tracker.calculate_drawdown_series(portfolio_values)
            
            # Calculate underwater periods
            underwater_periods = self.performance_tracker.calculate_underwater_periods(portfolio_values)
            
            # Get trade history
            trade_history = self.env.trade_history
            
            # Generate equity curve plot if enabled
            if plot_results:
                self._generate_backtest_plots(portfolio_values, drawdown_series, trade_history)
            
            # Add to results
            eval_results.update({
                "portfolio_values": portfolio_values,
                "drawdown_series": drawdown_series,
                "underwater_periods": underwater_periods,
                "trade_history": trade_history
            })
        
        return eval_results
    
    def _generate_backtest_plots(self, portfolio_values, drawdown_series, trade_history):
        """
        Generate detailed backtest plots.
        
        Args:
            portfolio_values: List of portfolio values
            drawdown_series: List of drawdown values
            trade_history: List of trade dictionaries
        """
        # Create figure directory if it doesn't exist
        fig_dir = os.path.join(self.config["performance_dir"], 'backtest_figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values)
        plt.xlabel('Step')
        plt.ylabel('Portfolio Value')
        plt.title('Equity Curve')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(fig_dir, 'equity_curve.png'))
        plt.close()
        
        # Plot drawdown
        plt.figure(figsize=(12, 6))
        plt.plot(drawdown_series)
        plt.xlabel('Step')
        plt.ylabel('Drawdown')
        plt.title('Drawdown')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(fig_dir, 'drawdown.png'))
        plt.close()
        
        # Plot trade outcomes
        if trade_history:
            # Extract trade profits
            trade_profits = [trade.get('profit_loss', 0) for trade in trade_history if 'profit_loss' in trade]
            
            if trade_profits:
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(trade_profits)), trade_profits)
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.xlabel('Trade Number')
                plt.ylabel('Profit/Loss')
                plt.title('Trade Outcomes')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(fig_dir, 'trade_outcomes.png'))
                plt.close()
                
                # Plot cumulative trade profits
                plt.figure(figsize=(12, 6))
                plt.plot(np.cumsum(trade_profits))
                plt.xlabel('Trade Number')
                plt.ylabel('Cumulative Profit/Loss')
                plt.title('Cumulative Trade Profits')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(fig_dir, 'cumulative_trade_profits.png'))
                plt.close()
        
        logger.info(f"Backtest plots saved to {fig_dir}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and evaluate the trading bot on historical data')
    
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to historical data file')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'backtest'],
                        default='train', help='Operation mode')
    
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes to train')
    
    parser.add_argument('--save-interval', type=int, default=None,
                        help='Interval for saving model')
    
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model file for evaluation or backtest')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create trainer
    trainer = EnhancedTrainer(args.config)
    
    # Run in specified mode
    if args.mode == 'train':
        results = trainer.train(args.data_file, args.episodes, args.save_interval)
        if results["success"]:
            logger.info(f"Training completed successfully. Best performance: {results['best_performance']:.4f}")
        else:
            logger.error(f"Training failed: {results['error']}")
    
    elif args.mode == 'evaluate':
        results = trainer.evaluate(args.data_file, args.model_path)
        if results["success"]:
            logger.info(f"Evaluation completed successfully. Return: {results['total_return']:.4f}")
        else:
            logger.error(f"Evaluation failed: {results['error']}")
    
    elif args.mode == 'backtest':
        results = trainer.backtest(args.data_file, args.model_path)
        if results["success"]:
            logger.info(f"Backtest completed successfully. Return: {results['total_return']:.4f}")
        else:
            logger.error(f"Backtest failed: {results['error']}")

if __name__ == "__main__":
    main()
