"""
Backtesting Module for the Autonomous Trading Bot

This module implements comprehensive backtesting functionality
for evaluating trading strategies on historical data.

Features:
- Train/test split for proper validation
- Out-of-sample performance evaluation
- Detailed performance metrics
- Overfitting detection
- Walk-forward testing
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json

# Import custom modules
from trading_environment import TradingEnvironment
from rl_model import PPOAgent
from risk_management import RiskManager, PositionManager
from performance_tracking import PerformanceTracker

# Configure logging
logger = logging.getLogger(__name__)

class Backtester:
    """
    Backtester for evaluating trading strategies on historical data.
    
    Features:
    - Train/test split for proper validation
    - Out-of-sample performance evaluation
    - Detailed performance metrics
    - Overfitting detection
    - Walk-forward testing
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        results_dir: str = "backtest_results",
        train_test_split: float = 0.7,
        use_walk_forward: bool = True,
        walk_forward_windows: int = 5
    ):
        """
        Initialize the backtester.
        
        Args:
            data_dir: Directory with historical data
            results_dir: Directory to save backtest results
            train_test_split: Fraction of data to use for training
            use_walk_forward: Whether to use walk-forward testing
            walk_forward_windows: Number of windows for walk-forward testing
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.train_test_split = train_test_split
        self.use_walk_forward = use_walk_forward
        self.walk_forward_windows = walk_forward_windows
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize components
        self.performance_tracker = PerformanceTracker(data_dir=results_dir)
        
        logger.info("Backtester initialized")
    
    def load_data(self, data_file: str) -> pd.DataFrame:
        """
        Load historical data from file.
        
        Args:
            data_file: Path to data file
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Check if file exists
            if not os.path.exists(data_file):
                logger.error(f"Data file not found: {data_file}")
                return None
            
            # Load data
            df = pd.read_csv(data_file)
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"Loaded data from {data_file} with {len(df)} rows")
            return df
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for backtesting.
        
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
            
            # Add technical indicators
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
            
            logger.info(f"Preprocessed data shape: {df.shape}")
            return df
        
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            df: DataFrame with preprocessed data
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Calculate split index
        split_idx = int(len(df) * self.train_test_split)
        
        # Split data
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Split data into train ({len(train_df)} rows) and test ({len(test_df)} rows)")
        return train_df, test_df
    
    def create_walk_forward_windows(
        self,
        df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward testing windows.
        
        Args:
            df: DataFrame with preprocessed data
            
        Returns:
            List of (train_df, test_df) tuples
        """
        windows = []
        
        # Calculate window size
        window_size = len(df) // self.walk_forward_windows
        
        for i in range(self.walk_forward_windows):
            # Calculate indices
            test_start = i * window_size
            test_end = (i + 1) * window_size
            
            # For the first window, use only future data for training
            if i == 0:
                train_df = df.iloc[test_end:].copy()
                test_df = df.iloc[test_start:test_end].copy()
            # For the last window, use only past data for training
            elif i == self.walk_forward_windows - 1:
                train_df = df.iloc[:test_start].copy()
                test_df = df.iloc[test_start:].copy()
            # For middle windows, use both past and future data for training
            else:
                train_df = pd.concat([
                    df.iloc[:test_start].copy(),
                    df.iloc[test_end:].copy()
                ])
                test_df = df.iloc[test_start:test_end].copy()
            
            windows.append((train_df, test_df))
            
            logger.debug(f"Window {i+1}: Train {len(train_df)} rows, Test {len(test_df)} rows")
        
        logger.info(f"Created {len(windows)} walk-forward windows")
        return windows
    
    def backtest_strategy(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        agent: PPOAgent,
        config: Dict[str, Any],
        window_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Backtest a trading strategy.
        
        Args:
            train_df: Training data
            test_df: Testing data
            agent: PPO agent
            config: Configuration dictionary
            window_id: Window ID for walk-forward testing
            
        Returns:
            Dictionary with backtest results
        """
        # Create environments
        train_env = TradingEnvironment(
            df=train_df,
            initial_balance=config.get('initial_balance', 10000.0),
            transaction_fee=config.get('transaction_fee', 0.0004),
            window_size=config.get('window_size', 20),
            max_position_size=config.get('max_position_size', 0.1),
            max_drawdown=config.get('max_drawdown', 0.15),
            use_risk_adjusted_rewards=config.get('use_risk_adjusted_rewards', True),
            use_position_features=config.get('use_position_features', True),
            use_technical_indicators=config.get('use_technical_indicators', True)
        )
        
        test_env = TradingEnvironment(
            df=test_df,
            initial_balance=config.get('initial_balance', 10000.0),
            transaction_fee=config.get('transaction_fee', 0.0004),
            window_size=config.get('window_size', 20),
            max_position_size=config.get('max_position_size', 0.1),
            max_drawdown=config.get('max_drawdown', 0.15),
            use_risk_adjusted_rewards=config.get('use_risk_adjusted_rewards', True),
            use_position_features=config.get('use_position_features', True),
            use_technical_indicators=config.get('use_technical_indicators', True)
        )
        
        # Train agent
        logger.info("Training agent on training data...")
        self._train_agent(agent, train_env, config)
        
        # Test agent
        logger.info("Testing agent on testing data...")
        test_results = self._test_agent(agent, test_env)
        
        # Calculate overfitting metrics
        train_results = self._test_agent(agent, train_env)
        overfitting_metrics = self._calculate_overfitting_metrics(train_results, test_results)
        
        # Combine results
        results = {
            'train_results': train_results,
            'test_results': test_results,
            'overfitting_metrics': overfitting_metrics,
            'window_id': window_id
        }
        
        # Save results
        self._save_results(results, window_id)
        
        return results
    
    def _train_agent(
        self,
        agent: PPOAgent,
        env: TradingEnvironment,
        config: Dict[str, Any]
    ):
        """
        Train agent on environment.
        
        Args:
            agent: PPO agent
            env: Trading environment
            config: Configuration dictionary
        """
        # Get training parameters
        episodes = config.get('episodes', 100)
        
        for episode in range(1, episodes + 1):
            # Reset environment
            state = env.reset()
            done = False
            episode_reward = 0
            
            # Episode loop
            while not done:
                # Select action
                action, action_prob, value = agent.select_action(state)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                agent.store_transition(state, action, action_prob, reward, value, done)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
            
            # Update agent
            if len(agent.states) > 0:
                # Get value estimate of final state
                _, _, next_value = agent.select_action(state)
                
                # Update policy and value networks
                agent.update(next_value)
            
            # Log progress
            if episode % 10 == 0:
                logger.info(f"Episode {episode}/{episodes} - Reward: {episode_reward:.4f}")
    
    def _test_agent(
        self,
        agent: PPOAgent,
        env: TradingEnvironment
    ) -> Dict[str, Any]:
        """
        Test agent on environment.
        
        Args:
            agent: PPO agent
            env: Trading environment
            
        Returns:
            Dictionary with test results
        """
        # Reset environment
        state = env.reset()
        done = False
        
        # Start tracking performance
        self.performance_tracker.start_episode()
        
        # Episode loop
        while not done:
            # Select action (no exploration)
            action, _, _ = agent.select_action(state, training=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update state
            state = next_state
            
            # Update performance tracker
            self.performance_tracker.update_portfolio_value(info["portfolio_value"])
            self.performance_tracker.update_reward(reward)
            
            # Add trade to performance tracker if made
            if info["trade_made"]:
                self.performance_tracker.add_trade({
                    "step": env.current_step,
                    "action": action,
                    "price": info["current_price"],
                    "profit_loss": info.get("trade_profit", 0),
                    "portfolio_value": info["portfolio_value"]
                })
        
        # End episode and get metrics
        metrics = self.performance_tracker.end_episode()
        
        # Get environment performance stats
        env_stats = env.get_performance_stats()
        
        # Combine metrics
        results = {**metrics, **env_stats}
        
        return results
    
    def _calculate_overfitting_metrics(
        self,
        train_results: Dict[str, Any],
        test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate overfitting metrics.
        
        Args:
            train_results: Training results
            test_results: Testing results
            
        Returns:
            Dictionary with overfitting metrics
        """
        # Calculate performance differences
        return_diff = train_results['total_return'] - test_results['total_return']
        sharpe_diff = train_results['sharpe_ratio'] - test_results['sharpe_ratio']
        drawdown_diff = test_results['max_drawdown'] - train_results['max_drawdown']
        
        # Calculate overfitting score
        # Higher score indicates more overfitting
        overfitting_score = (
            0.4 * return_diff +  # 40% weight on return difference
            0.3 * sharpe_diff +  # 30% weight on Sharpe difference
            0.3 * drawdown_diff  # 30% weight on drawdown difference
        )
        
        # Determine overfitting level
        if overfitting_score < 0.1:
            overfitting_level = "Low"
        elif overfitting_score < 0.3:
            overfitting_level = "Medium"
        else:
            overfitting_level = "High"
        
        return {
            'return_diff': return_diff,
            'sharpe_diff': sharpe_diff,
            'drawdown_diff': drawdown_diff,
            'overfitting_score': overfitting_score,
            'overfitting_level': overfitting_level
        }
    
    def _save_results(self, results: Dict[str, Any], window_id: Optional[int] = None):
        """
        Save backtest results to disk.
        
        Args:
            results: Backtest results
            window_id: Window ID for walk-forward testing
        """
        # Create results directory
        if window_id is not None:
            results_dir = os.path.join(self.results_dir, f"window_{window_id}")
        else:
            results_dir = os.path.join(self.results_dir, "single_backtest")
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results as JSON
        with open(os.path.join(results_dir, "results.json"), "w") as f:
            # Convert numpy values to Python types
            results_json = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    results_json[k] = {}
                    for kk, vv in v.items():
                        if isinstance(vv, (np.int64, np.int32, np.int16, np.int8)):
                            results_json[k][kk] = int(vv)
                        elif isinstance(vv, (np.float64, np.float32, np.float16)):
                            results_json[k][kk] = float(vv)
                        else:
                            results_json[k][kk] = vv
                else:
                    results_json[k] = v
            
            json.dump(results_json, f, indent=4)
        
        logger.info(f"Results saved to {results_dir}")
    
    def run_backtest(
        self,
        data_file: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a backtest on historical data.
        
        Args:
            data_file: Path to data file
            config: Configuration dictionary
            
        Returns:
            Dictionary with backtest results
        """
        # Load and preprocess data
        df = self.load_data(data_file)
        if df is None:
            return {"success": False, "error": "Failed to load data"}
        
        df = self.preprocess_data(df)
        if df is None:
            return {"success": False, "error": "Failed to preprocess data"}
        
        # Initialize agent
        agent = PPOAgent(
            state_dim=config.get('state_dim', 30),
            action_dim=config.get('action_dim', 3),
            hidden_dims=config.get('hidden_dims', [256, 128, 64]),
            lr_policy=config.get('learning_rate', 1e-4),
            gamma=config.get('gamma', 0.99),
            gae_lambda=config.get('gae_lambda', 0.95),
            clip_ratio=config.get('clip_ratio', 0.2),
            value_coef=config.get('value_coef', 0.5),
            entropy_coef=config.get('entropy_coef', 0.01),
            device=config.get('device', None)
        )
        
        # Run backtest
        if self.use_walk_forward:
            # Create walk-forward windows
            windows = self.create_walk_forward_windows(df)
            
            # Run backtest on each window
            window_results = []
            for i, (train_df, test_df) in enumerate(windows):
                logger.info(f"Running backtest on window {i+1}/{len(windows)}")
                results = self.backtest_strategy(train_df, test_df, agent, config, i)
                window_results.append(results)
            
            # Aggregate results
            aggregated_results = self._aggregate_window_results(window_results)
            
            return {
                "success": True,
                "window_results": window_results,
                "aggregated_results": aggregated_results
            }
        else:
            # Split data
            train_df, test_df = self.split_data(df)
            
            # Run backtest
            results = self.backtest_strategy(train_df, test_df, agent, config)
            
            return {
                "success": True,
                "results": results
            }
    
    def _aggregate_window_results(
        self,
        window_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple windows.
        
        Args:
            window_results: List of window results
            
        Returns:
            Dictionary with aggregated results
        """
        # Extract test results from each window
        test_returns = [r['test_results']['total_return'] for r in window_results]
        test_sharpes = [r['test_results']['sharpe_ratio'] for r in window_results]
        test_drawdowns = [r['test_results']['max_drawdown'] for r in window_results]
        test_win_rates = [r['test_results']['win_rate'] for r in window_results]
        
        # Extract overfitting metrics from each window
        overfitting_scores = [r['overfitting_metrics']['overfitting_score'] for r in window_results]
        
        # Calculate aggregated metrics
        avg_return = np.mean(test_returns)
        avg_sharpe = np.mean(test_sharpes)
        avg_drawdown = np.mean(test_drawdowns)
        avg_win_rate = np.mean(test_win_rates)
        avg_overfitting = np.mean(overfitting_scores)
        
        # Calculate consistency metrics
        return_std = np.std(test_returns)
        sharpe_std = np.std(test_sharpes)
        drawdown_std = np.std(test_drawdowns)
        
        # Calculate consistency score
        # Lower score indicates more consistency
        consistency_score = (
            0.4 * return_std / (abs(avg_return) + 1e-8) +  # 40% weight on return consistency
            0.3 * sharpe_std / (abs(avg_sharpe) + 1e-8) +  # 30% weight on Sharpe consistency
            0.3 * drawdown_std / (avg_drawdown + 1e-8)     # 30% weight on drawdown consistency
        )
        
        # Determine consistency level
        if consistency_score < 0.3:
            consistency_level = "High"
        elif consistency_score < 0.7:
            consistency_level = "Medium"
        else:
            consistency_level = "Low"
        
        return {
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_drawdown': avg_drawdown,
            'avg_win_rate': avg_win_rate,
            'avg_overfitting': avg_overfitting,
            'return_std': return_std,
            'sharpe_std': sharpe_std,
            'drawdown_std': drawdown_std,
            'consistency_score': consistency_score,
            'consistency_level': consistency_level
        }
    
    def generate_backtest_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a backtest report.
        
        Args:
            results: Backtest results
            
        Returns:
            Path to report file
        """
        # Create report directory
        report_dir = os.path.join(self.results_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(report_dir, f"backtest_report_{timestamp}.html")
        
        # Generate HTML report
        with open(report_file, "w") as f:
            f.write("<html>\n")
            f.write("<head>\n")
            f.write("<title>Backtest Report</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
            f.write("h1 { color: #333366; }\n")
            f.write("h2 { color: #666699; }\n")
            f.write("table { border-collapse: collapse; width: 100%; }\n")
            f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
            f.write("th { background-color: #f2f2f2; }\n")
            f.write("tr:nth-child(even) { background-color: #f9f9f9; }\n")
            f.write(".good { color: green; }\n")
            f.write(".bad { color: red; }\n")
            f.write(".warning { color: orange; }\n")
            f.write("</style>\n")
            f.write("</head>\n")
            f.write("<body>\n")
            
            # Report header
            f.write(f"<h1>Backtest Report - {timestamp}</h1>\n")
            
            if self.use_walk_forward:
                # Walk-forward results
                f.write("<h2>Walk-Forward Test Results</h2>\n")
                
                # Aggregated results
                agg = results['aggregated_results']
                f.write("<h3>Aggregated Results</h3>\n")
                f.write("<table>\n")
                f.write("<tr><th>Metric</th><th>Value</th></tr>\n")
                f.write(f"<tr><td>Average Return</td><td>{agg['avg_return']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Average Sharpe Ratio</td><td>{agg['avg_sharpe']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Average Max Drawdown</td><td>{agg['avg_drawdown']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Average Win Rate</td><td>{agg['avg_win_rate']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Average Overfitting Score</td><td>{agg['avg_overfitting']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Return Standard Deviation</td><td>{agg['return_std']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Sharpe Standard Deviation</td><td>{agg['sharpe_std']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Drawdown Standard Deviation</td><td>{agg['drawdown_std']:.4f}</td></tr>\n")
                
                # Add consistency level with color
                consistency_class = "good" if agg['consistency_level'] == "High" else \
                                   "warning" if agg['consistency_level'] == "Medium" else "bad"
                f.write(f"<tr><td>Consistency Level</td><td class='{consistency_class}'>{agg['consistency_level']}</td></tr>\n")
                
                f.write("</table>\n")
                
                # Window results
                f.write("<h3>Window Results</h3>\n")
                f.write("<table>\n")
                f.write("<tr><th>Window</th><th>Train Return</th><th>Test Return</th><th>Overfitting Score</th><th>Overfitting Level</th></tr>\n")
                
                for i, window in enumerate(results['window_results']):
                    train_return = window['train_results']['total_return']
                    test_return = window['test_results']['total_return']
                    overfitting_score = window['overfitting_metrics']['overfitting_score']
                    overfitting_level = window['overfitting_metrics']['overfitting_level']
                    
                    # Add color based on overfitting level
                    overfitting_class = "good" if overfitting_level == "Low" else \
                                       "warning" if overfitting_level == "Medium" else "bad"
                    
                    f.write(f"<tr><td>{i+1}</td><td>{train_return:.4f}</td><td>{test_return:.4f}</td>"
                           f"<td>{overfitting_score:.4f}</td><td class='{overfitting_class}'>{overfitting_level}</td></tr>\n")
                
                f.write("</table>\n")
            else:
                # Single backtest results
                f.write("<h2>Backtest Results</h2>\n")
                
                # Training results
                train = results['results']['train_results']
                f.write("<h3>Training Results</h3>\n")
                f.write("<table>\n")
                f.write("<tr><th>Metric</th><th>Value</th></tr>\n")
                f.write(f"<tr><td>Total Return</td><td>{train['total_return']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Sharpe Ratio</td><td>{train['sharpe_ratio']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Max Drawdown</td><td>{train['max_drawdown']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Win Rate</td><td>{train['win_rate']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Trade Count</td><td>{train['trade_count']}</td></tr>\n")
                f.write("</table>\n")
                
                # Testing results
                test = results['results']['test_results']
                f.write("<h3>Testing Results</h3>\n")
                f.write("<table>\n")
                f.write("<tr><th>Metric</th><th>Value</th></tr>\n")
                f.write(f"<tr><td>Total Return</td><td>{test['total_return']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Sharpe Ratio</td><td>{test['sharpe_ratio']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Max Drawdown</td><td>{test['max_drawdown']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Win Rate</td><td>{test['win_rate']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Trade Count</td><td>{test['trade_count']}</td></tr>\n")
                f.write("</table>\n")
                
                # Overfitting metrics
                overfitting = results['results']['overfitting_metrics']
                f.write("<h3>Overfitting Analysis</h3>\n")
                f.write("<table>\n")
                f.write("<tr><th>Metric</th><th>Value</th></tr>\n")
                f.write(f"<tr><td>Return Difference</td><td>{overfitting['return_diff']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Sharpe Difference</td><td>{overfitting['sharpe_diff']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Drawdown Difference</td><td>{overfitting['drawdown_diff']:.4f}</td></tr>\n")
                f.write(f"<tr><td>Overfitting Score</td><td>{overfitting['overfitting_score']:.4f}</td></tr>\n")
                
                # Add overfitting level with color
                overfitting_class = "good" if overfitting['overfitting_level'] == "Low" else \
                                   "warning" if overfitting['overfitting_level'] == "Medium" else "bad"
                f.write(f"<tr><td>Overfitting Level</td><td class='{overfitting_class}'>{overfitting['overfitting_level']}</td></tr>\n")
                
                f.write("</table>\n")
            
            # Conclusion
            f.write("<h2>Conclusion</h2>\n")
            
            if self.use_walk_forward:
                agg = results['aggregated_results']
                
                if agg['avg_return'] > 0 and agg['consistency_level'] in ["High", "Medium"] and agg['avg_overfitting'] < 0.3:
                    f.write("<p class='good'>The strategy shows positive returns with good consistency and low overfitting. "
                           "It may be suitable for live trading after further validation.</p>\n")
                elif agg['avg_return'] > 0 and agg['consistency_level'] == "Medium" and agg['avg_overfitting'] < 0.5:
                    f.write("<p class='warning'>The strategy shows positive returns with moderate consistency and moderate overfitting. "
                           "Further optimization and validation is recommended before live trading.</p>\n")
                else:
                    f.write("<p class='bad'>The strategy shows poor performance, inconsistency, or high overfitting. "
                           "It is not recommended for live trading without significant improvements.</p>\n")
            else:
                overfitting = results['results']['overfitting_metrics']
                test = results['results']['test_results']
                
                if test['total_return'] > 0 and overfitting['overfitting_level'] == "Low":
                    f.write("<p class='good'>The strategy shows positive returns on out-of-sample data with low overfitting. "
                           "It may be suitable for live trading after further validation.</p>\n")
                elif test['total_return'] > 0 and overfitting['overfitting_level'] == "Medium":
                    f.write("<p class='warning'>The strategy shows positive returns on out-of-sample data but with moderate overfitting. "
                           "Further optimization and validation is recommended before live trading.</p>\n")
                else:
                    f.write("<p class='bad'>The strategy shows poor performance on out-of-sample data or high overfitting. "
                           "It is not recommended for live trading without significant improvements.</p>\n")
            
            f.write("</body>\n")
            f.write("</html>\n")
        
        logger.info(f"Backtest report generated: {report_file}")
        return report_file
