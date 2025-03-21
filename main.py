"""
Main module for the Telegram bot for trading on Binance Futures.

This module integrates all components of the bot and manages its execution.
"""

import asyncio
import json
import logging
import os
import sys
from queue import Queue

# Import the enhanced logging setup
from debug_logger import setup_enhanced_logging

# Set up enhanced logging
logger = setup_enhanced_logging()

# Import components after setting up logging
from src.signal_parser import SignalParser
from src.telegram_client import TelegramClient, SignalProcessor
from src.binance_client import BinanceClient
from src.trade_executor import TradeExecutor
from src.risk_manager import RiskManager


async def main():
    """Main function to run the Telegram bot."""
    try:
        logger.info("Starting Telegram trading bot")
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Set logging level based on debug mode
            if config.get("debug", False):
                logger.setLevel(logging.DEBUG)
                logger.debug("Debug mode enabled")
        
        logger.info("Configuration loaded")
        
        # Initialize components
        signal_queue = Queue()
        
        # Initialize Binance client
        logger.info("Initializing Binance client with API key: " + config['binance']['api_key'][:5] + "...")
        binance_client = BinanceClient(
            api_key=config['binance']['api_key'],
            api_secret=config['binance']['api_secret'],
            testnet=config['binance']['testnet']
        )
        
        logger.info("Binance client initialized")
        
        # Initialize risk manager
        risk_manager = RiskManager(
            binance_client=binance_client,
            risk_per_trade=config['trading']['risk_per_trade'],
            max_total_risk=config['trading']['max_total_risk'],
            max_positions=config['trading']['max_positions']
        )
        
        logger.info("Risk manager initialized")
        
        # Initialize trade executor
        trade_executor = TradeExecutor(
            binance_client=binance_client,
            risk_manager=risk_manager,
            validate_trend=config['trading']['validate_trend'],
            use_volatility_sl=config['trading']['use_volatility_sl']
        )
        
        logger.info("Trade executor initialized")
        
        # Initialize Telegram client
        telegram_client = TelegramClient(
            token=config['telegram']['token'],
            signal_queue=signal_queue,
            allowed_chats=config['telegram']['allowed_chats'],
            allowed_users=config['telegram']['allowed_users'],
            admin_users=config['telegram']['admin_users']
        )
        
        logger.info("Telegram client initialized")
        
        # Initialize signal parser
        signal_parser = SignalParser()
        
        logger.info("Signal parser initialized")
        
        # Define callback function for signal parsing
        def parse_signal_callback(message_text):
            return signal_parser.parse_signal(message_text)
        
        # Initialize signal processor
        signal_processor = SignalProcessor(
            signal_queue=signal_queue,
            parser_callback=parse_signal_callback,
            telegram_client=telegram_client
        )
        
        logger.info("Signal processor initialized")
        
        # Start signal processor
        signal_processor.start()
        
        logger.info("Signal processor started")
        
        # Callback function to process parsed signals
        async def process_signal(signal):
            try:
                # Execute signal
                result = await trade_executor.execute_signal(signal)
                
                # Send result to Telegram client
                chat_id = signal.chat_id if hasattr(signal, 'chat_id') else None
                if chat_id and telegram_client:
                    await telegram_client.send_signal_result(chat_id, result.signal_id, result.to_dict())
                
                return result
            except Exception as e:
                logger.error(f"Error processing signal: {str(e)}")
                return None
        
        # Start Telegram client
        logger.info("Starting Telegram client...")
        telegram_task = telegram_client.start_async()
        
        # Wait for Telegram client to be ready
        await asyncio.sleep(2)
        
        logger.info("Bot started and waiting for signals")
        
        # Wait for Telegram client to stop
        await telegram_task
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        # Run main function
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        logger.critical("Stack trace:", exc_info=True)
