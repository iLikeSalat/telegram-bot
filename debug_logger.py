import logging
import os
import sys
import traceback

def setup_enhanced_logging():
    """Set up enhanced logging with detailed error tracking"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Create formatters
    standard_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    
    # File handler for all logs
    file_handler = logging.FileHandler('logs/telegram_binance_bot.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # File handler for errors only
    error_handler = logging.FileHandler('logs/errors.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(standard_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    # Set up exception hook to log uncaught exceptions
    def exception_hook(exc_type, exc_value, exc_traceback):
        root_logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    sys.excepthook = exception_hook
    
    return root_logger

def log_function_call(logger):
    """Decorator to log function calls with parameters and return values"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Calling {func_name} with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} returned: {result}")
                return result
            except Exception as e:
                logger.error(f"{func_name} raised exception: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        return wrapper
    return decorator
