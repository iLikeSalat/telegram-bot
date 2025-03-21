"""
Diagnostic tool to check the environment and configuration.
"""

import os
import sys
import json
import logging
import platform
import requests
import socket
import traceback

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check Python version."""
    logger.info(f"Python version: {platform.python_version()}")
    if sys.version_info < (3, 7):
        logger.warning("Python version is below 3.7, which may cause compatibility issues")
    return True

def check_dependencies():
    """Check required dependencies."""
    required_packages = [
        "python-telegram-bot",
        "python-binance",
        "numpy",
        "pandas",
        "aiohttp",
        "asyncio"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_") )
            logger.info(f"Package {package} is installed")
        except ImportError:
            logger.error(f"Package {package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with: pip install " + " ".join(missing_packages))
        return False
    return True

def check_config():
    """Check configuration file."""
    if not os.path.exists("config.json"):
        logger.error("config.json file not found")
        return False
    
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        
        # Check Telegram token
        if not config.get("telegram", {}).get("token"):
            logger.error("Telegram token is missing in config.json")
            return False
        
        # Check Binance API keys
        if not config.get("binance", {}).get("api_key") or not config.get("binance", {}).get("api_secret"):
            logger.error("Binance API keys are missing in config.json")
            return False
        
        logger.info("Configuration file is valid")
        return True
    except json.JSONDecodeError:
        logger.error("config.json is not a valid JSON file")
        return False
    except Exception as e:
        logger.error(f"Error checking config: {str(e)}")
        return False

def check_internet_connection():
    """Check internet connection."""
    try:
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=5) 
        if response.status_code == 200:
            logger.info("Internet connection is working")
            logger.info("Binance API is accessible")
            return True
        else:
            logger.error(f"Binance API returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Internet connection error: {str(e)}")
        return False

def check_ip_address():
    """Check public IP address."""
    try:
        response = requests.get("https://api.ipify.org?format=json", timeout=5) 
        if response.status_code == 200:
            ip = response.json()["ip"]
            logger.info(f"Your public IP address is: {ip}")
            logger.info("Make sure this IP is whitelisted in your Binance API settings")
            return True
        else:
            logger.error("Could not determine public IP address")
            return False
    except Exception as e:
        logger.error(f"Error checking IP address: {str(e)}")
        return False

def run_diagnostics():
    """Run all diagnostic checks."""
    logger.info("Starting diagnostic checks...")
    
    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Configuration", check_config),
        ("Internet connection", check_internet_connection),
        ("IP address", check_ip_address)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Error during {name} check: {str(e)}")
            logger.error(traceback.format_exc())
            results.append((name, False))
    
    # Print summary
    logger.info("\n=== Diagnostic Summary ===")
    all_passed = True
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\nAll checks passed! Your environment should be ready to run the bot.")
    else:
        logger.warning("\nSome checks failed. Please fix the issues before running the bot.")

if __name__ == "__main__":
    run_diagnostics()
