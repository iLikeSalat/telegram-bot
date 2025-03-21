"""
Module for Binance API integration.

This module contains classes and functions needed to interact
with the Binance API for cryptocurrency trading.
Enhanced version with advanced order types and error handling.
"""

import logging
import time
import json
import asyncio
import hashlib
import hmac
import requests
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from binance.client import Client
from binance import AsyncClient
from binance.exceptions import BinanceAPIException, BinanceRequestException

# Logging configuration
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filename='telegram_binance_bot.log'
)
logger = logging.getLogger(__name__)

# Add handler to display logs in console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class OrderSide(str, Enum):
    """Order side (buy or sell)."""
    
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""
    
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"


class PositionSide(str, Enum):
    """Position side."""
    
    BOTH = "BOTH"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class MarketData:
    """Market data for a symbol."""
    
    symbol: str
    price: float
    timestamp: float
    volume_24h: float = 0.0
    price_change_24h: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    trend: str = "unknown"


class OrderExecutor:
    """
    Class for executing orders on Binance.
    
    This class provides methods for executing different types of orders
    on Binance, handling the specific details for each order type.
    """
    
    def __init__(self, client: Client):
        """
        Initialize the order executor.
        
        Args:
            client: The Binance client to use for executing orders.
        """
        self.client = client
        self.symbol_info_cache = {}
        self.cache_expiry = 3600  # 1 hour
        self.cache_timestamp = 0
    
    async def execute_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                           quantity: float, price: Optional[float] = None,
                           stop_price: Optional[float] = None, reduce_only: bool = False) -> Dict[str, Any]:
        """
        Execute an order on Binance.
        
        Args:
            symbol: The symbol to execute the order on.
            side: The order side (buy or sell).
            order_type: The order type.
            quantity: The quantity to buy or sell.
            price: The order price (for limit orders).
            stop_price: The trigger price (for stop orders).
            reduce_only: If True, the order can only reduce an existing position.
            
        Returns:
            A dictionary containing the executed order information.
        """
        logger.info(f"Executing order: {symbol} {side} {order_type}")
        logger.info(f"Parameters: quantity={quantity}, price={price}, stop_price={stop_price}, reduce_only={reduce_only}")
        
        try:
            # Round quantity and price according to symbol rules
            quantity = self._round_quantity(symbol, quantity)
            if price is not None:
                price = self._round_price(symbol, price)
            if stop_price is not None:
                stop_price = self._round_price(symbol, stop_price)
            
            # Prepare order parameters
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
                'reduceOnly': reduce_only
            }
            
            # Add parameters specific to order type
            if order_type == OrderType.LIMIT:
                params['timeInForce'] = 'GTC'
                params['price'] = price
            elif order_type in [OrderType.STOP_MARKET, OrderType.TAKE_PROFIT_MARKET]:
                params['stopPrice'] = stop_price
            elif order_type == OrderType.TRAILING_STOP_MARKET:
                params['callbackRate'] = price  # In this case, price is the callback rate
            
            # Execute the order
            logger.info(f"Sending order to Binance: {params}")
            response = await self.client.futures_create_order(**params)
            logger.info(f"Order executed successfully: {response}")
            
            return response
        
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e.status_code} - {e.message}")
            raise
        except BinanceRequestException as e:
            logger.error(f"Binance request error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error executing order: {str(e)}")
            raise
    
    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """
        Round quantity according to symbol rules.
        
        Args:
            symbol: The symbol to round the quantity for.
            quantity: The quantity to round.
            
        Returns:
            The rounded quantity.
        """
        try:
            # Get symbol information
            symbol_info = self._get_symbol_info(symbol)
            
            # Get quantity precision
            quantity_precision = symbol_info.get('quantityPrecision', 8)
            
            # Round quantity
            rounded_quantity = round(quantity, quantity_precision)
            
            return rounded_quantity
        
        except Exception as e:
            logger.warning(f"Error rounding quantity: {str(e)}. Using original quantity.")
            return quantity
    
    def _round_price(self, symbol: str, price: float) -> float:
        """
        Round price according to symbol rules.
        
        Args:
            symbol: The symbol to round the price for.
            price: The price to round.
            
        Returns:
            The rounded price.
        """
        try:
            # Get symbol information
            symbol_info = self._get_symbol_info(symbol)
            
            # Get price precision
            price_precision = symbol_info.get('pricePrecision', 8)
            
            # Round price
            rounded_price = round(price, price_precision)
            
            return rounded_price
        
        except Exception as e:
            logger.warning(f"Error rounding price: {str(e)}. Using original price.")
            return price
    
    def _get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol information.
        
        Args:
            symbol: The symbol to get information for.
            
        Returns:
            A dictionary containing symbol information.
        """
        # Check if information is in cache and still valid
        now = time.time()
        if now - self.cache_timestamp > self.cache_expiry:
            # Get information for all symbols
            exchange_info = self.client.futures_exchange_info()
            
            # Update cache
            self.symbol_info_cache = {}
            for symbol_info in exchange_info['symbols']:
                self.symbol_info_cache[symbol_info['symbol']] = symbol_info
            
            self.cache_timestamp = now
        
        # Get symbol information from cache
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]
        
        # If symbol is not in cache, get information directly
        exchange_info = self.client.futures_exchange_info()
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                return symbol_info
        
        # If symbol is not found, return empty dictionary
        return {}


class BinanceClient:
    """
    Client for interacting with the Binance API.
    
    This class provides methods for interacting with the Binance API,
    handling authentication, requests, and responses.
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, tld: str = 'com'):
        """
        Initialize the Binance client.
        
        Args:
            api_key: The Binance API key.
            api_secret: The Binance API secret.
            testnet: If True, use the Binance testnet.
            tld: The TLD to use for the Binance API.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.tld = tld
        
        # Initialize Binance client
        self.client = Client(api_key, api_secret, testnet=testnet, tld=tld)
        
        # Initialize order executor
        self.order_executor = OrderExecutor(self.client)
        
        # Update exchange information
        self._update_exchange_info()
        
        logger.info(f"Binance client initialized (testnet: {testnet})")
    
    def _update_exchange_info(self):
        """Update exchange information."""
        try:
            self.exchange_info = self.client.futures_exchange_info()
            logger.info("Exchange information updated")
        except Exception as e:
            logger.error(f"Error updating exchange information: {str(e)}")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            A dictionary containing account information.
        """
        try:
            logger.info("Retrieving account information")
            account_info = self.client.futures_account()
            logger.info(f"Account information retrieved successfully: Balance={account_info.get('totalWalletBalance', 'N/A')}")
            return account_info
        except Exception as e:
            logger.error(f"Error retrieving account information: {str(e)}")
            raise
    
    async def get_symbol_price(self, symbol: str) -> float:
        """
        Get the current price of a symbol.
        
        Args:
            symbol: The symbol to get the price for.
            
        Returns:
            The current price of the symbol.
        """
        try:
            logger.info(f"Retrieving price for {symbol}")
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            logger.info(f"Current price for {symbol}: {price}")
            return price
        except Exception as e:
            logger.error(f"Error retrieving price for {symbol}: {str(e)}")
            raise
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """
        Get market data for a symbol.
        
        Args:
            symbol: The symbol to get data for.
            
        Returns:
            A MarketData object containing market data.
        """
        try:
            logger.info(f"Retrieving market data for {symbol}")
            
            # Get 24h ticker
            ticker_24h = self.client.futures_ticker(symbol=symbol)
            
            # Get order book
            order_book = self.client.futures_order_book(symbol=symbol)
            
            # Create MarketData object
            market_data = MarketData(
                symbol=symbol,
                price=float(ticker_24h['lastPrice']),
                timestamp=time.time(),
                volume_24h=float(ticker_24h['volume']),
                price_change_24h=float(ticker_24h['priceChangePercent']),
                high_24h=float(ticker_24h['highPrice']),
                low_24h=float(ticker_24h['lowPrice']),
                bid=float(order_book['bids'][0][0]) if order_book['bids'] else 0.0,
                ask=float(order_book['asks'][0][0]) if order_book['asks'] else 0.0,
                trend="uptrend" if float(ticker_24h['priceChangePercent']) > 0 else "downtrend"
            )
            
            logger.info(f"Market data retrieved for {symbol}: price={market_data.price}, trend={market_data.trend}")
            return market_data
        
        except Exception as e:
            logger.error(f"Error retrieving market data for {symbol}: {str(e)}")
            # In case of error, return minimal data
            return MarketData(
                symbol=symbol,
                price=0.0,
                timestamp=time.time()
            )
    
    async def get_klines(self, symbol: str, interval: str, limit: int) -> List[List[Any]]:
        """
        Get klines (candles) for a symbol.
        
        Args:
            symbol: The symbol to get klines for.
            interval: The kline interval (1m, 5m, 15m, 1h, 4h, 1d, etc.).
            limit: The number of klines to get.
            
        Returns:
            A list of klines.
        """
        try:
            logger.info(f"Retrieving {limit} klines for {symbol} with interval {interval}")
            klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            logger.info(f"Retrieved {len(klines)} klines for {symbol}")
            return klines
        except Exception as e:
            logger.error(f"Error retrieving klines for {symbol}: {str(e)}")
            return []
    
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                         quantity: float, price: Optional[float] = None,
                         stop_price: Optional[float] = None, reduce_only: bool = False) -> Dict[str, Any]:
        """
        Place an order on Binance.
        
        Args:
            symbol: The symbol to place the order on.
            side: The order side (buy or sell).
            order_type: The order type.
            quantity: The quantity to buy or sell.
            price: The order price (for limit orders).
            stop_price: The trigger price (for stop orders).
            reduce_only: If True, the order can only reduce an existing position.
            
        Returns:
            A dictionary containing the placed order information.
        """
        logger.info(f"Attempting to execute order: {symbol} {side} {order_type}")
        logger.info(f"Parameters: quantity={quantity}, price={price}, stop_price={stop_price}, reduce_only={reduce_only}")
        try:
            result = await self.order_executor.execute_order(symbol, side, order_type, quantity, price, stop_price, reduce_only)
            logger.info(f"Order executed successfully: {result}")
            return result
        except Exception as e:
            logger.error(f"Error executing order: {str(e)}")
            logger.error(f"Binance API details: {e}")
            raise
    
    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        Set leverage for a symbol.
        
        Args:
            symbol: The symbol to set leverage for.
            leverage: The leverage to set.
            
        Returns:
            A dictionary containing the response information.
        """
        try:
            logger.info(f"Setting leverage for {symbol} to {leverage}x")
            response = self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            logger.info(f"Leverage set successfully for {symbol}: {leverage}x")
            return response
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {str(e)}")
            raise
    
    async def set_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """
        Set margin type for a symbol.
        
        Args:
            symbol: The symbol to set margin type for.
            margin_type: The margin type to set (ISOLATED or CROSSED).
            
        Returns:
            A dictionary containing the response information.
        """
        try:
            logger.info(f"Setting margin type for {symbol} to {margin_type}")
            response = self.client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
            logger.info(f"Margin type set successfully for {symbol}: {margin_type}")
            return response
        except BinanceAPIException as e:
            # Ignore error if margin type is already set
            if e.code == -4046:  # "No need to change margin type."
                logger.info(f"Margin type already set to {margin_type} for {symbol}")
                return {"msg": "Margin type already set", "code": 0}
            else:
                logger.error(f"Error setting margin type for {symbol}: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Error setting margin type for {symbol}: {str(e)}")
            raise
    
    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get open positions.
        
        Returns:
            A list of open positions.
        """
        try:
            logger.info("Retrieving open positions")
            account_info = await self.get_account_info()
            positions = account_info['positions']
            
            # Filter positions with non-zero quantity
            open_positions = [p for p in positions if float(p['positionAmt']) != 0]
            
            logger.info(f"Retrieved {len(open_positions)} open positions")
            return open_positions
        except Exception as e:
            logger.error(f"Error retrieving open positions: {str(e)}")
            return []
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders.
        
        Args:
            symbol: The symbol to get orders for (optional).
            
        Returns:
            A list of open orders.
        """
        try:
            logger.info(f"Retrieving open orders for {symbol if symbol else 'all symbols'}")
            
            if symbol:
                open_orders = self.client.futures_get_open_orders(symbol=symbol)
            else:
                open_orders = self.client.futures_get_open_orders()
            
            logger.info(f"Retrieved {len(open_orders)} open orders")
            return open_orders
        except Exception as e:
            logger.error(f"Error retrieving open orders: {str(e)}")
            return []
    
    async def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            symbol: The symbol of the order to cancel.
            order_id: The ID of the order to cancel.
            
        Returns:
            A dictionary containing the response information.
        """
        try:
            logger.info(f"Cancelling order {order_id} for {symbol}")
            response = self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"Order {order_id} cancelled successfully")
            return response
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            raise
    
    async def close_position(self, symbol: str, position_side: PositionSide) -> Dict[str, Any]:
        """
        Close a position.
        
        Args:
            symbol: The symbol of the position to close.
            position_side: The side of the position to close.
            
        Returns:
            A dictionary containing the response information.
        """
        try:
            logger.info(f"Closing position for {symbol} (side: {position_side})")
            
            # Get open positions
            positions = await self.get_open_positions()
            
            # Find position to close
            position = None
            for p in positions:
                if p['symbol'] == symbol and p['positionSide'] == position_side:
                    position = p
                    break
            
            if not position:
                logger.warning(f"No open position found for {symbol} (side: {position_side})")
                return {"msg": "No open position found", "code": 0}
            
            # Determine order side to close position
            position_amt = float(position['positionAmt'])
            side = OrderSide.SELL if position_amt > 0 else OrderSide.BUY
            
            # Place market order to close position
            response = await self.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=abs(position_amt),
                reduce_only=True
            )
            
            logger.info(f"Position closed successfully for {symbol}")
            return response
        
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {str(e)}")
            raise
    
    def close(self):
        """Close the Binance client."""
        try:
            self.client.close_connection()
            logger.info("Binance client connection closed")
        except Exception as e:
            logger.error(f"Error closing Binance client connection: {str(e)}")


def main():
    """Main function to test the module."""
    # Create a Binance client (replace with your API keys)
    client = BinanceClient(
        api_key="YOUR_API_KEY",
        api_secret="YOUR_API_SECRET",
        testnet=True  # Use testnet for testing
    )
    
    # Example usage
    async def test():
        try:
            # Get BTC price
            btc_price = await client.get_symbol_price("BTCUSDT")
            print(f"BTC price: {btc_price}")
            
            # Get market data
            market_data = await client.get_market_data("BTCUSDT")
            print(f"Market data: {market_data}")
            
            # Get klines
            klines = await client.get_klines("BTCUSDT", "1h", 10)
            print(f"Klines: {klines[:2]}...")
            
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            # Close client
            client.close()
    
    # Run test function
    import asyncio
    asyncio.run(test())


if __name__ == "__main__":
    main()
