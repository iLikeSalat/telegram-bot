"""
Module d'intégration avec l'API Binance.

Ce module contient les classes et fonctions nécessaires pour interagir
avec l'API Binance pour le trading de crypto-monnaies.
Version améliorée avec types d'ordres avancés et gestion des erreurs.
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

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filename='telegram_binance_bot.log'
)
logger = logging.getLogger(__name__)

# Ajouter un handler pour afficher les logs dans la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class OrderSide(str, Enum):
    """Côté de l'ordre (achat ou vente)."""
    
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Type d'ordre."""
    
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"


class PositionSide(str, Enum):
    """Côté de la position."""
    
    BOTH = "BOTH"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class MarketData:
    """Données de marché pour un symbole."""
    
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
    Classe pour exécuter des ordres sur Binance.
    
    Cette classe fournit des méthodes pour exécuter différents types d'ordres
    sur Binance, en gérant les détails spécifiques à chaque type d'ordre.
    """
    
    def __init__(self, client: Client):
        """
        Initialise l'exécuteur d'ordres.
        
        Args:
            client: Le client Binance à utiliser pour exécuter les ordres.
        """
        self.client = client
        self.symbol_info_cache = {}
        self.cache_expiry = 3600  # 1 heure
        self.cache_timestamp = 0
    
    async def execute_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                           quantity: float, price: Optional[float] = None,
                           stop_price: Optional[float] = None, reduce_only: bool = False) -> Dict[str, Any]:
        """
        Exécute un ordre sur Binance.
        
        Args:
            symbol: Le symbole sur lequel exécuter l'ordre.
            side: Le côté de l'ordre (achat ou vente).
            order_type: Le type d'ordre.
            quantity: La quantité à acheter ou vendre.
            price: Le prix de l'ordre (pour les ordres limites).
            stop_price: Le prix de déclenchement (pour les ordres stop).
            reduce_only: Si True, l'ordre ne peut que réduire une position existante.
            
        Returns:
            Un dictionnaire contenant les informations de l'ordre exécuté.
        """
        logger.info(f"Executing order: {symbol} {side} {order_type}")
        logger.info(f"Parameters: quantity={quantity}, price={price}, stop_price={stop_price}, reduce_only={reduce_only}")
        
        try:
            # Arrondir la quantité et le prix selon les règles du symbole
            quantity = self._round_quantity(symbol, quantity)
            if price is not None:
                price = self._round_price(symbol, price)
            if stop_price is not None:
                stop_price = self._round_price(symbol, stop_price)
            
            # Préparer les paramètres de l'ordre
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
                'reduceOnly': reduce_only
            }
            
            # Ajouter les paramètres spécifiques au type d'ordre
            if order_type == OrderType.LIMIT:
                params['timeInForce'] = 'GTC'
                params['price'] = price
            elif order_type in [OrderType.STOP_MARKET, OrderType.TAKE_PROFIT_MARKET]:
                params['stopPrice'] = stop_price
            elif order_type == OrderType.TRAILING_STOP_MARKET:
                params['callbackRate'] = price  # Dans ce cas, price est le taux de callback
            
            # Exécuter l'ordre
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
        Arrondit la quantité selon les règles du symbole.
        
        Args:
            symbol: Le symbole pour lequel arrondir la quantité.
            quantity: La quantité à arrondir.
            
        Returns:
            La quantité arrondie.
        """
        try:
            # Récupérer les informations du symbole
            symbol_info = self._get_symbol_info(symbol)
            
            # Récupérer la précision de la quantité
            quantity_precision = symbol_info.get('quantityPrecision', 8)
            
            # Arrondir la quantité
            rounded_quantity = round(quantity, quantity_precision)
            
            return rounded_quantity
        
        except Exception as e:
            logger.warning(f"Error rounding quantity: {str(e)}. Using original quantity.")
            return quantity
    
    def _round_price(self, symbol: str, price: float) -> float:
        """
        Arrondit le prix selon les règles du symbole.
        
        Args:
            symbol: Le symbole pour lequel arrondir le prix.
            price: Le prix à arrondir.
            
        Returns:
            Le prix arrondi.
        """
        try:
            # Récupérer les informations du symbole
            symbol_info = self._get_symbol_info(symbol)
            
            # Récupérer la précision du prix
            price_precision = symbol_info.get('pricePrecision', 8)
            
            # Arrondir le prix
            rounded_price = round(price, price_precision)
            
            return rounded_price
        
        except Exception as e:
            logger.warning(f"Error rounding price: {str(e)}. Using original price.")
            return price
    
    def _get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère les informations d'un symbole.
        
        Args:
            symbol: Le symbole pour lequel récupérer les informations.
            
        Returns:
            Un dictionnaire contenant les informations du symbole.
        """
        # Vérifier si les informations sont dans le cache et si elles sont encore valides
        now = time.time()
        if now - self.cache_timestamp > self.cache_expiry:
            # Récupérer les informations de tous les symboles
            exchange_info = self.client.futures_exchange_info()
            
            # Mettre à jour le cache
            self.symbol_info_cache = {}
            for symbol_info in exchange_info['symbols']:
                self.symbol_info_cache[symbol_info['symbol']] = symbol_info
            
            self.cache_timestamp = now
        
        # Récupérer les informations du symbole depuis le cache
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]
        
        # Si le symbole n'est pas dans le cache, récupérer les informations directement
        exchange_info = self.client.futures_exchange_info()
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                return symbol_info
        
        # Si le symbole n'est pas trouvé, retourner un dictionnaire vide
        return {}


class BinanceClient:
    """
    Client pour interagir avec l'API Binance.
    
    Cette classe fournit des méthodes pour interagir avec l'API Binance,
    en gérant l'authentification, les requêtes et les réponses.
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, tld: str = 'com'):
        """
        Initialise le client Binance.
        
        Args:
            api_key: La clé API Binance.
            api_secret: La clé secrète API Binance.
            testnet: Si True, utilise le testnet de Binance.
            tld: Le TLD à utiliser pour l'API Binance.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.tld = tld
        
        # Initialiser le client Binance
        self.client = Client(api_key, api_secret, testnet=testnet, tld=tld)
        
        # Initialiser l'exécuteur d'ordres
        self.order_executor = OrderExecutor(self.client)
        
        # Mettre à jour les informations d'échange
        self._update_exchange_info()
        
        logger.info(f"Client Binance initialisé (testnet: {testnet})")
    
    def _update_exchange_info(self):
        """Met à jour les informations d'échange."""
        try:
            self.exchange_info = self.client.futures_exchange_info()
            logger.info("Informations d'échange mises à jour")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des informations d'échange: {str(e)}")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Récupère les informations du compte.
        
        Returns:
            Un dictionnaire contenant les informations du compte.
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
        Récupère le prix actuel d'un symbole.
        
        Args:
            symbol: Le symbole pour lequel récupérer le prix.
            
        Returns:
            Le prix actuel du symbole.
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
        Récupère les données de marché pour un symbole.
        
        Args:
            symbol: Le symbole pour lequel récupérer les données.
            
        Returns:
            Un objet MarketData contenant les données de marché.
        """
        try:
            logger.info(f"Retrieving market data for {symbol}")
            
            # Récupérer le ticker 24h
            ticker_24h = self.client.futures_ticker(symbol=symbol)
            
            # Récupérer le carnet d'ordres
            order_book = self.client.futures_order_book(symbol=symbol)
            
            # Créer l'objet MarketData
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
            # En cas d'erreur, retourner des données minimales
            return MarketData(
                symbol=symbol,
                price=0.0,
                timestamp=time.time()
            )
    
    async def get_klines(self, symbol: str, interval: str, limit: int) -> List[List[Any]]:
        """
        Récupère les klines (chandeliers) pour un symbole.
        
        Args:
            symbol: Le symbole pour lequel récupérer les klines.
            interval: L'intervalle des klines (1m, 5m, 15m, 1h, 4h, 1d, etc.).
            limit: Le nombre de klines à récupérer.
            
        Returns:
            Une liste de klines.
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
        Place un ordre sur Binance.
        
        Args:
            symbol: Le symbole sur lequel placer l'ordre.
            side: Le côté de l'ordre (achat ou vente).
            order_type: Le type d'ordre.
            quantity: La quantité à acheter ou vendre.
            price: Le prix de l'ordre (pour les ordres limites).
            stop_price: Le prix de déclenchement (pour les ordres stop).
            reduce_only: Si True, l'ordre ne peut que réduire une position existante.
            
        Returns:
            Un dictionnaire contenant les informations de l'ordre placé.
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
        Définit le levier pour un symbole.
        
        Args:
            symbol: Le symbole pour lequel définir le levier.
            leverage: Le levier à définir.
            
        Returns:
            Un dictionnaire contenant les informations de la réponse.
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
        Définit le type de marge pour un symbole.
        
        Args:
            symbol: Le symbole pour lequel définir le type de marge.
            margin_type: Le type de marge à définir (ISOLATED ou CROSSED).
            
        Returns:
            Un dictionnaire contenant les informations de la réponse.
        """
        try:
            logger.info(f"Setting margin type for {symbol} to {margin_type}")
            response = self.client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
            logger.info(f"Margin type set successfully for {symbol}: {margin_type}")
            return response
        except BinanceAPIException as e:
            # Ignorer l'erreur si le type de marge est déjà défini
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
        Récupère les positions ouvertes.
        
        Returns:
            Une liste de positions ouvertes.
        """
        try:
            logger.info("Retrieving open positions")
            account_info = await self.get_account_info()
            positions = account_info['positions']
            
            # Filtrer les positions avec une quantité non nulle
            open_positions = [p for p in positions if float(p['positionAmt']) != 0]
            
            logger.info(f"Retrieved {len(open_positions)} open positions")
            return open_positions
        except Exception as e:
            logger.error(f"Error retrieving open positions: {str(e)}")
            return []
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Récupère les ordres ouverts.
        
        Args:
            symbol: Le symbole pour lequel récupérer les ordres (optionnel).
            
        Returns:
            Une liste d'ordres ouverts.
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
        Annule un ordre.
        
        Args:
            symbol: Le symbole de l'ordre à annuler.
            order_id: L'ID de l'ordre à annuler.
            
        Returns:
            Un dictionnaire contenant les informations de la réponse.
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
        Ferme une position.
        
        Args:
            symbol: Le symbole de la position à fermer.
            position_side: Le côté de la position à fermer.
            
        Returns:
            Un dictionnaire contenant les informations de la réponse.
        """
        try:
            logger.info(f"Closing position for {symbol} (side: {position_side})")
            
            # Récupérer les positions ouvertes
            positions = await self.get_open_positions()
            
            # Trouver la position à fermer
            position = None
            for p in positions:
                if p['symbol'] == symbol and p['positionSide'] == position_side:
                    position = p
                    break
            
            if not position:
                logger.warning(f"No open position found for {symbol} (side: {position_side})")
                return {"msg": "No open position found", "code": 0}
            
            # Déterminer le côté de l'ordre pour fermer la position
            position_amt = float(position['positionAmt'])
            side = OrderSide.SELL if position_amt > 0 else OrderSide.BUY
            
            # Placer un ordre de marché pour fermer la position
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
        """Ferme le client Binance."""
        try:
            self.client.close_connection()
            logger.info("Binance client connection closed")
        except Exception as e:
            logger.error(f"Error closing Binance client connection: {str(e)}")


def main():
    """Fonction principale pour tester le module."""
    # Créer un client Binance (remplacer par vos clés API)
    client = BinanceClient(
        api_key="YOUR_API_KEY",
        api_secret="YOUR_API_SECRET",
        testnet=True  # Utiliser le testnet pour les tests
    )
    
    # Exemple d'utilisation
    async def test():
        try:
            # Récupérer le prix de BTC
            btc_price = await client.get_symbol_price("BTCUSDT")
            print(f"Prix de BTC: {btc_price}")
            
            # Récupérer les données de marché
            market_data = await client.get_market_data("BTCUSDT")
            print(f"Données de marché: {market_data}")
            
            # Récupérer les klines
            klines = await client.get_klines("BTCUSDT", "1h", 10)
            print(f"Klines: {klines[:2]}...")
            
        except Exception as e:
            print(f"Erreur: {str(e)}")
        finally:
            # Fermer le client
            client.close()
    
    # Exécuter la fonction de test
    import asyncio
    asyncio.run(test())


if __name__ == "__main__":
    main()
