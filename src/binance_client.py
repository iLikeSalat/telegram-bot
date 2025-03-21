"""
Module d'interaction avec l'API Binance.

Ce module contient les classes et fonctions nÃ©cessaires pour interagir
avec l'API Binance et exÃ©cuter des ordres de trading.
Version amÃ©liorÃ©e avec support d'ordres avancÃ©s et meilleure gestion des erreurs.
"""

import time
import logging
import json
import hmac
import hashlib
import requests
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from binance.client import Client
from binance import ThreadedWebsocketManager
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


class OrderType(Enum):
    """Types d'ordres supportÃ©s par Binance."""
    
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"


class OrderSide(Enum):
    """CÃ´tÃ©s d'ordre (achat/vente)."""
    
    BUY = "BUY"
    SELL = "SELL"


class PositionSide(Enum):
    """CÃ´tÃ©s de position (long/short/both)."""
    
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"


@dataclass
class MarketData:
    """DonnÃ©es de marchÃ© pour un symbole."""
    
    symbol: str
    price: float
    timestamp: float
    volume_24h: Optional[float] = None
    price_change_24h: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    trend: Optional[str] = None  # "uptrend", "downtrend", "sideways"


class ExchangeClient:
    """
    Classe abstraite pour les clients d'Ã©change.
    
    Cette classe dÃ©finit l'interface commune pour tous les clients d'Ã©change,
    ce qui permet d'ajouter facilement le support pour d'autres Ã©changes
    que Binance Ã  l'avenir.
    """
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialise le client d'Ã©change.
        
        Args:
            api_key: La clÃ© API de l'Ã©change.
            api_secret: Le secret API de l'Ã©change.
        """
        self.api_key = api_key
        self.api_secret = api_secret
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        RÃ©cupÃ¨re les informations du compte.
        
        Returns:
            Un dictionnaire contenant les informations du compte.
        """
        raise NotImplementedError("Cette mÃ©thode doit Ãªtre implÃ©mentÃ©e par les sous-classes")
    
    async def get_symbol_price(self, symbol: str) -> float:
        """
        RÃ©cupÃ¨re le prix actuel d'un symbole.
        
        Args:
            symbol: Le symbole dont on veut connaÃ®tre le prix.
            
        Returns:
            Le prix actuel du symbole.
        """
        raise NotImplementedError("Cette mÃ©thode doit Ãªtre implÃ©mentÃ©e par les sous-classes")
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """
        RÃ©cupÃ¨re les donnÃ©es de marchÃ© pour un symbole.
        
        Args:
            symbol: Le symbole dont on veut connaÃ®tre les donnÃ©es de marchÃ©.
            
        Returns:
            Un objet MarketData contenant les donnÃ©es de marchÃ©.
        """
        raise NotImplementedError("Cette mÃ©thode doit Ãªtre implÃ©mentÃ©e par les sous-classes")
    
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
        logger.info(f"Attempting to execute order on Binance: {symbol} {side} {order_type}")
        logger.info(f"Parameters: quantity={quantity}, price={price}, stop_price={stop_price}, reduce_only={reduce_only}")
        try:
        logger.info(f"Attempting d'exécution d'ordre sur Binance: {symbol} {side} {order_type}")
        logger.info(f"Paramètres: quantity={quantity}, price={price}, stop_price={stop_price}, reduce_only={reduce_only}")
        try:
                         quantity: float, price: Optional[float] = None,
                         stop_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place un ordre sur l'Ã©change.
        
        Args:
            symbol: Le symbole sur lequel placer l'ordre.
            side: Le cÃ´tÃ© de l'ordre (achat/vente).
            order_type: Le type d'ordre.
            quantity: La quantitÃ© Ã  acheter/vendre.
            price: Le prix de l'ordre (pour les ordres limites).
            stop_price: Le prix de dÃ©clenchement (pour les ordres stop).
            
        Returns:
            Un dictionnaire contenant les informations de l'ordre placÃ©.
        """
        raise NotImplementedError("Cette mÃ©thode doit Ãªtre implÃ©mentÃ©e par les sous-classes")
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Annule un ordre sur l'Ã©change.
        
        Args:
            symbol: Le symbole de l'ordre Ã  annuler.
            order_id: L'ID de l'ordre Ã  annuler.
            
        Returns:
            Un dictionnaire contenant les informations de l'ordre annulÃ©.
        """
        raise NotImplementedError("Cette mÃ©thode doit Ãªtre implÃ©mentÃ©e par les sous-classes")
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        RÃ©cupÃ¨re les ordres ouverts.
        
        Args:
            symbol: Le symbole pour lequel rÃ©cupÃ©rer les ordres ouverts.
                   Si None, rÃ©cupÃ¨re tous les ordres ouverts.
            
        Returns:
            Une liste de dictionnaires contenant les informations des ordres ouverts.
        """
        raise NotImplementedError("Cette mÃ©thode doit Ãªtre implÃ©mentÃ©e par les sous-classes")
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        RÃ©cupÃ¨re les positions ouvertes.
        
        Returns:
            Une liste de dictionnaires contenant les informations des positions ouvertes.
        """
        raise NotImplementedError("Cette mÃ©thode doit Ãªtre implÃ©mentÃ©e par les sous-classes")


class BinanceClient(ExchangeClient):
    """
    Client pour l'API Binance.
    
    Cette classe implÃ©mente l'interface ExchangeClient pour Binance.
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        Initialise le client Binance.
        
        Args:
            api_key: La clÃ© API Binance.
            api_secret: Le secret API Binance.
            testnet: Si True, utilise le testnet Binance.
        """
        super().__init__(api_key, api_secret)
        self.testnet = testnet
        
        # Initialiser le client Binance
        self.client = Client(api_key, api_secret, testnet=testnet)
        
        # Initialiser le gestionnaire de websocket
        self.ws_manager = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret, testnet=testnet)
        self.ws_manager.start()
        
        # Cache pour les donnÃ©es de marchÃ©
        self.market_data_cache = {}
        self.cache_expiry = 5  # Expiration du cache en secondes
        
        # Cache pour les informations de symbole
        self.symbol_info_cache = {}
        
        # Initialiser les informations d'Ã©change
        self.exchange_info = None
        self.update_exchange_info()
        
        logger.info(f"Client Binance initialisÃ© (testnet: {testnet})")
    
    def update_exchange_info(self) -> None:
        """Met Ã  jour les informations d'Ã©change."""
        try:
            self.exchange_info = self.client.futures_exchange_info()
            logger.info("Informations d'Ã©change mises Ã  jour")
        except Exception as e:
            logger.error(f"Erreur lors de la mise Ã  jour des informations d'Ã©change: {str(e)}")
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        RÃ©cupÃ¨re les informations d'un symbole.
        
        Args:
            symbol: Le symbole dont on veut connaÃ®tre les informations.
            
        Returns:
            Un dictionnaire contenant les informations du symbole.
        """
        # VÃ©rifier si les informations sont dans le cache
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]
        
        # VÃ©rifier si les informations d'Ã©change sont disponibles
        if not self.exchange_info:
            self.update_exchange_info()
        
        # Rechercher le symbole dans les informations d'Ã©change
        for sym_info in self.exchange_info['symbols']:
            if sym_info['symbol'] == symbol:
                # Mettre en cache les informations
                self.symbol_info_cache[symbol] = sym_info
                return sym_info
        
        # Si le symbole n'est pas trouvÃ©, lever une exception
        raise ValueError(f"Symbole non trouvÃ©: {symbol}")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        RÃ©cupÃ¨re les informations du compte.
        
        Returns:
            Un dictionnaire contenant les informations du compte.
        """
        try:
            account_info = self.client.futures_account()
            return account_info
        except Exception as e:
            logger.error(f"Erreur lors de la rÃ©cupÃ©ration des informations du compte: {str(e)}")
            raise
    
    async def get_symbol_price(self, symbol: str) -> float:
        """
        RÃ©cupÃ¨re le prix actuel d'un symbole.
        
        Args:
            symbol: Le symbole dont on veut connaÃ®tre le prix.
            
        Returns:
            Le prix actuel du symbole.
        """
        try:
            # VÃ©rifier si les donnÃ©es sont dans le cache et si elles sont encore valides
            now = time.time()
            if symbol in self.market_data_cache and now - self.market_data_cache[symbol].timestamp < self.cache_expiry:
                return self.market_data_cache[symbol].price
            
            # RÃ©cupÃ©rer le prix
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            
            # Mettre Ã  jour le cache
            if symbol in self.market_data_cache:
                self.market_data_cache[symbol].price = price
                self.market_data_cache[symbol].timestamp = now
            else:
                self.market_data_cache[symbol] = MarketData(symbol=symbol, price=price, timestamp=now)
            
            return price
        except Exception as e:
            logger.error(f"Erreur lors de la rÃ©cupÃ©ration du prix de {symbol}: {str(e)}")
            raise
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """
        RÃ©cupÃ¨re les donnÃ©es de marchÃ© pour un symbole.
        
        Args:
            symbol: Le symbole dont on veut connaÃ®tre les donnÃ©es de marchÃ©.
            
        Returns:
            Un objet MarketData contenant les donnÃ©es de marchÃ©.
        """
        try:
            # VÃ©rifier si les donnÃ©es sont dans le cache et si elles sont encore valides
            now = time.time()
            if symbol in self.market_data_cache and now - self.market_data_cache[symbol].timestamp < self.cache_expiry:
                return self.market_data_cache[symbol]
            
            # RÃ©cupÃ©rer les donnÃ©es de marchÃ©
            ticker_24h = self.client.futures_ticker(symbol=symbol)
            
            # RÃ©cupÃ©rer le carnet d'ordres pour le bid/ask
            order_book = self.client.futures_order_book(symbol=symbol, limit=5)
            
            # CrÃ©er l'objet MarketData
            market_data = MarketData(
                symbol=symbol,
                price=float(ticker_24h['lastPrice']),
                timestamp=now,
                volume_24h=float(ticker_24h['volume']),
                price_change_24h=float(ticker_24h['priceChangePercent']),
                high_24h=float(ticker_24h['highPrice']),
                low_24h=float(ticker_24h['lowPrice']),
                bid=float(order_book['bids'][0][0]) if order_book['bids'] else None,
                ask=float(order_book['asks'][0][0]) if order_book['asks'] else None
            )
            
            # DÃ©terminer la tendance
            if market_data.price_change_24h > 1.0:
                market_data.trend = "uptrend"
            elif market_data.price_change_24h < -1.0:
                market_data.trend = "downtrend"
            else:
                market_data.trend = "sideways"
            
            # Mettre Ã  jour le cache
            self.market_data_cache[symbol] = market_data
            
            return market_data
        except Exception as e:
            logger.error(f"Erreur lors de la rÃ©cupÃ©ration des donnÃ©es de marchÃ© de {symbol}: {str(e)}")
            raise
    
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
        logger.info(f"Attempting to execute order on Binance: {symbol} {side} {order_type}")
        logger.info(f"Parameters: quantity={quantity}, price={price}, stop_price={stop_price}, reduce_only={reduce_only}")
        try:
        logger.info(f"Attempting d'exécution d'ordre sur Binance: {symbol} {side} {order_type}")
        logger.info(f"Paramètres: quantity={quantity}, price={price}, stop_price={stop_price}, reduce_only={reduce_only}")
        try:
                         quantity: float, price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         position_side: PositionSide = PositionSide.BOTH,
                         reduce_only: bool = False,
                         close_position: bool = False,
                         callback_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        Place un ordre sur Binance Futures.
        
        Args:
            symbol: Le symbole sur lequel placer l'ordre.
            side: Le cÃ´tÃ© de l'ordre (achat/vente).
            order_type: Le type d'ordre.
            quantity: La quantitÃ© Ã  acheter/vendre.
            price: Le prix de l'ordre (pour les ordres limites).
            stop_price: Le prix de dÃ©clenchement (pour les ordres stop).
            position_side: Le cÃ´tÃ© de la position (long/short/both).
            reduce_only: Si True, l'ordre ne peut que rÃ©duire une position existante.
            close_position: Si True, ferme la position entiÃ¨re.
            callback_rate: Taux de callback pour les ordres trailing stop.
            
        Returns:
            Un dictionnaire contenant les informations de l'ordre placÃ©.
        """
        try:
            # PrÃ©parer les paramÃ¨tres de l'ordre
            params = {
                'symbol': symbol,
                'side': side.value,
                'type': order_type.value,
                'quantity': quantity,
                'positionSide': position_side.value,
                'reduceOnly': reduce_only,
                'closePosition': close_position
            }
            
            # Ajouter les paramÃ¨tres optionnels
            if price is not None and order_type != OrderType.MARKET:
                params['price'] = price
            
            if stop_price is not None and order_type in [OrderType.STOP, OrderType.STOP_MARKET, 
                                                        OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_MARKET]:
                params['stopPrice'] = stop_price
            
            if callback_rate is not None and order_type == OrderType.TRAILING_STOP_MARKET:
                params['callbackRate'] = callback_rate
            
            # Placer l'ordre
            if order_type == OrderType.MARKET:
                response = self.client.futures_create_order(
                    symbol=symbol,
                    side=side.value,
                    type=order_type.value,
                    quantity=quantity,
                    positionSide=position_side.value,
                    reduceOnly=reduce_only,
                    closePosition=close_position
                )
            elif order_type == OrderType.LIMIT:
                response = self.client.futures_create_order(
                    symbol=symbol,
                    side=side.value,
                    type=order_type.value,
                    quantity=quantity,
                    price=price,
                    positionSide=position_side.value,
                    reduceOnly=reduce_only,
                    timeInForce='GTC'  # Good Till Cancelled
                )
            elif order_type in [OrderType.STOP, OrderType.STOP_MARKET]:
                response = self.client.futures_create_order(
                    symbol=symbol,
                    side=side.value,
                    type=order_type.value,
                    quantity=quantity,
                    price=price if order_type == OrderType.STOP else None,
                    stopPrice=stop_price,
                    positionSide=position_side.value,
                    reduceOnly=reduce_only,
                    timeInForce='GTC' if order_type == OrderType.STOP else None
                )
            elif order_type in [OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_MARKET]:
                response = self.client.futures_create_order(
                    symbol=symbol,
                    side=side.value,
                    type=order_type.value,
                    quantity=quantity,
                    price=price if order_type == OrderType.TAKE_PROFIT else None,
                    stopPrice=stop_price,
                    positionSide=position_side.value,
                    reduceOnly=reduce_only,
                    timeInForce='GTC' if order_type == OrderType.TAKE_PROFIT else None
                )
            elif order_type == OrderType.TRAILING_STOP_MARKET:
                response = self.client.futures_create_order(
                    symbol=symbol,
                    side=side.value,
                    type=order_type.value,
                    quantity=quantity,
                    callbackRate=callback_rate,
                    positionSide=position_side.value,
                    reduceOnly=reduce_only
                )
            else:
                raise ValueError(f"Type d'ordre non supportÃ©: {order_type}")
            
            logger.info(f"Ordre placÃ©: {response}")
            return response
        except Exception as e:
            logger.error(f"Erreur lors du placement de l'ordre: {str(e)}")
            raise
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Annule un ordre sur Binance Futures.
        
        Args:
            symbol: Le symbole de l'ordre Ã  annuler.
            order_id: L'ID de l'ordre Ã  annuler.
            
        Returns:
            Un dictionnaire contenant les informations de l'ordre annulÃ©.
        """
        try:
            response = self.client.futures_cancel_order(
                symbol=symbol,
                orderId=order_id
            )
            logger.info(f"Ordre annulÃ©: {response}")
            return response
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation de l'ordre: {str(e)}")
            raise
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        RÃ©cupÃ¨re les ordres ouverts sur Binance Futures.
        
        Args:
            symbol: Le symbole pour lequel rÃ©cupÃ©rer les ordres ouverts.
                   Si None, rÃ©cupÃ¨re tous les ordres ouverts.
            
        Returns:
            Une liste de dictionnaires contenant les informations des ordres ouverts.
        """
        try:
            if symbol:
                response = self.client.futures_get_open_orders(symbol=symbol)
            else:
                response = self.client.futures_get_open_orders()
            
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la rÃ©cupÃ©ration des ordres ouverts: {str(e)}")
            raise
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        RÃ©cupÃ¨re les positions ouvertes sur Binance Futures.
        
        Returns:
            Une liste de dictionnaires contenant les informations des positions ouvertes.
        """
        try:
            account_info = await self.get_account_info()
            positions = account_info['positions']
            
            # Filtrer les positions avec une quantitÃ© non nulle
            open_positions = [p for p in positions if float(p['positionAmt']) != 0]
            
            return open_positions
        except Exception as e:
            logger.error(f"Erreur lors de la rÃ©cupÃ©ration des positions: {str(e)}")
            raise
    
    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        DÃ©finit le levier pour un symbole.
        
        Args:
            symbol: Le symbole pour lequel dÃ©finir le levier.
            leverage: Le levier Ã  dÃ©finir (1-125).
            
        Returns:
            Un dictionnaire contenant la rÃ©ponse de l'API.
        """
        try:
            response = self.client.futures_change_leverage(
                symbol=symbol,
                leverage=leverage
            )
            logger.info(f"Levier dÃ©fini pour {symbol}: {leverage}x")
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la dÃ©finition du levier pour {symbol}: {str(e)}")
            raise
    
    async def set_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """
        DÃ©finit le type de marge pour un symbole.
        
        Args:
            symbol: Le symbole pour lequel dÃ©finir le type de marge.
            margin_type: Le type de marge Ã  dÃ©finir ('ISOLATED' ou 'CROSSED').
            
        Returns:
            Un dictionnaire contenant la rÃ©ponse de l'API.
        """
        try:
            response = self.client.futures_change_margin_type(
                symbol=symbol,
                marginType=margin_type
            )
            logger.info(f"Type de marge dÃ©fini pour {symbol}: {margin_type}")
            return response
        except BinanceAPIException as e:
            # Ignorer l'erreur si le type de marge est dÃ©jÃ  dÃ©fini
            if e.code == -4046:  # "No need to change margin type."
                logger.info(f"Type de marge dÃ©jÃ  dÃ©fini pour {symbol}: {margin_type}")
                return {"msg": "No need to change margin type."}
            else:
                logger.error(f"Erreur lors de la dÃ©finition du type de marge pour {symbol}: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Erreur lors de la dÃ©finition du type de marge pour {symbol}: {str(e)}")
            raise
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[List[Any]]:
        """
        RÃ©cupÃ¨re les klines (bougies) pour un symbole.
        
        Args:
            symbol: Le symbole pour lequel rÃ©cupÃ©rer les klines.
            interval: L'intervalle des klines (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M).
            limit: Le nombre de klines Ã  rÃ©cupÃ©rer (max 1500).
            
        Returns:
            Une liste de klines. Chaque kline est une liste contenant:
            [open_time, open, high, low, close, volume, close_time, quote_asset_volume,
             number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
        """
        try:
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            return klines
        except Exception as e:
            logger.error(f"Erreur lors de la rÃ©cupÃ©ration des klines pour {symbol}: {str(e)}")
            raise
    
    def subscribe_to_klines(self, symbol: str, interval: str, callback: Callable[[Dict[str, Any]], None]) -> int:
        """
        S'abonne aux klines (bougies) pour un symbole.
        
        Args:
            symbol: Le symbole pour lequel s'abonner aux klines.
            interval: L'intervalle des klines (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M).
            callback: La fonction de callback Ã  appeler lorsqu'une nouvelle kline est reÃ§ue.
            
        Returns:
            L'ID de la connexion websocket.
        """
        try:
            # Convertir le symbole en minuscules pour le stream
            stream_symbol = symbol.lower()
            
            # S'abonner au stream de klines
            stream_name = f"{stream_symbol}@kline_{interval}"
            conn_key = self.ws_manager.start_kline_socket(
                callback=callback,
                symbol=symbol,
                interval=interval
            )
            
            logger.info(f"AbonnÃ© au stream de klines pour {symbol} ({interval})")
            return conn_key
        except Exception as e:
            logger.error(f"Erreur lors de l'abonnement au stream de klines pour {symbol}: {str(e)}")
            raise
    
    def subscribe_to_user_data(self, callback: Callable[[Dict[str, Any]], None]) -> int:
        """
        S'abonne aux donnÃ©es utilisateur (ordres, positions, solde).
        
        Args:
            callback: La fonction de callback Ã  appeler lorsque de nouvelles donnÃ©es sont reÃ§ues.
            
        Returns:
            L'ID de la connexion websocket.
        """
        try:
            conn_key = self.ws_manager.start_user_socket(callback=callback)
            logger.info("AbonnÃ© au stream de donnÃ©es utilisateur")
            return conn_key
        except Exception as e:
            logger.error(f"Erreur lors de l'abonnement au stream de donnÃ©es utilisateur: {str(e)}")
            raise
    
    def stop_socket(self, conn_key: int) -> None:
        """
        ArrÃªte une connexion websocket.
        
        Args:
            conn_key: L'ID de la connexion websocket Ã  arrÃªter.
        """
        try:
            self.ws_manager.stop_socket(conn_key)
            logger.info(f"Socket arrÃªtÃ©: {conn_key}")
        except Exception as e:
            logger.error(f"Erreur lors de l'arrÃªt du socket {conn_key}: {str(e)}")
    
    def close(self) -> None:
        """Ferme toutes les connexions websocket et le client."""
        try:
            self.ws_manager.stop()
            logger.info("Client Binance fermÃ©")
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture du client Binance: {str(e)}")


class OrderExecutor:
    """
    Classe pour exÃ©cuter des ordres sur Binance Futures.
    
    Cette classe fournit des mÃ©thodes de haut niveau pour exÃ©cuter
    diffÃ©rents types d'ordres et de stratÃ©gies de trading.
    """
    
    def __init__(self, binance_client: BinanceClient):
        """
        Initialise l'exÃ©cuteur d'ordres.
        
        Args:
            binance_client: Le client Binance Ã  utiliser pour exÃ©cuter les ordres.
        """
        self.client = binance_client
        
        # Cache pour les informations de symbole
        self.symbol_precision_cache = {}
        
        logger.info("OrderExecutor initialisÃ©")
    
    def _get_symbol_precision(self, symbol: str) -> Tuple[int, int]:
        """
        RÃ©cupÃ¨re la prÃ©cision de la quantitÃ© et du prix pour un symbole.
        
        Args:
            symbol: Le symbole dont on veut connaÃ®tre la prÃ©cision.
            
        Returns:
            Un tuple (quantity_precision, price_precision).
        """
        # VÃ©rifier si les informations sont dans le cache
        if symbol in self.symbol_precision_cache:
            return self.symbol_precision_cache[symbol]
        
        # RÃ©cupÃ©rer les informations du symbole
        symbol_info = self.client.get_symbol_info(symbol)
        
        # Extraire la prÃ©cision de la quantitÃ© et du prix
        quantity_precision = 0
        price_precision = 0
        
        for filter_info in symbol_info['filters']:
            if filter_info['filterType'] == 'LOT_SIZE':
                step_size = float(filter_info['stepSize'])
                quantity_precision = len(str(step_size).rstrip('0').split('.')[-1]) if '.' in str(step_size) else 0
            elif filter_info['filterType'] == 'PRICE_FILTER':
                tick_size = float(filter_info['tickSize'])
                price_precision = len(str(tick_size).rstrip('0').split('.')[-1]) if '.' in str(tick_size) else 0
        
        # Mettre en cache les informations
        self.symbol_precision_cache[symbol] = (quantity_precision, price_precision)
        
        return quantity_precision, price_precision
    
    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """
        Arrondit une quantitÃ© selon la prÃ©cision du symbole.
        
        Args:
            symbol: Le symbole pour lequel arrondir la quantitÃ©.
            quantity: La quantitÃ© Ã  arrondir.
            
        Returns:
            La quantitÃ© arrondie.
        """
        quantity_precision, _ = self._get_symbol_precision(symbol)
        return round(quantity, quantity_precision)
    
    def _round_price(self, symbol: str, price: float) -> float:
        """
        Arrondit un prix selon la prÃ©cision du symbole.
        
        Args:
            symbol: Le symbole pour lequel arrondir le prix.
            price: Le prix Ã  arrondir.
            
        Returns:
            Le prix arrondi.
        """
        _, price_precision = self._get_symbol_precision(symbol)
        return round(price, price_precision)
    
    async def execute_market_order(self, symbol: str, side: OrderSide, quantity: float,
                                  position_side: PositionSide = PositionSide.BOTH,
                                  reduce_only: bool = False) -> Dict[str, Any]:
        """
        ExÃ©cute un ordre au marchÃ©.
        
        Args:
            symbol: Le symbole sur lequel exÃ©cuter l'ordre.
            side: Le cÃ´tÃ© de l'ordre (achat/vente).
            quantity: La quantitÃ© Ã  acheter/vendre.
            position_side: Le cÃ´tÃ© de la position (long/short/both).
            reduce_only: Si True, l'ordre ne peut que rÃ©duire une position existante.
            
        Returns:
            Un dictionnaire contenant les informations de l'ordre exÃ©cutÃ©.
        """
        # Arrondir la quantitÃ©
        quantity = self._round_quantity(symbol, quantity)
        
        # ExÃ©cuter l'ordre
        response = await self.client.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            position_side=position_side,
            reduce_only=reduce_only
        )
        
        return response
    
    async def execute_limit_order(self, symbol: str, side: OrderSide, quantity: float, price: float,
                                 position_side: PositionSide = PositionSide.BOTH,
                                 reduce_only: bool = False) -> Dict[str, Any]:
        """
        ExÃ©cute un ordre limite.
        
        Args:
            symbol: Le symbole sur lequel exÃ©cuter l'ordre.
            side: Le cÃ´tÃ© de l'ordre (achat/vente).
            quantity: La quantitÃ© Ã  acheter/vendre.
            price: Le prix de l'ordre.
            position_side: Le cÃ´tÃ© de la position (long/short/both).
            reduce_only: Si True, l'ordre ne peut que rÃ©duire une position existante.
            
        Returns:
            Un dictionnaire contenant les informations de l'ordre exÃ©cutÃ©.
        """
        # Arrondir la quantitÃ© et le prix
        quantity = self._round_quantity(symbol, quantity)
        price = self._round_price(symbol, price)
        
        # ExÃ©cuter l'ordre
        response = await self.client.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            position_side=position_side,
            reduce_only=reduce_only
        )
        
        return response
    
    async def execute_stop_market_order(self, symbol: str, side: OrderSide, quantity: float, stop_price: float,
                                      position_side: PositionSide = PositionSide.BOTH,
                                      reduce_only: bool = True) -> Dict[str, Any]:
        """
        ExÃ©cute un ordre stop au marchÃ©.
        
        Args:
            symbol: Le symbole sur lequel exÃ©cuter l'ordre.
            side: Le cÃ´tÃ© de l'ordre (achat/vente).
            quantity: La quantitÃ© Ã  acheter/vendre.
            stop_price: Le prix de dÃ©clenchement.
            position_side: Le cÃ´tÃ© de la position (long/short/both).
            reduce_only: Si True, l'ordre ne peut que rÃ©duire une position existante.
            
        Returns:
            Un dictionnaire contenant les informations de l'ordre exÃ©cutÃ©.
        """
        # Arrondir la quantitÃ© et le prix
        quantity = self._round_quantity(symbol, quantity)
        stop_price = self._round_price(symbol, stop_price)
        
        # ExÃ©cuter l'ordre
        response = await self.client.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_MARKET,
            quantity=quantity,
            stop_price=stop_price,
            position_side=position_side,
            reduce_only=reduce_only
        )
        
        return response
    
    async def execute_take_profit_market_order(self, symbol: str, side: OrderSide, quantity: float, stop_price: float,
                                             position_side: PositionSide = PositionSide.BOTH,
                                             reduce_only: bool = True) -> Dict[str, Any]:
        """
        ExÃ©cute un ordre take profit au marchÃ©.
        
        Args:
            symbol: Le symbole sur lequel exÃ©cuter l'ordre.
            side: Le cÃ´tÃ© de l'ordre (achat/vente).
            quantity: La quantitÃ© Ã  acheter/vendre.
            stop_price: Le prix de dÃ©clenchement.
            position_side: Le cÃ´tÃ© de la position (long/short/both).
            reduce_only: Si True, l'ordre ne peut que rÃ©duire une position existante.
            
        Returns:
            Un dictionnaire contenant les informations de l'ordre exÃ©cutÃ©.
        """
        # Arrondir la quantitÃ© et le prix
        quantity = self._round_quantity(symbol, quantity)
        stop_price = self._round_price(symbol, stop_price)
        
        # ExÃ©cuter l'ordre
        response = await self.client.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            quantity=quantity,
            stop_price=stop_price,
            position_side=position_side,
            reduce_only=reduce_only
        )
        
        return response
    
    async def execute_trailing_stop_order(self, symbol: str, side: OrderSide, quantity: float, callback_rate: float,
                                        position_side: PositionSide = PositionSide.BOTH,
                                        reduce_only: bool = True) -> Dict[str, Any]:
        """
        ExÃ©cute un ordre trailing stop.
        
        Args:
            symbol: Le symbole sur lequel exÃ©cuter l'ordre.
            side: Le cÃ´tÃ© de l'ordre (achat/vente).
            quantity: La quantitÃ© Ã  acheter/vendre.
            callback_rate: Le taux de callback en pourcentage (0.1 Ã  5).
            position_side: Le cÃ´tÃ© de la position (long/short/both).
            reduce_only: Si True, l'ordre ne peut que rÃ©duire une position existante.
            
        Returns:
            Un dictionnaire contenant les informations de l'ordre exÃ©cutÃ©.
        """
        # Arrondir la quantitÃ©
        quantity = self._round_quantity(symbol, quantity)
        
        # VÃ©rifier que le taux de callback est dans la plage autorisÃ©e
        if callback_rate < 0.1 or callback_rate > 5:
            raise ValueError("Le taux de callback doit Ãªtre compris entre 0.1 et 5")
        
        # ExÃ©cuter l'ordre
        response = await self.client.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.TRAILING_STOP_MARKET,
            quantity=quantity,
            callback_rate=callback_rate,
            position_side=position_side,
            reduce_only=reduce_only
        )
        
        return response
    
    async def execute_oco_order(self, symbol: str, side: OrderSide, quantity: float,
                               price: float, stop_price: float, stop_limit_price: float,
                               position_side: PositionSide = PositionSide.BOTH) -> Dict[str, Any]:
        """
        ExÃ©cute un ordre OCO (One-Cancels-the-Other).
        
        Args:
            symbol: Le symbole sur lequel exÃ©cuter l'ordre.
            side: Le cÃ´tÃ© de l'ordre (achat/vente).
            quantity: La quantitÃ© Ã  acheter/vendre.
            price: Le prix de l'ordre limite.
            stop_price: Le prix de dÃ©clenchement de l'ordre stop.
            stop_limit_price: Le prix de l'ordre stop-limit.
            position_side: Le cÃ´tÃ© de la position (long/short/both).
            
        Returns:
            Un dictionnaire contenant les informations de l'ordre exÃ©cutÃ©.
        """
        # Arrondir la quantitÃ© et les prix
        quantity = self._round_quantity(symbol, quantity)
        price = self._round_price(symbol, price)
        stop_price = self._round_price(symbol, stop_price)
        stop_limit_price = self._round_price(symbol, stop_limit_price)
        
        # ExÃ©cuter l'ordre OCO
        try:
            response = self.client.client.futures_create_oco(
                symbol=symbol,
                side=side.value,
                quantity=quantity,
                price=price,
                stopPrice=stop_price,
                stopLimitPrice=stop_limit_price,
                stopLimitTimeInForce='GTC',
                positionSide=position_side.value
            )
            
            return response
        except Exception as e:
            logger.error(f"Erreur lors de l'exÃ©cution de l'ordre OCO: {str(e)}")
            
            # Si l'OCO n'est pas supportÃ©, exÃ©cuter deux ordres sÃ©parÃ©s
            logger.info("ExÃ©cution de deux ordres sÃ©parÃ©s (limite et stop-limit)")
            
            # Ordre limite
            limit_order = await self.execute_limit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                position_side=position_side,
                reduce_only=True
            )
            
            # Ordre stop-limit
            stop_limit_order = await self.client.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.STOP,
                quantity=quantity,
                price=stop_limit_price,
                stop_price=stop_price,
                position_side=position_side,
                reduce_only=True
            )
            
            return {
                "limit_order": limit_order,
                "stop_limit_order": stop_limit_order
            }
    
    async def execute_scaled_entry(self, symbol: str, side: OrderSide, total_quantity: float,
                                  start_price: float, end_price: float, num_orders: int,
                                  position_side: PositionSide = PositionSide.BOTH) -> List[Dict[str, Any]]:
        """
        ExÃ©cute une entrÃ©e Ã©chelonnÃ©e avec plusieurs ordres limites.
        
        Args:
            symbol: Le symbole sur lequel exÃ©cuter les ordres.
            side: Le cÃ´tÃ© des ordres (achat/vente).
            total_quantity: La quantitÃ© totale Ã  acheter/vendre.
            start_price: Le prix de dÃ©part.
            end_price: Le prix de fin.
            num_orders: Le nombre d'ordres Ã  placer.
            position_side: Le cÃ´tÃ© de la position (long/short/both).
            
        Returns:
            Une liste de dictionnaires contenant les informations des ordres exÃ©cutÃ©s.
        """
        # VÃ©rifier que le nombre d'ordres est valide
        if num_orders < 2:
            raise ValueError("Le nombre d'ordres doit Ãªtre au moins 2")
        
        # Calculer le pas de prix
        price_step = (end_price - start_price) / (num_orders - 1)
        
        # Calculer la quantitÃ© par ordre
        quantity_per_order = total_quantity / num_orders
        
        # Arrondir la quantitÃ©
        quantity_per_order = self._round_quantity(symbol, quantity_per_order)
        
        # Placer les ordres
        orders = []
        for i in range(num_orders):
            price = start_price + i * price_step
            price = self._round_price(symbol, price)
            
            order = await self.execute_limit_order(
                symbol=symbol,
                side=side,
                quantity=quantity_per_order,
                price=price,
                position_side=position_side
            )
            
            orders.append(order)
        
        return orders
    
    async def execute_scaled_exit(self, symbol: str, side: OrderSide, total_quantity: float,
                                 start_price: float, end_price: float, num_orders: int,
                                 position_side: PositionSide = PositionSide.BOTH) -> List[Dict[str, Any]]:
        """
        ExÃ©cute une sortie Ã©chelonnÃ©e avec plusieurs ordres take profit.
        
        Args:
            symbol: Le symbole sur lequel exÃ©cuter les ordres.
            side: Le cÃ´tÃ© des ordres (achat/vente).
            total_quantity: La quantitÃ© totale Ã  acheter/vendre.
            start_price: Le prix de dÃ©part.
            end_price: Le prix de fin.
            num_orders: Le nombre d'ordres Ã  placer.
            position_side: Le cÃ´tÃ© de la position (long/short/both).
            
        Returns:
            Une liste de dictionnaires contenant les informations des ordres exÃ©cutÃ©s.
        """
        # VÃ©rifier que le nombre d'ordres est valide
        if num_orders < 2:
            raise ValueError("Le nombre d'ordres doit Ãªtre au moins 2")
        
        # Calculer le pas de prix
        price_step = (end_price - start_price) / (num_orders - 1)
        
        # Calculer les quantitÃ©s pour une distribution pyramidale
        # Plus de quantitÃ© pour les premiers TP, moins pour les derniers
        total_parts = sum(range(1, num_orders + 1))
        quantities = []
        
        for i in range(num_orders, 0, -1):
            quantity = total_quantity * i / total_parts
            quantities.append(self._round_quantity(symbol, quantity))
        
        # Placer les ordres
        orders = []
        for i in range(num_orders):
            price = start_price + i * price_step
            price = self._round_price(symbol, price)
            
            order = await self.execute_take_profit_market_order(
                symbol=symbol,
                side=side,
                quantity=quantities[i],
                stop_price=price,
                position_side=position_side,
                reduce_only=True
            )
            
            orders.append(order)
        
        return orders
    
    async def close_position(self, symbol: str, position_side: PositionSide = PositionSide.BOTH) -> Dict[str, Any]:
        """
        Ferme une position ouverte.
        
        Args:
            symbol: Le symbole de la position Ã  fermer.
            position_side: Le cÃ´tÃ© de la position (long/short/both).
            
        Returns:
            Un dictionnaire contenant les informations de l'ordre exÃ©cutÃ©.
        """
        try:
            # RÃ©cupÃ©rer les positions ouvertes
            positions = await self.client.get_positions()
            
            # Trouver la position pour le symbole et le cÃ´tÃ© spÃ©cifiÃ©s
            position = None
            for p in positions:
                if p['symbol'] == symbol and (position_side == PositionSide.BOTH or p['positionSide'] == position_side.value):
                    position = p
                    break
            
            if not position or float(position['positionAmt']) == 0:
                logger.warning(f"Aucune position ouverte pour {symbol} ({position_side.value})")
                return {"status": "no_position", "message": f"Aucune position ouverte pour {symbol} ({position_side.value})"}
            
            # DÃ©terminer le cÃ´tÃ© de l'ordre pour fermer la position
            position_amt = float(position['positionAmt'])
            side = OrderSide.SELL if position_amt > 0 else OrderSide.BUY
            
            # Fermer la position avec un ordre au marchÃ©
            response = await self.client.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=abs(position_amt),
                position_side=position_side,
                reduce_only=True
            )
            
            logger.info(f"Position fermÃ©e pour {symbol} ({position_side.value})")
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture de la position pour {symbol}: {str(e)}")
            raise


def main():
    """Fonction principale pour tester le module."""
    # CrÃ©er un client Binance (remplacer par vos clÃ©s API)
    client = BinanceClient(
        api_key="YOUR_API_KEY",
        api_secret="YOUR_API_SECRET",
        testnet=True  # Utiliser le testnet pour les tests
    )
    
    # CrÃ©er un exÃ©cuteur d'ordres
    executor = OrderExecutor(client)
    
    # Exemple d'utilisation
    async def test():
        try:
            # RÃ©cupÃ©rer le prix de BTC
            btc_price = await client.get_symbol_price("BTCUSDT")
            print(f"Prix de BTC: {btc_price}")
            
            # RÃ©cupÃ©rer les donnÃ©es de marchÃ©
            btc_market_data = await client.get_market_data("BTCUSDT")
            print(f"DonnÃ©es de marchÃ© de BTC: {btc_market_data}")
            
            # DÃ©finir le levier
            await client.set_leverage("BTCUSDT", 10)
            
            # DÃ©finir le type de marge
            await client.set_margin_type("BTCUSDT", "ISOLATED")
            
            # Placer un ordre au marchÃ© (commentÃ© pour Ã©viter une exÃ©cution accidentelle)
            # order = await executor.execute_market_order(
            #     symbol="BTCUSDT",
            #     side=OrderSide.BUY,
            #     quantity=0.001
            # )
            # print(f"Ordre placÃ©: {order}")
            
        except Exception as e:
            print(f"Erreur: {str(e)}")
        finally:
            # Fermer le client
            client.close()
    
    # ExÃ©cuter la fonction de test
    import asyncio
    asyncio.run(test())


if __name__ == "__main__":
    main()
