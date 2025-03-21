"""
Module d'exécution des trades.

Ce module contient les classes et fonctions nécessaires pour exécuter
des trades basés sur les signaux reçus.
Version améliorée avec analyse technique et gestion avancée des positions.
"""

import logging
import time
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from src.signal_parser import TradingSignal
from src.binance_client import BinanceClient, OrderSide, OrderType, PositionSide, MarketData

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


@dataclass
class TradeResult:
    """Résultat d'un trade exécuté."""
    
    signal_id: str
    symbol: str
    direction: str
    status: str  # "success", "error", "rejected", "expired"
    message: str
    position_size: float
    leverage: int
    risk_percentage: float
    entry_price: float
    stop_loss: float
    take_profit_levels: List[float]
    orders: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'objet en dictionnaire."""
        return asdict(self)


class TechnicalAnalyzer:
    """
    Classe pour effectuer des analyses techniques sur les données de marché.
    
    Cette classe fournit des méthodes pour calculer divers indicateurs techniques
    et valider les signaux de trading en fonction des conditions de marché.
    """
    
    def __init__(self, binance_client: BinanceClient):
        """
        Initialise l'analyseur technique.
        
        Args:
            binance_client: Le client Binance à utiliser pour récupérer les données.
        """
        self.client = binance_client
        logger.info("TechnicalAnalyzer initialisé")
    
    async def validate_trend(self, signal: TradingSignal) -> Tuple[bool, str]:
        """
        Valide si le signal est dans la tendance actuelle.
        
        Args:
            signal: Le signal à valider.
            
        Returns:
            Un tuple (is_valid, message) indiquant si le signal est valide
            et un message explicatif.
        """
        try:
            # Récupérer les données de marché
            symbol = f"{signal.symbol}USDT"
            market_data = await self.client.get_market_data(symbol)
            
            # Récupérer les klines pour calculer les moyennes mobiles
            klines = await self.client.get_klines(symbol, "1h", 50)
            
            # Calculer les moyennes mobiles
            ma20 = self._calculate_ma(klines, 20)
            ma50 = self._calculate_ma(klines, 50)
            
            # Déterminer la tendance
            current_price = market_data.price
            trend = "uptrend" if ma20[-1] > ma50[-1] else "downtrend"
            
            # Valider le signal en fonction de la tendance
            if signal.direction == "LONG" and trend == "uptrend":
                return True, f"Signal LONG validé dans une tendance haussière (MA20: {ma20[-1]:.2f} > MA50: {ma50[-1]:.2f})"
            elif signal.direction == "SHORT" and trend == "downtrend":
                return True, f"Signal SHORT validé dans une tendance baissière (MA20: {ma20[-1]:.2f} < MA50: {ma50[-1]:.2f})"
            else:
                return False, f"Signal {signal.direction} contre la tendance {trend} (MA20: {ma20[-1]:.2f}, MA50: {ma50[-1]:.2f})"
        
        except Exception as e:
            logger.error(f"Erreur lors de la validation de la tendance: {str(e)}")
            # En cas d'erreur, on considère le signal comme valide
            return True, "Validation de tendance ignorée en raison d'une erreur"
    
    async def calculate_volatility_stop_loss(self, signal: TradingSignal) -> float:
        """
        Calcule un stop loss basé sur la volatilité (ATR).
        
        Args:
            signal: Le signal pour lequel calculer le stop loss.
            
        Returns:
            Le prix du stop loss calculé.
        """
        try:
            # Récupérer les klines pour calculer l'ATR
            symbol = f"{signal.symbol}USDT"
            klines = await self.client.get_klines(symbol, "1h", 14)
            
            # Calculer l'ATR
            atr = self._calculate_atr(klines)
            
            # Calculer le stop loss en fonction de la direction
            if signal.direction == "LONG":
                # Pour un LONG, le stop loss est le prix d'entrée moins un multiple de l'ATR
                stop_loss = signal.entry_min - (2 * atr[-1])
            else:  # SHORT
                # Pour un SHORT, le stop loss est le prix d'entrée plus un multiple de l'ATR
                stop_loss = signal.entry_max + (2 * atr[-1])
            
            logger.info(f"Stop loss basé sur la volatilité calculé: {stop_loss:.2f} (ATR: {atr[-1]:.2f})")
            return stop_loss
        
        except Exception as e:
            logger.error(f"Erreur lors du calcul du stop loss basé sur la volatilité: {str(e)}")
            # En cas d'erreur, on retourne le stop loss du signal
            return signal.stop_loss
    
    async def get_market_context(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère le contexte de marché pour un symbole.
        
        Args:
            symbol: Le symbole pour lequel récupérer le contexte.
            
        Returns:
            Un dictionnaire contenant les informations de contexte.
        """
        try:
            # Ajouter le suffixe USDT si nécessaire
            if not symbol.endswith("USDT"):
                symbol = f"{symbol}USDT"
            
            # Récupérer les données de marché
            market_data = await self.client.get_market_data(symbol)
            
            # Récupérer les klines pour différentes périodes
            klines_1h = await self.client.get_klines(symbol, "1h", 24)
            klines_4h = await self.client.get_klines(symbol, "4h", 24)
            klines_1d = await self.client.get_klines(symbol, "1d", 14)
            
            # Calculer les indicateurs
            ma20_1h = self._calculate_ma(klines_1h, min(20, len(klines_1h)))
            ma50_1h = self._calculate_ma(klines_1h, min(50, len(klines_1h)))
            
            ma20_4h = self._calculate_ma(klines_4h, min(20, len(klines_4h)))
            ma50_4h = self._calculate_ma(klines_4h, min(50, len(klines_4h)))
            
            ma20_1d = self._calculate_ma(klines_1d, min(20, len(klines_1d)))
            ma50_1d = self._calculate_ma(klines_1d, min(50, len(klines_1d)))
            
            atr_1h = self._calculate_atr(klines_1h)
            atr_4h = self._calculate_atr(klines_4h)
            atr_1d = self._calculate_atr(klines_1d)
            
            # Déterminer les tendances
            trend_1h = "uptrend" if ma20_1h[-1] > ma50_1h[-1] else "downtrend"
            trend_4h = "uptrend" if ma20_4h[-1] > ma50_4h[-1] else "downtrend"
            trend_1d = "uptrend" if ma20_1d[-1] > ma50_1d[-1] else "downtrend"
            
            # Créer le contexte
            context = {
                "symbol": symbol,
                "current_price": market_data.price,
                "volume_24h": market_data.volume_24h,
                "price_change_24h": market_data.price_change_24h,
                "high_24h": market_data.high_24h,
                "low_24h": market_data.low_24h,
                "trends": {
                    "1h": trend_1h,
                    "4h": trend_4h,
                    "1d": trend_1d
                },
                "indicators": {
                    "ma20_1h": ma20_1h[-1] if len(ma20_1h) > 0 else None,
                    "ma50_1h": ma50_1h[-1] if len(ma50_1h) > 0 else None,
                    "ma20_4h": ma20_4h[-1] if len(ma20_4h) > 0 else None,
                    "ma50_4h": ma50_4h[-1] if len(ma50_4h) > 0 else None,
                    "ma20_1d": ma20_1d[-1] if len(ma20_1d) > 0 else None,
                    "ma50_1d": ma50_1d[-1] if len(ma50_1d) > 0 else None,
                    "atr_1h": atr_1h[-1] if len(atr_1h) > 0 else None,
                    "atr_4h": atr_4h[-1] if len(atr_4h) > 0 else None,
                    "atr_1d": atr_1d[-1] if len(atr_1d) > 0 else None
                }
            }
            
            return context
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du contexte de marché: {str(e)}")
            # En cas d'erreur, on retourne un contexte minimal
            return {
                "symbol": symbol,
                "error": str(e)
            }
    
    def _calculate_ma(self, klines: List[List[Any]], period: int) -> List[float]:
        """
        Calcule la moyenne mobile sur une période donnée.
        
        Args:
            klines: Les klines à utiliser pour le calcul.
            period: La période de la moyenne mobile.
            
        Returns:
            Une liste contenant les valeurs de la moyenne mobile.
        """
        if len(klines) < period:
            return []
        
        # Extraire les prix de clôture
        closes = [float(kline[4]) for kline in klines]
        
        # Calculer la moyenne mobile
        ma = []
        for i in range(len(closes) - period + 1):
            ma.append(sum(closes[i:i+period]) / period)
        
        return ma
    
    def _calculate_atr(self, klines: List[List[Any]], period: int = 14) -> List[float]:
        """
        Calcule l'Average True Range (ATR) sur une période donnée.
        
        Args:
            klines: Les klines à utiliser pour le calcul.
            period: La période de l'ATR.
            
        Returns:
            Une liste contenant les valeurs de l'ATR.
        """
        if len(klines) < period + 1:
            return []
        
        # Extraire les prix high, low et close
        highs = [float(kline[2]) for kline in klines]
        lows = [float(kline[3]) for kline in klines]
        closes = [float(kline[4]) for kline in klines]
        
        # Calculer les True Ranges
        tr = []
        for i in range(1, len(klines)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr.append(max(tr1, tr2, tr3))
        
        # Calculer l'ATR
        atr = []
        atr.append(sum(tr[:period]) / period)  # Premier ATR (moyenne simple)
        
        for i in range(1, len(tr) - period + 1):
            atr.append((atr[-1] * (period - 1) + tr[i + period - 1]) / period)
        
        return atr


class TradeExecutor:
    """
    Classe pour exécuter des trades basés sur les signaux reçus.
    
    Cette classe coordonne l'exécution des trades en utilisant le client Binance,
    l'analyseur technique et le gestionnaire de risque.
    """
    
    def __init__(self, binance_client: BinanceClient, risk_manager: 'RiskManager',
                 validate_trend: bool = True, use_volatility_sl: bool = True):
        """
        Initialise l'exécuteur de trades.
        
        Args:
            binance_client: Le client Binance à utiliser pour exécuter les trades.
            risk_manager: Le gestionnaire de risque à utiliser pour calculer les tailles de position.
            validate_trend: Si True, valide que le signal est dans la tendance actuelle.
            use_volatility_sl: Si True, utilise un stop loss basé sur la volatilité.
        """
        self.client = binance_client
        self.risk_manager = risk_manager
        self.technical_analyzer = TechnicalAnalyzer(binance_client)
        self.validate_trend = validate_trend
        self.use_volatility_sl = use_volatility_sl
        
        # Historique des trades
        self.trade_history = []
        
        logger.info(f"TradeExecutor initialisé (validate_trend: {validate_trend}, use_volatility_sl: {use_volatility_sl})")
    
    async def execute_signal(self, signal: TradingSignal) -> TradeResult:
        logger.info(f"Debut de execution du signal: {signal.symbol} {signal.direction}")
        logger.info(f"Details du signal: entry_min={signal.entry_min}, entry_max={signal.entry_max}, stop_loss={signal.stop_loss}")
        logger.info(f"Take profits: {signal.take_profit_levels}")
        logger.info(f"D�but de l'ex�cution du signal: {signal.symbol} {signal.direction}")
        logger.info(f"D�tails du signal: entry_min={signal.entry_min}, entry_max={signal.entry_max}, stop_loss={signal.stop_loss}")
        logger.info(f"Take profits: {signal.take_profit_levels}")
        """
        Exécute un signal de trading.
        
        Args:
            signal: Le signal à exécuter.
            
        Returns:
            Un objet TradeResult contenant le résultat de l'exécution.
        """
        try:
            logger.info(f"Exécution du signal: {signal.symbol} {signal.direction}")
            
            # Générer un ID unique pour le signal
            signal_id = f"{signal.symbol}_{signal.direction}_{int(time.time())}"
            
            # Ajouter le suffixe USDT au symbole si nécessaire
            symbol = f"{signal.symbol}USDT"
            
            # Valider le signal
            if self.validate_trend:
                is_valid, message = await self.technical_analyzer.validate_trend(signal)
                if not is_valid:
                    logger.warning(f"Signal rejeté: {message}")
                    return TradeResult(
                        signal_id=signal_id,
                        symbol=symbol,
                        direction=signal.direction,
                        status="rejected",
                        message=message,
                        position_size=0,
                        leverage=0,
                        risk_percentage=0,
                        entry_price=0,
                        stop_loss=0,
                        take_profit_levels=[],
                        orders={}
                    )
            
            # Calculer le stop loss basé sur la volatilité si demandé
            if self.use_volatility_sl:
                volatility_sl = await self.technical_analyzer.calculate_volatility_stop_loss(signal)
                
                # Utiliser le stop loss le plus conservateur
                if signal.direction == "LONG":
                    stop_loss = max(signal.stop_loss, volatility_sl)
                else:  # SHORT
                    stop_loss = min(signal.stop_loss, volatility_sl)
            else:
                stop_loss = signal.stop_loss
            
            # Calculer la taille de position
            position_size, leverage, risk_percentage = await self.risk_manager.calculate_position_size(
            logger.info(f"Taille de position calculee: {position_size}, levier: {leverage}, risque: {risk_percentage}%")
            logger.info(f"Taille de position calcul�e: {position_size}, levier: {leverage}, risque: {risk_percentage}%")
                signal=signal,
                stop_loss=stop_loss
            )
            
            # Vérifier si la position est autorisée
            is_allowed, reason = await self.risk_manager.is_position_allowed(signal)
            if not is_allowed:
                logger.warning(f"Position non autorisée: {reason}")
                return TradeResult(
                    signal_id=signal_id,
                    symbol=symbol,
                    direction=signal.direction,
                    status="rejected",
                    message=reason,
                    position_size=0,
                    leverage=0,
                    risk_percentage=0,
                    entry_price=0,
                    stop_loss=0,
                    take_profit_levels=[],
                    orders={}
                )
            
            # Définir le levier
            await self.client.set_leverage(symbol, leverage)
            
            # Définir le type de marge (ISOLATED)
            await self.client.set_margin_type(symbol, "ISOLATED")
            
            # Déterminer le côté de l'ordre
            side = OrderSide.BUY if signal.direction == "LONG" else OrderSide.SELL
            
            # Déterminer le prix d'entrée (milieu de la plage)
            entry_price = (signal.entry_min + signal.entry_max) / 2
            
            # Calculer la distribution des TP
            tp_distribution = await self.risk_manager.calculate_tp_distribution(signal, position_size)
            
            # Exécuter l'ordre d'entrée
            entry_order = await self.client.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=position_size
            )
            
            # Placer l'ordre de stop loss
            sl_side = OrderSide.SELL if signal.direction == "LONG" else OrderSide.BUY
            sl_order = await self.client.place_order(
                symbol=symbol,
                side=sl_side,
                order_type=OrderType.STOP_MARKET,
                quantity=position_size,
                stop_price=stop_loss,
                reduce_only=True
            )
            
            # Placer les ordres de take profit
            tp_orders = []
            for i, (tp_level, tp_quantity) in enumerate(zip(signal.take_profit_levels, tp_distribution)):
                tp_side = OrderSide.SELL if signal.direction == "LONG" else OrderSide.BUY
                tp_order = await self.client.place_order(
                    symbol=symbol,
                    side=tp_side,
                    order_type=OrderType.TAKE_PROFIT_MARKET,
                    quantity=tp_quantity,
                    stop_price=tp_level,
                    reduce_only=True
                )
                tp_orders.append(tp_order)
            
            # Créer le résultat du trade
            result = TradeResult(
                signal_id=signal_id,
                symbol=symbol,
                direction=signal.direction,
                status="success",
                message="Trade exécuté avec succès",
                position_size=position_size,
                leverage=leverage,
                risk_percentage=risk_percentage,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_levels=signal.take_profit_levels,
                orders={
                    "entry": entry_order,
                    "sl": sl_order,
                    "tp": tp_orders
                }
            )
            
            # Ajouter le trade à l'historique
            self.trade_history.append(result)
            
            logger.info(f"Trade exécuté avec succès: {signal.symbol} {signal.direction}")
            return result
        
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du signal: {str(e)}")
            return TradeResult(
                signal_id=f"{signal.symbol}_{signal.direction}_{int(time.time())}",
                symbol=f"{signal.symbol}USDT",
                direction=signal.direction,
                status="error",
                message=f"Erreur lors de l'exécution: {str(e)}",
                position_size=0,
                leverage=0,
                risk_percentage=0,
                entry_price=0,
                stop_loss=0,
                take_profit_levels=[],
                orders={}
            )
    
    async def close_position(self, symbol: str, direction: str) -> Dict[str, Any]:
        """
        Ferme une position ouverte.
        
        Args:
            symbol: Le symbole de la position à fermer.
            direction: La direction de la position (LONG ou SHORT).
            
        Returns:
            Un dictionnaire contenant le résultat de l'opération.
        """
        try:
            # Ajouter le suffixe USDT au symbole si nécessaire
            if not symbol.endswith("USDT"):
                symbol = f"{symbol}USDT"
            
            # Déterminer le côté de la position
            position_side = PositionSide.BOTH
            
            # Fermer la position
            result = await self.client.close_position(symbol, position_side)
            
            logger.info(f"Position fermée: {symbol} {direction}")
            return {
                "status": "success",
                "message": f"Position fermée: {symbol} {direction}",
                "result": result
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture de la position: {str(e)}")
            return {
                "status": "error",
                "message": f"Erreur lors de la fermeture de la position: {str(e)}"
            }
    
    async def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """
        Annule tous les ordres ouverts pour un symbole.
        
        Args:
            symbol: Le symbole pour lequel annuler les ordres.
            
        Returns:
            Un dictionnaire contenant le résultat de l'opération.
        """
        try:
            # Ajouter le suffixe USDT au symbole si nécessaire
            if not symbol.endswith("USDT"):
                symbol = f"{symbol}USDT"
            
            # Récupérer les ordres ouverts
            open_orders = await self.client.get_open_orders(symbol)
            
            # Annuler chaque ordre
            results = []
            for order in open_orders:
                result = await self.client.cancel_order(symbol, order['orderId'])
                results.append(result)
            
            logger.info(f"Ordres annulés: {symbol} ({len(results)} ordres)")
            return {
                "status": "success",
                "message": f"Ordres annulés: {symbol} ({len(results)} ordres)",
                "results": results
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation des ordres: {str(e)}")
            return {
                "status": "error",
                "message": f"Erreur lors de l'annulation des ordres: {str(e)}"
            }
    
    async def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des trades.
        
        Returns:
            Une liste de dictionnaires contenant les informations des trades.
        """
        return [trade.to_dict() for trade in self.trade_history]
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calcule les métriques de performance des trades.
        
        Returns:
            Un dictionnaire contenant les métriques de performance.
        """
        try:
            # Récupérer l'historique des trades
            trades = self.trade_history
            
            if not trades:
                return {
                    "total_trades": 0,
                    "successful_trades": 0,
                    "failed_trades": 0,
                    "success_rate": 0,
                    "average_risk": 0,
                    "symbols": {}
                }
            
            # Calculer les métriques
            total_trades = len(trades)
            successful_trades = sum(1 for trade in trades if trade.status == "success")
            failed_trades = total_trades - successful_trades
            success_rate = successful_trades / total_trades if total_trades > 0 else 0
            average_risk = sum(trade.risk_percentage for trade in trades) / total_trades if total_trades > 0 else 0
            
            # Métriques par symbole
            symbols = {}
            for trade in trades:
                symbol = trade.symbol
                if symbol not in symbols:
                    symbols[symbol] = {
                        "total_trades": 0,
                        "successful_trades": 0,
                        "failed_trades": 0,
                        "success_rate": 0,
                        "average_risk": 0
                    }
                
                symbols[symbol]["total_trades"] += 1
                if trade.status == "success":
                    symbols[symbol]["successful_trades"] += 1
                else:
                    symbols[symbol]["failed_trades"] += 1
                
                symbols[symbol]["success_rate"] = symbols[symbol]["successful_trades"] / symbols[symbol]["total_trades"]
                symbols[symbol]["average_risk"] += trade.risk_percentage
            
            # Calculer la moyenne du risque par symbole
            for symbol in symbols:
                symbols[symbol]["average_risk"] /= symbols[symbol]["total_trades"]
            
            return {
                "total_trades": total_trades,
                "successful_trades": successful_trades,
                "failed_trades": failed_trades,
                "success_rate": success_rate,
                "average_risk": average_risk,
                "symbols": symbols
            }
        
        except Exception as e:
            logger.error(f"Erreur lors du calcul des métriques de performance: {str(e)}")
            return {
                "error": str(e)
            }


def main():
    """Fonction principale pour tester le module."""
    # Créer un client Binance (remplacer par vos clés API)
    from src.binance_client import BinanceClient
    from src.risk_manager import RiskManager
    
    client = BinanceClient(
        api_key="YOUR_API_KEY",
        api_secret="YOUR_API_SECRET",
        testnet=True  # Utiliser le testnet pour les tests
    )
    
    # Créer un gestionnaire de risque
    risk_manager = RiskManager(
        binance_client=client,
        risk_per_trade=1.0,
        max_total_risk=5.0
    )
    
    # Créer un exécuteur de trades
    executor = TradeExecutor(
        binance_client=client,
        risk_manager=risk_manager,
        validate_trend=True,
        use_volatility_sl=True
    )
    
    # Exemple d'utilisation
    async def test():
        try:
            # Créer un signal de test
            signal = TradingSignal(
                symbol="BTC",
                direction="LONG",
                entry_min=60000,
                entry_max=61000,
                take_profit_levels=[62000, 63000, 64000],
                stop_loss=59000,
                raw_text="Test signal"
            )
            
            # Exécuter le signal
            result = await executor.execute_signal(signal)
            print(f"Résultat: {result}")
            
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
