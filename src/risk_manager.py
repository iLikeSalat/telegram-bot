"""
Module de gestion des risques.

Ce module contient les classes et fonctions nécessaires pour gérer
les risques liés au trading.
Version améliorée avec gestion dynamique de la taille des positions et analyse de corrélation.
"""

import logging
import time
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict

from src.signal_parser import TradingSignal
from src.binance_client import BinanceClient

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
class Position:
    """Représente une position ouverte."""
    
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    position_size: float
    leverage: int
    risk_percentage: float
    stop_loss: float
    take_profit_levels: List[float]
    orders: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        """Initialisation après création."""
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'objet en dictionnaire."""
        return asdict(self)


class RiskManager:
    """
    Classe pour gérer les risques liés au trading.
    
    Cette classe fournit des méthodes pour calculer les tailles de position,
    gérer le risque global du portefeuille, et analyser les corrélations
    entre les différents actifs.
    """
    
    def __init__(self, binance_client: BinanceClient, risk_per_trade: float = 1.0,
                max_total_risk: float = 5.0, max_positions: int = 10):
        """
        Initialise le gestionnaire de risque.
        
        Args:
            binance_client: Le client Binance à utiliser pour récupérer les données.
            risk_per_trade: Le pourcentage de risque par trade (en % du capital).
            max_total_risk: Le pourcentage maximum de risque total (en % du capital).
            max_positions: Le nombre maximum de positions ouvertes simultanément.
        """
        self.client = binance_client
        self.risk_per_trade = risk_per_trade
        self.max_total_risk = max_total_risk
        self.max_positions = max_positions
        
        # Liste des positions ouvertes
        self.open_positions = []
        
        # Cache des corrélations
        self.correlation_cache = {}
        self.correlation_expiry = 3600  # 1 heure
        
        logger.info(f"RiskManager initialisé (risk_per_trade: {risk_per_trade}%, max_total_risk: {max_total_risk}%, max_positions: {max_positions})")
    
    async def calculate_position_size(self, signal: TradingSignal, stop_loss: Optional[float] = None) -> Tuple[float, int, float]:
        """
        Calcule la taille de position optimale pour un signal.
        
        Args:
            signal: Le signal pour lequel calculer la taille de position.
            stop_loss: Le prix du stop loss (si différent de celui du signal).
            
        Returns:
            Un tuple (position_size, leverage, risk_percentage).
        """
        try:
            # Utiliser le stop loss du signal si non spécifié
            if stop_loss is None:
                stop_loss = signal.stop_loss
            
            # Récupérer le solde du compte
            account_info = await self.client.get_account_info()
            balance = float(account_info['totalWalletBalance'])
            
            # Calculer le montant à risquer
            risk_amount = balance * (self.risk_per_trade / 100)
            
            # Ajouter le suffixe USDT au symbole si nécessaire
            symbol = f"{signal.symbol}USDT"
            
            # Récupérer le prix actuel
            current_price = await self.client.get_symbol_price(symbol)
            
            # Calculer la distance au stop loss en pourcentage
            if signal.direction == "LONG":
                entry_price = signal.entry_min  # Utiliser le prix d'entrée minimum pour un LONG
                stop_distance_percent = abs((entry_price - stop_loss) / entry_price) * 100
            else:  # SHORT
                entry_price = signal.entry_max  # Utiliser le prix d'entrée maximum pour un SHORT
                stop_distance_percent = abs((stop_loss - entry_price) / entry_price) * 100
            
            # Calculer le levier optimal (arrondi à l'entier inférieur)
            # Le levier est calculé pour que la distance au stop loss * levier soit environ 100%
            # Cela signifie qu'une variation de prix jusqu'au stop loss entraînerait une perte de 100% de la marge
            optimal_leverage = min(int(100 / stop_distance_percent), 20)  # Limiter à 20x
            
            # Calculer la taille de position
            position_size = risk_amount * optimal_leverage / current_price
            
            # Ajuster la taille de position en fonction du risque total actuel
            current_risk = await self.get_total_risk()
            if current_risk + self.risk_per_trade > self.max_total_risk:
                # Réduire la taille de position pour respecter le risque maximum
                adjusted_risk = max(0, self.max_total_risk - current_risk)
                position_size = position_size * (adjusted_risk / self.risk_per_trade)
                logger.warning(f"Taille de position réduite en raison du risque total ({current_risk}% + {self.risk_per_trade}% > {self.max_total_risk}%)")
            
            logger.info(f"Taille de position calculée: {position_size} {symbol} (levier: {optimal_leverage}x, risque: {self.risk_per_trade}%)")
            return position_size, optimal_leverage, self.risk_per_trade
        
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la taille de position: {str(e)}")
            # En cas d'erreur, retourner des valeurs par défaut
            return 0.01, 1, 0.1
    
    async def is_position_allowed(self, signal: TradingSignal) -> Tuple[bool, str]:
        """
        Vérifie si une nouvelle position est autorisée.
        
        Args:
            signal: Le signal pour lequel vérifier l'autorisation.
            
        Returns:
            Un tuple (is_allowed, reason) indiquant si la position est autorisée
            et la raison si elle ne l'est pas.
        """
        try:
            # Vérifier le nombre de positions ouvertes
            if len(self.open_positions) >= self.max_positions:
                return False, f"Nombre maximum de positions atteint ({self.max_positions})"
            
            # Vérifier si une position est déjà ouverte pour ce symbole
            symbol = f"{signal.symbol}USDT"
            for position in self.open_positions:
                if position.symbol == symbol:
                    return False, f"Position déjà ouverte pour {symbol}"
            
            # Vérifier le risque total
            current_risk = await self.get_total_risk()
            if current_risk + self.risk_per_trade > self.max_total_risk:
                return False, f"Risque total trop élevé ({current_risk}% + {self.risk_per_trade}% > {self.max_total_risk}%)"
            
            # Vérifier les corrélations avec les positions existantes
            correlated_symbols = await self.get_correlated_symbols(signal.symbol)
            for position in self.open_positions:
                position_symbol = position.symbol.replace("USDT", "")
                if position_symbol in correlated_symbols:
                    # Si la corrélation est forte et les directions sont identiques, c'est un risque accru
                    if position.direction == signal.direction:
                        return False, f"Corrélation forte avec {position_symbol} dans la même direction"
            
            return True, "Position autorisée"
        
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de l'autorisation de position: {str(e)}")
            # En cas d'erreur, autoriser la position
            return True, "Position autorisée (erreur ignorée)"
    
    async def add_position(self, position: Position) -> None:
        """
        Ajoute une position à la liste des positions ouvertes.
        
        Args:
            position: La position à ajouter.
        """
        self.open_positions.append(position)
        logger.info(f"Position ajoutée: {position.symbol} {position.direction} (taille: {position.position_size}, risque: {position.risk_percentage}%)")
    
    async def remove_position(self, signal_id: str) -> Optional[Position]:
        """
        Supprime une position de la liste des positions ouvertes.
        
        Args:
            signal_id: L'ID du signal associé à la position.
            
        Returns:
            La position supprimée, ou None si elle n'a pas été trouvée.
        """
        for i, position in enumerate(self.open_positions):
            if position.signal_id == signal_id:
                removed_position = self.open_positions.pop(i)
                logger.info(f"Position supprimée: {removed_position.symbol} {removed_position.direction}")
                return removed_position
        
        logger.warning(f"Position non trouvée pour suppression: {signal_id}")
        return None
    
    async def get_total_risk(self) -> float:
        """
        Calcule le risque total actuel.
        
        Returns:
            Le pourcentage de risque total.
        """
        return sum(position.risk_percentage for position in self.open_positions)
    
    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Récupère la liste des positions ouvertes.
        
        Returns:
            Une liste de dictionnaires contenant les informations des positions ouvertes.
        """
        return [position.to_dict() for position in self.open_positions]
    
    async def calculate_tp_distribution(self, signal: TradingSignal, total_quantity: float) -> List[float]:
        """
        Calcule la distribution des quantités pour les différents niveaux de take profit.
        
        Args:
            signal: Le signal pour lequel calculer la distribution.
            total_quantity: La quantité totale à distribuer.
            
        Returns:
            Une liste de quantités pour chaque niveau de take profit.
        """
        # Nombre de niveaux de TP
        num_levels = len(signal.take_profit_levels)
        
        if num_levels == 0:
            return []
        
        if num_levels == 1:
            return [total_quantity]
        
        # Distribution pyramidale (plus de quantité pour les premiers TP, moins pour les derniers)
        # Exemple pour 3 niveaux: 50%, 30%, 20%
        if num_levels == 2:
            distribution = [0.6, 0.4]
        elif num_levels == 3:
            distribution = [0.5, 0.3, 0.2]
        elif num_levels == 4:
            distribution = [0.4, 0.3, 0.2, 0.1]
        else:
            # Pour plus de niveaux, utiliser une distribution décroissante
            total_parts = sum(range(num_levels, 0, -1))
            distribution = [i / total_parts for i in range(num_levels, 0, -1)]
        
        # Calculer les quantités
        quantities = [total_quantity * dist for dist in distribution]
        
        return quantities
    
    async def get_correlated_symbols(self, symbol: str) -> List[str]:
        """
        Récupère la liste des symboles corrélés à un symbole donné.
        
        Args:
            symbol: Le symbole pour lequel récupérer les corrélations.
            
        Returns:
            Une liste de symboles corrélés.
        """
        # Vérifier si les corrélations sont dans le cache et si elles sont encore valides
        now = time.time()
        if symbol in self.correlation_cache and now - self.correlation_cache[symbol]["timestamp"] < self.correlation_expiry:
            return self.correlation_cache[symbol]["correlated_symbols"]
        
        # Dans un cas réel, on calculerait les corrélations à partir des données historiques
        # Pour cet exemple, on utilise des corrélations prédéfinies
        
        # Corrélations connues entre les principales crypto-monnaies
        correlations = {
            "BTC": ["ETH", "BNB", "SOL"],
            "ETH": ["BTC", "BNB", "SOL", "AVAX"],
            "BNB": ["BTC", "ETH", "SOL"],
            "SOL": ["BTC", "ETH", "BNB", "AVAX"],
            "ADA": ["DOT", "XRP", "MATIC"],
            "XRP": ["ADA", "XLM", "ALGO"],
            "DOT": ["ADA", "LINK", "ATOM"],
            "AVAX": ["ETH", "SOL", "MATIC"],
            "MATIC": ["ETH", "SOL", "AVAX", "ADA"],
            "LINK": ["DOT", "ATOM", "UNI"],
            "ATOM": ["DOT", "LINK", "NEAR"],
            "DOGE": ["SHIB", "FLOKI", "BABYDOGE"],
            "SHIB": ["DOGE", "FLOKI", "ELON"]
        }
        
        # Retourner les symboles corrélés ou une liste vide si le symbole n'est pas connu
        correlated_symbols = correlations.get(symbol.upper(), [])
        
        # Mettre en cache les corrélations
        self.correlation_cache[symbol] = {
            "correlated_symbols": correlated_symbols,
            "timestamp": now
        }
        
        return correlated_symbols
    
    async def calculate_portfolio_risk(self) -> Dict[str, Any]:
        """
        Calcule le risque global du portefeuille en tenant compte des corrélations.
        
        Returns:
            Un dictionnaire contenant les métriques de risque du portefeuille.
        """
        try:
            # Récupérer les positions ouvertes
            positions = self.open_positions
            
            if not positions:
                return {
                    "total_risk": 0,
                    "effective_risk": 0,
                    "correlation_factor": 1,
                    "positions": []
                }
            
            # Calculer le risque total (somme des risques individuels)
            total_risk = sum(position.risk_percentage for position in positions)
            
            # Calculer le facteur de corrélation
            # Plus les positions sont corrélées, plus le facteur est élevé
            correlation_factor = 1.0
            
            # Pour chaque paire de positions, vérifier si elles sont corrélées
            for i, pos1 in enumerate(positions):
                symbol1 = pos1.symbol.replace("USDT", "")
                for j, pos2 in enumerate(positions):
                    if i >= j:
                        continue  # Éviter les doublons et les comparaisons avec soi-même
                    
                    symbol2 = pos2.symbol.replace("USDT", "")
                    
                    # Vérifier si les symboles sont corrélés
                    correlated_symbols = await self.get_correlated_symbols(symbol1)
                    if symbol2 in correlated_symbols:
                        # Si les positions sont dans la même direction, augmenter le facteur de corrélation
                        if pos1.direction == pos2.direction:
                            correlation_factor += 0.1
                        # Si les positions sont dans des directions opposées, réduire le facteur de corrélation
                        else:
                            correlation_factor -= 0.05
            
            # Limiter le facteur de corrélation entre 0.5 et 2.0
            correlation_factor = max(0.5, min(2.0, correlation_factor))
            
            # Calculer le risque effectif (risque total ajusté par le facteur de corrélation)
            effective_risk = total_risk * correlation_factor
            
            # Préparer les informations de position
            position_info = []
            for position in positions:
                symbol = position.symbol.replace("USDT", "")
                correlated_symbols = await self.get_correlated_symbols(symbol)
                
                # Filtrer les symboles corrélés qui sont dans le portefeuille
                portfolio_correlated = [s for s in correlated_symbols if any(p.symbol.replace("USDT", "") == s for p in positions)]
                
                position_info.append({
                    "symbol": position.symbol,
                    "direction": position.direction,
                    "risk_percentage": position.risk_percentage,
                    "correlated_symbols": portfolio_correlated
                })
            
            return {
                "total_risk": total_risk,
                "effective_risk": effective_risk,
                "correlation_factor": correlation_factor,
                "positions": position_info
            }
        
        except Exception as e:
            logger.error(f"Erreur lors du calcul du risque du portefeuille: {str(e)}")
            return {
                "error": str(e)
            }
    
    async def adjust_position_sizes(self) -> Dict[str, Any]:
        """
        Ajuste les tailles de position en fonction du risque global.
        
        Returns:
            Un dictionnaire contenant les résultats de l'ajustement.
        """
        try:
            # Calculer le risque du portefeuille
            portfolio_risk = await self.calculate_portfolio_risk()
            
            # Si le risque effectif dépasse le risque maximum, ajuster les positions
            if portfolio_risk["effective_risk"] > self.max_total_risk:
                logger.warning(f"Risque effectif trop élevé: {portfolio_risk['effective_risk']}% > {self.max_total_risk}%")
                
                # Calculer le facteur de réduction
                reduction_factor = self.max_total_risk / portfolio_risk["effective_risk"]
                
                # Ajuster chaque position
                adjustments = []
                for position in self.open_positions:
                    # Calculer la nouvelle taille de position
                    new_size = position.position_size * reduction_factor
                    
                    # Enregistrer l'ajustement
                    adjustments.append({
                        "symbol": position.symbol,
                        "old_size": position.position_size,
                        "new_size": new_size,
                        "reduction_factor": reduction_factor
                    })
                    
                    # Mettre à jour la position
                    position.position_size = new_size
                    position.risk_percentage = position.risk_percentage * reduction_factor
                
                return {
                    "status": "adjusted",
                    "message": f"Positions ajustées (facteur: {reduction_factor:.2f})",
                    "adjustments": adjustments
                }
            else:
                return {
                    "status": "no_adjustment",
                    "message": f"Aucun ajustement nécessaire (risque effectif: {portfolio_risk['effective_risk']}%)"
                }
        
        except Exception as e:
            logger.error(f"Erreur lors de l'ajustement des tailles de position: {str(e)}")
            return {
                "status": "error",
                "message": f"Erreur lors de l'ajustement: {str(e)}"
            }


def main():
    """Fonction principale pour tester le module."""
    # Créer un client Binance (remplacer par vos clés API)
    from src.binance_client import BinanceClient
    
    client = BinanceClient(
        api_key="YOUR_API_KEY",
        api_secret="YOUR_API_SECRET",
        testnet=True  # Utiliser le testnet pour les tests
    )
    
    # Créer un gestionnaire de risque
    risk_manager = RiskManager(
        binance_client=client,
        risk_per_trade=1.0,
        max_total_risk=5.0,
        max_positions=10
    )
    
    # Exemple d'utilisation
    async def test():
        try:
            # Créer un signal de test
            from src.signal_parser import TradingSignal, SignalFormat
            
            signal = TradingSignal(
                symbol="BTC",
                direction="LONG",
                entry_min=60000,
                entry_max=61000,
                take_profit_levels=[62000, 63000, 64000],
                stop_loss=59000,
                raw_text="Test signal",
                format=SignalFormat.STANDARD
            )
            
            # Calculer la taille de position
            position_size, leverage, risk_percentage = await risk_manager.calculate_position_size(signal)
            print(f"Taille de position: {position_size}, Levier: {leverage}x, Risque: {risk_percentage}%")
            
            # Vérifier si la position est autorisée
            is_allowed, reason = await risk_manager.is_position_allowed(signal)
            print(f"Position autorisée: {is_allowed}, Raison: {reason}")
            
            # Calculer la distribution des TP
            tp_distribution = await risk_manager.calculate_tp_distribution(signal, position_size)
            print(f"Distribution des TP: {tp_distribution}")
            
            # Récupérer les symboles corrélés
            correlated_symbols = await risk_manager.get_correlated_symbols("BTC")
            print(f"Symboles corrélés à BTC: {correlated_symbols}")
            
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
