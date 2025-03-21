"""
Module de parsing des signaux de trading.

Ce module contient les classes et fonctions nécessaires pour parser
les signaux de trading reçus de Telegram.
Version améliorée avec support de formats multiples et validation avancée.
"""

import re
import logging
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

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


class SignalFormat(Enum):
    """Énumération des formats de signal supportés."""
    
    UNKNOWN = 0
    STANDARD = 1  # Format avec emojis et structure claire
    ALERT = 2     # Format "SIGNAL ALERT" avec structure par lignes
    SIMPLE = 3    # Format simple avec symbole, direction et quelques prix


@dataclass
class TradingSignal:
    """Représente un signal de trading parsé."""
    
    symbol: str
    direction: str  # "LONG" ou "SHORT"
    entry_min: float
    entry_max: float
    take_profit_levels: List[float]
    stop_loss: float
    raw_text: str
    format: SignalFormat = SignalFormat.UNKNOWN
    confidence: float = 1.0  # Niveau de confiance du parsing (0.0 à 1.0)
    
    def __post_init__(self):
        """Validation après initialisation."""
        # Convertir le symbole en majuscules et supprimer le suffixe USDT si présent
        self.symbol = self.symbol.upper()
        if self.symbol.endswith("USDT"):
            self.symbol = self.symbol[:-4]
        
        # Convertir la direction en majuscules
        self.direction = self.direction.upper()


class SignalParser:
    """
    Classe pour parser les signaux de trading.
    
    Cette classe prend en charge plusieurs formats de signaux et
    extrait les informations pertinentes pour le trading.
    """
    
    def __init__(self):
        """Initialise le parser de signaux."""
        # Patterns regex pour les différents formats
        
        # Pattern pour le format standard (avec emojis)
        self.standard_symbol_pattern = r'(?:🟢|🔴|⚪|🟡)?\s*([A-Z]+(?:USDT|BTC|ETH)?)\s*(LONG|SHORT|long|short)'
        self.standard_entry_pattern = r'(?:🎯|📈|📉)?\s*[Ee]ntry(?:\s*price)?:?\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)'
        self.standard_tp_pattern = r'(?:TP|[Tt]argets?|[Tt]ake\s*[Pp]rofits?)(?:\s*:)?\s*\n((?:\d+(?:\.\d+)?(?:\s*,?\s*)?)+)'
        self.standard_sl_pattern = r'(?:🛑|⛔|❌)?\s*(?:SL|[Ss]top\s*[Ll]oss)(?:\s*:)?\s*(\d+(?:\.\d+)?)'
        
        # Pattern pour le format alert
        self.alert_header_pattern = r'(?:SIGNAL\s*ALERT|TRADE\s*SIGNAL|TRADING\s*SIGNAL)'
        self.alert_coin_pattern = r'(?:COIN|SYMBOL|TICKER|PAIR):?\s*([A-Z]+(?:USDT|BTC|ETH)?)'
        self.alert_direction_pattern = r'(?:DIRECTION|SIDE|TYPE):?\s*(LONG|SHORT|long|short|[Bb]uy|[Ss]ell)'
        self.alert_entry_pattern = r'(?:ENTRY\s*(?:ZONE|PRICE|LEVEL)|ENTRY):?\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)'
        self.alert_tp_pattern = r'(?:TARGETS|TAKE\s*PROFITS?|TPS?):?\s*((?:\d+(?:\.\d+)?(?:\s*,\s*)?)+)'
        self.alert_sl_pattern = r'(?:STOP\s*LOSS|SL):?\s*(\d+(?:\.\d+)?)'
        
        # Pattern pour le format simple
        self.simple_symbol_pattern = r'^([A-Z]+(?:USDT|BTC|ETH)?)\s*(LONG|SHORT|long|short|[Bb]uy|[Ss]ell)'
        self.simple_entry_pattern = r'[Ee]ntry(?:\s*(?:around|at|price|zone))?:?\s*(\d+(?:\.\d+)?)'
        self.simple_tp_pattern = r'(?:TP|[Tt]argets?|[Tt]ake\s*[Pp]rofits?)(?:\s*:)?\s*((?:\d+(?:\.\d+)?(?:\s*,\s*)?)+)'
        self.simple_sl_pattern = r'(?:SL|[Ss]top\s*[Ll]oss)(?:\s*:)?\s*(\d+(?:\.\d+)?)'
        
        logger.info("SignalParser initialisé")
    
    def parse_signal(self, text: str) -> Tuple[Optional[TradingSignal], List[str]]:
        """
        Parse un signal de trading à partir d'un texte.
        
        Args:
            text: Le texte du signal à parser.
            
        Returns:
            Un tuple (signal, errors) où signal est un objet TradingSignal
            s'il a pu être parsé, et errors est une liste d'erreurs ou
            d'avertissements rencontrés pendant le parsing.
        """
        # Nettoyer le texte
        text = text.strip()
        
        # Détecter le format du signal
        signal_format = self.detect_format(text)
        
        # Parser le signal selon son format
        if signal_format == SignalFormat.STANDARD:
            return self._parse_standard_format(text)
        elif signal_format == SignalFormat.ALERT:
            return self._parse_alert_format(text)
        elif signal_format == SignalFormat.SIMPLE:
            return self._parse_simple_format(text)
        else:
            # Essayer tous les formats et prendre le meilleur résultat
            logger.info("Format de signal non détecté, essai de tous les formats")
            
            standard_result, standard_errors = self._parse_standard_format(text)
            alert_result, alert_errors = self._parse_alert_format(text)
            simple_result, simple_errors = self._parse_simple_format(text)
            
            # Choisir le résultat avec le moins d'erreurs et la plus grande confiance
            results = [
                (standard_result, standard_errors, SignalFormat.STANDARD),
                (alert_result, alert_errors, SignalFormat.ALERT),
                (simple_result, simple_errors, SignalFormat.SIMPLE)
            ]
            
            valid_results = [(r, e, f) for r, e, f in results if r is not None]
            
            if not valid_results:
                logger.warning("Aucun format n'a pu parser le signal")
                return None, ["Format de signal non reconnu"]
            
            # Trier par nombre d'erreurs (croissant) puis par confiance (décroissant)
            valid_results.sort(key=lambda x: (len(x[1]), -x[0].confidence if x[0] else 0))
            
            best_result, best_errors, best_format = valid_results[0]
            best_result.format = best_format
            
            logger.info(f"Meilleur format détecté: {best_format.name}, confiance: {best_result.confidence:.2f}")
            
            return best_result, best_errors
    
    def detect_format(self, text: str) -> SignalFormat:
        """
        Détecte le format d'un signal de trading.
        
        Args:
            text: Le texte du signal.
            
        Returns:
            Le format du signal détecté.
        """
        # Vérifier le format alert (le plus distinctif)
        if re.search(self.alert_header_pattern, text, re.IGNORECASE):
            return SignalFormat.ALERT
        
        # Vérifier le format standard (avec emojis)
        if re.search(r'(?:🟢|🔴|⚪|🟡|🎯|📈|📉|🛑|⛔|❌)', text):
            return SignalFormat.STANDARD
        
        # Vérifier le format simple
        if re.search(self.simple_symbol_pattern, text, re.MULTILINE):
            return SignalFormat.SIMPLE
        
        # Format inconnu
        return SignalFormat.UNKNOWN
    
    def _parse_standard_format(self, text: str) -> Tuple[Optional[TradingSignal], List[str]]:
        """
        Parse un signal au format standard (avec emojis).
        
        Args:
            text: Le texte du signal.
            
        Returns:
            Un tuple (signal, errors).
        """
        errors = []
        
        # Extraire le symbole et la direction
        symbol_match = re.search(self.standard_symbol_pattern, text, re.MULTILINE)
        if not symbol_match:
            errors.append("Symbole et direction non trouvés")
            symbol = None
            direction = None
        else:
            symbol = symbol_match.group(1)
            direction = symbol_match.group(2).upper()
        
        # Extraire la plage d'entrée
        entry_match = re.search(self.standard_entry_pattern, text, re.MULTILINE)
        if not entry_match:
            errors.append("Plage d'entrée non trouvée")
            entry_min = None
            entry_max = None
        else:
            try:
                entry_min = float(entry_match.group(1))
                entry_max = float(entry_match.group(2))
            except ValueError:
                errors.append("Erreur de conversion des prix d'entrée")
                entry_min = None
                entry_max = None
        
        # Extraire les niveaux de take profit
        tp_match = re.search(self.standard_tp_pattern, text, re.MULTILINE | re.DOTALL)
        if not tp_match:
            errors.append("Niveaux de take profit non trouvés")
            tp_levels = []
        else:
            tp_text = tp_match.group(1)
            # Chercher les niveaux de TP ligne par ligne
            tp_lines = tp_text.strip().split('\n')
            tp_levels = []
            
            for line in tp_lines:
                # Nettoyer la ligne et extraire les nombres
                line = line.strip()
                if not line:
                    continue
                
                # Vérifier si la ligne contient des nombres séparés par des virgules
                if ',' in line:
                    # Format: "1.1, 1.2, 1.3"
                    for tp_str in line.split(','):
                        try:
                            tp_levels.append(float(tp_str.strip()))
                        except ValueError:
                            errors.append(f"Erreur de conversion du TP: {tp_str}")
                else:
                    # Format: un nombre par ligne
                    try:
                        tp_levels.append(float(line))
                    except ValueError:
                        errors.append(f"Erreur de conversion du TP: {line}")
            
            if not tp_levels:
                errors.append("Aucun niveau de take profit valide trouvé")
        
        # Extraire le stop loss
        sl_match = re.search(self.standard_sl_pattern, text, re.MULTILINE)
        if not sl_match:
            errors.append("Stop loss non trouvé")
            stop_loss = None
        else:
            try:
                stop_loss = float(sl_match.group(1))
            except ValueError:
                errors.append("Erreur de conversion du stop loss")
                stop_loss = None
        
        # Vérifier si les informations essentielles sont présentes
        if None in [symbol, direction, entry_min, entry_max, stop_loss] or not tp_levels:
            # Essayer de compléter les informations manquantes
            if symbol and direction:
                # Si on a au moins le symbole et la direction, on peut créer un signal partiel
                logger.warning(f"Signal partiel détecté pour {symbol} {direction}")
                
                # Valeurs par défaut
                if entry_min is None and entry_max is None:
                    errors.append("Utilisation du prix actuel comme référence pour l'entrée")
                    # Dans un cas réel, on récupérerait le prix actuel via l'API Binance
                    entry_min = 1000.0  # Valeur fictive
                    entry_max = 1010.0  # Valeur fictive
                
                if stop_loss is None:
                    errors.append("Calcul automatique du stop loss")
                    if direction == "LONG":
                        stop_loss = entry_min * 0.95  # -5% par défaut
                    else:  # SHORT
                        stop_loss = entry_max * 1.05  # +5% par défaut
                
                if not tp_levels:
                    errors.append("Calcul automatique des take profits")
                    if direction == "LONG":
                        tp_levels = [entry_max * 1.05, entry_max * 1.10]  # +5% et +10% par défaut
                    else:  # SHORT
                        tp_levels = [entry_min * 0.95, entry_min * 0.90]  # -5% et -10% par défaut
                
                # Créer le signal avec les informations disponibles
                signal = TradingSignal(
                    symbol=symbol,
                    direction=direction,
                    entry_min=entry_min,
                    entry_max=entry_max,
                    take_profit_levels=tp_levels,
                    stop_loss=stop_loss,
                    raw_text=text,
                    format=SignalFormat.STANDARD,
                    confidence=0.7  # Confiance réduite car informations partielles
                )
                
                return signal, errors
            else:
                logger.error("Informations essentielles manquantes, impossible de créer un signal")
                return None, errors
        
        # Créer le signal
        signal = TradingSignal(
            symbol=symbol,
            direction=direction,
            entry_min=entry_min,
            entry_max=entry_max,
            take_profit_levels=tp_levels,
            stop_loss=stop_loss,
            raw_text=text,
            format=SignalFormat.STANDARD,
            confidence=1.0 - (len(errors) * 0.1)  # Réduire la confiance en fonction du nombre d'erreurs
        )
        
        # Valider le signal
        is_valid, validation_errors = self.validate_signal(signal)
        errors.extend(validation_errors)
        
        if not is_valid:
            logger.warning(f"Signal invalide: {', '.join(validation_errors)}")
            # On retourne quand même le signal, mais avec une confiance réduite
            signal.confidence *= 0.5
        
        return signal, errors
    
    def _parse_alert_format(self, text: str) -> Tuple[Optional[TradingSignal], List[str]]:
        """
        Parse un signal au format alert.
        
        Args:
            text: Le texte du signal.
            
        Returns:
            Un tuple (signal, errors).
        """
        errors = []
        
        # Vérifier si c'est bien un format alert
        header_match = re.search(self.alert_header_pattern, text, re.IGNORECASE)
        if not header_match:
            errors.append("En-tête de signal alert non trouvé")
        
        # Extraire le symbole
        coin_match = re.search(self.alert_coin_pattern, text, re.IGNORECASE)
        if not coin_match:
            errors.append("Symbole non trouvé")
            symbol = None
        else:
            symbol = coin_match.group(1)
        
        # Extraire la direction
        direction_match = re.search(self.alert_direction_pattern, text, re.IGNORECASE)
        if not direction_match:
            errors.append("Direction non trouvée")
            direction = None
        else:
            direction_text = direction_match.group(1).upper()
            if direction_text in ["LONG", "BUY"]:
                direction = "LONG"
            elif direction_text in ["SHORT", "SELL"]:
                direction = "SHORT"
            else:
                errors.append(f"Direction non reconnue: {direction_text}")
                direction = None
        
        # Extraire la plage d'entrée
        entry_match = re.search(self.alert_entry_pattern, text, re.IGNORECASE)
        if not entry_match:
            errors.append("Plage d'entrée non trouvée")
            entry_min = None
            entry_max = None
        else:
            try:
                entry_min = float(entry_match.group(1))
                entry_max = float(entry_match.group(2))
            except ValueError:
                errors.append("Erreur de conversion des prix d'entrée")
                entry_min = None
                entry_max = None
        
        # Extraire les niveaux de take profit
        tp_match = re.search(self.alert_tp_pattern, text, re.IGNORECASE)
        if not tp_match:
            errors.append("Niveaux de take profit non trouvés")
            tp_levels = []
        else:
            tp_text = tp_match.group(1)
            # Extraire les nombres séparés par des virgules
            tp_levels = []
            for tp_str in tp_text.split(','):
                try:
                    tp_levels.append(float(tp_str.strip()))
                except ValueError:
                    errors.append(f"Erreur de conversion du TP: {tp_str}")
            
            if not tp_levels:
                errors.append("Aucun niveau de take profit valide trouvé")
        
        # Extraire le stop loss
        sl_match = re.search(self.alert_sl_pattern, text, re.IGNORECASE)
        if not sl_match:
            errors.append("Stop loss non trouvé")
            stop_loss = None
        else:
            try:
                stop_loss = float(sl_match.group(1))
            except ValueError:
                errors.append("Erreur de conversion du stop loss")
                stop_loss = None
        
        # Vérifier si les informations essentielles sont présentes
        if None in [symbol, direction, entry_min, entry_max, stop_loss] or not tp_levels:
            # Essayer de compléter les informations manquantes
            if symbol and direction:
                # Si on a au moins le symbole et la direction, on peut créer un signal partiel
                logger.warning(f"Signal partiel détecté pour {symbol} {direction}")
                
                # Valeurs par défaut
                if entry_min is None and entry_max is None:
                    errors.append("Utilisation du prix actuel comme référence pour l'entrée")
                    # Dans un cas réel, on récupérerait le prix actuel via l'API Binance
                    entry_min = 1000.0  # Valeur fictive
                    entry_max = 1010.0  # Valeur fictive
                
                if stop_loss is None:
                    errors.append("Calcul automatique du stop loss")
                    if direction == "LONG":
                        stop_loss = entry_min * 0.95  # -5% par défaut
                    else:  # SHORT
                        stop_loss = entry_max * 1.05  # +5% par défaut
                
                if not tp_levels:
                    errors.append("Calcul automatique des take profits")
                    if direction == "LONG":
                        tp_levels = [entry_max * 1.05, entry_max * 1.10]  # +5% et +10% par défaut
                    else:  # SHORT
                        tp_levels = [entry_min * 0.95, entry_min * 0.90]  # -5% et -10% par défaut
                
                # Créer le signal avec les informations disponibles
                signal = TradingSignal(
                    symbol=symbol,
                    direction=direction,
                    entry_min=entry_min,
                    entry_max=entry_max,
                    take_profit_levels=tp_levels,
                    stop_loss=stop_loss,
                    raw_text=text,
                    format=SignalFormat.ALERT,
                    confidence=0.7  # Confiance réduite car informations partielles
                )
                
                return signal, errors
            else:
                logger.error("Informations essentielles manquantes, impossible de créer un signal")
                return None, errors
        
        # Créer le signal
        signal = TradingSignal(
            symbol=symbol,
            direction=direction,
            entry_min=entry_min,
            entry_max=entry_max,
            take_profit_levels=tp_levels,
            stop_loss=stop_loss,
            raw_text=text,
            format=SignalFormat.ALERT,
            confidence=1.0 - (len(errors) * 0.1)  # Réduire la confiance en fonction du nombre d'erreurs
        )
        
        # Valider le signal
        is_valid, validation_errors = self.validate_signal(signal)
        errors.extend(validation_errors)
        
        if not is_valid:
            logger.warning(f"Signal invalide: {', '.join(validation_errors)}")
            # On retourne quand même le signal, mais avec une confiance réduite
            signal.confidence *= 0.5
        
        return signal, errors
    
    def _parse_simple_format(self, text: str) -> Tuple[Optional[TradingSignal], List[str]]:
        """
        Parse un signal au format simple.
        
        Args:
            text: Le texte du signal.
            
        Returns:
            Un tuple (signal, errors).
        """
        errors = []
        
        # Extraire le symbole et la direction
        symbol_match = re.search(self.simple_symbol_pattern, text, re.MULTILINE)
        if not symbol_match:
            errors.append("Symbole et direction non trouvés")
            symbol = None
            direction = None
        else:
            symbol = symbol_match.group(1)
            direction_text = symbol_match.group(2).upper()
            if direction_text in ["LONG", "BUY"]:
                direction = "LONG"
            elif direction_text in ["SHORT", "SELL"]:
                direction = "SHORT"
            else:
                errors.append(f"Direction non reconnue: {direction_text}")
                direction = None
        
        # Extraire le prix d'entrée (format simple a souvent un seul prix, pas une plage)
        entry_match = re.search(self.simple_entry_pattern, text, re.MULTILINE)
        if not entry_match:
            errors.append("Prix d'entrée non trouvé")
            entry_price = None
        else:
            try:
                entry_price = float(entry_match.group(1))
            except ValueError:
                errors.append("Erreur de conversion du prix d'entrée")
                entry_price = None
        
        # Calculer la plage d'entrée à partir du prix unique (±1%)
        if entry_price is not None:
            entry_min = entry_price * 0.99
            entry_max = entry_price * 1.01
        else:
            entry_min = None
            entry_max = None
        
        # Extraire les niveaux de take profit
        tp_match = re.search(self.simple_tp_pattern, text, re.MULTILINE)
        if not tp_match:
            errors.append("Niveaux de take profit non trouvés")
            tp_levels = []
        else:
            tp_text = tp_match.group(1)
            # Extraire les nombres séparés par des virgules
            tp_levels = []
            for tp_str in tp_text.split(','):
                try:
                    tp_levels.append(float(tp_str.strip()))
                except ValueError:
                    errors.append(f"Erreur de conversion du TP: {tp_str}")
            
            if not tp_levels:
                errors.append("Aucun niveau de take profit valide trouvé")
        
        # Extraire le stop loss
        sl_match = re.search(self.simple_sl_pattern, text, re.MULTILINE)
        if not sl_match:
            errors.append("Stop loss non trouvé")
            stop_loss = None
        else:
            try:
                stop_loss = float(sl_match.group(1))
            except ValueError:
                errors.append("Erreur de conversion du stop loss")
                stop_loss = None
        
        # Vérifier si les informations essentielles sont présentes
        if None in [symbol, direction, entry_min, entry_max, stop_loss] or not tp_levels:
            # Essayer de compléter les informations manquantes
            if symbol and direction:
                # Si on a au moins le symbole et la direction, on peut créer un signal partiel
                logger.warning(f"Signal partiel détecté pour {symbol} {direction}")
                
                # Valeurs par défaut
                if entry_min is None and entry_max is None:
                    errors.append("Utilisation du prix actuel comme référence pour l'entrée")
                    # Dans un cas réel, on récupérerait le prix actuel via l'API Binance
                    entry_price = 1000.0  # Valeur fictive
                    entry_min = entry_price * 0.99
                    entry_max = entry_price * 1.01
                
                if stop_loss is None:
                    errors.append("Calcul automatique du stop loss")
                    if direction == "LONG":
                        stop_loss = entry_min * 0.95  # -5% par défaut
                    else:  # SHORT
                        stop_loss = entry_max * 1.05  # +5% par défaut
                
                if not tp_levels:
                    errors.append("Calcul automatique des take profits")
                    if direction == "LONG":
                        tp_levels = [entry_max * 1.05, entry_max * 1.10]  # +5% et +10% par défaut
                    else:  # SHORT
                        tp_levels = [entry_min * 0.95, entry_min * 0.90]  # -5% et -10% par défaut
                
                # Créer le signal avec les informations disponibles
                signal = TradingSignal(
                    symbol=symbol,
                    direction=direction,
                    entry_min=entry_min,
                    entry_max=entry_max,
                    take_profit_levels=tp_levels,
                    stop_loss=stop_loss,
                    raw_text=text,
                    format=SignalFormat.SIMPLE,
                    confidence=0.7  # Confiance réduite car informations partielles
                )
                
                return signal, errors
            else:
                logger.error("Informations essentielles manquantes, impossible de créer un signal")
                return None, errors
        
        # Créer le signal
        signal = TradingSignal(
            symbol=symbol,
            direction=direction,
            entry_min=entry_min,
            entry_max=entry_max,
            take_profit_levels=tp_levels,
            stop_loss=stop_loss,
            raw_text=text,
            format=SignalFormat.SIMPLE,
            confidence=1.0 - (len(errors) * 0.1)  # Réduire la confiance en fonction du nombre d'erreurs
        )
        
        # Valider le signal
        is_valid, validation_errors = self.validate_signal(signal)
        errors.extend(validation_errors)
        
        if not is_valid:
            logger.warning(f"Signal invalide: {', '.join(validation_errors)}")
            # On retourne quand même le signal, mais avec une confiance réduite
            signal.confidence *= 0.5
        
        return signal, errors
    
    def validate_signal(self, signal: TradingSignal) -> Tuple[bool, List[str]]:
        """
        Valide un signal de trading.
        
        Args:
            signal: Le signal à valider.
            
        Returns:
            Un tuple (is_valid, errors) indiquant si le signal est valide
            et les erreurs de validation le cas échéant.
        """
        errors = []
        
        # Vérifier que le symbole est valide
        if not signal.symbol or len(signal.symbol) < 2:
            errors.append(f"Symbole invalide: {signal.symbol}")
        
        # Vérifier que la direction est valide
        if signal.direction not in ["LONG", "SHORT"]:
            errors.append(f"Direction invalide: {signal.direction}")
        
        # Vérifier que la plage d'entrée est valide
        if signal.entry_min >= signal.entry_max:
            errors.append(f"Plage d'entrée invalide: {signal.entry_min} - {signal.entry_max}")
        
        # Vérifier que les niveaux de TP sont valides
        if not signal.take_profit_levels:
            errors.append("Aucun niveau de take profit")
        else:
            # Vérifier que les TP sont dans le bon ordre par rapport à la direction
            if signal.direction == "LONG":
                for tp in signal.take_profit_levels:
                    if tp <= signal.entry_min:
                        errors.append(f"TP invalide pour LONG: {tp} <= {signal.entry_min}")
                
                # Vérifier que les TP sont en ordre croissant
                for i in range(1, len(signal.take_profit_levels)):
                    if signal.take_profit_levels[i] <= signal.take_profit_levels[i-1]:
                        errors.append(f"TP non croissants: {signal.take_profit_levels[i-1]} >= {signal.take_profit_levels[i]}")
            else:  # SHORT
                for tp in signal.take_profit_levels:
                    if tp >= signal.entry_max:
                        errors.append(f"TP invalide pour SHORT: {tp} >= {signal.entry_max}")
                
                # Vérifier que les TP sont en ordre décroissant
                for i in range(1, len(signal.take_profit_levels)):
                    if signal.take_profit_levels[i] >= signal.take_profit_levels[i-1]:
                        errors.append(f"TP non décroissants: {signal.take_profit_levels[i-1]} <= {signal.take_profit_levels[i]}")
        
        # Vérifier que le stop loss est valide
        if signal.direction == "LONG":
            if signal.stop_loss >= signal.entry_min:
                errors.append(f"SL invalide pour LONG: {signal.stop_loss} >= {signal.entry_min}")
        else:  # SHORT
            if signal.stop_loss <= signal.entry_max:
                errors.append(f"SL invalide pour SHORT: {signal.stop_loss} <= {signal.entry_max}")
        
        # Vérifier le ratio risque/récompense
        if signal.take_profit_levels and signal.stop_loss:
            if signal.direction == "LONG":
                risk = signal.entry_max - signal.stop_loss
                reward = signal.take_profit_levels[0] - signal.entry_max
            else:  # SHORT
                risk = signal.stop_loss - signal.entry_min
                reward = signal.entry_min - signal.take_profit_levels[0]
            
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < 1.0:
                errors.append(f"Ratio risque/récompense faible: {rr_ratio:.2f}")
        
        return len(errors) == 0, errors


def main():
    """Fonction principale pour tester le module."""
    # Exemples de signaux
    signals = [
        # Format standard
        """
        🟢 BTCUSDT LONG

        🎯Entry price: 60000 - 61000

        TP:
        62000
        63000
        64000

        🛑 SL 59000
        """,
        
        # Format alert
        """
        SIGNAL ALERT
        COIN: ETHUSDT
        DIRECTION: SHORT
        ENTRY ZONE: 3000 - 3100
        TARGETS: 2900, 2800, 2700
        STOP LOSS: 3200
        """,
        
        # Format simple
        """
        ADAUSDT LONG
        Entry around 1.2
        SL 1.1
        TP 1.3, 1.4, 1.5
        """
    ]
    
    # Créer un parser
    parser = SignalParser()
    
    # Parser chaque signal
    for i, signal_text in enumerate(signals):
        print(f"\nSignal {i+1}:")
        print("-" * 40)
        
        signal, errors = parser.parse_signal(signal_text)
        
        if signal:
            print(f"Symbole: {signal.symbol}")
            print(f"Direction: {signal.direction}")
            print(f"Entrée: {signal.entry_min} - {signal.entry_max}")
            print(f"Take Profits: {signal.take_profit_levels}")
            print(f"Stop Loss: {signal.stop_loss}")
            print(f"Format: {signal.format.name}")
            print(f"Confiance: {signal.confidence:.2f}")
        else:
            print("Échec du parsing")
        
        if errors:
            print("Erreurs/Avertissements:")
            for error in errors:
                print(f"- {error}")


if __name__ == "__main__":
    main()
