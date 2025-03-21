"""
Tests pour les composants du bot Telegram.

Ce module contient les tests pour vérifier le bon fonctionnement
des différents composants du bot Telegram.
"""

import asyncio
import logging
import time
import json
import os
import sys
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from queue import Queue

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les modules
from src.signal_parser import SignalParser, TradingSignal, SignalFormat
from src.telegram_client import TelegramClient, SignalMessage, SignalProcessor
from src.binance_client import BinanceClient, OrderExecutor, MarketData
from src.trade_executor import TradeExecutor, TechnicalAnalyzer, TradeResult
from src.risk_manager import RiskManager, Position

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TestRunner:
    """
    Classe pour exécuter les tests des composants.
    """
    
    def __init__(self):
        """Initialise le runner de tests."""
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
    
    def assert_equal(self, actual, expected, test_name):
        """
        Vérifie que deux valeurs sont égales.
        
        Args:
            actual: La valeur actuelle.
            expected: La valeur attendue.
            test_name: Le nom du test.
        """
        try:
            assert actual == expected
            self.tests_passed += 1
            self.test_results.append({
                "name": test_name,
                "status": "PASSED",
                "message": f"Expected: {expected}, Got: {actual}"
            })
            logger.info(f"✅ Test '{test_name}' PASSED")
        except AssertionError:
            self.tests_failed += 1
            self.test_results.append({
                "name": test_name,
                "status": "FAILED",
                "message": f"Expected: {expected}, Got: {actual}"
            })
            logger.error(f"❌ Test '{test_name}' FAILED: Expected {expected}, got {actual}")
    
    def assert_true(self, condition, test_name):
        """
        Vérifie qu'une condition est vraie.
        
        Args:
            condition: La condition à vérifier.
            test_name: Le nom du test.
        """
        try:
            assert condition
            self.tests_passed += 1
            self.test_results.append({
                "name": test_name,
                "status": "PASSED",
                "message": "Condition is True"
            })
            logger.info(f"✅ Test '{test_name}' PASSED")
        except AssertionError:
            self.tests_failed += 1
            self.test_results.append({
                "name": test_name,
                "status": "FAILED",
                "message": "Condition is False"
            })
            logger.error(f"❌ Test '{test_name}' FAILED: Condition is False")
    
    def assert_not_none(self, value, test_name):
        """
        Vérifie qu'une valeur n'est pas None.
        
        Args:
            value: La valeur à vérifier.
            test_name: Le nom du test.
        """
        try:
            assert value is not None
            self.tests_passed += 1
            self.test_results.append({
                "name": test_name,
                "status": "PASSED",
                "message": "Value is not None"
            })
            logger.info(f"✅ Test '{test_name}' PASSED")
        except AssertionError:
            self.tests_failed += 1
            self.test_results.append({
                "name": test_name,
                "status": "FAILED",
                "message": "Value is None"
            })
            logger.error(f"❌ Test '{test_name}' FAILED: Value is None")
    
    def assert_in_range(self, value, min_val, max_val, test_name):
        """
        Vérifie qu'une valeur est dans une plage.
        
        Args:
            value: La valeur à vérifier.
            min_val: La valeur minimale de la plage.
            max_val: La valeur maximale de la plage.
            test_name: Le nom du test.
        """
        try:
            assert min_val <= value <= max_val
            self.tests_passed += 1
            self.test_results.append({
                "name": test_name,
                "status": "PASSED",
                "message": f"Value {value} is in range [{min_val}, {max_val}]"
            })
            logger.info(f"✅ Test '{test_name}' PASSED")
        except AssertionError:
            self.tests_failed += 1
            self.test_results.append({
                "name": test_name,
                "status": "FAILED",
                "message": f"Value {value} is not in range [{min_val}, {max_val}]"
            })
            logger.error(f"❌ Test '{test_name}' FAILED: Value {value} is not in range [{min_val}, {max_val}]")
    
    def print_summary(self):
        """Affiche un résumé des tests."""
        total_tests = self.tests_passed + self.tests_failed
        logger.info(f"\n===== TEST SUMMARY =====")
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed: {self.tests_passed}")
        logger.info(f"Failed: {self.tests_failed}")
        logger.info(f"Success rate: {self.tests_passed / total_tests * 100:.2f}%")
        logger.info(f"========================\n")
        
        # Afficher les tests qui ont échoué
        if self.tests_failed > 0:
            logger.error("Failed tests:")
            for result in self.test_results:
                if result["status"] == "FAILED":
                    logger.error(f"- {result['name']}: {result['message']}")


class TestSignalParser:
    """
    Classe pour tester le module SignalParser.
    """
    
    def __init__(self, test_runner: TestRunner):
        """
        Initialise les tests du parser de signaux.
        
        Args:
            test_runner: Le runner de tests.
        """
        self.test_runner = test_runner
        self.parser = SignalParser()
    
    def run_tests(self):
        """Exécute tous les tests du parser de signaux."""
        logger.info("Running SignalParser tests...")
        
        self.test_standard_format()
        self.test_alert_format()
        self.test_simple_format()
        self.test_partial_parsing()
        self.test_format_detection()
        self.test_validation()
        
        logger.info("SignalParser tests completed.")
    
    def test_standard_format(self):
        """Teste le parsing du format standard."""
        # Signal au format standard
        signal_text = """
        🟢 BTCUSDT LONG

        🎯Entry price: 60000 - 61000

        TP:
        62000
        63000
        64000

        🛑 SL 59000
        """
        
        signal, errors = self.parser.parse_signal(signal_text)
        
        self.test_runner.assert_not_none(signal, "Standard format parsing should return a signal")
        
        if signal:
            self.test_runner.assert_equal(signal.symbol, "BTC", "Standard format should extract correct symbol")
            self.test_runner.assert_equal(signal.direction, "LONG", "Standard format should extract correct direction")
            self.test_runner.assert_equal(signal.entry_min, 60000, "Standard format should extract correct entry_min")
            self.test_runner.assert_equal(signal.entry_max, 61000, "Standard format should extract correct entry_max")
            self.test_runner.assert_equal(signal.stop_loss, 59000, "Standard format should extract correct stop_loss")
            self.test_runner.assert_equal(len(signal.take_profit_levels), 3, "Standard format should extract correct number of TP levels")
            self.test_runner.assert_equal(signal.take_profit_levels[0], 62000, "Standard format should extract correct TP1")
            self.test_runner.assert_equal(signal.format, SignalFormat.STANDARD, "Signal format should be STANDARD")
    
    def test_alert_format(self):
        """Teste le parsing du format alert."""
        # Signal au format alert
        signal_text = """
        SIGNAL ALERT
        COIN: ETHUSDT
        DIRECTION: SHORT
        ENTRY ZONE: 3000 - 3100
        TARGETS: 2900, 2800, 2700
        STOP LOSS: 3200
        """
        
        signal, errors = self.parser.parse_signal(signal_text)
        
        self.test_runner.assert_not_none(signal, "Alert format parsing should return a signal")
        
        if signal:
            self.test_runner.assert_equal(signal.symbol, "ETH", "Alert format should extract correct symbol")
            self.test_runner.assert_equal(signal.direction, "SHORT", "Alert format should extract correct direction")
            self.test_runner.assert_equal(signal.entry_min, 3000, "Alert format should extract correct entry_min")
            self.test_runner.assert_equal(signal.entry_max, 3100, "Alert format should extract correct entry_max")
            self.test_runner.assert_equal(signal.stop_loss, 3200, "Alert format should extract correct stop_loss")
            self.test_runner.assert_equal(len(signal.take_profit_levels), 3, "Alert format should extract correct number of TP levels")
            self.test_runner.assert_equal(signal.format, SignalFormat.ALERT, "Signal format should be ALERT")
    
    def test_simple_format(self):
        """Teste le parsing du format simple."""
        # Signal au format simple
        signal_text = """
        ADAUSDT LONG
        Entry around 1.2
        SL 1.1
        TP 1.3, 1.4, 1.5
        """
        
        signal, errors = self.parser.parse_signal(signal_text)
        
        self.test_runner.assert_not_none(signal, "Simple format parsing should return a signal")
        
        if signal:
            self.test_runner.assert_equal(signal.symbol, "ADA", "Simple format should extract correct symbol")
            self.test_runner.assert_equal(signal.direction, "LONG", "Simple format should extract correct direction")
            self.test_runner.assert_equal(signal.entry_min, 1.2 * 0.99, "Simple format should calculate correct entry_min")
            self.test_runner.assert_equal(signal.entry_max, 1.2 * 1.01, "Simple format should calculate correct entry_max")
            self.test_runner.assert_equal(signal.stop_loss, 1.1, "Simple format should extract correct stop_loss")
            self.test_runner.assert_equal(len(signal.take_profit_levels), 3, "Simple format should extract correct number of TP levels")
            self.test_runner.assert_equal(signal.format, SignalFormat.SIMPLE, "Signal format should be SIMPLE")
    
    def test_partial_parsing(self):
        """Teste le parsing partiel avec des informations manquantes."""
        # Signal incomplet
        signal_text = """
        SOLUSDT LONG
        Entry around 100
        """
        
        signal, errors = self.parser.parse_signal(signal_text)
        
        self.test_runner.assert_not_none(signal, "Partial parsing should return a signal")
        self.test_runner.assert_true(len(errors) > 0, "Partial parsing should return errors")
        
        if signal:
            self.test_runner.assert_equal(signal.symbol, "SOL", "Partial parsing should extract correct symbol")
            self.test_runner.assert_equal(signal.direction, "LONG", "Partial parsing should extract correct direction")
            self.test_runner.assert_equal(signal.entry_min, 100 * 0.99, "Partial parsing should calculate correct entry_min")
            self.test_runner.assert_equal(signal.entry_max, 100 * 1.01, "Partial parsing should calculate correct entry_max")
            # Les valeurs par défaut doivent être utilisées pour SL et TP
            self.test_runner.assert_true(signal.stop_loss < signal.entry_min, "Partial parsing should set default SL")
            self.test_runner.assert_true(len(signal.take_profit_levels) > 0, "Partial parsing should set default TP levels")
    
    def test_format_detection(self):
        """Teste la détection automatique du format."""
        # Signal au format standard
        standard_text = "🟢 BTCUSDT LONG\n🎯Entry price: 60000 - 61000\nTP:\n62000\n63000\n🛑 SL 59000"
        
        # Signal au format alert
        alert_text = "SIGNAL ALERT\nCOIN: ETHUSDT\nDIRECTION: SHORT\nENTRY ZONE: 3000 - 3100"
        
        # Signal au format simple
        simple_text = "ADAUSDT LONG\nEntry around 1.2\nSL 1.1\nTP 1.3, 1.4"
        
        # Tester la détection de format
        standard_format = self.parser.detect_format(standard_text)
        alert_format = self.parser.detect_format(alert_text)
        simple_format = self.parser.detect_format(simple_text)
        
        self.test_runner.assert_equal(standard_format, SignalFormat.STANDARD, "Should detect STANDARD format")
        self.test_runner.assert_equal(alert_format, SignalFormat.ALERT, "Should detect ALERT format")
        self.test_runner.assert_equal(simple_format, SignalFormat.SIMPLE, "Should detect SIMPLE format")
    
    def test_validation(self):
        """Teste la validation des signaux."""
        # Signal valide
        valid_signal = TradingSignal(
            symbol="BTC",
            direction="LONG",
            entry_min=60000,
            entry_max=61000,
            take_profit_levels=[62000, 63000, 64000],
            stop_loss=59000,
            raw_text="Test signal",
            format=SignalFormat.STANDARD
        )
        
        # Signal invalide (stop loss au-dessus de l'entrée pour un LONG)
        invalid_signal = TradingSignal(
            symbol="BTC",
            direction="LONG",
            entry_min=60000,
            entry_max=61000,
            take_profit_levels=[62000, 63000, 64000],
            stop_loss=62000,  # SL au-dessus de l'entrée pour un LONG
            raw_text="Test signal",
            format=SignalFormat.STANDARD
        )
        
        valid_result, valid_errors = self.parser.validate_signal(valid_signal)
        invalid_result, invalid_errors = self.parser.validate_signal(invalid_signal)
        
        self.test_runner.assert_true(valid_result, "Valid signal should pass validation")
        self.test_runner.assert_true(len(valid_errors) == 0, "Valid signal should have no validation errors")
        
        self.test_runner.assert_true(not invalid_result, "Invalid signal should fail validation")
        self.test_runner.assert_true(len(invalid_errors) > 0, "Invalid signal should have validation errors")


class TestTelegramClient:
    """
    Classe pour tester le module TelegramClient.
    """
    
    def __init__(self, test_runner: TestRunner):
        """
        Initialise les tests du client Telegram.
        
        Args:
            test_runner: Le runner de tests.
        """
        self.test_runner = test_runner
        self.signal_queue = Queue()
    
    def run_tests(self):
        """Exécute tous les tests du client Telegram."""
        logger.info("Running TelegramClient tests...")
        
        self.test_signal_message()
        self.test_signal_processor()
        
        logger.info("TelegramClient tests completed.")
    
    def test_signal_message(self):
        """Teste la classe SignalMessage."""
        # Créer un message de signal
        signal_message = SignalMessage(
            text="Test signal",
            chat_id=123456789,
            message_id=1,
            timestamp=time.time(),
            user_id=987654321,
            username="test_user"
        )
        
        # Tester la conversion en dictionnaire
        message_dict = signal_message.to_dict()
        
        self.test_runner.assert_equal(message_dict["text"], "Test signal", "SignalMessage to_dict should preserve text")
        self.test_runner.assert_equal(message_dict["chat_id"], 123456789, "SignalMessage to_dict should preserve chat_id")
        
        # Tester la création à partir d'un dictionnaire
        recreated_message = SignalMessage.from_dict(message_dict)
        
        self.test_runner.assert_equal(recreated_message.text, "Test signal", "SignalMessage from_dict should restore text")
        self.test_runner.assert_equal(recreated_message.chat_id, 123456789, "SignalMessage from_dict should restore chat_id")
        self.test_runner.assert_equal(recreated_message.user_id, 987654321, "SignalMessage from_dict should restore user_id")
    
    def test_signal_processor(self):
        """Teste la classe SignalProcessor."""
        # Fonction de callback de test
        def test_parser(message_text):
            if "LONG" in message_text:
                return TradingSignal(
                    symbol="BTC",
                    direction="LONG",
                    entry_min=60000,
                    entry_max=61000,
                    take_profit_levels=[62000, 63000],
                    stop_loss=59000,
                    raw_text=message_text
                ), []
            else:
                return None, ["Invalid signal format"]
        
        # Créer un processeur de signaux
        signal_processor = SignalProcessor(
            self.signal_queue,
            test_parser,
            max_retries=2,
            retry_delay=0.1
        )
        
        # Ajouter un signal valide à la file d'attente
        valid_message = SignalMessage(
            text="BTCUSDT LONG",
            chat_id=123456789,
            message_id=1,
            timestamp=time.time()
        )
        self.signal_queue.put(valid_message)
        
        # Ajouter un signal invalide à la file d'attente
        invalid_message = SignalMessage(
            text="Invalid signal",
            chat_id=123456789,
            message_id=2,
            timestamp=time.time()
        )
        self.signal_queue.put(invalid_message)
        
        # Démarrer le processeur
        signal_processor.start()
        
        # Attendre que les signaux soient traités
        time.sleep(1)
        
        # Arrêter le processeur
        signal_processor.stop()
        
        # Vérifier les statistiques
        stats = signal_processor.get_stats()
        
        self.test_runner.assert_equal(stats["processed"], 2, "SignalProcessor should process all signals")
        self.test_runner.assert_equal(stats["success"], 1, "SignalProcessor should successfully parse valid signals")
        self.test_runner.assert_equal(stats["failed"], 1, "SignalProcessor should fail to parse invalid signals")


class TestBinanceClient:
    """
    Classe pour tester le module BinanceClient.
    
    Note: Ces tests sont principalement des tests unitaires et ne font pas d'appels réels à l'API Binance.
    """
    
    def __init__(self, test_runner: TestRunner):
        """
        Initialise les tests du client Binance.
        
        Args:
            test_runner: Le runner de tests.
        """
        self.test_runner = test_runner
        
        # Créer un client Binance avec des clés factices
        self.binance_client = BinanceClient(
            api_key="fake_api_key",
            api_secret="fake_api_secret",
            testnet=True
        )
    
    def run_tests(self):
        """Exécute tous les tests du client Binance."""
        logger.info("Running BinanceClient tests...")
        
        self.test_market_data()
        self.test_order_executor()
        
        logger.info("BinanceClient tests completed.")
    
    def test_market_data(self):
        """Teste la classe MarketData."""
        # Créer un objet MarketData
        market_data = MarketData(
            symbol="BTCUSDT",
            price=60000.0,
            timestamp=time.time(),
            volume_24h=1000.0,
            price_change_24h=5.0,
            high_24h=61000.0,
            low_24h=59000.0,
            bid=59990.0,
            ask=60010.0,
            trend="uptrend"
        )
        
        self.test_runner.assert_equal(market_data.symbol, "BTCUSDT", "MarketData should store symbol")
        self.test_runner.assert_equal(market_data.price, 60000.0, "MarketData should store price")
        self.test_runner.assert_equal(market_data.trend, "uptrend", "MarketData should store trend")
    
    def test_order_executor(self):
        """Teste la classe OrderExecutor."""
        # Créer un exécuteur d'ordres
        order_executor = OrderExecutor(self.binance_client)
        
        # Tester les méthodes de formatage
        rounded_quantity = order_executor._round_quantity("BTCUSDT", 1.23456789)
        rounded_price = order_executor._round_price("BTCUSDT", 60000.123456789)
        
        self.test_runner.assert_true(isinstance(rounded_quantity, float), "Rounded quantity should be a float")
        self.test_runner.assert_true(isinstance(rounded_price, float), "Rounded price should be a float")


class TestTradeExecutor:
    """
    Classe pour tester le module TradeExecutor.
    
    Note: Ces tests sont principalement des tests unitaires et ne font pas d'appels réels à l'API Binance.
    """
    
    def __init__(self, test_runner: TestRunner):
        """
        Initialise les tests de l'exécuteur de trades.
        
        Args:
            test_runner: Le runner de tests.
        """
        self.test_runner = test_runner
        
        # Créer un client Binance avec des clés factices
        self.binance_client = BinanceClient(
            api_key="fake_api_key",
            api_secret="fake_api_secret",
            testnet=True
        )
        
        # Créer un gestionnaire de risque
        self.risk_manager = RiskManager(
            self.binance_client,
            risk_per_trade=1.0,
            max_total_risk=5.0
        )
    
    def run_tests(self):
        """Exécute tous les tests de l'exécuteur de trades."""
        logger.info("Running TradeExecutor tests...")
        
        self.test_technical_analyzer()
        self.test_trade_result()
        
        logger.info("TradeExecutor tests completed.")
    
    def test_technical_analyzer(self):
        """Teste la classe TechnicalAnalyzer."""
        # Créer un analyseur technique
        technical_analyzer = TechnicalAnalyzer(self.binance_client)
        
        # Tester la méthode de calcul de l'ATR
        klines = [
            [0, 0, "100", "90", "95", 0, 0],  # [open_time, open, high, low, close, volume, ...]
            [0, 0, "110", "95", "105", 0, 0],
            [0, 0, "115", "100", "110", 0, 0],
            [0, 0, "120", "105", "115", 0, 0],
            [0, 0, "125", "110", "120", 0, 0]
        ]
        
        atr = technical_analyzer._calculate_atr(klines, period=3)
        
        self.test_runner.assert_true(len(atr) > 0, "ATR calculation should return non-empty list")
        self.test_runner.assert_true(atr[0] > 0, "ATR values should be positive")
    
    def test_trade_result(self):
        """Teste la classe TradeResult."""
        # Créer un résultat de trade
        trade_result = TradeResult(
            signal_id="test_signal_1",
            symbol="BTCUSDT",
            direction="LONG",
            status="success",
            message="Trade executed successfully",
            position_size=0.1,
            leverage=10,
            risk_percentage=1.0,
            entry_price=60000.0,
            stop_loss=59000.0,
            take_profit_levels=[61000.0, 62000.0, 63000.0],
            orders={"entry": {}, "sl": {}, "tp": []}
        )
        
        # Tester la conversion en dictionnaire
        result_dict = trade_result.to_dict()
        
        self.test_runner.assert_equal(result_dict["signal_id"], "test_signal_1", "TradeResult to_dict should preserve signal_id")
        self.test_runner.assert_equal(result_dict["symbol"], "BTCUSDT", "TradeResult to_dict should preserve symbol")
        self.test_runner.assert_equal(result_dict["status"], "success", "TradeResult to_dict should preserve status")
        self.test_runner.assert_equal(result_dict["position_size"], 0.1, "TradeResult to_dict should preserve position_size")


class TestRiskManager:
    """
    Classe pour tester le module RiskManager.
    
    Note: Ces tests sont principalement des tests unitaires et ne font pas d'appels réels à l'API Binance.
    """
    
    def __init__(self, test_runner: TestRunner):
        """
        Initialise les tests du gestionnaire de risque.
        
        Args:
            test_runner: Le runner de tests.
        """
        self.test_runner = test_runner
        
        # Créer un client Binance avec des clés factices
        self.binance_client = BinanceClient(
            api_key="fake_api_key",
            api_secret="fake_api_secret",
            testnet=True
        )
        
        # Créer un gestionnaire de risque
        self.risk_manager = RiskManager(
            self.binance_client,
            risk_per_trade=1.0,
            max_total_risk=5.0,
            max_positions=10
        )
    
    def run_tests(self):
        """Exécute tous les tests du gestionnaire de risque."""
        logger.info("Running RiskManager tests...")
        
        self.test_position()
        self.test_tp_distribution()
        self.test_correlated_symbols()
        
        logger.info("RiskManager tests completed.")
    
    def test_position(self):
        """Teste la classe Position."""
        # Créer une position
        position = Position(
            signal_id="test_signal_1",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=60000.0,
            position_size=0.1,
            leverage=10,
            risk_percentage=1.0,
            stop_loss=59000.0,
            take_profit_levels=[61000.0, 62000.0, 63000.0],
            orders={"entry": {}, "sl": {}, "tp": []}
        )
        
        # Tester la conversion en dictionnaire
        position_dict = position.to_dict()
        
        self.test_runner.assert_equal(position_dict["signal_id"], "test_signal_1", "Position to_dict should preserve signal_id")
        self.test_runner.assert_equal(position_dict["symbol"], "BTCUSDT", "Position to_dict should preserve symbol")
        self.test_runner.assert_equal(position_dict["direction"], "LONG", "Position to_dict should preserve direction")
        self.test_runner.assert_equal(position_dict["position_size"], 0.1, "Position to_dict should preserve position_size")
    
    def test_tp_distribution(self):
        """Teste la méthode de calcul de la distribution des TP."""
        # Créer un signal
        signal = TradingSignal(
            symbol="BTC",
            direction="LONG",
            entry_min=60000,
            entry_max=61000,
            take_profit_levels=[62000, 63000, 64000],
            stop_loss=59000,
            raw_text="Test signal"
        )
        
        # Calculer la distribution des TP
        tp_distribution = asyncio.run(self.risk_manager.calculate_tp_distribution(signal, 1.0))
        
        self.test_runner.assert_equal(len(tp_distribution), 3, "TP distribution should have 3 levels")
        self.test_runner.assert_true(sum(tp_distribution) <= 1.0, "Sum of TP quantities should not exceed total position size")
        self.test_runner.assert_true(tp_distribution[0] > tp_distribution[1], "First TP should have larger quantity than second TP")
    
    def test_correlated_symbols(self):
        """Teste la méthode de récupération des symboles corrélés."""
        # Tester avec différents symboles
        btc_correlated = asyncio.run(self.risk_manager.get_correlated_symbols("BTC"))
        xrp_correlated = asyncio.run(self.risk_manager.get_correlated_symbols("XRP"))
        doge_correlated = asyncio.run(self.risk_manager.get_correlated_symbols("DOGE"))
        
        self.test_runner.assert_true("ETH" in btc_correlated, "BTC should be correlated with ETH")
        self.test_runner.assert_true("ADA" in xrp_correlated, "XRP should be correlated with ADA")
        self.test_runner.assert_true("SHIB" in doge_correlated, "DOGE should be correlated with SHIB")


async def main():
    """Fonction principale pour exécuter tous les tests."""
    logger.info("Starting tests for Telegram bot components...")
    
    # Créer un runner de tests
    test_runner = TestRunner()
    
    # Exécuter les tests pour chaque module
    TestSignalParser(test_runner).run_tests()
    TestTelegramClient(test_runner).run_tests()
    TestBinanceClient(test_runner).run_tests()
    TestTradeExecutor(test_runner).run_tests()
    TestRiskManager(test_runner).run_tests()
    
    # Afficher le résumé des tests
    test_runner.print_summary()
    
    logger.info("All tests completed.")


if __name__ == "__main__":
    # Exécuter la fonction principale de manière asynchrone
    asyncio.run(main())
