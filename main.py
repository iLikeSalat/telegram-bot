"""
Module principal du bot Telegram pour le trading sur Binance Futures.

Ce module intègre tous les composants du bot et gère son exécution.
"""

import asyncio
import json
import logging
import os
import sys
from queue import Queue

from src.signal_parser import SignalParser
from src.telegram_client import TelegramClient, SignalProcessor
from src.binance_client import BinanceClient
from src.trade_executor import TradeExecutor
from src.risk_manager import RiskManager

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


async def main():
    """Fonction principale pour exécuter le bot Telegram."""
    try:
        # Charger la configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Set logging level based on debug mode
            # Set logging level based on debug mode
            if config.get("debug", False):
                logger.setLevel(logging.DEBUG)
                console_handler.setLevel(logging.DEBUG)
                logger.debug("Debug mode enabled")
        
        logger.info("Configuration chargée")
        
        # Initialiser les composants
        signal_queue = Queue()
        
        # Initialiser le client Binance
        binance_client = BinanceClient(
            api_key=config['binance']['api_key'],
            api_secret=config['binance']['api_secret'],
            testnet=config['binance']['testnet']
        )
        
        logger.info("Client Binance initialisé")
        
        # Initialiser le gestionnaire de risque
        risk_manager = RiskManager(
            binance_client=binance_client,
            risk_per_trade=config['trading']['risk_per_trade'],
            max_total_risk=config['trading']['max_total_risk'],
            max_positions=config['trading']['max_positions']
        )
        
        logger.info("Gestionnaire de risque initialisé")
        
        # Initialiser l'exécuteur de trades
        trade_executor = TradeExecutor(
            binance_client=binance_client,
            risk_manager=risk_manager,
            validate_trend=config['trading']['validate_trend'],
            use_volatility_sl=config['trading']['use_volatility_sl']
        )
        
        logger.info("Exécuteur de trades initialisé")
        
        # Initialiser le client Telegram
        telegram_client = TelegramClient(
            token=config['telegram']['token'],
            signal_queue=signal_queue,
            allowed_chats=config['telegram']['allowed_chats'],
            allowed_users=config['telegram']['allowed_users'],
            admin_users=config['telegram']['admin_users']
        )
        
        logger.info("Client Telegram initialisé")
        
        # Initialiser le parser de signaux
        signal_parser = SignalParser()
        
        logger.info("Parser de signaux initialisé")
        
        # Définir la fonction de callback pour le parsing des signaux
        def parse_signal_callback(message_text):
            return signal_parser.parse_signal(message_text)
        
        # Initialiser le processeur de signaux
        signal_processor = SignalProcessor(
            signal_queue=signal_queue,
            parser_callback=parse_signal_callback,
            telegram_client=telegram_client
        )
        
        logger.info("Processeur de signaux initialisé")
        
        # Démarrer le processeur de signaux
        signal_processor.start()
        
        logger.info("Processeur de signaux démarré")
        
        # Fonction de callback pour traiter les signaux parsés
        async def process_signal(signal):
            try:
                # Exécuter le signal
                result = await trade_executor.execute_signal(signal)
                
                # Envoyer le résultat au client Telegram
                chat_id = signal.chat_id if hasattr(signal, 'chat_id') else None
                if chat_id and telegram_client:
                    await telegram_client.send_signal_result(chat_id, result.signal_id, result.to_dict())
                
                return result
            except Exception as e:
                logger.error(f"Erreur lors du traitement du signal: {str(e)}")
                return None
        
        # Démarrer le client Telegram
        logger.info("Démarrage du client Telegram...")
        telegram_task = telegram_client.start_async()
        
        # Attendre que le client Telegram soit prêt
        await asyncio.sleep(2)
        
        logger.info("Bot démarré et en attente de signaux")
        
        # Attendre que le client Telegram s'arrête
        await telegram_task
        
    except Exception as e:
        logger.error(f"Erreur dans la fonction principale: {str(e)}")
        raise


if __name__ == "__main__":
    # Exécuter la fonction principale
    asyncio.run(main())
