"""
Module de connexion à Telegram pour recevoir les signaux de trading.

Ce module contient les classes et fonctions nécessaires pour se connecter
à Telegram et recevoir les messages contenant des signaux de trading.
Version améliorée avec meilleure gestion des erreurs et support de formats multiples.
"""

import asyncio
import logging
import time
import json
from typing import List, Callable, Optional, Dict, Any, Tuple
from queue import Queue
from dataclasses import dataclass, asdict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, MessageHandler, CommandHandler, CallbackQueryHandler, ContextTypes, filters

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
class SignalMessage:
    """Représente un message de signal reçu de Telegram."""
    
    text: str
    chat_id: int
    message_id: int
    timestamp: float
    user_id: Optional[int] = None
    username: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'objet en dictionnaire."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalMessage':
        """Crée un objet à partir d'un dictionnaire."""
        return cls(**data)


class TelegramClient:
    """
    Classe pour gérer la connexion à Telegram et recevoir les messages.
    
    Cette classe utilise python-telegram-bot pour se connecter à l'API Telegram
    et recevoir les messages contenant des signaux de trading.
    """
    
    def __init__(self, token: str, signal_queue: Queue, allowed_chats: Optional[List[int]] = None,
                allowed_users: Optional[List[int]] = None, admin_users: Optional[List[int]] = None):
        """
        Initialise le client Telegram.
        
        Args:
            token: Le token d'API du bot Telegram.
            signal_queue: Une file d'attente pour stocker les signaux reçus.
            allowed_chats: Liste des IDs de chat autorisés à envoyer des signaux.
                           Si None, tous les chats sont autorisés.
            allowed_users: Liste des IDs d'utilisateurs autorisés à envoyer des signaux.
                           Si None, tous les utilisateurs sont autorisés.
            admin_users: Liste des IDs d'utilisateurs administrateurs.
                         Si None, aucun utilisateur n'est administrateur.
        """
        self.token = token
        self.signal_queue = signal_queue
        self.allowed_chats = allowed_chats or []
        self.allowed_users = allowed_users or []
        self.admin_users = admin_users or []
        self.application = Application.builder().token(token).build()
        self.setup_handlers()
        self.bot_username = None
        self.bot_info = None
        
    async def initialize(self):
        """Initialise le bot et récupère ses informations."""
        try:
            self.bot_info = await self.application.bot.get_me()
            self.bot_username = self.bot_info.username
            logger.info(f"Bot initialisé: @{self.bot_username}")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du bot: {str(e)}")
            raise
        
    def setup_handlers(self):
        """Configure les gestionnaires de messages et de commandes."""
        # Gestionnaire pour les commandes
        self.application.add_handler(CommandHandler("start", self.handle_start))
        self.application.add_handler(CommandHandler("help", self.handle_help))
        self.application.add_handler(CommandHandler("status", self.handle_status))
        self.application.add_handler(CommandHandler("settings", self.handle_settings))
        self.application.add_handler(CommandHandler("admin", self.handle_admin, filters=filters.User(user_id=self.admin_users) if self.admin_users else None))
        
        # Gestionnaire pour les messages texte (potentiels signaux)
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Gestionnaire pour les boutons inline
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Gestionnaire d'erreurs
        self.application.add_error_handler(self.handle_error)
    
    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Gère la commande /start."""
        user = update.effective_user
        
        # Vérifier si l'utilisateur est autorisé
        if self.allowed_users and user.id not in self.allowed_users:
            await update.message.reply_text(
                "Désolé, vous n'êtes pas autorisé à utiliser ce bot. "
                "Contactez l'administrateur pour obtenir l'accès."
            )
            logger.warning(f"Tentative d'accès non autorisée: {user.id} (@{user.username})")
            return
        
        # Message de bienvenue
        welcome_text = (
            f"Bonjour {user.first_name}! Je suis votre bot de trading Binance Futures. "
            f"Envoyez-moi des signaux de trading et je les exécuterai automatiquement.\n\n"
            f"Utilisez /help pour voir les commandes disponibles et le format des signaux attendu."
        )
        
        # Créer un clavier inline pour les actions rapides
        keyboard = [
            [
                InlineKeyboardButton("Aide", callback_data="help"),
                InlineKeyboardButton("Statut", callback_data="status")
            ],
            [
                InlineKeyboardButton("Paramètres", callback_data="settings")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_text, reply_markup=reply_markup)
        logger.info(f"Nouvel utilisateur: {user.id} (@{user.username})")
    
    async def handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Gère la commande /help."""
        help_text = (
            "Je peux exécuter automatiquement des trades sur Binance Futures "
            "basés sur les signaux que vous m'envoyez.\n\n"
            "📝 Formats de signal acceptés:\n\n"
            "1️⃣ Format standard:\n"
            "🟢 SYMBOLE DIRECTION\n\n"
            "🎯Entry price: MIN - MAX\n\n"
            "TP:\n"
            "NIVEAU1\n"
            "NIVEAU2\n"
            "...\n\n"
            "🛑 SL NIVEAU\n\n"
            "2️⃣ Format alert:\n"
            "SIGNAL ALERT\n"
            "COIN: SYMBOLE\n"
            "DIRECTION: DIRECTION\n"
            "ENTRY ZONE: MIN - MAX\n"
            "TARGETS: NIVEAU1, NIVEAU2, ...\n"
            "STOP LOSS: NIVEAU\n\n"
            "3️⃣ Format simple:\n"
            "SYMBOLE DIRECTION\n"
            "Entry around PRIX\n"
            "SL NIVEAU\n"
            "TP NIVEAU1, NIVEAU2, ...\n\n"
            "📋 Commandes disponibles:\n"
            "/start - Démarrer le bot\n"
            "/help - Afficher cette aide\n"
            "/status - Vérifier le statut du bot\n"
            "/settings - Configurer les paramètres du bot\n"
        )
        
        # Ajouter les commandes admin si l'utilisateur est admin
        user_id = update.effective_user.id
        if user_id in self.admin_users:
            help_text += (
                "\n🔐 Commandes administrateur:\n"
                "/admin - Accéder au panneau d'administration\n"
            )
        
        await update.message.reply_text(help_text)
    
    async def handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Gère la commande /status."""
        # Récupérer les informations de statut
        queue_size = self.signal_queue.qsize()
        
        status_text = (
            "📊 Statut du bot:\n\n"
            f"🤖 Bot: {'En ligne ✅' if self.bot_info else 'Hors ligne ❌'}\n"
            f"📨 Signaux en attente: {queue_size}\n"
            f"🔒 Chats autorisés: {len(self.allowed_chats)}\n"
            f"👥 Utilisateurs autorisés: {len(self.allowed_users)}\n"
        )
        
        await update.message.reply_text(status_text)
    
    async def handle_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Gère la commande /settings."""
        # Créer un clavier inline pour les paramètres
        keyboard = [
            [
                InlineKeyboardButton("Format de signal", callback_data="settings_format"),
                InlineKeyboardButton("Notifications", callback_data="settings_notifications")
            ],
            [
                InlineKeyboardButton("Risque", callback_data="settings_risk"),
                InlineKeyboardButton("Compte Binance", callback_data="settings_binance")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "⚙️ Paramètres du bot\n\n"
            "Sélectionnez une catégorie de paramètres à configurer:",
            reply_markup=reply_markup
        )
    
    async def handle_admin(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Gère la commande /admin."""
        # Vérifier si l'utilisateur est admin (double vérification)
        user_id = update.effective_user.id
        if user_id not in self.admin_users:
            await update.message.reply_text("Vous n'avez pas les droits d'administration.")
            logger.warning(f"Tentative d'accès admin non autorisée: {user_id}")
            return
        
        # Créer un clavier inline pour les actions admin
        keyboard = [
            [
                InlineKeyboardButton("Gérer les utilisateurs", callback_data="admin_users"),
                InlineKeyboardButton("Gérer les chats", callback_data="admin_chats")
            ],
            [
                InlineKeyboardButton("Logs", callback_data="admin_logs"),
                InlineKeyboardButton("Redémarrer", callback_data="admin_restart")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🔐 Panneau d'administration\n\n"
            "Sélectionnez une action:",
            reply_markup=reply_markup
        )
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Gère les callbacks des boutons inline."""
        query = update.callback_query
        await query.answer()  # Répondre au callback pour éviter le spinner
        
        callback_data = query.data
        
        if callback_data == "help":
            # Simuler la commande /help
            await self.handle_help(update, context)
        
        elif callback_data == "status":
            # Simuler la commande /status
            await self.handle_status(update, context)
        
        elif callback_data == "settings":
            # Simuler la commande /settings
            await self.handle_settings(update, context)
        
        elif callback_data.startswith("settings_"):
            # Gérer les sous-menus des paramètres
            setting_type = callback_data.split("_")[1]
            await self._handle_settings_submenu(update, context, setting_type)
        
        elif callback_data.startswith("admin_"):
            # Gérer les sous-menus d'administration
            admin_action = callback_data.split("_")[1]
            await self._handle_admin_submenu(update, context, admin_action)
    
    async def _handle_settings_submenu(self, update: Update, context: ContextTypes.DEFAULT_TYPE, setting_type: str) -> None:
        """Gère les sous-menus des paramètres."""
        query = update.callback_query
        
        if setting_type == "format":
            await query.edit_message_text(
                "📝 Format de signal\n\n"
                "Le bot supporte plusieurs formats de signaux de trading:\n"
                "- Format standard (avec emoji et structure claire)\n"
                "- Format alert (avec 'SIGNAL ALERT' et structure par lignes)\n"
                "- Format simple (avec symbole, direction et quelques prix)\n\n"
                "Le format est détecté automatiquement."
            )
        
        elif setting_type == "notifications":
            # Créer un clavier inline pour les options de notification
            keyboard = [
                [
                    InlineKeyboardButton("Toutes ✅", callback_data="notif_all"),
                    InlineKeyboardButton("Importantes ⚠️", callback_data="notif_important")
                ],
                [
                    InlineKeyboardButton("Aucune ❌", callback_data="notif_none"),
                    InlineKeyboardButton("Retour", callback_data="settings")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                "🔔 Paramètres de notification\n\n"
                "Choisissez quand vous souhaitez recevoir des notifications:",
                reply_markup=reply_markup
            )
        
        elif setting_type == "risk":
            # Créer un clavier inline pour les options de risque
            keyboard = [
                [
                    InlineKeyboardButton("Risque par trade", callback_data="risk_per_trade"),
                    InlineKeyboardButton("Risque total", callback_data="risk_total")
                ],
                [
                    InlineKeyboardButton("Retour", callback_data="settings")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                "⚠️ Paramètres de risque\n\n"
                "Configurez les paramètres de risque pour vos trades:",
                reply_markup=reply_markup
            )
        
        elif setting_type == "binance":
            await query.edit_message_text(
                "🔑 Compte Binance\n\n"
                "Pour configurer votre compte Binance, vous devez définir vos clés API.\n"
                "Envoyez un message au format suivant:\n\n"
                "/set_api_key VOTRE_CLE_API VOTRE_SECRET_API\n\n"
                "⚠️ Ne partagez jamais vos clés API en public!\n"
                "Utilisez uniquement des clés avec des permissions limitées (lecture et trading)."
            )
    
    async def _handle_admin_submenu(self, update: Update, context: ContextTypes.DEFAULT_TYPE, admin_action: str) -> None:
        """Gère les sous-menus d'administration."""
        query = update.callback_query
        
        # Vérifier si l'utilisateur est admin
        user_id = update.effective_user.id
        if user_id not in self.admin_users:
            await query.edit_message_text("Vous n'avez pas les droits d'administration.")
            logger.warning(f"Tentative d'accès admin non autorisée: {user_id}")
            return
        
        if admin_action == "users":
            # Liste des utilisateurs autorisés
            users_text = "👥 Gestion des utilisateurs\n\n"
            
            if not self.allowed_users:
                users_text += "Aucun utilisateur spécifique autorisé (tous les utilisateurs peuvent utiliser le bot).\n\n"
            else:
                users_text += "Utilisateurs autorisés:\n"
                for user_id in self.allowed_users:
                    users_text += f"- {user_id}\n"
                users_text += "\n"
            
            users_text += "Pour ajouter un utilisateur: /add_user ID\n"
            users_text += "Pour supprimer un utilisateur: /remove_user ID"
            
            await query.edit_message_text(users_text)
        
        elif admin_action == "chats":
            # Liste des chats autorisés
            chats_text = "💬 Gestion des chats\n\n"
            
            if not self.allowed_chats:
                chats_text += "Aucun chat spécifique autorisé (tous les chats sont acceptés).\n\n"
            else:
                chats_text += "Chats autorisés:\n"
                for chat_id in self.allowed_chats:
                    chats_text += f"- {chat_id}\n"
                chats_text += "\n"
            
            chats_text += "Pour ajouter un chat: /add_chat ID\n"
            chats_text += "Pour supprimer un chat: /remove_chat ID"
            
            await query.edit_message_text(chats_text)
        
        elif admin_action == "logs":
            # Envoyer les dernières lignes du log
            try:
                with open('telegram_binance_bot.log', 'r') as f:
                    logs = f.readlines()
                
                # Prendre les 20 dernières lignes
                last_logs = logs[-20:]
                logs_text = "📋 Derniers logs:\n\n" + "".join(last_logs)
                
                # Tronquer si trop long
                if len(logs_text) > 4000:
                    logs_text = logs_text[:3997] + "..."
                
                await query.edit_message_text(logs_text)
            except Exception as e:
                await query.edit_message_text(f"Erreur lors de la lecture des logs: {str(e)}")
        
        elif admin_action == "restart":
            await query.edit_message_text(
                "🔄 Redémarrage du bot\n\n"
                "Êtes-vous sûr de vouloir redémarrer le bot?\n\n"
                "⚠️ Tous les signaux en attente seront perdus."
            )
            
            # Ajouter des boutons de confirmation
            keyboard = [
                [
                    InlineKeyboardButton("Oui, redémarrer", callback_data="confirm_restart"),
                    InlineKeyboardButton("Non, annuler", callback_data="admin")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                "🔄 Redémarrage du bot\n\n"
                "Êtes-vous sûr de vouloir redémarrer le bot?\n\n"
                "⚠️ Tous les signaux en attente seront perdus.",
                reply_markup=reply_markup
            )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Gère les messages reçus et identifie les potentiels signaux de trading.
        
        Args:
            update: L'objet Update contenant les informations du message.
            context: Le contexte de l'application.
        """
        # Récupérer les informations du message
        message = update.message
        chat_id = update.effective_chat.id
        user = update.effective_user
        message_text = message.text
        
        # Vérifier si le message provient d'une source autorisée
        if self.allowed_chats and chat_id not in self.allowed_chats:
            if self.allowed_users and user.id not in self.allowed_users:
                logger.info(f"Message ignoré de chat/utilisateur non autorisé: {chat_id}/{user.id}")
                return
        
        # Vérifier si c'est un signal potentiel (filtrage basique)
        is_potential_signal = (
            "LONG" in message_text.upper() or 
            "SHORT" in message_text.upper() or
            "SIGNAL ALERT" in message_text.upper() or
            "TRADE SIGNAL" in message_text.upper()
        )
        
        if is_potential_signal:
            logger.info(f"Signal potentiel détecté de {chat_id}/{user.id}")
            
            # Créer un objet SignalMessage
            signal_data = SignalMessage(
                text=message_text,
                chat_id=chat_id,
                message_id=message.message_id,
                timestamp=message.date.timestamp(),
                user_id=user.id,
                username=user.username
            )
            
            # Ajouter à la file d'attente pour traitement
            self.signal_queue.put(signal_data)
            
            # Accuser réception du signal
            await message.reply_text(
                "📨 Signal reçu et en cours de traitement.\n"
                "Je vous tiendrai informé de son exécution."
            )
        else:
            logger.debug(f"Message non reconnu comme signal: {message_text[:20]}...")
    
    async def handle_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Gère les erreurs survenues pendant le traitement des mises à jour."""
        error = context.error
        
        # Journaliser l'erreur
        logger.error(f"L'erreur {error} est survenue", exc_info=context.error)
        
        # Déterminer le type d'erreur et répondre en conséquence
        error_message = "Désolé, une erreur s'est produite lors du traitement de votre message."
        
        if "Forbidden" in str(error):
            error_message = "Erreur: Le bot n'a pas les permissions nécessaires pour effectuer cette action."
        elif "Timed out" in str(error):
            error_message = "Erreur: La connexion a expiré. Veuillez réessayer."
        elif "Not enough rights" in str(error):
            error_message = "Erreur: Le bot n'a pas assez de droits dans ce chat."
        
        # Envoyer un message à l'utilisateur si possible
        if update and isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(error_message)
    
    async def send_notification(self, chat_id: int, message: str, reply_markup: Optional[InlineKeyboardMarkup] = None) -> None:
        """
        Envoie une notification à un chat spécifique.
        
        Args:
            chat_id: L'ID du chat auquel envoyer la notification.
            message: Le message à envoyer.
            reply_markup: Clavier inline optionnel à joindre au message.
        """
        try:
            await self.application.bot.send_message(
                chat_id=chat_id, 
                text=message,
                reply_markup=reply_markup
            )
            logger.info(f"Notification envoyée à {chat_id}")
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de la notification à {chat_id}: {str(e)}")
    
    async def send_signal_result(self, chat_id: int, signal_id: str, result: Dict[str, Any]) -> None:
        # Use a shorter timeout for buttons
        # Telegram buttons expire after some time
        """
        Envoie le résultat du traitement d'un signal.
        
        Args:
            chat_id: L'ID du chat auquel envoyer le résultat.
            signal_id: L'ID du signal traité.
            result: Le résultat du traitement du signal.
        """
        status = result.get("status", "unknown")
        
        if status == "success":
            # Créer un message de succès détaillé
            symbol = result.get("symbol", "")
            direction = result.get("direction", "")
            position_size = result.get("position_size", 0)
            leverage = result.get("leverage", 1)
            
            message = (
                f"✅ Signal exécuté avec succès!\n\n"
                f"🔹 ID: {signal_id}\n"
                f"🔹 Paire: {symbol}\n"
                f"🔹 Direction: {direction}\n"
                f"🔹 Taille: {position_size}\n"
                f"🔹 Levier: {leverage}x\n\n"
                f"Les ordres ont été placés sur Binance Futures."
            )
            
            # Ajouter des boutons pour gérer la position
            keyboard = [
                [
                    InlineKeyboardButton("Voir position", callback_data=f"view_position_{signal_id}"),
                    InlineKeyboardButton("Fermer position", callback_data=f"close_position_{signal_id}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
        elif status == "rejected":
            message = (
                f"⚠️ Signal rejeté\n\n"
                f"Raison: {result.get('message', 'Raison inconnue')}"
            )
            reply_markup = None
            
        elif status == "error":
            message = (
                f"❌ Erreur lors de l'exécution du signal\n\n"
                f"Détails: {result.get('message', 'Erreur inconnue')}"
            )
            reply_markup = None
            
        elif status == "expired":
            message = (
                f"⏱️ Signal expiré\n\n"
                f"Détails: {result.get('message', 'Le signal a expiré')}"
            )
            reply_markup = None
            
        else:
            message = (
                f"❓ Statut inconnu pour le signal\n\n"
                f"Détails: {json.dumps(result, indent=2)}"
            )
            reply_markup = None
        
        await self.send_notification(chat_id, message, reply_markup)
    
    def start(self) -> None:
        """Démarre le bot en mode polling."""
        logger.info("Démarrage du bot Telegram...")
        
        # Initialiser le bot de manière asynchrone
        async def start_bot():
            await self.initialize()
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
        
        # Exécuter la fonction asynchrone
        asyncio.run(start_bot())
    
    def start_async(self) -> None:
        """Démarre le bot de manière asynchrone."""
        logger.info("Démarrage asynchrone du bot Telegram...")
        
        # Créer une tâche asynchrone pour initialiser et démarrer le bot
        async def start_bot_async():
            await self.initialize()
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
        
        # Créer et retourner la tâche
        return asyncio.create_task(start_bot_async())
    
    async def stop(self) -> None:
        """Arrête le bot."""
        logger.info("Arrêt du bot Telegram...")
        await self.application.stop()
        await self.application.shutdown()


class SignalProcessor:
    """
    Classe pour traiter les signaux reçus de Telegram.
    
    Cette classe prend les signaux de la file d'attente, les parse,
    et les transmet au gestionnaire de trading.
    """
    
    def __init__(self, signal_queue: Queue, parser_callback: Callable, 
                 telegram_client: Optional[TelegramClient] = None,
                 max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialise le processeur de signaux.
        
        Args:
            signal_queue: La file d'attente contenant les signaux à traiter.
            parser_callback: La fonction de callback pour parser les signaux.
            telegram_client: Le client Telegram pour envoyer des notifications.
            max_retries: Nombre maximum de tentatives de parsing.
            retry_delay: Délai entre les tentatives en secondes.
        """
        self.signal_queue = signal_queue
        self.parser_callback = parser_callback
        self.telegram_client = telegram_client
        self.running = False
        self.processing_task = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Statistiques de traitement
        self.stats = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "retried": 0
        }
    
    async def process_signals(self) -> None:
        """Traite les signaux de la file d'attente en continu."""
        self.running = True
        
        while self.running:
            try:
                # Vérifier si la file d'attente contient des signaux
                if not self.signal_queue.empty():
                    # Récupérer le prochain signal
                    signal_data = self.signal_queue.get()
                    
                    # Si c'est un dictionnaire, le convertir en objet SignalMessage
                    if isinstance(signal_data, dict):
                        signal_data = SignalMessage.from_dict(signal_data)
                    
                    # Extraire les informations du signal
                    message_text = signal_data.text
                    chat_id = signal_data.chat_id
                    
                    logger.info(f"Traitement du signal de {chat_id}: {message_text[:20]}...")
                    self.stats["processed"] += 1
                    
                    # Parser le signal avec plusieurs tentatives si nécessaire
                    parsed_signal = None
                    parsing_errors = []
                    retries = 0
                    
                    while parsed_signal is None and retries < self.max_retries:
                        if retries > 0:
                            logger.info(f"Tentative {retries+1}/{self.max_retries} de parsing du signal")
                            self.stats["retried"] += 1
                            await asyncio.sleep(self.retry_delay)
                        
                        parsed_signal, errors = self.parser_callback(message_text)
                        parsing_errors = errors
                        retries += 1
                    
                    if parsed_signal:
                        logger.info(f"Signal parsé avec succès: {parsed_signal}")
                        self.stats["success"] += 1
                        
                        # Envoyer une notification si le client Telegram est disponible
                        if self.telegram_client:
                            # Créer un message de confirmation détaillé
                            confirmation = (
                                f"✅ Signal parsé avec succès:\n\n"
                                f"🔹 Symbole: {parsed_signal.symbol}\n"
                                f"🔹 Direction: {parsed_signal.direction}\n"
                                f"🔹 Entrée: {parsed_signal.entry_min} - {parsed_signal.entry_max}\n"
                                f"🔹 Stop Loss: {parsed_signal.stop_loss}\n"
                                f"🔹 Take Profits: {', '.join([str(tp) for tp in parsed_signal.take_profit_levels])}\n"
                                f"🔹 Confiance: {parsed_signal.confidence:.2f}\n"
                            )
                            
                            # Ajouter des avertissements si présents
                            if parsing_errors:
                                confirmation += "\n⚠️ Avertissements:\n"
                                for error in parsing_errors:
                                    confirmation += f"- {error}\n"
                            
                            # Ajouter des boutons pour confirmer ou annuler
                            keyboard = [
                                [
                                    InlineKeyboardButton("Exécuter", callback_data="execute_signal"),
                                    InlineKeyboardButton("Annuler", callback_data="cancel_signal")
                                ]
                            ]
                            reply_markup = InlineKeyboardMarkup(keyboard)
                            
                            await self.telegram_client.send_notification(
                                chat_id,
                                confirmation,
                                reply_markup
                            )
                    else:
                        logger.warning(f"Échec du parsing du signal après {retries} tentatives: {message_text[:50]}...")
                        self.stats["failed"] += 1
                        
                        # Envoyer une notification d'échec
                        if self.telegram_client:
                            failure_message = (
                                "❌ Échec du parsing du signal.\n\n"
                                "Le format du signal n'a pas pu être reconnu. "
                                "Veuillez vérifier le format et réessayer.\n\n"
                                "Utilisez /help pour voir les formats acceptés."
                            )
                            
                            # Ajouter les erreurs spécifiques
                            if parsing_errors:
                                failure_message += "\n\nErreurs détectées:\n"
                                for error in parsing_errors[:5]:  # Limiter à 5 erreurs
                                    failure_message += f"- {error}\n"
                                
                                if len(parsing_errors) > 5:
                                    failure_message += f"- ... et {len(parsing_errors) - 5} autres erreurs\n"
                            
                            await self.telegram_client.send_notification(
                                chat_id,
                                failure_message
                            )
                    
                    # Marquer le signal comme traité
                    self.signal_queue.task_done()
                
                # Attendre un court instant avant de vérifier à nouveau
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement du signal: {str(e)}")
                await asyncio.sleep(1)  # Attendre un peu plus longtemps en cas d'erreur
    
    def start(self) -> None:
        """Démarre le traitement des signaux dans une tâche asynchrone."""
        if not self.processing_task or self.processing_task.done():
            self.processing_task = asyncio.create_task(self.process_signals())
            logger.info("Processeur de signaux démarré")
    
    def stop(self) -> None:
        """Arrête le traitement des signaux."""
        self.running = False
        if self.processing_task:
            self.processing_task.cancel()
            logger.info("Processeur de signaux arrêté")
    
    def get_stats(self) -> Dict[str, int]:
        """Retourne les statistiques de traitement."""
        return self.stats


def main():
    """Fonction principale pour tester le module."""
    # Créer une file d'attente pour les signaux
    signal_queue = Queue()
    
    # Créer un client Telegram (remplacer 'YOUR_TOKEN' par un vrai token)
    # Pour les tests, vous pouvez utiliser un token de test obtenu via @BotFather
    telegram_client = TelegramClient(
        "YOUR_TOKEN", 
        signal_queue,
        admin_users=[123456789]  # Remplacer par votre ID Telegram
    )
    
    # Fonction de callback de test pour le parsing
    def test_parser(message_text):
        print(f"Parsing du message: {message_text[:50]}...")
        return None, ["Test parser not implemented"]
    
    # Créer un processeur de signaux
    signal_processor = SignalProcessor(
        signal_queue,
        test_parser,
        telegram_client
    )
    
    # Démarrer le bot et le processeur
    try:
        # Démarrer le processeur de signaux
        signal_processor.start()
        
        # Démarrer le bot Telegram
        telegram_client.start()
    except KeyboardInterrupt:
        print("Arrêt du bot...")
        signal_processor.stop()


if __name__ == "__main__":
    main()
