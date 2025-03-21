# Telegram Bot for Trading Signals

This bot monitors Telegram channels for trading signals, parses them, and executes trades on Binance Futures.

## Features

- Multi-format signal parsing with automatic detection
- Enhanced validation with risk-based and market context checks
- Advanced order types (trailing stops, OCO orders, scaled entry/exit)
- Dynamic position sizing based on market conditions
- Technical analysis integration for trend validation
- Comprehensive risk management
- Performance analytics and strategy optimization

## Directory Structure

```
telegram-bot/
├── config.json           # Configuration file
├── main.py               # Main entry point
├── src/                  # Source code
│   ├── __init__.py       # Package initialization
│   ├── signal_parser.py  # Signal parsing module
│   ├── telegram_client.py # Telegram integration
│   ├── binance_client.py # Binance API integration
│   ├── trade_executor.py # Trade execution logic
│   └── risk_manager.py   # Risk management
└── tests/                # Test suite
    └── test_components.py # Component tests
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/telegram-bot.git
cd telegram-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the bot:
Edit `config.json` with your Telegram bot token and Binance API credentials.

## Usage

1. Start the bot:
```bash
python main.py
```

2. The bot will monitor configured Telegram channels for trading signals.

3. When a valid signal is detected, it will be parsed and executed on Binance.

## Configuration

Edit `config.json` to configure:

- Telegram bot token
- Allowed Telegram chats and users
- Binance API credentials
- Risk management parameters
- Trading preferences

## Testing

Run the test suite:
```bash
python -m tests.test_components
```

## License

MIT
