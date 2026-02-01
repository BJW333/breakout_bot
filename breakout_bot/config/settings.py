"""
Consolidation Breakout Strategy - Configuration Settings
=========================================================
All tunable parameters in one place. Modify here, not in code.
"""

# =============================================================================
# STRATEGY PARAMETERS
# =============================================================================

# Plateau Detection
LOOKBACK_PERIOD = 20              # Candles to check for squeeze
RANGE_THRESHOLD = 0.04            # 4% - max price range for consolidation
MIN_CONSOLIDATION_CANDLES = 20    # Minimum candles in consolidation

# Bollinger Bands
BB_LENGTH = 20                    # Period for BB calculation
BB_STD = 2.0                      # Standard deviations

# Trend Filter
TREND_MA_LENGTH = 50              # Moving average for trend direction
TREND_MA_SLOPE_LOOKBACK = 5       # Candles to measure MA slope

# Volume Confirmation
VOLUME_MA_LENGTH = 20             # Period for volume moving average
VOLUME_THRESHOLD = 1.2           # Multiplier (1.2 = 120% of average)

# ATR (Average True Range)
ATR_PERIOD = 14                   # Period for ATR calculation

# =============================================================================
# RISK MANAGEMENT
# =============================================================================

STOP_LOSS_ATR = 2.0               # Stop loss distance in ATR multiples
PROFIT_TARGET_ATR = 2.2           # Profit target in ATR multiples
BREAKEVEN_TRIGGER_ATR = 1.5       # Move stop to breakeven after this profit
TRAILING_STOP_ATR = 1.0           # Trail stop this far below price (optional)
#RISK_PER_TRADE can be changed for how much to risk per trade so you make more money faster or more slowly
RISK_PER_TRADE = 0.10        # % of account per trade
MAX_DRAWDOWN = 0.15               # 15% - kill switch threshold

# =============================================================================
# DATA SETTINGS
# =============================================================================

# Primary trading pair
SYMBOL = "LINK/USD"               # Best performer from backtests
TIMEFRAME = "4h"                  # 1m, 5m, 15m, 30m, 1h, 4h, 1d
EXCHANGE = "kraken"               # ccxt exchange id (kraken works in US)

# Assets to monitor with tuned parameters (used by live_trader.py)
# Parameters determined via --auto-tune backtests
ASSET_CONFIGS = {
    'LINK/USD': {'range_threshold': 0.0853, 'volume_threshold': 1.0, 'enabled': True},
    'SOL/USD': {'range_threshold': 0.08, 'volume_threshold': 1.0, 'enabled': True},
    'ETH/USD': {'range_threshold': 0.0723, 'volume_threshold': 1.2, 'enabled': True},
    'XRP/USD': {'range_threshold': 0.0501, 'volume_threshold': 1.0, 'enabled': False},
    'BTC/USD': {'range_threshold': 0.039, 'volume_threshold': 1.0, 'enabled': False},
}

# Timeframe options
TIMEFRAMES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

# =============================================================================
# ACCOUNT SETTINGS
# =============================================================================

STARTING_CAPITAL = 1000          # Initial account balance for backtesting
PAPER_TRADING = True              # True = paper, False = live

# =============================================================================
# EXCHANGE API (load from environment variables in production)
# =============================================================================
# 
# To get Kraken API keys:
# 1. Go to https://www.kraken.com/u/security/api
# 2. Create a new API key with "Query Funds" and "Create & Modify Orders" permissions
# 3. Set them below or use environment variables:
#    export KRAKEN_API_KEY="your-key"
#    export KRAKEN_API_SECRET="your-secret"

import os
API_KEY = "MGbucIVKckIb4j4INGGdTtYYw1cwt4NmxGNJJMDmu3HW83vWwG3Ko6aY"
API_SECRET = "CXZvroZlYun0LsoNnXk17xh9jrFrq+RGuYVT7SrG1F2E0ujx0XiVF/EWGItlgmigIXa1UqmtJy6/kUHBqEIlNQ=="

# =============================================================================
# NOTIFICATIONS
# =============================================================================
#
# TELEGRAM SETUP:
# 1. Message @BotFather on Telegram
# 2. Send /newbot and follow instructions
# 3. Copy the token (looks like: 123456789:ABCdefGHIjklmNOpqrsTUVwxyz)
# 4. Message your new bot, then visit:
#    https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
# 5. Find your chat_id in the response
#
# DISCORD SETUP:
# 1. Go to Server Settings > Integrations > Webhooks
# 2. Create a new webhook and copy the URL

ENABLE_NOTIFICATIONS = True
VERBOSE = True                    # Print to console

# Telegram
#TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_BOT_TOKEN = "8576313126:AAHJ2TuV2Bqv7nIHDHLhUNUCQmK3mZluYQU"
#TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
TELEGRAM_CHAT_ID= "5711640022"

# Discord (optional)
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '')

# =============================================================================
# LOGGING
# =============================================================================

LOG_TRADES = True
LOG_FILE = "logs/trades.csv"

# =============================================================================
# BACKTESTING
# =============================================================================

BACKTEST_START = "2024-01-01"
BACKTEST_END = "2025-01-01"
COMMISSION = 0.0026               # 0.26% per trade (Kraken taker fee for market orders)
SLIPPAGE = 0.0005                 # 0.05% estimated slippage
