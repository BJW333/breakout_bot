"""
Breakout Bot Source Package
===========================
"""

# Core modules (no external dependencies beyond pandas/numpy)
from .indicators import add_all_indicators, get_indicator_summary
from .signals import generate_signals, find_all_signals, count_signals
from .risk_manager import RiskManager, TradePosition, TradeDirection, ExitReason
from .backtester import Backtester

__all__ = [
    'add_all_indicators',
    'get_indicator_summary',
    'generate_signals',
    'find_all_signals',
    'count_signals',
    'RiskManager',
    'TradePosition',
    'TradeDirection',
    'ExitReason',
    'Backtester'
]

# Optional: Data fetcher (requires ccxt)
try:
    from .data_fetcher import DataFetcher, fetch_eth_data, fetch_btc_data
    __all__.extend(['DataFetcher', 'fetch_eth_data', 'fetch_btc_data'])
except ImportError:
    pass  # ccxt not installed, data fetcher unavailable
