"""
Utilities Module
================
Helper functions for logging, notifications, and formatting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import csv
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import LOG_FILE, VERBOSE


def setup_logging(log_dir: str = "logs") -> Path:
    """
    Set up logging directory.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Path to log directory
    """
    log_path = Path(__file__).parent.parent / log_dir
    log_path.mkdir(parents=True, exist_ok=True)
    return log_path


def log_trade(trade_data: dict, filename: str = LOG_FILE):
    """
    Log a trade to CSV file.
    
    Args:
        trade_data: Dictionary containing trade information
        filename: Path to log file
    """
    filepath = Path(__file__).parent.parent / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to determine if we need headers
    file_exists = filepath.exists()
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=trade_data.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(trade_data)


def load_trade_log(filename: str = LOG_FILE) -> pd.DataFrame:
    """
    Load trade log from CSV.
    
    Args:
        filename: Path to log file
        
    Returns:
        DataFrame of trades
    """
    filepath = Path(__file__).parent.parent / filename
    
    if not filepath.exists():
        return pd.DataFrame()
    
    return pd.read_csv(filepath, parse_dates=['entry_time', 'exit_time'])


def format_results(results: dict) -> str:
    """
    Format results dictionary as readable string.
    
    Args:
        results: Dictionary of results
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("RESULTS")
    lines.append("=" * 50)
    
    for key, value in results.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.2f}")
        elif isinstance(value, dict):
            lines.append(f"  {key}:")
            for k, v in value.items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append(f"  {key}: {value}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


def print_signal(signal_type: str, price: float, timestamp: pd.Timestamp,
                 details: dict = None):
    """
    Print a trading signal to console.
    
    Args:
        signal_type: 'LONG', 'SHORT', or 'EXIT'
        price: Current price
        timestamp: Signal timestamp
        details: Additional details
    """
    if not VERBOSE:
        return
    
    emoji = "🟢" if signal_type == "LONG" else "🔴" if signal_type == "SHORT" else "⚪"
    
    print(f"\n{emoji} {signal_type} SIGNAL")
    print(f"   Time: {timestamp}")
    print(f"   Price: ${price:,.2f}")
    
    if details:
        for key, value in details.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")


def calculate_performance_stats(trades: pd.DataFrame) -> dict:
    """
    Calculate detailed performance statistics.
    
    Args:
        trades: DataFrame of trades
        
    Returns:
        Dictionary of statistics
    """
    if len(trades) == 0:
        return {'error': 'No trades to analyze'}
    
    # Basic stats
    stats = {
        'total_trades': len(trades),
        'total_pnl': trades['pnl'].sum(),
        'avg_pnl': trades['pnl'].mean(),
        'std_pnl': trades['pnl'].std(),
    }
    
    # Win/loss
    winners = trades[trades['pnl'] > 0]
    losers = trades[trades['pnl'] <= 0]
    
    stats['win_rate'] = len(winners) / len(trades) * 100
    stats['avg_winner'] = winners['pnl'].mean() if len(winners) > 0 else 0
    stats['avg_loser'] = losers['pnl'].mean() if len(losers) > 0 else 0
    stats['largest_winner'] = winners['pnl'].max() if len(winners) > 0 else 0
    stats['largest_loser'] = losers['pnl'].min() if len(losers) > 0 else 0
    
    # Streaks
    trades['win'] = trades['pnl'] > 0
    
    def count_streaks(series):
        streaks = []
        current = 0
        for val in series:
            if val:
                current += 1
            else:
                if current > 0:
                    streaks.append(current)
                current = 0
        if current > 0:
            streaks.append(current)
        return streaks
    
    win_streaks = count_streaks(trades['win'])
    lose_streaks = count_streaks(~trades['win'])
    
    stats['max_win_streak'] = max(win_streaks) if win_streaks else 0
    stats['max_lose_streak'] = max(lose_streaks) if lose_streaks else 0
    
    # Expectancy
    stats['expectancy'] = (
        (stats['win_rate'] / 100 * stats['avg_winner']) +
        ((100 - stats['win_rate']) / 100 * stats['avg_loser'])
    )
    
    return stats


def format_currency(value: float) -> str:
    """Format value as currency string."""
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    """Format value as percentage string."""
    return f"{value:.2f}%"


def time_since(timestamp: pd.Timestamp) -> str:
    """
    Get human-readable time since timestamp.
    
    Args:
        timestamp: Past timestamp
        
    Returns:
        String like "5 minutes ago"
    """
    now = pd.Timestamp.now()
    diff = now - timestamp
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        return f"{int(seconds / 60)} minutes ago"
    elif seconds < 86400:
        return f"{int(seconds / 3600)} hours ago"
    else:
        return f"{int(seconds / 86400)} days ago"


class ConsoleColors:
    """ANSI color codes for terminal output."""
    
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    @classmethod
    def green(cls, text: str) -> str:
        return f"{cls.GREEN}{text}{cls.RESET}"
    
    @classmethod
    def red(cls, text: str) -> str:
        return f"{cls.RED}{text}{cls.RESET}"
    
    @classmethod
    def yellow(cls, text: str) -> str:
        return f"{cls.YELLOW}{text}{cls.RESET}"
    
    @classmethod
    def blue(cls, text: str) -> str:
        return f"{cls.BLUE}{text}{cls.RESET}"
    
    @classmethod
    def bold(cls, text: str) -> str:
        return f"{cls.BOLD}{text}{cls.RESET}"


def print_banner():
    """Print application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║      CONSOLIDATION BREAKOUT TRADING BOT               ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    print(ConsoleColors.blue(banner))


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    print("Testing Utils Module...")
    print("=" * 50)
    
    # Test logging setup
    log_path = setup_logging()
    print(f"Log directory: {log_path}")
    
    # Test trade logging
    sample_trade = {
        'entry_time': datetime.now().isoformat(),
        'exit_time': datetime.now().isoformat(),
        'direction': 'LONG',
        'entry_price': 3000,
        'exit_price': 3100,
        'pnl': 100,
        'r_multiple': 1.5
    }
    
    log_trade(sample_trade, "logs/test_trades.csv")
    print("Logged sample trade")
    
    # Test formatting
    results = {
        'total_return': 15.5,
        'win_rate': 45.0,
        'trades': 20
    }
    print("\nFormatted results:")
    print(format_results(results))
    
    # Test signal printing
    print_signal('LONG', 3000.50, pd.Timestamp.now(), {'atr': 50.0, 'volume_ratio': 1.8})
    
    # Test colors
    print("\nColor test:")
    print(ConsoleColors.green("This is green (profit)"))
    print(ConsoleColors.red("This is red (loss)"))
    print(ConsoleColors.yellow("This is yellow (warning)"))
    print(ConsoleColors.bold("This is bold"))
    
    # Print banner
    print_banner()
