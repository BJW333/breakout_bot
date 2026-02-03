#!/usr/bin/env python3
"""
Backtest Runner
===============
Entry point for running strategy backtests.

Usage:
    python backtest.py                          # Run with defaults (Kraken ETH/USD)
    python backtest.py --symbol SOL/USD --auto-tune   # Auto-tune parameters for SOL
    python backtest.py --symbol BTC/USD         # Different symbol
    python backtest.py --timeframe 1h           # Different timeframe
    python backtest.py --start 2024-01-01       # Custom date range
    python backtest.py --capital 50000          # Different starting capital
    python backtest.py --exchange binanceus     # Different exchange

Exchanges that work in the US:
    - kraken (default) - symbols like ETH/USD, BTC/USD
    - binanceus - symbols like ETH/USD, BTC/USD
    - coinbase - symbols like ETH/USD, BTC/USD (limited timeframes)
    
Exchanges that DON'T work in the US:
    - binance (blocked)

Notes:
- Make sure to set correct symbol format for the exchange.
    For example, Kraken and Coinbase use ETH/USD, while Binance uses ETH/USDT.
- Use --auto-tune flag to find optimal parameters for the selected asset.
- Results and trade logs can be saved to CSV files using --save-data and --save-trades flags.

"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    SYMBOL, TIMEFRAME, STARTING_CAPITAL,
    BACKTEST_START, BACKTEST_END, EXCHANGE,
    RANGE_THRESHOLD, VOLUME_THRESHOLD
)
from src.data_fetcher import DataFetcher
from src.indicators import add_all_indicators
from src.signals import generate_signals, count_signals
from src.backtester import Backtester
from src.utils import print_banner, ConsoleColors


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run backtest on consolidation breakout strategy'
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default=SYMBOL,
        help=f'Trading symbol (default: {SYMBOL})'
    )
    
    parser.add_argument(
        '--timeframe', '-t',
        type=str,
        default=TIMEFRAME,
        help=f'Candle timeframe (default: {TIMEFRAME})'
    )
    
    parser.add_argument(
        '--exchange', '-e',
        type=str,
        default=EXCHANGE,
        help=f'Exchange to use (default: {EXCHANGE}). US-friendly: kraken, binanceus, coinbase'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default=BACKTEST_START,
        help=f'Start date YYYY-MM-DD (default: {BACKTEST_START})'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default=BACKTEST_END,
        help=f'End date YYYY-MM-DD (default: {BACKTEST_END})'
    )
    
    parser.add_argument(
        '--capital', '-c',
        type=float,
        default=STARTING_CAPITAL,
        help=f'Starting capital (default: {STARTING_CAPITAL})'
    )
    
    parser.add_argument(
        '--from-file', '-f',
        type=str,
        default=None,
        help='Load data from CSV file instead of fetching'
    )
    
    parser.add_argument(
        '--save-data',
        action='store_true',
        help='Save fetched data to CSV'
    )
    
    parser.add_argument(
        '--save-trades',
        action='store_true',
        help='Save trade log to CSV'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )
    
    parser.add_argument(
        '--auto-tune', '-a',
        action='store_true',
        help='Auto-tune parameters for the selected asset'
    )
    
    return parser.parse_args()


def auto_tune_parameters(df, symbol):
    """
    Analyze asset and find optimal parameters.
    
    Returns:
        tuple: (best_range_threshold, best_volume_threshold, diagnosis_info)
    """
    print(f"\n🔧 AUTO-TUNING PARAMETERS FOR {symbol}...")
    
    # Analyze the data
    range_min = df['price_range'].min()
    range_10pct = df['price_range'].quantile(0.10)
    range_20pct = df['price_range'].quantile(0.20)
    range_30pct = df['price_range'].quantile(0.30)
    
    print(f"   Asset volatility analysis:")
    print(f"   - Min 20-candle range: {range_min*100:.2f}%")
    print(f"   - 10th percentile: {range_10pct*100:.2f}%")
    print(f"   - 20th percentile: {range_20pct*100:.2f}%")
    print(f"   - 30th percentile: {range_30pct*100:.2f}%")
    
    # Test parameter combinations
    print(f"\n   Testing parameter combinations...")
    
    best_result = None
    best_params = None
    best_score = -999
    
    range_options = [range_10pct, range_20pct, range_30pct, 0.05, 0.06, 0.08, 0.10]
    volume_options = [1.0, 1.2, 1.3, 1.5]
    
    for rt in range_options:
        for vt in volume_options:
            result = backtest_with_params(df.copy(), rt, vt)
            
            trades = result.get('total_trades', 0)
            pf = result.get('profit_factor', 0)
            ret = result.get('total_return_pct', -100)
            dd = result.get('max_drawdown_pct', 100)
            
            # Score: prioritize returns, profit factor, and reasonable trade count
            if trades >= 2:
                score = ret * 2 + (pf * 5) + (trades * 0.3) - (dd * 0.5)
            else:
                score = -100
            
            if score > best_score:
                best_score = score
                best_result = result
                best_params = {'range_threshold': rt, 'volume_threshold': vt}
    
    if best_params:
        print(f"\n   ✅ OPTIMAL PARAMETERS FOUND:")
        print(f"      RANGE_THRESHOLD = {best_params['range_threshold']*100:.2f}%")
        print(f"      VOLUME_THRESHOLD = {best_params['volume_threshold']}")
        
        if best_result and best_result.get('total_trades', 0) > 0:
            print(f"\n   📊 Expected performance:")
            print(f"      Trades: {best_result['total_trades']}")
            print(f"      Return: {best_result['total_return_pct']:.2f}%")
            print(f"      Win Rate: {best_result['win_rate']:.1f}%")
            print(f"      Profit Factor: {best_result['profit_factor']:.2f}")
    else:
        print(f"\n   ⚠️  Could not find profitable parameters for {symbol}")
        best_params = {'range_threshold': range_20pct, 'volume_threshold': 1.2}
    
    return best_params['range_threshold'], best_params['volume_threshold'], best_result


def backtest_with_params(df, range_threshold, volume_threshold, capital=10000):
    """
    Run backtest with specific parameters.
    """
    df = df.copy()
    
    # Apply plateau detection with custom threshold
    bb_squeeze = df['bb_width_percentile'] < 25
    tight_range = df['price_range'] < range_threshold
    df['plateau_active'] = bb_squeeze & tight_range
    
    # Generate signals
    df['signal'] = 0
    plateau_was_active = df['plateau_active'].shift(1).fillna(False)
    
    long_trend = (df['ma_slope'] > 0) & (df['close'] > df['ma_50'])
    short_trend = (df['ma_slope'] < 0) & (df['close'] < df['ma_50'])
    
    prev_range_high = df['range_high'].shift(1)
    prev_range_low = df['range_low'].shift(1)
    volume_ok = df['volume_ratio'] >= volume_threshold
    
    long_breakout = (df['high'] > prev_range_high) & volume_ok
    short_breakout = (df['low'] < prev_range_low) & volume_ok
    
    long_signal = plateau_was_active & long_trend & long_breakout
    short_signal = plateau_was_active & short_trend & short_breakout
    
    df.loc[long_signal, 'signal'] = 1
    # figure this out later  # Disabled - live_trader only does LONG
    #df.loc[short_signal, 'signal'] = -1
    
    # Run backtest
    bt = Backtester(df, starting_capital=capital)
    return bt.run()


def main():
    """Main entry point."""
    args = parse_args()
    
    if not args.quiet:
        print_banner()
    
    print(f"\n📊 BACKTEST CONFIGURATION")
    print(f"   Exchange: {args.exchange}")
    print(f"   Symbol: {args.symbol}")
    print(f"   Timeframe: {args.timeframe}")
    print(f"   Period: {args.start} to {args.end}")
    print(f"   Starting Capital: ${args.capital:,.2f}")
    if args.auto_tune:
        print(f"   Auto-tune: ENABLED")
    
    # Load or fetch data
    print(f"\n📥 LOADING DATA...")
    
    if args.from_file:
        # Load from file
        fetcher = DataFetcher()
        df = fetcher.load_from_csv(args.from_file)
    else:
        # Fetch from exchange
        fetcher = DataFetcher(exchange_id=args.exchange)
        df = fetcher.fetch_historical(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end
        )
        
        if len(df) == 0:
            print(ConsoleColors.red(f"\n❌ No data returned from {args.exchange}"))
            print(f"\n💡 Troubleshooting tips:")
            print(f"   - Make sure symbol format is correct for {args.exchange}")
            print(f"   - Kraken/Coinbase use: ETH/USD, BTC/USD")
            print(f"   - Binance uses: ETH/USDT, BTC/USDT")
            print(f"   - If in the US, avoid 'binance' (use 'kraken' or 'binanceus')")
            sys.exit(1)
        
        if args.save_data:
            fetcher.save_to_csv(df)
    
    print(f"   Loaded {len(df)} candles")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # Add indicators
    print(f"\n📈 CALCULATING INDICATORS...")
    df = add_all_indicators(df)
    
    # Auto-tune if requested
    if args.auto_tune:
        range_thresh, vol_thresh, tuned_result = auto_tune_parameters(df, args.symbol)
        
        # Run final backtest with tuned parameters
        print(f"\n🚀 RUNNING BACKTEST WITH TUNED PARAMETERS...")
        results = backtest_with_params(df, range_thresh, vol_thresh, args.capital)
        
        # Print results manually since we bypassed normal flow
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS (AUTO-TUNED)")
        print("=" * 60)
        
        if results.get('total_trades', 0) == 0:
            print("\nNo trades executed")
        else:
            print(f"\n📊 PERFORMANCE SUMMARY")
            print(f"   Total Return: {results['total_return_pct']}%")
            print(f"   Monthly Return: {results['monthly_return_pct']}%")
            print(f"   Max Drawdown: {results['max_drawdown_pct']}%")
            print(f"   Sharpe Ratio: {results['sharpe_ratio']}")
            
            print(f"\n💰 CAPITAL")
            print(f"   Starting: ${args.capital:,.2f}")
            print(f"   Final: ${results['final_capital']:,.2f}")
            print(f"   Total P&L: ${results['total_pnl']:,.2f}")
            
            print(f"\n📈 TRADES")
            print(f"   Total Trades: {results['total_trades']}")
            print(f"   Winners: {results['winning_trades']}")
            print(f"   Losers: {results['losing_trades']}")
            print(f"   Win Rate: {results['win_rate']}%")
            print(f"   Profit Factor: {results['profit_factor']}")
            
            print(f"\n🎯 TUNED PARAMETERS")
            print(f"   RANGE_THRESHOLD = {range_thresh:.4f} ({range_thresh*100:.2f}%)")
            print(f"   VOLUME_THRESHOLD = {vol_thresh}")
            print(f"\n   💡 Add these to config/settings.py for {args.symbol}")
        
        print("=" * 60)
        return results
    
    # Standard flow without auto-tune
    print(f"⚡ GENERATING SIGNALS...")
    df = generate_signals(df)
    
    # Signal summary
    signal_stats = count_signals(df)
    print(f"   Plateau candles: {signal_stats['plateau_candles']}")
    print(f"   Long signals: {signal_stats['long_signals']}")
    print(f"   Short signals: {signal_stats['short_signals']}")
    
    # Run backtest
    print(f"\n🚀 RUNNING BACKTEST...")
    
    backtester = Backtester(df, starting_capital=args.capital)
    results = backtester.run()
    
    # Print results
    backtester.print_results()
    
    # Save trade log if requested
    if args.save_trades:
        trade_log = backtester.get_trade_log()
        if len(trade_log) > 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"logs/trades_{args.symbol.replace('/', '_')}_{timestamp}.csv"
            trade_log.to_csv(Path(__file__).parent / filename, index=False)
            print(f"\n💾 Trade log saved to: {filename}")
    
    # Return results for programmatic use
    return results


if __name__ == "__main__":
    try:
        results = main()
        
        # Exit with appropriate code
        if results.get('total_trades', 0) == 0:
            print("\n⚠️  No trades executed. Try --auto-tune flag.")
            sys.exit(1)
        elif results.get('total_return_pct', 0) > 0:
            print(ConsoleColors.green("\n✅ Backtest completed successfully!"))
            sys.exit(0)
        else:
            print(ConsoleColors.yellow("\n⚠️  Backtest completed with negative returns."))
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\n⛔ Backtest interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(ConsoleColors.red(f"\n❌ Error: {e}"))
        raise
