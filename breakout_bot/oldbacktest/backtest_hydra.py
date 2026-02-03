"""
HYDRA-SPOT Backtester
====================
Backtest the HYDRA-SPOT strategy on Kraken data.

Usage:
    python backtest_hydra.py --symbol SOL/USD --days 14
    python backtest_hydra.py --multi
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.hydra_spot import (
    HydraSpotStrategy, HydraConfig, FeatureEngine, 
    RegimeDetector, AlphaEngines, check_hydra_signal
)
from src.data_fetcher import DataFetcher

# Config
STARTING_CAPITAL = 10000.0
TIMEFRAME = '15m'  # Changed from 5m - bigger moves, less noise
ASSETS = ['SOL/USD', 'ETH/USD', 'BTC/USD']  # Removed LINK, added BTC


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    symbol: str
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    total_pnl: float
    final_equity: float
    return_pct: float
    profit_factor: float
    avg_winner: float
    avg_loser: float
    max_drawdown: float
    trades_per_week: float
    by_engine: Dict
    by_exit_reason: Dict


class HydraBacktester:
    """Backtest the HYDRA-SPOT strategy."""
    
    def __init__(self, df: pd.DataFrame, cfg: HydraConfig = None, symbol: str = 'TEST'):
        self.df = df.copy()
        self.cfg = cfg or HydraConfig()
        self.symbol = symbol
        
        # Create strategy
        self.strategy = HydraSpotStrategy(self.cfg)
        
        # Results tracking
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []
    
    def run(self) -> BacktestResult:
        """Run the backtest."""
        print(f"   Running HYDRA backtest on {len(self.df)} bars...")
        
        # Ensure index is datetime
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'time' in self.df.columns:
                self.df['time'] = pd.to_datetime(self.df['time'])
                self.df.set_index('time', inplace=True)
            else:
                self.df.index = pd.to_datetime(self.df.index)
        
        warmup = 150  # Need enough data for features
        peak_equity = self.strategy.equity
        max_dd = 0
        
        for i in range(warmup, len(self.df)):
            # Get candles up to this point
            candles = self.df.iloc[:i+1].copy()
            
            # Process bar
            actions = self.strategy.on_bar(self.symbol, candles, i)
            
            # Track equity and drawdown
            current_equity = self.strategy.equity
            self.equity_curve.append(current_equity)
            
            peak_equity = max(peak_equity, current_equity)
            dd = (peak_equity - current_equity) / peak_equity
            max_dd = max(max_dd, dd)
            self.drawdown_curve.append(dd)
            
            # Log trades
            for action in actions:
                if action.get('action') == 'OPEN':
                    self.trades.append({
                        'type': 'OPEN',
                        'bar': i,
                        'time': candles.index[-1],
                        'symbol': self.symbol,
                        'direction': action['direction'],
                        'engine': action['engine'],
                        'entry': action['entry'],
                        'stop': action['stop'],
                        'tp1': action['tp1'],
                        'tp2': action['tp2'],
                        'quality': action['quality'],
                        'regime': action['regime']
                    })
                elif action.get('action') == 'CLOSE':
                    self.trades.append({
                        'type': 'CLOSE',
                        'bar': i,
                        'time': candles.index[-1],
                        'reason': action['reason'],
                        'price': action['price']
                    })
        
        # Close any remaining positions
        if self.strategy.positions:
            final_price = float(self.df.iloc[-1]['close'])
            for pos in self.strategy.positions:
                if pos.active:
                    self.strategy._close_position(pos, final_price, 'END')
        
        # Calculate results
        return self._calculate_results(max_dd)
    
    def _calculate_results(self, max_dd: float) -> BacktestResult:
        """Calculate backtest statistics."""
        # Get trade results from strategy
        results = self.strategy.trade_results
        
        total = len(results)
        winners = sum(results) if results else 0
        losers = total - winners
        win_rate = (winners / total * 100) if total > 0 else 0
        
        # Calculate P&L
        final_equity = self.strategy.equity
        total_pnl = final_equity - STARTING_CAPITAL
        return_pct = (final_equity / STARTING_CAPITAL - 1) * 100
        
        # Trades per week
        if len(self.df) > 0:
            days = len(self.df) * 15 / (24 * 60)  # 15m candles
            weeks = max(days / 7, 0.1)
            trades_per_week = total / weeks
        else:
            trades_per_week = 0
        
        # By engine analysis
        engine_trades = {}
        exit_reasons = {}
        
        for trade in self.trades:
            if trade['type'] == 'OPEN':
                eng = trade['engine']
                if eng not in engine_trades:
                    engine_trades[eng] = {'count': 0}
                engine_trades[eng]['count'] += 1
            elif trade['type'] == 'CLOSE':
                reason = trade['reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return BacktestResult(
            symbol=self.symbol,
            total_trades=total,
            winners=winners,
            losers=losers,
            win_rate=win_rate,
            total_pnl=total_pnl,
            final_equity=final_equity,
            return_pct=return_pct,
            profit_factor=0,  # Would need trade-level P&L
            avg_winner=0,
            avg_loser=0,
            max_drawdown=max_dd * 100,
            trades_per_week=trades_per_week,
            by_engine=engine_trades,
            by_exit_reason=exit_reasons
        )


def fetch_data(symbol: str, days: int = 14) -> pd.DataFrame:
    """Fetch data from Kraken."""
    print(f"📥 Fetching {symbol}...")
    
    try:
        fetcher = DataFetcher()
        
        # Calculate candles needed
        # 15m = 96 candles per day, max 720 from Kraken
        candles_needed = min(days * 96, 720)
        
        df = fetcher.fetch_live(symbol, timeframe=TIMEFRAME, num_candles=candles_needed)
        
        if df is None or len(df) == 0:
            print(f"   ❌ No data for {symbol}")
            return None
        
        actual_days = len(df) * 15 / (24 * 60)  # 15m candles
        print(f"   Fetched {len(df)} candles (~{actual_days:.1f} days)")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error fetching {symbol}: {e}")
        return None


def run_single_backtest(symbol: str, days: int = 14) -> Optional[BacktestResult]:
    """Run backtest on a single symbol."""
    df = fetch_data(symbol, days)
    
    if df is None or len(df) < 200:
        print(f"   ❌ Insufficient data for {symbol}")
        return None
    
    try:
        cfg = HydraConfig()
        bt = HydraBacktester(df, cfg, symbol)
        result = bt.run()
        
        print(f"   ✅ {symbol}: {result.total_trades} trades, ${result.total_pnl:.2f} P&L, "
              f"{result.win_rate:.1f}% win rate, {result.trades_per_week:.0f} trades/wk")
        
        if result.by_engine:
            print(f"      Engines: {result.by_engine}")
        if result.by_exit_reason:
            print(f"      Exits: {result.by_exit_reason}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Error with {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_multi_asset_backtest():
    """Run backtest on all configured assets."""
    print("\n🐉 HYDRA-SPOT MULTI-ASSET BACKTEST")
    print("=" * 50)
    print(f"   Assets: {', '.join(ASSETS)}")
    print(f"   Timeframe: {TIMEFRAME}")
    print("=" * 50)
    print()
    
    results = []
    
    for symbol in ASSETS:
        result = run_single_backtest(symbol, days=14)
        if result:
            results.append(result)
        print()
    
    # Aggregate results
    print("=" * 50)
    print("📊 AGGREGATE RESULTS")
    print("=" * 50)
    
    if results:
        total_trades = sum(r.total_trades for r in results)
        total_pnl = sum(r.total_pnl for r in results)
        avg_win_rate = np.mean([r.win_rate for r in results])
        total_trades_per_week = sum(r.trades_per_week for r in results)
        
        print(f"   Total Trades: {total_trades}")
        print(f"   Total P&L: ${total_pnl:.2f}")
        print(f"   Avg Win Rate: {avg_win_rate:.1f}%")
        print(f"   Total Trades/Week: {total_trades_per_week:.0f}")
        
        # Engine breakdown
        all_engines = {}
        all_exits = {}
        for r in results:
            for eng, data in r.by_engine.items():
                if eng not in all_engines:
                    all_engines[eng] = 0
                all_engines[eng] += data['count']
            for reason, count in r.by_exit_reason.items():
                all_exits[reason] = all_exits.get(reason, 0) + count
        
        if all_engines:
            print(f"\n   By Engine:")
            for eng, count in sorted(all_engines.items(), key=lambda x: -x[1]):
                print(f"      {eng}: {count} trades")
        
        if all_exits:
            print(f"\n   By Exit:")
            for reason, count in sorted(all_exits.items(), key=lambda x: -x[1]):
                print(f"      {reason}: {count}")
    else:
        print("   No results")


def main():
    parser = argparse.ArgumentParser(description='HYDRA-SPOT Backtester')
    parser.add_argument('--symbol', type=str, default='SOL/USD', help='Symbol to test')
    parser.add_argument('--days', type=int, default=14, help='Days of data')
    parser.add_argument('--multi', action='store_true', help='Run on all assets')
    
    args = parser.parse_args()
    
    if args.multi:
        run_multi_asset_backtest()
    else:
        print(f"\n🐉 HYDRA-SPOT BACKTEST: {args.symbol}")
        print("=" * 50)
        run_single_backtest(args.symbol, args.days)


if __name__ == "__main__":
    main()
