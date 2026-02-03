#!/usr/bin/env python3
"""
Combined Strategy Backtester
============================
Backtest BREAKOUT, MEAN_REV, and HYDRA strategies side-by-side for fair comparison.

Usage:
    python backtest_combined.py                          # Run all strategies on all assets
    python backtest_combined.py --strategy breakout      # BREAKOUT only
    python backtest_combined.py --strategy mean_rev      # MEAN_REV only
    python backtest_combined.py --strategy hydra         # HYDRA only
    python backtest_combined.py --symbol SOL/USD         # Single asset
    python backtest_combined.py --auto-tune              # Auto-tune BREAKOUT params
    python backtest_combined.py --save-report            # Save detailed report to CSV
    python backtest_combined.py --debug                  # Show debug info (why no trades)

    # BREAKOUT-specific options (from original backtest.py)
    python backtest_combined.py --strategy breakout --timeframe 1h
    python backtest_combined.py --strategy breakout --exchange binanceus
    python backtest_combined.py --strategy breakout --save-trades
    python backtest_combined.py --strategy breakout --from-file data.csv

    # HYDRA-specific options (from original backtest_hydra.py)  
    python backtest_combined.py --strategy hydra --days 7

Strategies:
    BREAKOUT: 4-hour consolidation breakout with BB squeeze detection
    MEAN_REV: 4-hour oversold bounce (RSI + Bollinger Band)
    HYDRA:    15-minute regime-gated multi-engine strategy
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    SYMBOL, TIMEFRAME, STARTING_CAPITAL,
    BACKTEST_START, BACKTEST_END, EXCHANGE,
    RANGE_THRESHOLD, VOLUME_THRESHOLD, ASSET_CONFIGS,
    COMMISSION, SLIPPAGE
)
from src.data_fetcher import DataFetcher
from src.indicators import add_all_indicators
from src.signals import generate_signals, count_signals
from src.backtester import Backtester
from src.hydra_spot import (
    HydraSpotStrategy, HydraConfig, FeatureEngine, 
    RegimeDetector, AlphaEngines
)
from src.utils import print_banner, ConsoleColors


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default assets to test (use settings.py configs)
DEFAULT_ASSETS = ['SOL/USD', 'ETH/USD', 'LINK/USD', 'BTC/USD', 'XRP/USD']

# Strategy timeframes
BREAKOUT_TIMEFRAME = '4h'
MEAN_REV_TIMEFRAME = '4h'  # Same as BREAKOUT
HYDRA_TIMEFRAME = '15m'
HYDRA_STARTING_CAPITAL = 10000.0  # HYDRA's internal default

# Mean reversion parameters (per asset)
MEAN_REV_PARAMS = {
    'LINK/USD': {'rsi_oversold': 30, 'rsi_overbought': 70, 'enabled': True},
    'SOL/USD': {'rsi_oversold': 30, 'rsi_overbought': 70, 'enabled': True},
    'XRP/USD': {'rsi_oversold': 30, 'rsi_overbought': 70, 'enabled': True},
    'BTC/USD': {'rsi_oversold': 25, 'rsi_overbought': 75, 'enabled': True},
    'ETH/USD': {'rsi_oversold': 25, 'rsi_overbought': 75, 'enabled': True},
}


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class StrategyResult:
    """Results from a single strategy backtest."""
    strategy: str
    symbol: str
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    total_pnl: float
    final_equity: float
    return_pct: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    avg_winner: float
    avg_loser: float
    avg_r_multiple: float
    trades_per_week: float
    days_tested: int
    monthly_return: float = 0.0
    peak_capital: float = 0.0
    exit_reasons: Dict = field(default_factory=dict)
    by_engine: Dict = field(default_factory=dict)  # HYDRA only
    params: Dict = field(default_factory=dict)
    debug_info: Dict = field(default_factory=dict)


@dataclass
class CombinedResult:
    """Combined results across all strategies and assets."""
    breakout_results: List[StrategyResult]
    mean_rev_results: List[StrategyResult]
    hydra_results: List[StrategyResult]
    aggregate_breakout: Dict
    aggregate_mean_rev: Dict
    aggregate_hydra: Dict


# =============================================================================
# HYDRA BACKTESTER
# =============================================================================

class HydraBacktester:
    """Backtest the HYDRA-SPOT strategy with debugging support."""
    
    def __init__(self, df: pd.DataFrame, cfg: HydraConfig = None, 
                 symbol: str = 'TEST', starting_capital: float = HYDRA_STARTING_CAPITAL,
                 debug: bool = False):
        self.df = df.copy()
        self.cfg = cfg or HydraConfig()
        self.symbol = symbol
        self.starting_capital = starting_capital
        self.debug = debug
        
        self.strategy = HydraSpotStrategy(self.cfg)
        if starting_capital != HYDRA_STARTING_CAPITAL:
            self.strategy.equity = starting_capital
            self.strategy.starting_equity = starting_capital
            self.strategy.peak_equity = starting_capital
            self.strategy.day_start_equity = starting_capital
        
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []
        
        self.debug_stats = {
            'bars_processed': 0,
            'regime_failures': 0,
            'no_candidates': 0,
            'quality_filtered': 0,
            'signals_generated': 0,
            'regime_samples': [],
            'candidate_samples': []
        }
    
    def run(self) -> StrategyResult:
        """Run the backtest."""
        print(f"      Running HYDRA backtest on {len(self.df)} bars...")
        
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'time' in self.df.columns:
                self.df['time'] = pd.to_datetime(self.df['time'])
                self.df.set_index('time', inplace=True)
            else:
                self.df.index = pd.to_datetime(self.df.index)
        
        warmup = 150
        peak_equity = self.strategy.equity
        max_dd = 0
        
        if self.debug and len(self.df) > warmup:
            features = FeatureEngine()
            regime_detector = RegimeDetector(self.cfg)
            engines = AlphaEngines(self.cfg)
        
        for i in range(warmup, len(self.df)):
            candles = self.df.iloc[:i+1].copy()
            
            if self.debug:
                self._debug_bar(candles, i, features, regime_detector, engines)
            
            actions = self.strategy.on_bar(self.symbol, candles, i)
            self.debug_stats['bars_processed'] += 1
            
            current_equity = self.strategy.equity
            self.equity_curve.append(current_equity)
            
            peak_equity = max(peak_equity, current_equity)
            dd = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
            max_dd = max(max_dd, dd)
            self.drawdown_curve.append(dd)
            
            for action in actions:
                if action.get('action') == 'OPEN':
                    self.debug_stats['signals_generated'] += 1
                    self.trades.append({
                        'type': 'OPEN', 'bar': i, 'time': candles.index[-1],
                        'symbol': self.symbol, 'direction': action['direction'],
                        'engine': action['engine'], 'entry': action['entry'],
                        'stop': action['stop'], 'tp1': action['tp1'],
                        'tp2': action['tp2'], 'quality': action['quality'],
                        'regime': action['regime']
                    })
                elif action.get('action') == 'CLOSE':
                    self.trades.append({
                        'type': 'CLOSE', 'bar': i, 'time': candles.index[-1],
                        'reason': action['reason'], 'price': action['price']
                    })
        
        if self.strategy.positions:
            final_price = float(self.df.iloc[-1]['close'])
            for pos in self.strategy.positions:
                if pos.active:
                    self.strategy._close_position(pos, final_price, 'END')
        
        return self._calculate_results(max_dd)
    
    def _debug_bar(self, candles, bar_idx, features, regime_detector, engines):
        if len(candles) < 100:
            return
        feats = features.compute(candles)
        if len(feats) < 100:
            return
        regime = regime_detector.detect(feats)
        max_regime_conf = max(regime.values()) if regime else 0
        
        if bar_idx % 100 == 0 and len(self.debug_stats['regime_samples']) < 5:
            self.debug_stats['regime_samples'].append({
                'bar': bar_idx, 'regime': dict(regime), 'max_conf': max_regime_conf
            })
        
        if max_regime_conf < self.cfg.min_regime_confidence:
            self.debug_stats['regime_failures'] += 1
            return
        
        candidates = engines.generate(self.symbol, candles, feats, regime)
        if not candidates:
            self.debug_stats['no_candidates'] += 1
            return
        
        if len(self.debug_stats['candidate_samples']) < 5:
            self.debug_stats['candidate_samples'].append({
                'bar': bar_idx, 'num_candidates': len(candidates),
                'qualities': [c.quality_score for c in candidates],
                'engines': [c.engine for c in candidates]
            })
        
        qualified = [c for c in candidates if c.quality_score >= self.cfg.min_quality_score]
        if not qualified:
            self.debug_stats['quality_filtered'] += 1
    
    def _calculate_results(self, max_dd: float) -> StrategyResult:
        results = self.strategy.trade_results
        total = len(results)
        winners = sum(results) if results else 0
        losers = total - winners
        win_rate = (winners / total * 100) if total > 0 else 0
        
        final_equity = self.strategy.equity
        total_pnl = final_equity - self.strategy.starting_equity
        return_pct = (final_equity / self.strategy.starting_equity - 1) * 100
        peak_capital = max(self.equity_curve) if self.equity_curve else self.starting_capital
        
        if len(self.df) > 1:
            days = (self.df.index[-1] - self.df.index[0]).days
            weeks = max(days / 7, 0.1)
            trades_per_week = total / weeks if weeks > 0 else 0
            monthly_return = (return_pct / days) * 30 if days > 0 else 0
        else:
            days = trades_per_week = monthly_return = 0
        
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0
        
        engine_trades = {}
        exit_reasons = {}
        for trade in self.trades:
            if trade['type'] == 'OPEN':
                eng = trade['engine']
                engine_trades[eng] = engine_trades.get(eng, {'count': 0})
                engine_trades[eng]['count'] += 1
            elif trade['type'] == 'CLOSE':
                reason = trade['reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return StrategyResult(
            strategy='HYDRA', symbol=self.symbol, total_trades=total,
            winners=winners, losers=losers, win_rate=win_rate,
            total_pnl=total_pnl, final_equity=final_equity, return_pct=return_pct,
            profit_factor=0, max_drawdown=max_dd * 100, sharpe_ratio=round(sharpe, 2),
            avg_winner=0, avg_loser=0, avg_r_multiple=0, trades_per_week=trades_per_week,
            days_tested=days, monthly_return=round(monthly_return, 2),
            peak_capital=round(peak_capital, 2), exit_reasons=exit_reasons,
            by_engine=engine_trades, debug_info=self.debug_stats
        )


# =============================================================================
# BREAKOUT BACKTESTER
# =============================================================================

def run_breakout_backtest(df: pd.DataFrame, symbol: str, 
                          range_threshold: float = RANGE_THRESHOLD,
                          volume_threshold: float = VOLUME_THRESHOLD,
                          capital: float = STARTING_CAPITAL,
                          show_signals: bool = False) -> Tuple[StrategyResult, Backtester]:
    df = df.copy()
    if 'bb_width' not in df.columns:
        df = add_all_indicators(df)
    
    bb_squeeze = df['bb_width_percentile'] < 25
    tight_range = df['price_range'] < range_threshold
    df['plateau_active'] = bb_squeeze & tight_range
    
    df['signal'] = 0
    plateau_was_active = df['plateau_active'].shift(1).fillna(False)
    long_trend = (df['ma_slope'] > 0) & (df['close'] > df['ma_50'])
    prev_range_high = df['range_high'].shift(1)
    volume_ok = df['volume_ratio'] >= volume_threshold
    long_breakout = (df['high'] > prev_range_high) & volume_ok
    long_signal = plateau_was_active & long_trend & long_breakout
    df.loc[long_signal, 'signal'] = 1
    
    if show_signals:
        print(f"      Plateau candles: {df['plateau_active'].sum()}")
        print(f"      Long signals: {(df['signal'] == 1).sum()}")
    
    bt = Backtester(df, starting_capital=capital)
    results = bt.run()
    
    days = (df.index[-1] - df.index[0]).days if len(df) > 1 else 0
    weeks = max(days / 7, 0.1)
    total_trades = results.get('total_trades', 0)
    trades_per_week = total_trades / weeks if weeks > 0 else 0
    
    return StrategyResult(
        strategy='BREAKOUT', symbol=symbol, total_trades=total_trades,
        winners=results.get('winning_trades', 0), losers=results.get('losing_trades', 0),
        win_rate=results.get('win_rate', 0), total_pnl=results.get('total_pnl', 0),
        final_equity=results.get('final_capital', capital),
        return_pct=results.get('total_return_pct', 0),
        profit_factor=results.get('profit_factor', 0),
        max_drawdown=results.get('max_drawdown_pct', 0),
        sharpe_ratio=results.get('sharpe_ratio', 0),
        avg_winner=results.get('avg_winner', 0), avg_loser=results.get('avg_loser', 0),
        avg_r_multiple=results.get('avg_r_multiple', 0), trades_per_week=trades_per_week,
        days_tested=days, monthly_return=results.get('monthly_return_pct', 0),
        peak_capital=results.get('peak_capital', capital),
        exit_reasons=results.get('exit_reasons', {}),
        params={'range_threshold': range_threshold, 'volume_threshold': volume_threshold}
    ), bt


def auto_tune_breakout(df: pd.DataFrame, symbol: str, 
                       capital: float = STARTING_CAPITAL,
                       verbose: bool = True) -> Tuple[float, float, StrategyResult]:
    if verbose:
        print(f"\n   🔧 AUTO-TUNING PARAMETERS FOR {symbol}...")
        range_10pct = df['price_range'].quantile(0.10)
        range_20pct = df['price_range'].quantile(0.20)
        range_30pct = df['price_range'].quantile(0.30)
        print(f"      Testing parameter combinations...")
    else:
        range_10pct = df['price_range'].quantile(0.10)
        range_20pct = df['price_range'].quantile(0.20)
        range_30pct = df['price_range'].quantile(0.30)
    
    best_result = None
    best_params = None
    best_score = -999
    
    range_options = [range_10pct, range_20pct, range_30pct, 0.05, 0.06, 0.08, 0.10]
    volume_options = [1.0, 1.2, 1.3, 1.5]
    
    for rt in range_options:
        for vt in volume_options:
            result, _ = run_breakout_backtest(df.copy(), symbol, rt, vt, capital)
            trades = result.total_trades
            pf = result.profit_factor
            ret = result.return_pct
            dd = result.max_drawdown
            
            if trades >= 2:
                score = ret * 2 + (pf * 5) + (trades * 0.3) - (dd * 0.5)
            else:
                score = -100
            
            if score > best_score:
                best_score = score
                best_result = result
                best_params = {'range_threshold': rt, 'volume_threshold': vt}
    
    if best_params and verbose:
        print(f"\n      ✅ OPTIMAL: RT={best_params['range_threshold']*100:.2f}%, VT={best_params['volume_threshold']}")
    
    if best_params:
        return best_params['range_threshold'], best_params['volume_threshold'], best_result
    return range_20pct, 1.2, None


# =============================================================================
# MEAN REVERSION BACKTESTER
# =============================================================================

class MeanRevBacktester:
    """Backtest the MEAN_REV strategy."""
    
    def __init__(self, df: pd.DataFrame, symbol: str, 
                 capital: float = STARTING_CAPITAL,
                 rsi_oversold: int = 30, rsi_overbought: int = 70,
                 debug: bool = False):
        self.df = df.copy()
        self.symbol = symbol
        self.capital = capital
        self.starting_capital = capital
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.debug = debug
        
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [capital]
        self.peak_capital = capital
        self.debug_stats = {'bars_processed': 0, 'oversold_signals': 0, 'trades_opened': 0}
    
    def _add_indicators(self):
        df = self.df
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['momentum'] = df['close'].pct_change(periods=5) * 100
        self.df = df
    
    def run(self) -> StrategyResult:
        print(f"      Running MEAN_REV backtest on {len(self.df)} bars...")
        
        if 'bb_width' not in self.df.columns:
            self.df = add_all_indicators(self.df)
        self._add_indicators()
        
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'time' in self.df.columns:
                self.df['time'] = pd.to_datetime(self.df['time'])
                self.df.set_index('time', inplace=True)
            else:
                self.df.index = pd.to_datetime(self.df.index)
        
        warmup = 25
        position = None
        exit_reasons = {}
        winners = losers = 0
        total_pnl = 0
        max_dd = 0
        
        for i in range(warmup, len(self.df) - 1):
            self.debug_stats['bars_processed'] += 1
            current = self.df.iloc[i]
            previous = self.df.iloc[i - 1]
            current_price = current['close']
            
            if position:
                unrealized = (current_price - position['entry']) * position['size']
                current_equity = self.capital + unrealized
            else:
                current_equity = self.capital
            
            self.equity_curve.append(current_equity)
            self.peak_capital = max(self.peak_capital, current_equity)
            dd = (self.peak_capital - current_equity) / self.peak_capital if self.peak_capital > 0 else 0
            max_dd = max(max_dd, dd)
            
            # Check exits
            if position:
                exit_reason = exit_price = None
                if current['low'] <= position['stop_loss']:
                    exit_reason, exit_price = 'STOP_LOSS', position['stop_loss']
                elif current['high'] >= position['take_profit']:
                    exit_reason, exit_price = 'TAKE_PROFIT', position['take_profit']
                elif current['rsi'] > 50:
                    exit_reason, exit_price = 'RSI_NORMALIZED', current_price
                elif current_price >= current['bb_middle']:
                    exit_reason, exit_price = 'MEAN_REACHED', current_price
                
                if exit_reason:
                    pnl = (exit_price - position['entry']) * position['size']
                    commission = (position['entry'] * position['size'] * 0.0026) + (exit_price * position['size'] * 0.0026)
                    pnl -= commission
                    self.capital += pnl
                    total_pnl += pnl
                    if pnl > 0: winners += 1
                    else: losers += 1
                    exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
                    self.trades.append({'type': 'CLOSE', 'time': self.df.index[i], 'reason': exit_reason, 'entry': position['entry'], 'exit': exit_price, 'pnl': pnl})
                    position = None
                    continue
            
            # Check entry
            if position is None:
                rsi_oversold_now = current['rsi'] < self.rsi_oversold
                at_lower_bb = current['bb_position'] < 0.2
                not_freefall = current['momentum'] > -10.0
                rsi_recovering = current['rsi'] > previous['rsi']
                
                if rsi_oversold_now and at_lower_bb and not_freefall and rsi_recovering:
                    self.debug_stats['oversold_signals'] += 1
                    entry_price = current_price
                    atr = current['atr']
                    stop_loss = entry_price - (atr * 1.5)
                    take_profit = entry_price + (atr * 2.0)
                    
                    risk_amount = self.capital * 0.02
                    risk_per_unit = abs(entry_price - stop_loss)
                    size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                    max_size = (self.capital * 0.95) / entry_price
                    size = min(size, max_size)
                    
                    if size > 0:
                        position = {'entry': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit, 'size': size, 'time': self.df.index[i]}
                        self.debug_stats['trades_opened'] += 1
                        self.trades.append({'type': 'OPEN', 'time': self.df.index[i], 'entry': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit, 'size': size})
        
        # Close remaining
        if position:
            final_price = self.df.iloc[-1]['close']
            pnl = (final_price - position['entry']) * position['size']
            commission = (position['entry'] * position['size'] * 0.0026) + (final_price * position['size'] * 0.0026)
            pnl -= commission
            self.capital += pnl
            total_pnl += pnl
            if pnl > 0: winners += 1
            else: losers += 1
            exit_reasons['END'] = exit_reasons.get('END', 0) + 1
        
        total_trades = winners + losers
        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
        return_pct = ((self.capital / self.starting_capital) - 1) * 100
        
        if len(self.df) > 1:
            days = (self.df.index[-1] - self.df.index[0]).days
            weeks = max(days / 7, 0.1)
            trades_per_week = total_trades / weeks if weeks > 0 else 0
            monthly_return = (return_pct / days) * 30 if days > 0 else 0
        else:
            days = trades_per_week = monthly_return = 0
        
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0
        
        winning_pnl = sum(t['pnl'] for t in self.trades if t['type'] == 'CLOSE' and t['pnl'] > 0)
        losing_pnl = abs(sum(t['pnl'] for t in self.trades if t['type'] == 'CLOSE' and t['pnl'] < 0))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else 0
        
        return StrategyResult(
            strategy='MEAN_REV', symbol=self.symbol, total_trades=total_trades,
            winners=winners, losers=losers, win_rate=win_rate,
            total_pnl=total_pnl, final_equity=self.capital, return_pct=return_pct,
            profit_factor=round(profit_factor, 2), max_drawdown=max_dd * 100,
            sharpe_ratio=round(sharpe, 2),
            avg_winner=winning_pnl / winners if winners > 0 else 0,
            avg_loser=losing_pnl / losers if losers > 0 else 0,
            avg_r_multiple=0, trades_per_week=trades_per_week, days_tested=days,
            monthly_return=round(monthly_return, 2), peak_capital=round(self.peak_capital, 2),
            exit_reasons=exit_reasons,
            params={'rsi_oversold': self.rsi_oversold, 'rsi_overbought': self.rsi_overbought},
            debug_info=self.debug_stats
        )


def run_mean_rev_backtest(df: pd.DataFrame, symbol: str, 
                          capital: float = STARTING_CAPITAL,
                          rsi_oversold: int = 30, rsi_overbought: int = 70,
                          debug: bool = False) -> Tuple[StrategyResult, MeanRevBacktester]:
    bt = MeanRevBacktester(df, symbol, capital, rsi_oversold, rsi_overbought, debug)
    result = bt.run()
    return result, bt


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_data(symbol: str, timeframe: str, start_date: str = None, 
               end_date: str = None, days: int = None,
               exchange: str = EXCHANGE, from_file: str = None,
               save_data: bool = False) -> Optional[pd.DataFrame]:
    try:
        fetcher = DataFetcher(exchange_id=exchange)
        
        if from_file:
            return fetcher.load_from_csv(from_file)
        
        if days:
            if timeframe == '15m':
                candles_needed = min(days * 96, 720)
                df = fetcher.fetch_live(symbol, timeframe=timeframe, num_candles=candles_needed)
            else:
                start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                end = datetime.now().strftime('%Y-%m-%d')
                df = fetcher.fetch_historical(symbol, timeframe, start, end)
        else:
            df = fetcher.fetch_historical(symbol, timeframe, start_date, end_date)
        
        if df is None or len(df) == 0:
            print(f"      ❌ No data returned from {exchange}")
            print(f"      💡 Troubleshooting: Check symbol format (Kraken: ETH/USD, Binance: ETH/USDT)")
            return None
        
        if save_data:
            fetcher.save_to_csv(df, symbol=symbol, timeframe=timeframe)
        
        return df
        
    except Exception as e:
        print(f"      ❌ Error fetching {symbol}: {e}")
        return None


# =============================================================================
# MAIN BACKTESTING LOGIC
# =============================================================================

def run_single_asset(symbol: str, strategies: List[str], args) -> Dict[str, Tuple[StrategyResult, any]]:
    results = {}
    print(f"\n{'='*60}")
    print(f"📊 {symbol}")
    print(f"{'='*60}")
    
    asset_cfg = ASSET_CONFIGS.get(symbol, {})
    
    # Shared 4h data for BREAKOUT and MEAN_REV
    df_4h = None
    if 'breakout' in strategies or 'mean_rev' in strategies:
        timeframe = getattr(args, 'timeframe', None) or BREAKOUT_TIMEFRAME
        df_4h = fetch_data(symbol, timeframe, args.start, args.end,
                           exchange=args.exchange,
                           from_file=getattr(args, 'from_file', None),
                           save_data=getattr(args, 'save_data', False))
        
        if df_4h is not None and len(df_4h) >= 100:
            print(f"\n   📈 4h Data: {len(df_4h)} candles ({df_4h.index.min()} to {df_4h.index.max()})")
            df_4h = add_all_indicators(df_4h)
    
    # BREAKOUT
    if 'breakout' in strategies:
        print(f"\n   🔷 BREAKOUT Strategy ({BREAKOUT_TIMEFRAME})")
        if df_4h is not None and len(df_4h) >= 100:
            if args.auto_tune:
                rt, vt, result = auto_tune_breakout(df_4h, symbol, args.capital, verbose=True)
                if result is None:
                    result, bt = run_breakout_backtest(df_4h, symbol, RANGE_THRESHOLD, VOLUME_THRESHOLD, args.capital, show_signals=True)
                else:
                    result, bt = run_breakout_backtest(df_4h, symbol, rt, vt, args.capital, show_signals=True)
            else:
                rt = asset_cfg.get('range_threshold', RANGE_THRESHOLD)
                vt = asset_cfg.get('volume_threshold', VOLUME_THRESHOLD)
                result, bt = run_breakout_backtest(df_4h, symbol, rt, vt, args.capital, show_signals=args.debug)
            results['breakout'] = (result, bt)
            _print_strategy_result(result, args.debug)
        else:
            print(f"      ❌ Insufficient 4h data")
    
    # MEAN_REV
    if 'mean_rev' in strategies:
        print(f"\n   🔄 MEAN_REV Strategy ({MEAN_REV_TIMEFRAME})")
        if df_4h is not None and len(df_4h) >= 100:
            mr_params = MEAN_REV_PARAMS.get(symbol, {'rsi_oversold': 30, 'rsi_overbought': 70, 'enabled': True})
            if mr_params.get('enabled', True):
                result, bt = run_mean_rev_backtest(df_4h, symbol, args.capital,
                    rsi_oversold=mr_params['rsi_oversold'],
                    rsi_overbought=mr_params['rsi_overbought'], debug=args.debug)
                results['mean_rev'] = (result, bt)
                _print_strategy_result(result, args.debug)
            else:
                print(f"      ⚠️  MEAN_REV disabled for {symbol}")
        else:
            print(f"      ❌ Insufficient 4h data")
    
    # HYDRA
    if 'hydra' in strategies:
        print(f"\n   🐉 HYDRA Strategy ({HYDRA_TIMEFRAME})")
        hydra_days = getattr(args, 'days', 14) or 14
        df = fetch_data(symbol, HYDRA_TIMEFRAME, days=hydra_days, exchange=args.exchange)
        
        if df is not None and len(df) >= 200:
            actual_days = len(df) * 15 / (24 * 60)
            print(f"      Loaded {len(df)} candles (~{actual_days:.1f} days)")
            cfg = HydraConfig()
            bt = HydraBacktester(df, cfg, symbol, HYDRA_STARTING_CAPITAL, debug=args.debug)
            result = bt.run()
            results['hydra'] = (result, bt)
            _print_strategy_result(result, args.debug)
        else:
            print(f"      ❌ Insufficient data (need 200+, got {len(df) if df is not None else 0})")
    
    return results


def _print_strategy_result(result: StrategyResult, show_debug: bool = False):
    if result.total_trades == 0:
        print(f"      ⚠️  No trades")
        if show_debug and result.debug_info:
            d = result.debug_info
            print(f"      🔍 DEBUG: Bars={d.get('bars_processed', 0)}")
            if 'oversold_signals' in d:
                print(f"         Oversold signals: {d.get('oversold_signals', 0)}")
            if 'regime_failures' in d:
                print(f"         Regime failures: {d.get('regime_failures', 0)}, Quality filtered: {d.get('quality_filtered', 0)}")
        return
    
    pnl_str = ConsoleColors.green(f"${result.total_pnl:+,.2f}") if result.total_pnl > 0 else ConsoleColors.red(f"${result.total_pnl:+,.2f}")
    print(f"      Trades: {result.total_trades} ({result.winners}W/{result.losers}L)")
    print(f"      Win Rate: {result.win_rate:.1f}%")
    print(f"      P&L: {pnl_str} ({result.return_pct:+.2f}%)")
    print(f"      Max DD: {result.max_drawdown:.1f}%")
    if result.profit_factor > 0:
        print(f"      Profit Factor: {result.profit_factor:.2f}")
    if result.exit_reasons:
        exits_str = ", ".join(f"{r}:{c}" for r, c in result.exit_reasons.items())
        print(f"      Exits: {exits_str}")


def run_multi_asset(assets: List[str], strategies: List[str], args) -> CombinedResult:
    breakout_results, mean_rev_results, hydra_results = [], [], []
    breakout_backtesters = []
    
    for symbol in assets:
        results = run_single_asset(symbol, strategies, args)
        if 'breakout' in results:
            result, bt = results['breakout']
            breakout_results.append(result)
            breakout_backtesters.append((symbol, bt))
        if 'mean_rev' in results:
            result, bt = results['mean_rev']
            mean_rev_results.append(result)
        if 'hydra' in results:
            result, bt = results['hydra']
            hydra_results.append(result)
    
    if getattr(args, 'save_trades', False):
        for symbol, bt in breakout_backtesters:
            trade_log = bt.get_trade_log()
            if len(trade_log) > 0:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"logs/trades_{symbol.replace('/', '_')}_{timestamp}.csv"
                filepath = Path(__file__).parent / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                trade_log.to_csv(filepath, index=False)
                print(f"\n💾 Trade log saved: {filename}")
    
    return CombinedResult(
        breakout_results=breakout_results, mean_rev_results=mean_rev_results,
        hydra_results=hydra_results,
        aggregate_breakout=_aggregate_results(breakout_results, 'BREAKOUT'),
        aggregate_mean_rev=_aggregate_results(mean_rev_results, 'MEAN_REV'),
        aggregate_hydra=_aggregate_results(hydra_results, 'HYDRA')
    )


def _aggregate_results(results: List[StrategyResult], strategy_name: str) -> Dict:
    if not results:
        return {'strategy': strategy_name, 'message': 'No results'}
    
    valid = [r for r in results if r.total_trades > 0]
    if not valid:
        return {'strategy': strategy_name, 'message': 'No trades across all assets'}
    
    total_trades = sum(r.total_trades for r in valid)
    total_winners = sum(r.winners for r in valid)
    total_pnl = sum(r.total_pnl for r in valid)
    avg_win_rate = np.mean([r.win_rate for r in valid])
    avg_max_dd = np.mean([r.max_drawdown for r in valid])
    total_trades_per_week = sum(r.trades_per_week for r in valid)
    
    all_engines = {}
    for r in valid:
        for eng, data in r.by_engine.items():
            all_engines[eng] = all_engines.get(eng, 0) + data['count']
    
    all_exits = {}
    for r in valid:
        for reason, count in r.exit_reasons.items():
            all_exits[reason] = all_exits.get(reason, 0) + count
    
    return {
        'strategy': strategy_name, 'assets_tested': len(valid),
        'total_trades': total_trades, 'total_winners': total_winners,
        'total_losers': total_trades - total_winners, 'avg_win_rate': avg_win_rate,
        'total_pnl': total_pnl, 'avg_max_drawdown': avg_max_dd,
        'total_trades_per_week': total_trades_per_week,
        'by_engine': all_engines, 'exit_reasons': all_exits
    }


def print_aggregate_results(combined: CombinedResult):
    print("\n" + "=" * 70)
    print("📊 AGGREGATE RESULTS")
    print("=" * 70)
    
    for agg in [combined.aggregate_breakout, combined.aggregate_mean_rev, combined.aggregate_hydra]:
        if 'message' in agg:
            print(f"\n{agg['strategy']}: {agg['message']}")
            continue
        
        print(f"\n🔹 {agg['strategy']}")
        print(f"   Assets: {agg['assets_tested']} | Trades: {agg['total_trades']} ({agg['total_winners']}W/{agg['total_losers']}L)")
        print(f"   Win Rate: {agg['avg_win_rate']:.1f}%")
        pnl_str = ConsoleColors.green(f"${agg['total_pnl']:+,.2f}") if agg['total_pnl'] >= 0 else ConsoleColors.red(f"${agg['total_pnl']:+,.2f}")
        print(f"   P&L: {pnl_str} | Max DD: {agg['avg_max_drawdown']:.1f}%")
        
        if agg['exit_reasons']:
            exits_str = ", ".join(f"{r}:{c}" for r, c in sorted(agg['exit_reasons'].items(), key=lambda x: -x[1]))
            print(f"   Exits: {exits_str}")
    
    # Comparison table
    active = [(n, a) for n, a in [('BREAKOUT', combined.aggregate_breakout), 
                                   ('MEAN_REV', combined.aggregate_mean_rev),
                                   ('HYDRA', combined.aggregate_hydra)] if a.get('total_trades', 0) > 0]
    
    if len(active) >= 2:
        print("\n" + "-" * 70)
        print("📈 STRATEGY COMPARISON")
        print("-" * 70)
        header = f"{'Metric':<20}" + "".join(f" {n:>12}" for n, _ in active)
        print(header)
        print("-" * (20 + 13 * len(active)))
        for label, key in [('Trades', 'total_trades'), ('Win Rate', 'avg_win_rate'), ('P&L', 'total_pnl'), ('Max DD', 'avg_max_drawdown')]:
            row = f"{label:<20}"
            for _, agg in active:
                v = agg.get(key, 0)
                if key == 'total_pnl': row += f" ${v:>10,.0f}"
                elif key in ['avg_win_rate', 'avg_max_drawdown']: row += f" {v:>11.1f}%"
                else: row += f" {v:>12}"
            print(row)


def save_report(combined: CombinedResult, filename: str = None):
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"logs/backtest_report_{timestamp}.csv"
    
    all_results = []
    for r in combined.breakout_results + combined.mean_rev_results + combined.hydra_results:
        all_results.append({
            'strategy': r.strategy, 'symbol': r.symbol, 'total_trades': r.total_trades,
            'winners': r.winners, 'losers': r.losers, 'win_rate': r.win_rate,
            'total_pnl': r.total_pnl, 'final_equity': r.final_equity,
            'return_pct': r.return_pct, 'max_drawdown': r.max_drawdown,
            'profit_factor': r.profit_factor, 'sharpe_ratio': r.sharpe_ratio,
            'trades_per_week': r.trades_per_week, 'days_tested': r.days_tested,
            'range_threshold': r.params.get('range_threshold', ''),
            'volume_threshold': r.params.get('volume_threshold', ''),
            'rsi_oversold': r.params.get('rsi_oversold', ''),
            'rsi_overbought': r.params.get('rsi_overbought', '')
        })
    
    if all_results:
        df = pd.DataFrame(all_results)
        filepath = Path(__file__).parent / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"\n💾 Report saved to: {filename}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Combined Strategy Backtester - Test BREAKOUT, MEAN_REV, and HYDRA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest_combined.py                          # All strategies, all assets
  python backtest_combined.py --strategy breakout -a   # BREAKOUT with auto-tune
  python backtest_combined.py --strategy mean_rev      # MEAN_REV only
  python backtest_combined.py --strategy hydra --debug # HYDRA with debug info
  python backtest_combined.py --symbol SOL/USD         # Single asset
        """
    )
    
    parser.add_argument('--strategy', '-s', type=str, choices=['breakout', 'mean_rev', 'hydra', 'all'], default='all', help='Strategy to test (default: all)')
    parser.add_argument('--symbol', type=str, default=None, help='Single symbol to test')
    parser.add_argument('--assets', type=str, nargs='+', default=None, help='List of assets')
    parser.add_argument('--enabled-only', action='store_true', help='Only enabled assets')
    parser.add_argument('--start', type=str, default=BACKTEST_START, help=f'Start date (default: {BACKTEST_START})')
    parser.add_argument('--end', type=str, default=BACKTEST_END, help=f'End date (default: {BACKTEST_END})')
    parser.add_argument('--days', type=int, default=14, help='Days for HYDRA (default: 14)')
    parser.add_argument('--timeframe', '-t', type=str, default=None, help=f'Timeframe (default: {BREAKOUT_TIMEFRAME})')
    parser.add_argument('--exchange', '-e', type=str, default=EXCHANGE, help=f'Exchange (default: {EXCHANGE})')
    parser.add_argument('--from-file', '-f', type=str, default=None, help='Load from CSV')
    parser.add_argument('--capital', '-c', type=float, default=STARTING_CAPITAL, help=f'Capital (default: {STARTING_CAPITAL})')
    parser.add_argument('--auto-tune', '-a', action='store_true', help='Auto-tune BREAKOUT')
    parser.add_argument('--save-report', action='store_true', help='Save report to CSV')
    parser.add_argument('--save-trades', action='store_true', help='Save trades to CSV')
    parser.add_argument('--save-data', action='store_true', help='Save data to CSV')
    parser.add_argument('--debug', '-d', action='store_true', help='Show debug info')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.quiet:
        print_banner()
    
    strategies = ['breakout', 'mean_rev', 'hydra'] if args.strategy == 'all' else [args.strategy]
    
    if args.symbol:
        assets = [args.symbol]
    elif args.assets:
        assets = args.assets
    elif args.enabled_only:
        assets = [sym for sym, cfg in ASSET_CONFIGS.items() if cfg.get('enabled', True)]
    else:
        assets = list(ASSET_CONFIGS.keys())
    
    print(f"\n🔬 COMBINED BACKTEST")
    print(f"   Strategies: {', '.join(s.upper() for s in strategies)}")
    print(f"   Assets: {', '.join(assets)}")
    print(f"   Capital: ${args.capital:,.2f}")
    
    combined = run_multi_asset(assets, strategies, args)
    print_aggregate_results(combined)
    
    if args.save_report:
        save_report(combined)
    
    print("\n" + "=" * 70)
    print(ConsoleColors.green("✅ Backtest complete!"))
    print("=" * 70)
    
    return combined


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ Backtest interrupted.")
        sys.exit(130)
    except Exception as e:
        print(ConsoleColors.red(f"\n❌ Error: {e}"))
        import traceback
        traceback.print_exc()
        sys.exit(1)
