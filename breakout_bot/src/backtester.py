"""
Backtester Module
=================
Simulates strategy performance on historical data.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    STARTING_CAPITAL, RISK_PER_TRADE, COMMISSION, SLIPPAGE
)
from src.indicators import add_all_indicators
from src.signals import generate_signals
from src.risk_manager import (
    RiskManager, TradePosition, TradeDirection, ExitReason
)


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_percent: float
    r_multiple: float
    exit_reason: str
    bars_held: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'size': self.size,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'r_multiple': self.r_multiple,
            'exit_reason': self.exit_reason,
            'bars_held': self.bars_held
        }


class Backtester:
    """
    Simulates trading strategy on historical data.
    
    Handles:
    - Order execution with slippage and commission
    - Position tracking
    - Performance metrics calculation
    - Equity curve generation
    """
    
    def __init__(self, df: pd.DataFrame, starting_capital: float = STARTING_CAPITAL,
                 risk_per_trade: float = RISK_PER_TRADE,
                 commission: float = COMMISSION, slippage: float = SLIPPAGE):
        """
        Initialize the backtester.
        
        Args:
            df: DataFrame with OHLCV data (will add indicators if not present)
            starting_capital: Initial account balance
            risk_per_trade: Fraction of account to risk per trade
            commission: Commission per trade as decimal (0.001 = 0.1%)
            slippage: Slippage estimate as decimal (0.0005 = 0.05%)
        """
        self.starting_capital = starting_capital
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        
        # Prepare data
        self.df = self._prepare_data(df)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(starting_capital, risk_per_trade)
        
        # State tracking
        self.current_position: Optional[TradePosition] = None
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []
        self.current_bar_index: int = 0
        
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators and signals if not present."""
        df = df.copy()
        
        # Check if indicators need to be added
        if 'bb_width' not in df.columns:
            df = add_all_indicators(df)
        
        # Check if signals need to be added
        if 'signal' not in df.columns:
            df = generate_signals(df)
        
        return df
    
    def _apply_slippage(self, price: float, direction: TradeDirection,
                        is_entry: bool) -> float:
        """
        Apply slippage to price.
        
        For entries: slippage works against you
        For exits: slippage works against you
        
        Args:
            price: Base price
            direction: Trade direction
            is_entry: True if entry, False if exit
            
        Returns:
            Price with slippage applied
        """
        if direction == TradeDirection.LONG:
            if is_entry:
                return price * (1 + self.slippage)  # Buy higher
            else:
                return price * (1 - self.slippage)  # Sell lower
        else:
            if is_entry:
                return price * (1 - self.slippage)  # Sell lower
            else:
                return price * (1 + self.slippage)  # Buy higher
    
    def _apply_commission(self, trade_value: float) -> float:
        """Calculate commission cost."""
        return trade_value * self.commission
    
    def _enter_trade(self, row: pd.Series, bar_index: int):
        """
        Execute trade entry.
        
        Args:
            row: Current bar data
            bar_index: Index in DataFrame
        """
        direction = TradeDirection.LONG if row['signal'] == 1 else TradeDirection.SHORT
        
        # Apply slippage to entry
        entry_price = self._apply_slippage(row['close'], direction, is_entry=True)
        
        # Create position
        self.current_position = self.risk_manager.create_position(
            entry_price=entry_price,
            entry_time=row.name,
            atr=row['atr'],
            direction=direction
        )
        
        # Apply commission
        commission_cost = self._apply_commission(entry_price * self.current_position.size)
        self.risk_manager.update_capital(self.risk_manager.current_capital - commission_cost)
        
        self.current_bar_index = bar_index
    
    def _exit_trade(self, row: pd.Series, bar_index: int, exit_reason: ExitReason,
                    exit_price: float):
        """
        Execute trade exit.
        
        Args:
            row: Current bar data
            bar_index: Index in DataFrame
            exit_reason: Reason for exit
            exit_price: Price at which to exit
        """
        pos = self.current_position
        
        # Apply slippage to exit
        actual_exit = self._apply_slippage(exit_price, pos.direction, is_entry=False)
        
        # Calculate P&L
        if pos.direction == TradeDirection.LONG:
            raw_pnl = (actual_exit - pos.entry_price) * pos.size
        else:
            raw_pnl = (pos.entry_price - actual_exit) * pos.size
        
        # Apply exit commission
        commission_cost = self._apply_commission(actual_exit * pos.size)
        net_pnl = raw_pnl - commission_cost
        
        # Calculate metrics
        pnl_percent = (net_pnl / (pos.entry_price * pos.size)) * 100
        initial_risk = abs(pos.entry_price - pos.stop_loss)
        if pos.direction == TradeDirection.LONG:
            profit = actual_exit - pos.entry_price
        else:
            profit = pos.entry_price - actual_exit
        r_multiple = profit / initial_risk if initial_risk > 0 else 0
        
        # Record trade
        trade = TradeRecord(
            entry_time=pos.entry_time,
            exit_time=row.name,
            direction='LONG' if pos.direction == TradeDirection.LONG else 'SHORT',
            entry_price=pos.entry_price,
            exit_price=actual_exit,
            size=pos.size,
            pnl=net_pnl,
            pnl_percent=pnl_percent,
            r_multiple=r_multiple,
            exit_reason=exit_reason.value,
            bars_held=bar_index - self.current_bar_index
        )
        self.trades.append(trade)
        
        # Update capital
        self.risk_manager.update_capital(self.risk_manager.current_capital + net_pnl)
        
        # Clear position
        self.current_position = None
    
    def run(self) -> Dict:
        """
        Run the backtest.
        
        Returns:
            Dictionary of performance results
        """
        # Reset state
        self.current_position = None
        self.trades = []
        self.equity_curve = [self.starting_capital]
        self.risk_manager = RiskManager(self.starting_capital, self.risk_per_trade)
        
        # Skip first N bars to allow indicators to warm up
        start_bar = 50
        
        for i in range(start_bar, len(self.df)):
            row = self.df.iloc[i]
            
            # Check max drawdown
            if self.risk_manager.check_max_drawdown():
                print(f"Max drawdown exceeded at {row.name}. Stopping.")
                break
            
            # If in position, check for exit
            if self.current_position is not None:
                # Update trailing stop
                self.risk_manager.update_trailing_stop(self.current_position)
                
                # Check exit conditions
                should_exit, exit_reason, exit_price = self.risk_manager.check_exit(
                    self.current_position,
                    high=row['high'],
                    low=row['low'],
                    close=row['close']
                )
                
                if should_exit:
                    self._exit_trade(row, i, exit_reason, exit_price)
            
            # If not in position, check for entry signal
            if self.current_position is None:
                if row['signal'] != 0:
                    self._enter_trade(row, i)
            
            # Record equity
            if self.current_position is not None:
                unrealized = self.current_position.unrealized_pnl(row['close'])
                self.equity_curve.append(self.risk_manager.current_capital + unrealized)
            else:
                self.equity_curve.append(self.risk_manager.current_capital)
        
        # Close any open position at end
        if self.current_position is not None:
            last_row = self.df.iloc[-1]
            self._exit_trade(last_row, len(self.df) - 1, ExitReason.MANUAL, last_row['close'])
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        """
        Calculate and return performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'message': 'No trades executed'
            }
        
        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
        
        # Basic metrics
        total_trades = len(trades_df)
        winners = trades_df[trades_df['pnl'] > 0]
        losers = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0
        
        # R metrics
        avg_r = trades_df['r_multiple'].mean()
        
        # Profit factor
        gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Equity curve metrics
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max() * 100
        
        # Returns
        total_return = (equity[-1] - self.starting_capital) / self.starting_capital * 100
        
        # Trading days
        days = (self.df.index[-1] - self.df.index[0]).days
        monthly_return = (total_return / days) * 30 if days > 0 else 0
        
        # Sharpe ratio (simplified, assuming risk-free rate = 0)
        returns = np.diff(equity) / equity[:-1]
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Exit reason breakdown
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        # Long vs Short breakdown
        long_trades = trades_df[trades_df['direction'] == 'LONG']
        short_trades = trades_df[trades_df['direction'] == 'SHORT']
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': round(win_rate, 2),
            
            'total_pnl': round(total_pnl, 2),
            'avg_winner': round(avg_win, 2),
            'avg_loser': round(avg_loss, 2),
            'avg_r_multiple': round(avg_r, 2),
            'profit_factor': round(profit_factor, 2),
            
            'total_return_pct': round(total_return, 2),
            'monthly_return_pct': round(monthly_return, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 2),
            
            'final_capital': round(equity[-1], 2),
            'peak_capital': round(peak.max(), 2),
            
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'exit_reasons': exit_reasons,
            
            'avg_bars_held': round(trades_df['bars_held'].mean(), 1),
            'days_tested': days
        }
    
    def get_trade_log(self) -> pd.DataFrame:
        """
        Get detailed trade log.
        
        Returns:
            DataFrame of all trades
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([t.to_dict() for t in self.trades])
    
    def get_equity_curve(self) -> pd.Series:
        """
        Get equity curve as Series.
        
        Returns:
            Equity curve Series
        """
        # Align with DataFrame index
        start_idx = 50  # Warmup period
        index = self.df.index[start_idx - 1:start_idx - 1 + len(self.equity_curve)]
        
        return pd.Series(self.equity_curve, index=index)
    
    def print_results(self):
        """Print formatted results to console."""
        results = self.get_results()
        
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        
        if results.get('message'):
            print(f"\n{results['message']}")
            return
        
        print(f"\n📊 PERFORMANCE SUMMARY")
        print(f"   Total Return: {results['total_return_pct']}%")
        print(f"   Monthly Return: {results['monthly_return_pct']}%")
        print(f"   Max Drawdown: {results['max_drawdown_pct']}%")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']}")
        
        print(f"\n💰 CAPITAL")
        print(f"   Starting: ${self.starting_capital:,.2f}")
        print(f"   Final: ${results['final_capital']:,.2f}")
        print(f"   Peak: ${results['peak_capital']:,.2f}")
        print(f"   Total P&L: ${results['total_pnl']:,.2f}")
        
        print(f"\n📈 TRADES")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Winners: {results['winning_trades']}")
        print(f"   Losers: {results['losing_trades']}")
        print(f"   Win Rate: {results['win_rate']}%")
        print(f"   Avg R-Multiple: {results['avg_r_multiple']}")
        print(f"   Profit Factor: {results['profit_factor']}")
        
        print(f"\n📋 BREAKDOWN")
        print(f"   Long Trades: {results['long_trades']}")
        print(f"   Short Trades: {results['short_trades']}")
        print(f"   Avg Bars Held: {results['avg_bars_held']}")
        
        print(f"\n🚪 EXIT REASONS")
        for reason, count in results['exit_reasons'].items():
            print(f"   {reason}: {count}")
        
        print("\n" + "=" * 60)


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    print("Testing Backtester Module...")
    print("=" * 50)
    
    # Generate synthetic data with patterns
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range(start='2024-01-01', periods=n, freq='4h')
    
    # Create trending price with consolidations
    price = np.zeros(n)
    price[0] = 3000
    
    in_consolidation = False
    consolidation_start = 0
    
    for i in range(1, n):
        # Randomly enter/exit consolidation phases
        if not in_consolidation and np.random.random() < 0.02:
            in_consolidation = True
            consolidation_start = i
        
        if in_consolidation:
            # Low volatility during consolidation
            price[i] = price[i-1] + np.random.randn() * 5
            
            # Break out after 20-40 bars
            if i - consolidation_start > np.random.randint(20, 40):
                # Breakout move
                price[i] = price[i-1] + np.random.choice([-1, 1]) * np.random.uniform(30, 60)
                in_consolidation = False
        else:
            # Normal volatility
            trend = 0.1  # Slight upward bias
            price[i] = price[i-1] + np.random.randn() * 20 + trend
    
    # Build OHLCV
    df = pd.DataFrame({
        'open': price + np.random.randn(n) * 5,
        'high': price + abs(np.random.randn(n) * 15),
        'low': price - abs(np.random.randn(n) * 15),
        'close': price,
        'volume': np.random.randint(1000, 10000, n).astype(float)
    }, index=dates)
    
    # Run backtest
    bt = Backtester(df, starting_capital=10000)
    results = bt.run()
    
    # Print results
    bt.print_results()
    
    # Show sample trades
    trade_log = bt.get_trade_log()
    if len(trade_log) > 0:
        print("\nSample Trades:")
        print(trade_log.head(10).to_string())
