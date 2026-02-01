"""
Signals Module
==============
Detects consolidation patterns and generates entry signals.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    RANGE_THRESHOLD, MIN_CONSOLIDATION_CANDLES, VOLUME_THRESHOLD,
    LOOKBACK_PERIOD
)


def detect_plateau(df: pd.DataFrame, 
                   range_threshold: float = RANGE_THRESHOLD,
                   min_candles: int = MIN_CONSOLIDATION_CANDLES,
                   bb_width_percentile_threshold: float = 25.0) -> pd.Series:
    """
    Detect consolidation/plateau conditions.
    
    A plateau is detected when:
    1. BB width is in the lowest 25% of its recent range (squeeze)
    2. Price range over lookback period is below threshold
    
    Args:
        df: DataFrame with indicators
        range_threshold: Maximum price range for consolidation (e.g., 0.02 = 2%)
        min_candles: Minimum candles required in consolidation
        bb_width_percentile_threshold: BB width must be below this percentile (default 25%)
        
    Returns:
        Boolean series (True = plateau detected)
    """
    # Condition 1: BB width is in low percentile (squeeze)
    # Use percentile-based detection instead of exact minimum
    bb_squeeze = df['bb_width_percentile'] < bb_width_percentile_threshold
    
    # Condition 2: Price range is tight
    tight_range = df['price_range'] < range_threshold
    
    # Combine conditions
    plateau = bb_squeeze & tight_range
    
    return plateau


def check_trend_alignment(df: pd.DataFrame, direction: str = 'long') -> pd.Series:
    """
    Check if trend aligns with trade direction.
    
    For LONG: MA should be sloping up, price above MA
    For SHORT: MA should be sloping down, price below MA
    
    Args:
        df: DataFrame with indicators
        direction: 'long' or 'short'
        
    Returns:
        Boolean series (True = trend aligned)
    """
    if direction.lower() == 'long':
        # MA sloping up and price above MA
        trend_aligned = (df['ma_slope'] > 0) & (df['close'] > df['ma_50'])
    else:
        # MA sloping down and price below MA
        trend_aligned = (df['ma_slope'] < 0) & (df['close'] < df['ma_50'])
    
    return trend_aligned


def detect_breakout(df: pd.DataFrame, direction: str = 'long',
                    volume_threshold: float = VOLUME_THRESHOLD) -> pd.Series:
    """
    Detect breakout from consolidation range.
    
    LONG breakout: Candle closes above range high with volume
    SHORT breakout: Candle closes below range low with volume
    
    Args:
        df: DataFrame with indicators
        direction: 'long' or 'short'
        volume_threshold: Minimum volume ratio (e.g., 1.5 = 150% of average)
        
    Returns:
        Boolean series (True = breakout detected)
    """
    # Get the previous candle's range boundaries (not current, to avoid lookahead)
    prev_range_high = df['range_high'].shift(1)
    prev_range_low = df['range_low'].shift(1)
    
    # Volume confirmation
    volume_confirmed = df['volume_ratio'] >= volume_threshold
    
    if direction.lower() == 'long':
        # High touched above the previous range high (intrabar breakout)
        price_breakout = df['high'] > prev_range_high
    else:
        # Low touched below the previous range low (intrabar breakout)
        price_breakout = df['low'] < prev_range_low
    
    breakout = price_breakout & volume_confirmed
    
    return breakout


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on all conditions.
    
    Signal values:
    - 1 = Long entry
    - -1 = Short entry
    - 0 = No signal
    
    Args:
        df: DataFrame with indicators
        
    Returns:
        DataFrame with 'signal' column added
    """
    df = df.copy()
    
    # Initialize signal column
    df['signal'] = 0
    
    # Track if we're in a plateau state
    df['plateau_active'] = detect_plateau(df)
    
    # Detect long setups
    long_trend = check_trend_alignment(df, 'long')
    long_breakout = detect_breakout(df, 'long')
    
    # Plateau must have been active on the PREVIOUS candle
    # (breakout candle exits the plateau)
    plateau_was_active = df['plateau_active'].shift(1).fillna(False)
    
    # Long signal: plateau was forming, now breaking out with trend
    long_signal = plateau_was_active & long_trend & long_breakout
    
    # Detect short setups
    short_trend = check_trend_alignment(df, 'short')
    short_breakout = detect_breakout(df, 'short')
    
    # Short signal: plateau was forming, now breaking down with trend
    short_signal = plateau_was_active & short_trend & short_breakout
    
    # Apply signals
    df.loc[long_signal, 'signal'] = 1
    df.loc[short_signal, 'signal'] = -1
    
    # Add signal metadata for analysis
    df['long_trend_ok'] = long_trend
    df['short_trend_ok'] = short_trend
    df['long_breakout'] = long_breakout
    df['short_breakout'] = short_breakout
    
    return df


def get_signal_details(df: pd.DataFrame, index: int) -> dict:
    """
    Get detailed information about a specific signal.
    
    Args:
        df: DataFrame with signals
        index: Row index to examine
        
    Returns:
        Dictionary with signal details
    """
    row = df.iloc[index]
    
    return {
        'timestamp': df.index[index],
        'signal': 'LONG' if row['signal'] == 1 else 'SHORT' if row['signal'] == -1 else 'NONE',
        'price': row['close'],
        'volume_ratio': row['volume_ratio'],
        'bb_width': row['bb_width'],
        'bb_squeeze': row['bb_squeeze'],
        'price_range': row['price_range'],
        'ma_slope': row['ma_slope'],
        'range_high': row['range_high'],
        'range_low': row['range_low'],
        'atr': row['atr'],
        'trend': 'UP' if row['ma_slope'] > 0 else 'DOWN',
    }


def count_signals(df: pd.DataFrame) -> dict:
    """
    Count total signals in the dataset.
    
    Args:
        df: DataFrame with signals
        
    Returns:
        Dictionary with signal counts
    """
    return {
        'total': len(df),
        'long_signals': (df['signal'] == 1).sum(),
        'short_signals': (df['signal'] == -1).sum(),
        'plateau_candles': df['plateau_active'].sum(),
        'signal_rate': ((df['signal'] != 0).sum() / len(df)) * 100
    }


def find_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all rows where a signal occurred.
    
    Args:
        df: DataFrame with signals
        
    Returns:
        DataFrame containing only signal rows
    """
    return df[df['signal'] != 0].copy()


def validate_signal_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quality metrics to signals for filtering.
    
    Higher quality signals have:
    - Tighter consolidation
    - Stronger volume confirmation
    - Clearer trend alignment
    
    Args:
        df: DataFrame with signals
        
    Returns:
        DataFrame with 'signal_quality' column (0-100)
    """
    df = df.copy()
    
    # Only calculate for signals
    signal_mask = df['signal'] != 0
    
    # Initialize quality score
    df['signal_quality'] = 0
    
    if signal_mask.any():
        # Volume score (higher is better, cap at 3x)
        volume_score = (df.loc[signal_mask, 'volume_ratio'].clip(upper=3) / 3) * 30
        
        # Squeeze tightness score (lower BB width percentile is better)
        squeeze_score = (100 - df.loc[signal_mask, 'bb_width_percentile'].fillna(50)) * 0.3
        
        # Range tightness score (tighter range is better)
        range_score = (1 - (df.loc[signal_mask, 'price_range'] / RANGE_THRESHOLD).clip(upper=1)) * 20
        
        # Trend clarity score (stronger slope is better)
        ma_slope_abs = df.loc[signal_mask, 'ma_slope'].abs()
        trend_score = (ma_slope_abs / ma_slope_abs.max()).fillna(0) * 20 if ma_slope_abs.max() > 0 else 0
        
        df.loc[signal_mask, 'signal_quality'] = volume_score + squeeze_score + range_score + trend_score
    
    return df


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    print("Testing Signals Module...")
    print("=" * 50)
    
    # Import indicators to prepare test data
    from indicators import add_all_indicators
    
    # Generate synthetic price data with a consolidation pattern
    np.random.seed(42)
    n = 200
    
    dates = pd.date_range(start='2024-01-01', periods=n, freq='4h')
    
    # Create price with consolidation then breakout pattern
    price = np.zeros(n)
    price[0] = 3000
    
    for i in range(1, n):
        if 50 <= i < 80:  # Consolidation phase
            price[i] = price[i-1] + np.random.randn() * 5  # Low volatility
        elif i == 80:  # Breakout
            price[i] = price[i-1] + 50  # Big move up
        else:
            price[i] = price[i-1] + np.random.randn() * 20  # Normal volatility
    
    # Build OHLCV DataFrame
    df = pd.DataFrame({
        'open': price + np.random.randn(n) * 5,
        'high': price + abs(np.random.randn(n) * 10),
        'low': price - abs(np.random.randn(n) * 10),
        'close': price,
        'volume': np.where(
            (np.arange(n) >= 50) & (np.arange(n) < 80),
            np.random.randint(500, 1500, n),  # Low volume in consolidation
            np.random.randint(1000, 5000, n)  # Normal volume
        ).astype(float)
    }, index=dates)
    
    # Spike volume on breakout
    df.iloc[80, df.columns.get_loc('volume')] = 10000
    
    # Add indicators
    df = add_all_indicators(df)
    
    # Generate signals
    df = generate_signals(df)
    
    # Add quality scores
    df = validate_signal_quality(df)
    
    # Print results
    print("\nSignal Summary:")
    summary = count_signals(df)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\nAll Signals Found:")
    signals = find_all_signals(df)
    if len(signals) > 0:
        for idx in signals.index:
            details = get_signal_details(df, df.index.get_loc(idx))
            print(f"\n  {details['timestamp']}")
            print(f"    Signal: {details['signal']}")
            print(f"    Price: {details['price']:.2f}")
            print(f"    Volume Ratio: {details['volume_ratio']:.2f}x")
            print(f"    Trend: {details['trend']}")
    else:
        print("  No signals detected in test data")
    
    print("\nPlateau periods detected:", df['plateau_active'].sum(), "candles")
