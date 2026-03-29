"""
SIGMA Technical Analysis Engine.
Computes indicators, detects breakouts, and analyzes support/resistance levels.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append technical indicators to the DataFrame using pandas-ta.

    Args:
        df: DataFrame with columns [Date, Open, High, Low, Close, Volume]

    Returns:
        DataFrame with all indicator columns appended.
    """
    # Ensure we have the required columns
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Make a copy to avoid modifying original
    df = df.copy()

    try:
        import pandas_ta as ta

        # RSI
        df["RSI_14"] = ta.rsi(df["Close"], length=14)

        # MACD
        macd = ta.macd(df["Close"])
        if macd is not None and not macd.empty:
            df["MACD_line"] = macd.iloc[:, 0]
            df["MACD_signal"] = macd.iloc[:, 1]
            df["MACD_hist"] = macd.iloc[:, 2]

        # Bollinger Bands
        bbands = ta.bbands(df["Close"])
        if bbands is not None and not bbands.empty:
            df["BB_lower"] = bbands.iloc[:, 0]
            df["BB_mid"] = bbands.iloc[:, 1]
            df["BB_upper"] = bbands.iloc[:, 2]

        # SMAs
        df["SMA_20"] = ta.sma(df["Close"], length=20)
        df["SMA_50"] = ta.sma(df["Close"], length=50)
        df["SMA_200"] = ta.sma(df["Close"], length=200)

        # ATR
        df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    except ImportError:
        # Fallback to manual calculation if pandas-ta not available
        df["RSI_14"] = _calculate_rsi(df["Close"], 14)
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean()

    # Volume analysis (always use manual calc for consistency)
    df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"]

    return df


def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Manual RSI calculation as fallback."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def detect_52w_breakout(df: pd.DataFrame) -> dict[str, Any] | None:
    """
    Detect 52-week (252 trading day) breakout.

    Args:
        df: DataFrame with OHLCV data and computed indicators.

    Returns:
        Dict with breakout details, or None if no breakout detected.
    """
    if len(df) < 253:  # Need at least 252 + 1 days
        return None

    # Compute 252-day rolling max (excluding current bar)
    rolling_max = df["Close"].iloc[:-1].rolling(252, min_periods=200).max()

    if rolling_max.empty or pd.isna(rolling_max.iloc[-1]):
        return None

    prev_high = rolling_max.iloc[-1]
    current_close = df["Close"].iloc[-1]

    # Check if current close exceeds previous 252-day max
    if current_close <= prev_high:
        return None

    breakout_pct = ((current_close - prev_high) / prev_high) * 100

    # Volume confirmation
    volume_ratio = df["Volume_Ratio"].iloc[-1] if "Volume_Ratio" in df.columns else 1.0
    if pd.isna(volume_ratio):
        volume_ratio = 1.0

    # Only confirm breakout if volume is elevated
    if volume_ratio < 1.5:
        return None

    # Calculate strength
    strength = min(1.0, (volume_ratio - 1.0) * 0.3 + breakout_pct * 2 / 100)

    return {
        "breakout": True,
        "prev_high": float(prev_high),
        "breakout_pct": float(breakout_pct),
        "volume_ratio": float(volume_ratio),
        "strength": float(strength),
    }


def check_rsi_status(df: pd.DataFrame) -> dict[str, Any]:
    """
    Check RSI status for overbought/oversold conditions.

    Args:
        df: DataFrame with RSI_14 column.

    Returns:
        Dict with RSI status and strength.
    """
    if "RSI_14" not in df.columns or df["RSI_14"].empty:
        return {"rsi": 50.0, "status": "neutral", "strength": 0.0}

    rsi = df["RSI_14"].iloc[-1]
    if pd.isna(rsi):
        return {"rsi": 50.0, "status": "neutral", "strength": 0.0}

    rsi = float(rsi)

    if rsi > 70:
        status = "overbought"
        strength = min(1.0, (rsi - 70) / 30)
    elif rsi < 30:
        status = "oversold"
        strength = min(1.0, (30 - rsi) / 30)
    else:
        status = "neutral"
        strength = 0.0

    return {"rsi": rsi, "status": status, "strength": float(strength)}


def detect_support_resistance(
    df: pd.DataFrame, tolerance_pct: float = 0.5
) -> dict[str, list[dict]]:
    """
    Detect support and resistance levels using local extrema.

    Args:
        df: DataFrame with OHLCV data.
        tolerance_pct: Percentage band for clustering price levels.

    Returns:
        Dict with support_levels and resistance_levels lists.
    """
    if len(df) < 20:
        return {"support_levels": [], "resistance_levels": []}

    closes = df["Close"].values

    # Find local maxima (resistance) and minima (support)
    order = 5
    local_max_idx = argrelextrema(closes, np.greater, order=order)[0]
    local_min_idx = argrelextrema(closes, np.less, order=order)[0]

    def cluster_levels(indices: np.ndarray, prices: np.ndarray) -> list[dict]:
        """Cluster price levels within tolerance band."""
        if len(indices) == 0:
            return []

        levels = prices[indices]
        clusters = []
        used = set()

        for i, level in enumerate(levels):
            if i in used:
                continue

            # Find all levels within tolerance
            tolerance = level * (tolerance_pct / 100)
            cluster_indices = []
            cluster_prices = []

            for j, other_level in enumerate(levels):
                if j not in used and abs(other_level - level) <= tolerance:
                    cluster_indices.append(j)
                    cluster_prices.append(other_level)
                    used.add(j)

            if cluster_prices:
                avg_price = np.mean(cluster_prices)
                touches = len(cluster_prices)
                strength = "confirmed" if touches >= 3 else "weak"

                clusters.append(
                    {"price": float(avg_price), "touches": touches, "strength": strength}
                )

        return sorted(clusters, key=lambda x: x["price"], reverse=True)

    resistance_levels = cluster_levels(local_max_idx, closes)
    support_levels = cluster_levels(local_min_idx, closes)

    return {"support_levels": support_levels, "resistance_levels": resistance_levels}


def compute_historical_pattern_success_rate(
    df: pd.DataFrame, signal_type: str, lookback_days: int = 756
) -> dict[str, Any]:
    """
    Compute historical success rate for a given signal type.

    Args:
        df: DataFrame with OHLCV data (must have sufficient history).
        signal_type: Type of signal to analyze (e.g., "52W_BREAKOUT").
        lookback_days: Number of days to look back for pattern analysis.

    Returns:
        Dict with success rate statistics.
    """
    if len(df) < lookback_days or len(df) < 300:
        return {"signal_type": signal_type, "occurrences": 0, "insufficient_data": True}

    if signal_type == "52W_BREAKOUT":
        breakout_dates = []
        forward_returns = []

        # Scan through history for breakouts
        for i in range(252, len(df) - 30):
            subset = df.iloc[: i + 1].copy()
            subset = compute_all_indicators(subset)
            breakout = detect_52w_breakout(subset)

            if breakout:
                breakout_dates.append(i)
                # Compute 30-day forward return
                future_close = df["Close"].iloc[min(i + 30, len(df) - 1)]
                current_close = df["Close"].iloc[i]
                forward_return = ((future_close - current_close) / current_close) * 100
                forward_returns.append(forward_return)

        if len(forward_returns) < 5:
            return {"signal_type": signal_type, "occurrences": 0, "insufficient_data": True}

        returns_arr = np.array(forward_returns)
        positive_rate = np.mean(returns_arr > 0)
        median_return = float(np.median(returns_arr))
        worst_quartile = float(np.percentile(returns_arr, 25))

        return {
            "signal_type": signal_type,
            "occurrences": len(forward_returns),
            "positive_rate": float(positive_rate),
            "median_return": median_return,
            "worst_quartile": worst_quartile,
        }

    return {"signal_type": signal_type, "occurrences": 0, "insufficient_data": True}
