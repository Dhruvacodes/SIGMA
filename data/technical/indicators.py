"""
SIGMA Technical Analysis Engine.
Computes indicators, detects breakouts, and analyzes support/resistance levels.
Serverless-optimized: No heavy dependencies (numpy/scipy/pandas-ta optional).
"""

from typing import Any

import pandas as pd


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append technical indicators to the DataFrame.

    Args:
        df: DataFrame with columns [Date, Open, High, Low, Close, Volume]

    Returns:
        DataFrame with all indicator columns appended.
    """
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()

    # RSI
    df["RSI_14"] = _calculate_rsi(df["Close"], 14)

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_line"] = exp1 - exp2
    df["MACD_signal"] = df["MACD_line"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD_line"] - df["MACD_signal"]

    # Bollinger Bands
    df["BB_mid"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + (std * 2)
    df["BB_lower"] = df["BB_mid"] - (std * 2)

    # SMAs
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    # Volume analysis
    df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"]

    return df


def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Manual RSI calculation."""
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
    if len(df) < 253:
        return None

    rolling_max = df["Close"].iloc[:-1].rolling(252, min_periods=200).max()

    if rolling_max.empty or pd.isna(rolling_max.iloc[-1]):
        return None

    prev_high = rolling_max.iloc[-1]
    current_close = df["Close"].iloc[-1]

    if current_close <= prev_high:
        return None

    breakout_pct = ((current_close - prev_high) / prev_high) * 100

    volume_ratio = df["Volume_Ratio"].iloc[-1] if "Volume_Ratio" in df.columns else 1.0
    if pd.isna(volume_ratio):
        volume_ratio = 1.0

    if volume_ratio < 1.5:
        return None

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
    Pure pandas implementation without scipy.

    Args:
        df: DataFrame with OHLCV data.
        tolerance_pct: Percentage band for clustering price levels.

    Returns:
        Dict with support_levels and resistance_levels lists.
    """
    if len(df) < 20:
        return {"support_levels": [], "resistance_levels": []}

    closes = df["Close"].tolist()
    order = 5

    # Find local maxima (resistance)
    local_max_idx = []
    for i in range(order, len(closes) - order):
        if all(closes[i] > closes[i - j] for j in range(1, order + 1)) and \
           all(closes[i] > closes[i + j] for j in range(1, order + 1)):
            local_max_idx.append(i)

    # Find local minima (support)
    local_min_idx = []
    for i in range(order, len(closes) - order):
        if all(closes[i] < closes[i - j] for j in range(1, order + 1)) and \
           all(closes[i] < closes[i + j] for j in range(1, order + 1)):
            local_min_idx.append(i)

    def cluster_levels(indices: list, prices: list) -> list[dict]:
        """Cluster price levels within tolerance band."""
        if len(indices) == 0:
            return []

        levels = [prices[i] for i in indices]
        clusters = []
        used = set()

        for i, level in enumerate(levels):
            if i in used:
                continue

            tolerance = level * (tolerance_pct / 100)
            cluster_prices = []

            for j, other_level in enumerate(levels):
                if j not in used and abs(other_level - level) <= tolerance:
                    cluster_prices.append(other_level)
                    used.add(j)

            if cluster_prices:
                avg_price = sum(cluster_prices) / len(cluster_prices)
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
        forward_returns = []

        for i in range(252, len(df) - 30):
            subset = df.iloc[: i + 1].copy()
            subset = compute_all_indicators(subset)
            breakout = detect_52w_breakout(subset)

            if breakout:
                future_close = df["Close"].iloc[min(i + 30, len(df) - 1)]
                current_close = df["Close"].iloc[i]
                forward_return = ((future_close - current_close) / current_close) * 100
                forward_returns.append(forward_return)

        if len(forward_returns) < 5:
            return {"signal_type": signal_type, "occurrences": 0, "insufficient_data": True}

        sorted_returns = sorted(forward_returns)
        positive_rate = sum(1 for r in forward_returns if r > 0) / len(forward_returns)
        median_idx = len(sorted_returns) // 2
        median_return = sorted_returns[median_idx]
        worst_quartile_idx = len(sorted_returns) // 4
        worst_quartile = sorted_returns[worst_quartile_idx]

        return {
            "signal_type": signal_type,
            "occurrences": len(forward_returns),
            "positive_rate": float(positive_rate),
            "median_return": float(median_return),
            "worst_quartile": float(worst_quartile),
        }

    return {"signal_type": signal_type, "occurrences": 0, "insufficient_data": True}
