"""
Cálculo de features técnicas a partir de velas (OHLCV).

Cada función recibe un DataFrame de klines y devuelve el mismo DataFrame
enriquecido con columnas adicionales.  El conjunto final de features se
usa como entrada al modelo predictivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todas las features técnicas sobre un DataFrame de klines.

    Requiere columnas: open, high, low, close, volume.
    Devuelve una copia con las features añadidas y sin filas NaN.
    """
    df = df.copy()

    df = _add_trend_indicators(df)
    df = _add_momentum_indicators(df)
    df = _add_volatility_indicators(df)
    df = _add_volume_indicators(df)
    df = _add_price_change_features(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


# ------------------------------------------------------------------
# Indicadores de tendencia
# ------------------------------------------------------------------

def _add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["sma_7"] = ta.trend.sma_indicator(df["close"], window=7)
    df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)
    df["ema_7"] = ta.trend.ema_indicator(df["close"], window=7)
    df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)

    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    df["price_vs_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
    df["price_vs_sma50"] = (df["close"] - df["sma_50"]) / df["sma_50"]
    df["sma7_vs_sma20"] = (df["sma_7"] - df["sma_20"]) / df["sma_20"]

    return df


# ------------------------------------------------------------------
# Indicadores de momentum
# ------------------------------------------------------------------

def _add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
    df["stoch_k"] = ta.momentum.stoch(df["high"], df["low"], df["close"], window=14)
    df["stoch_d"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"], window=14)

    df["roc_1"] = df["close"].pct_change(periods=1)
    df["roc_3"] = df["close"].pct_change(periods=3)
    df["roc_7"] = df["close"].pct_change(periods=7)
    df["roc_24"] = df["close"].pct_change(periods=24)

    return df


# ------------------------------------------------------------------
# Indicadores de volatilidad
# ------------------------------------------------------------------

def _add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct"] = bb.bollinger_pband()

    df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["volatility_7"] = df["close"].rolling(window=7).std() / df["close"].rolling(window=7).mean()
    df["volatility_24"] = (
        df["close"].rolling(window=24).std() / df["close"].rolling(window=24).mean()
    )

    return df


# ------------------------------------------------------------------
# Indicadores de volumen
# ------------------------------------------------------------------

def _add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["vol_sma_7"] = df["volume"].rolling(window=7).mean()
    df["vol_sma_24"] = df["volume"].rolling(window=24).mean()
    df["vol_relative_7"] = df["volume"] / df["vol_sma_7"]
    df["vol_relative_24"] = df["volume"] / df["vol_sma_24"]
    df["vol_change"] = df["volume"].pct_change()

    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    return df


# ------------------------------------------------------------------
# Features derivadas del precio
# ------------------------------------------------------------------

def _add_price_change_features(df: pd.DataFrame) -> pd.DataFrame:
    df["high_low_pct"] = (df["high"] - df["low"]) / df["low"]
    df["close_open_pct"] = (df["close"] - df["open"]) / df["open"]

    df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
    df["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)

    return df


def get_feature_columns() -> list[str]:
    """Devuelve la lista de nombres de columnas que se usan como features."""
    return [
        "sma_7", "sma_20", "sma_50", "ema_7", "ema_20",
        "macd", "macd_signal", "macd_diff",
        "price_vs_sma20", "price_vs_sma50", "sma7_vs_sma20",
        "rsi_14", "stoch_k", "stoch_d",
        "roc_1", "roc_3", "roc_7", "roc_24",
        "bb_upper", "bb_lower", "bb_width", "bb_pct",
        "atr_14", "volatility_7", "volatility_24",
        "vol_sma_7", "vol_sma_24", "vol_relative_7", "vol_relative_24", "vol_change",
        "obv", "vwap",
        "high_low_pct", "close_open_pct",
        "higher_high", "lower_low",
    ]
