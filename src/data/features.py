"""
Calculo de features tecnicas a partir de velas (OHLCV).

Cada funcion recibe un DataFrame de klines y devuelve el mismo DataFrame
enriquecido con columnas adicionales.  El conjunto final de features se
usa como entrada al modelo predictivo.

Features incluidas:
- Indicadores de tendencia (SMA, EMA, MACD)
- Indicadores de momentum (RSI, Stochastic, ROC)
- Indicadores de volatilidad (Bollinger Bands, ATR)
- Indicadores de volumen (OBV, VWAP, volumen relativo)
- Features derivadas del precio
- Lag features a corto, medio y largo plazo
- Features temporales (hora del dia, dia de la semana)
- Features de BTC (correlacion, momentum del mercado)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta


# Minimo de filas necesarias para calcular todos los indicadores.
# La ventana mas grande es roc_72 (necesita 73 filas).
MIN_ROWS_FOR_FEATURES = 75


def compute_features(
    df: pd.DataFrame,
    btc_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calcula todas las features tecnicas sobre un DataFrame de klines.

    Requiere columnas: open, high, low, close, volume.
    Devuelve una copia con las features anadidas y sin filas NaN.
    Si el DataFrame tiene menos de MIN_ROWS_FOR_FEATURES filas,
    devuelve un DataFrame vacio (no hay suficientes datos para los indicadores).

    Parameters
    ----------
    df : pd.DataFrame
        Klines de la moneda objetivo.
    btc_df : pd.DataFrame | None
        Klines de BTCUSDT para calcular features de mercado.
        Si es None, las features de BTC no se calculan.
    """
    if len(df) < MIN_ROWS_FOR_FEATURES:
        return df.iloc[0:0].copy()

    df = df.copy()

    df = _add_trend_indicators(df)
    df = _add_momentum_indicators(df)
    df = _add_volatility_indicators(df)
    df = _add_volume_indicators(df)
    df = _add_price_change_features(df)
    df = _add_lag_features(df)
    df = _add_time_features(df)
    if btc_df is not None and len(btc_df) >= MIN_ROWS_FOR_FEATURES:
        df = _add_btc_features(df, btc_df)

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


# ------------------------------------------------------------------
# Lag features (corto, medio, largo plazo)
# ------------------------------------------------------------------

def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Anade features con retardos a distintos horizontes temporales."""
    # Corto plazo (1-6h): momentum reciente
    df["roc_6"] = df["close"].pct_change(periods=6)

    # Medio plazo (12-24h): tendencia intradia
    df["roc_12"] = df["close"].pct_change(periods=12)

    # Largo plazo (48-72h): tendencia de varios dias
    df["roc_48"] = df["close"].pct_change(periods=48)
    df["roc_72"] = df["close"].pct_change(periods=72)

    # RSI retardado (donde estaba el RSI hace unas horas)
    df["rsi_14_lag_6"] = df["rsi_14"].shift(6)
    df["rsi_14_lag_24"] = df["rsi_14"].shift(24)

    # Volatilidad retardada
    df["volatility_24_lag_24"] = df["volatility_24"].shift(24)

    # Volumen retardado
    df["vol_change_6h"] = df["volume"].pct_change(periods=6)
    df["vol_change_24h"] = df["volume"].pct_change(periods=24)
    df["vol_change_48h"] = df["volume"].pct_change(periods=48)

    return df


# ------------------------------------------------------------------
# Features temporales (patrones ciclicos)
# ------------------------------------------------------------------

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Anade hora del dia y dia de la semana como features ciclicas."""
    hour = df.index.hour
    dow = df.index.dayofweek

    # Codificacion sinusoidal para capturar ciclicidad
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return df


# ------------------------------------------------------------------
# Features de BTC (proxy del mercado global)
# ------------------------------------------------------------------

def _add_btc_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """Anade features de BTC como proxy del estado del mercado."""
    # Alinear precios de BTC con el indice de la moneda
    btc_close = btc_df["close"].reindex(df.index, method="ffill")

    # Momentum de BTC a distintos horizontes
    df["btc_roc_1"] = btc_close.pct_change(periods=1)
    df["btc_roc_6"] = btc_close.pct_change(periods=6)
    df["btc_roc_12"] = btc_close.pct_change(periods=12)
    df["btc_roc_24"] = btc_close.pct_change(periods=24)

    # RSI de BTC
    df["btc_rsi_14"] = ta.momentum.rsi(btc_close, window=14)

    # Volatilidad de BTC
    df["btc_volatility_24"] = (
        btc_close.rolling(window=24).std() / btc_close.rolling(window=24).mean()
    )

    # Correlacion moneda-BTC (rolling 24h)
    df["coin_btc_corr_24"] = df["close"].rolling(window=24).corr(btc_close)

    return df


# ------------------------------------------------------------------
# Lista de feature columns
# ------------------------------------------------------------------

def get_feature_columns(include_btc: bool = True) -> list[str]:
    """Devuelve la lista de nombres de columnas que se usan como features.

    Parameters
    ----------
    include_btc : bool
        Si True, incluye las features de BTC (requiere btc_df en compute_features).
    """
    base = [
        # Tendencia
        "sma_7", "sma_20", "sma_50", "ema_7", "ema_20",
        "macd", "macd_signal", "macd_diff",
        "price_vs_sma20", "price_vs_sma50", "sma7_vs_sma20",
        # Momentum
        "rsi_14", "stoch_k", "stoch_d",
        "roc_1", "roc_3", "roc_7", "roc_24",
        # Volatilidad
        "bb_upper", "bb_lower", "bb_width", "bb_pct",
        "atr_14", "volatility_7", "volatility_24",
        # Volumen
        "vol_sma_7", "vol_sma_24", "vol_relative_7", "vol_relative_24", "vol_change",
        "obv", "vwap",
        # Precio
        "high_low_pct", "close_open_pct",
        "higher_high", "lower_low",
        # Lags -- corto
        "roc_6",
        # Lags -- medio
        "roc_12",
        # Lags -- largo
        "roc_48", "roc_72",
        # Lags -- RSI retardado
        "rsi_14_lag_6", "rsi_14_lag_24",
        # Lags -- volatilidad retardada
        "volatility_24_lag_24",
        # Lags -- volumen retardado
        "vol_change_6h", "vol_change_24h", "vol_change_48h",
        # Temporales
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]

    if include_btc:
        base.extend([
            "btc_roc_1", "btc_roc_6", "btc_roc_12", "btc_roc_24",
            "btc_rsi_14", "btc_volatility_24",
            "coin_btc_corr_24",
        ])

    return base
