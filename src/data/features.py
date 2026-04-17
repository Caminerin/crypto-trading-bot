"""
Calculo de features tecnicas a partir de velas (OHLCV).

Cada funcion recibe un DataFrame de klines y devuelve el mismo DataFrame
enriquecido con columnas adicionales.  El conjunto final de features se
usa como entrada al modelo predictivo.

Features incluidas:
- Indicadores de tendencia (SMA, EMA, MACD)
- Indicadores de momentum (RSI, Stochastic, ROC, MFI)
- Indicadores de volatilidad (Bollinger Bands, ATR)
- Indicadores de volumen (OBV, VWAP, volumen relativo)
- Features derivadas del precio y soporte/resistencia
- Lag features a corto, medio y largo plazo
- Features temporales (hora del dia, dia de la semana)
- Features de BTC (correlacion, momentum del mercado, dominancia)
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
    market_df: pd.DataFrame | None = None,
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
    market_df : pd.DataFrame | None
        DataFrame pre-calculado con features de mercado global
        (market_pct_up_24, market_mean_roc_24, etc.).
        Si es None, las features de mercado global no se añaden.
    """
    if len(df) < MIN_ROWS_FOR_FEATURES:
        return df.iloc[0:0].copy()

    df = df.copy()

    df = _add_trend_indicators(df)
    df = _add_momentum_indicators(df)
    df = _add_volatility_indicators(df)
    df = _add_volume_indicators(df)
    df = _add_price_change_features(df)
    df = _add_microstructure_features(df)
    df = _add_support_resistance_features(df)
    df = _add_lag_features(df)
    df = _add_time_features(df)
    if btc_df is not None and len(btc_df) >= MIN_ROWS_FOR_FEATURES:
        df = _add_btc_features(df, btc_df)

    if market_df is not None:
        df = _add_market_features(df, market_df)

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

    # Money Flow Index (combina precio + volumen)
    df["mfi_14"] = ta.volume.money_flow_index(
        df["high"], df["low"], df["close"], df["volume"], window=14,
    )

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
    # VWAP rolling 24h (evita sesgo acumulativo que depende del inicio de ventana)
    cum_pv = (df["close"] * df["volume"]).rolling(window=24, min_periods=1).sum()
    cum_vol = df["volume"].rolling(window=24, min_periods=1).sum()
    df["vwap"] = np.where(cum_vol > 0, cum_pv / cum_vol, df["close"])

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
# Features de microestructura (velas + volumen)
# ------------------------------------------------------------------

def _add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de microestructura derivadas de velas y volumen."""
    # Volume spike: ratio volumen actual vs media 3h
    vol_ma3 = df["volume"].rolling(window=3).mean()
    df["volume_spike_3h"] = np.where(vol_ma3 > 0, df["volume"] / vol_ma3, 1.0)

    # Price acceleration: segunda derivada del precio
    roc1 = df["close"].pct_change()
    df["price_acceleration"] = roc1 - roc1.shift(1)

    # Sombras de la vela (upper/lower shadow)
    body_high = df[["open", "close"]].max(axis=1)
    body_low = df[["open", "close"]].min(axis=1)
    df["upper_shadow_pct"] = (df["high"] - body_high) / df["close"]
    df["lower_shadow_pct"] = (body_low - df["low"]) / df["close"]

    # Cuerpo de la vela (body size)
    df["body_pct"] = (df["close"] - df["open"]).abs() / df["close"]

    # Velas verdes consecutivas
    is_green = (df["close"] > df["open"]).astype(int)
    groups = (is_green != is_green.shift()).cumsum()
    df["consecutive_up"] = is_green.groupby(groups).cumsum()

    # RSI divergence: precio sube pero RSI baja (senal bajista)
    price_up = df["close"].diff() > 0
    rsi_down = df["rsi_14"].diff() < 0
    df["rsi_divergence"] = (price_up & rsi_down).astype(int)

    # Volume-price trend: correlacion rolling entre cambio de precio y volumen
    df["volume_price_trend"] = (
        df["close"].pct_change()
        .rolling(window=12)
        .corr(df["volume"].pct_change())
    )

    return df


# ------------------------------------------------------------------
# Soporte / Resistencia dinamico
# ------------------------------------------------------------------

def _add_support_resistance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Distancia del precio actual a maximos/minimos recientes.

    Usa ventanas de 24h y 72h (3 dias) para ser compatible con
    el lookback de 120h usado en prediccion.
    """
    # Maximos y minimos rolling
    high_24h = df["high"].rolling(window=24).max()
    low_24h = df["low"].rolling(window=24).min()
    high_72h = df["high"].rolling(window=72).max()
    low_72h = df["low"].rolling(window=72).min()

    # Distancia relativa al soporte/resistencia
    df["dist_resistance_24h"] = (high_24h - df["close"]) / df["close"]
    df["dist_support_24h"] = (df["close"] - low_24h) / df["close"]
    df["dist_resistance_72h"] = (high_72h - df["close"]) / df["close"]
    df["dist_support_72h"] = (df["close"] - low_72h) / df["close"]

    # Posicion relativa dentro del rango (0=en minimo, 1=en maximo)
    range_24h = high_24h - low_24h
    df["range_position_24h"] = np.where(
        range_24h > 0, (df["close"] - low_24h) / range_24h, 0.5,
    )
    range_72h = high_72h - low_72h
    df["range_position_72h"] = np.where(
        range_72h > 0, (df["close"] - low_72h) / range_72h, 0.5,
    )

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

    # Dominancia BTC proxy: ratio de volatilidad BTC vs moneda
    # Cuando BTC es menos volatil que altcoins, las altcoins tienden a moverse mas
    coin_vol = df["close"].rolling(window=24).std() / df["close"].rolling(window=24).mean()
    btc_vol = btc_close.rolling(window=24).std() / btc_close.rolling(window=24).mean()
    df["btc_dominance_proxy"] = np.where(coin_vol > 0, btc_vol / coin_vol, 1.0)

    # Performance relativa moneda vs BTC (rendimiento de la moneda menos el de BTC)
    coin_ret_24 = df["close"].pct_change(periods=24)
    btc_ret_24 = btc_close.pct_change(periods=24)
    df["relative_perf_btc_24"] = coin_ret_24 - btc_ret_24

    return df


# ------------------------------------------------------------------
# Features de mercado global (cross-coin)
# ------------------------------------------------------------------

def compute_market_features(klines_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calcula features agregadas de todo el mercado.

    Recibe las klines de todas las monedas y devuelve un DataFrame indexado
    por timestamp con métricas globales del mercado:
    - % de monedas subiendo en 24h
    - Retorno medio del mercado en 24h
    - Dispersión de retornos (volatilidad cross-section)
    - Volumen total del mercado (normalizado)
    """
    roc_24_frames: list[pd.Series] = []
    vol_frames: list[pd.Series] = []

    for symbol, df in klines_by_symbol.items():
        if len(df) < 25:
            continue
        roc_24 = df["close"].pct_change(periods=24).rename(symbol)
        roc_24_frames.append(roc_24)
        vol_frames.append(df["volume"].rename(symbol))

    if not roc_24_frames:
        return pd.DataFrame()

    roc_matrix = pd.concat(roc_24_frames, axis=1)
    vol_matrix = pd.concat(vol_frames, axis=1)

    market = pd.DataFrame(index=roc_matrix.index)
    # % de monedas con retorno positivo en 24h
    market["market_pct_up_24"] = (roc_matrix > 0).sum(axis=1) / roc_matrix.count(axis=1)
    # Retorno medio del mercado
    market["market_mean_roc_24"] = roc_matrix.mean(axis=1)
    # Dispersión de retornos (std cross-section)
    market["market_dispersion_24"] = roc_matrix.std(axis=1)
    # Volumen total normalizado (z-score rolling 72h)
    total_vol = vol_matrix.sum(axis=1)
    vol_mean = total_vol.rolling(window=72, min_periods=24).mean()
    vol_std = total_vol.rolling(window=72, min_periods=24).std()
    market["market_vol_zscore"] = np.where(vol_std > 0, (total_vol - vol_mean) / vol_std, 0.0)

    return market


def _add_market_features(df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """Añade features de mercado global pre-calculadas al DataFrame de una moneda."""
    for col in market_df.columns:
        df[col] = market_df[col].reindex(df.index, method="ffill")
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
        # MFI
        "mfi_14",
        # Soporte / Resistencia
        "dist_resistance_24h", "dist_support_24h",
        "dist_resistance_72h", "dist_support_72h",
        "range_position_24h", "range_position_72h",
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
            "btc_dominance_proxy", "relative_perf_btc_24",
        ])

    # Microestructura
    base.extend([
        "volume_spike_3h", "price_acceleration",
        "upper_shadow_pct", "lower_shadow_pct", "body_pct",
        "consecutive_up", "rsi_divergence", "volume_price_trend",
    ])

    # Features de mercado global
    base.extend([
        "market_pct_up_24",
        "market_mean_roc_24",
        "market_dispersion_24",
        "market_vol_zscore",
    ])

    return base
