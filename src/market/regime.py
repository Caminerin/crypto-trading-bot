"""Filtro de régimen de mercado.

Evalúa las condiciones generales del mercado antes de abrir nuevas
posiciones.  Cuando el mercado está en caída, el bot no compra pero
sí mantiene la gestión de posiciones existentes (TP/SL, expiración,
reconciliación).

Indicadores utilizados (basados en datos ya disponibles en el bot):
- BTC ROC 24h: variación de BTC en las últimas 24 horas.
- BTC RSI 14: índice de fuerza relativa de BTC (sobreventa extrema).
- % de monedas subiendo en 24h: amplitud de mercado.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import ta

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MarketRegimeResult:
    """Resultado de la evaluación del régimen de mercado."""

    allow_buys: bool
    btc_roc_24h: float
    btc_rsi_14: float
    pct_coins_up_24h: float
    reasons: tuple[str, ...]


# Umbrales por defecto (conservadores — evitan operar en caídas
# fuertes pero no bloquean correcciones normales).
BTC_ROC_24H_THRESHOLD = -0.03     # BTC cayendo >3% en 24h
BTC_RSI_THRESHOLD = 30.0          # RSI por debajo de 30 = pánico
MARKET_BREADTH_THRESHOLD = 0.25   # <25% de monedas subiendo


def evaluate_market_regime(
    btc_df: pd.DataFrame,
    klines_by_symbol: dict[str, pd.DataFrame],
    *,
    btc_roc_threshold: float = BTC_ROC_24H_THRESHOLD,
    btc_rsi_threshold: float = BTC_RSI_THRESHOLD,
    breadth_threshold: float = MARKET_BREADTH_THRESHOLD,
) -> MarketRegimeResult:
    """Decide si el mercado es propicio para abrir nuevas posiciones.

    Parameters
    ----------
    btc_df:
        Klines horarias de BTC (mínimo 25 filas para RSI + ROC 24h).
    klines_by_symbol:
        Klines de todas las monedas (para calcular amplitud de mercado).
    btc_roc_threshold:
        Umbral de caída de BTC en 24h para bloquear compras.
    btc_rsi_threshold:
        RSI de BTC por debajo del cual se bloquean compras.
    breadth_threshold:
        Porcentaje mínimo de monedas subiendo para permitir compras.

    Returns
    -------
    MarketRegimeResult con la decisión y los valores de cada indicador.
    """
    reasons: list[str] = []

    # --- BTC ROC 24h ---
    btc_roc_24h = _compute_btc_roc_24h(btc_df)

    # --- BTC RSI 14 ---
    btc_rsi_14 = _compute_btc_rsi(btc_df)

    # --- Amplitud de mercado (% monedas subiendo) ---
    pct_up = _compute_market_breadth(klines_by_symbol)

    # Evaluar condiciones
    if btc_roc_24h <= btc_roc_threshold:
        reasons.append(
            f"BTC cayendo {btc_roc_24h:.1%} en 24h (umbral: {btc_roc_threshold:.1%})"
        )

    if btc_rsi_14 <= btc_rsi_threshold:
        reasons.append(
            f"BTC RSI en {btc_rsi_14:.1f} (umbral: {btc_rsi_threshold:.0f})"
        )

    if pct_up <= breadth_threshold:
        reasons.append(
            f"Solo {pct_up:.0%} de monedas subiendo (umbral: {breadth_threshold:.0%})"
        )

    allow = len(reasons) == 0

    if allow:
        logger.info(
            "Mercado OK — BTC ROC 24h: %+.1f%%, RSI: %.1f, "
            "monedas subiendo: %.0f%%",
            btc_roc_24h * 100, btc_rsi_14, pct_up * 100,
        )
    else:
        logger.warning(
            "Mercado ADVERSO — compras bloqueadas. Razones: %s",
            " | ".join(reasons),
        )

    return MarketRegimeResult(
        allow_buys=allow,
        btc_roc_24h=btc_roc_24h,
        btc_rsi_14=btc_rsi_14,
        pct_coins_up_24h=pct_up,
        reasons=tuple(reasons),
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _compute_btc_roc_24h(btc_df: pd.DataFrame) -> float:
    """Retorno de BTC en las últimas 24 horas."""
    if len(btc_df) < 25:
        logger.warning("BTC: datos insuficientes para ROC 24h (%d filas)", len(btc_df))
        return 0.0
    close = btc_df["close"]
    return float((close.iloc[-1] - close.iloc[-25]) / close.iloc[-25])


def _compute_btc_rsi(btc_df: pd.DataFrame) -> float:
    """RSI 14 de BTC (último valor)."""
    if len(btc_df) < 15:
        logger.warning("BTC: datos insuficientes para RSI (%d filas)", len(btc_df))
        return 50.0  # valor neutro si no hay datos
    rsi = ta.momentum.rsi(btc_df["close"], window=14)
    last_valid = rsi.dropna()
    if last_valid.empty:
        return 50.0
    return float(last_valid.iloc[-1])


def _compute_market_breadth(
    klines_by_symbol: dict[str, pd.DataFrame],
) -> float:
    """Porcentaje de monedas que han subido en las últimas 24 horas."""
    up = 0
    total = 0
    for symbol, df in klines_by_symbol.items():
        if len(df) < 25:
            continue
        close = df["close"]
        roc_24 = (close.iloc[-1] - close.iloc[-25]) / close.iloc[-25]
        total += 1
        if roc_24 > 0:
            up += 1
    if total == 0:
        return 0.5  # valor neutro si no hay datos
    return up / total
