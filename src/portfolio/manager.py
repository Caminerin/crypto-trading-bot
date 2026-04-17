"""
Gestor de cartera.

Decide qué comprar y qué vender basándose en las recomendaciones del modelo
y en las reglas de gestión de cartera (máx. posiciones, reserva, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import PortfolioConfig, RiskConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeAction:
    """Representa una orden que el bot quiere ejecutar."""

    action: str  # "BUY" o "SELL"
    symbol: str
    quote_qty: float  # USDT a gastar (compra) o 0 (venta)
    base_qty: float  # Cantidad de cripto a vender (venta) o 0 (compra)
    reason: str
    probability: float  # Probabilidad del modelo (solo compras)


# Activos fiat / no-crypto que pueden aparecer en la cartera de Binance
# (depósitos EUR, USD, etc.) y que NO deben generar órdenes de venta.
_FIAT_ASSETS = frozenset(
    {
        "EUR",
        "USD",
        "GBP",
        "TRY",
        "BRL",
        "ARS",
        "RUB",
        "UAH",
        "NGN",
        "AUD",
        "JPY",
        "KRW",
        "INR",
        "PLN",
        "RON",
        "CZK",
        "HUF",
        "BGN",
        "SEK",
        "NOK",
        "DKK",
        "CHF",
        "CAD",
        "NZD",
        "MXN",
        "ZAR",
        "SGD",
        "HKD",
        "TWD",
        "THB",
        "IDR",
        "PHP",
        "VND",
        "MYR",
        "PKR",
        "BDT",
        "EGP",
        "CLP",
        "COP",
        "PEN",
    }
)


class PortfolioManager:
    """Genera las acciones de trading a partir del estado actual y las predicciones."""

    def __init__(
        self,
        portfolio_cfg: PortfolioConfig,
        risk_cfg: RiskConfig,
    ) -> None:
        self._pcfg = portfolio_cfg
        self._rcfg = risk_cfg

    def decide_actions(
        self,
        current_portfolio: dict[str, float],
        total_value_usdt: float,
        recommendations: list[tuple[str, float]],
        current_prices: dict[str, float],
        strategy_quote_available: float | None = None,
    ) -> list[TradeAction]:
        """Calcula la lista de acciones (compras y ventas) a ejecutar.

        Parámetros:
        - current_portfolio: {asset: cantidad} — solo posiciones de ESTA
          estrategia (no la cartera global de Binance).
        - total_value_usdt: budget total asignado a esta estrategia.
        - recommendations: [(symbol, probability)] ya filtradas y ordenadas.
        - current_prices: {symbol: price} para los pares USDT.
        - strategy_quote_available: USDT disponible para esta estrategia.
          Si es None, se lee del quote asset en current_portfolio (legacy).
        """
        actions: list[TradeAction] = []
        quote = self._pcfg.quote_asset

        # Posiciones actuales (excluir stablecoins)
        stablecoins = {"USDT", "USDC", "BUSD", "FDUSD", "DAI", "TUSD"}
        current_positions = {
            asset: qty
            for asset, qty in current_portfolio.items()
            if asset not in stablecoins and asset not in _FIAT_ASSETS and qty > 0
        }

        # ------------------------------------------------------------------
        # COMPRAS: nuevas recomendaciones
        # ------------------------------------------------------------------
        # Las ventas se gestionan externamente:
        #   - OCO en Binance (TP / SL automáticos)
        #   - Expiración de ventana temporal (target_horizon_hours)
        if strategy_quote_available is not None:
            usdt_available = strategy_quote_available
        else:
            usdt_available = current_portfolio.get(quote, 0.0)

        # Reserva mínima de stablecoins
        min_reserve = total_value_usdt * self._pcfg.min_stablecoin_reserve
        usdt_for_trading = max(0, usdt_available - min_reserve)

        # Todas las posiciones abiertas cuentan como slots ocupados
        open_slots = self._pcfg.max_positions - len(current_positions)

        if open_slots <= 0 or usdt_for_trading <= 0:
            logger.info(
                "Sin slots (slots=%d) o sin USDT disponible (%.2f)",
                open_slots,
                usdt_for_trading,
            )
            return actions

        # Máximo por posición
        max_per_position = total_value_usdt * self._pcfg.max_pct_per_coin

        # Solo comprar las top *open_slots* recomendaciones que no tenemos ya
        new_buys: list[tuple[str, float]] = []
        for symbol, prob in recommendations:
            asset = symbol.replace(quote, "")
            if asset not in current_positions and len(new_buys) < open_slots:
                new_buys.append((symbol, prob))

        if not new_buys:
            return actions

        # Repartir USDT equitativamente entre las compras (sin exceder max_per_position)
        per_buy = min(usdt_for_trading / len(new_buys), max_per_position)

        for symbol, prob in new_buys:
            if per_buy < 10:  # Binance mínimo ~10 USDT por orden
                logger.info("Insuficiente USDT para comprar %s (%.2f)", symbol, per_buy)
                continue
            actions.append(
                TradeAction(
                    action="BUY",
                    symbol=symbol,
                    quote_qty=per_buy,
                    base_qty=0,
                    reason=f"Modelo predice subida con prob={prob:.1%}",
                    probability=prob,
                )
            )

        return actions
