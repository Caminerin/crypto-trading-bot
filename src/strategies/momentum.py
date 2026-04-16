"""Estrategia Momentum (compra en tendencia alcista).

Compra BTC, ETH y BNB cuando suben fuerte en 24h Y mantienen
tendencia positiva en los últimos N días.
Vende automáticamente al take-profit o stop-loss (por moneda).

Posiciones se persisten en data/momentum_positions.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import _QUOTE, DEFAULT_MOMENTUM_POLICIES, MomentumAssetPolicy
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
MOMENTUM_POSITIONS_FILE = DATA_DIR / "momentum_positions.json"

# Monedas objetivo para momentum (usa el quote asset configurado)
MOMENTUM_ASSETS = [f"BTC{_QUOTE}", f"ETH{_QUOTE}", f"BNB{_QUOTE}"]

# Umbrales globales (fallback — se usan si no hay política por moneda)
MOMENTUM_THRESHOLD = 0.05   # Comprar cuando sube más del 5% en 24h
TAKE_PROFIT_PCT = 0.10      # Vender cuando sube 10% desde precio de compra
STOP_LOSS_PCT = -0.05       # Vender si cae 5% desde precio de compra
TREND_DAYS = 7              # Días para confirmar tendencia alcista
MIN_ORDER_USDT = 10.0       # Mínimo por orden en Binance


@dataclass
class MomentumAction:
    """Acción generada por la estrategia Momentum."""

    action: str          # "BUY" o "SELL"
    symbol: str          # Ej. "BTCUSDT"
    quote_qty: float     # USDT a gastar (compra) o 0 (venta)
    base_qty: float      # Cantidad cripto a vender o 0 (compra)
    reason: str
    entry_price: float   # Precio de compra original (para ventas)


class MomentumPosition:
    """Representa una posición abierta de Momentum."""

    def __init__(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        invested_usdt: float,
        entry_date: str,
    ) -> None:
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.invested_usdt = invested_usdt
        self.entry_date = entry_date

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "invested_usdt": self.invested_usdt,
            "entry_date": self.entry_date,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MomentumPosition:
        return cls(
            symbol=data["symbol"],
            entry_price=data["entry_price"],
            quantity=data["quantity"],
            invested_usdt=data["invested_usdt"],
            entry_date=data["entry_date"],
        )


class MomentumStrategy:
    """Estrategia Momentum: compra en tendencia alcista fuerte."""

    def __init__(
        self,
        budget_usdt: float,
        momentum_threshold: float = MOMENTUM_THRESHOLD,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        stop_loss_pct: float = STOP_LOSS_PCT,
        trend_days: int = TREND_DAYS,
        assets: list[str] | None = None,
        asset_policies: dict[str, MomentumAssetPolicy] | None = None,
    ) -> None:
        self._budget = budget_usdt
        self._momentum_threshold = momentum_threshold
        self._take_profit_pct = take_profit_pct
        self._stop_loss_pct = stop_loss_pct
        self._trend_days = trend_days
        self._assets = assets or MOMENTUM_ASSETS.copy()
        self._policies = asset_policies or DEFAULT_MOMENTUM_POLICIES
        self._positions: list[MomentumPosition] = []
        self._load_positions()

    def _get_policy(self, symbol: str) -> MomentumAssetPolicy:
        """Devuelve la política para un symbol, o la global por defecto."""
        if symbol in self._policies:
            return self._policies[symbol]
        return MomentumAssetPolicy(
            momentum_threshold=self._momentum_threshold,
            take_profit_pct=self._take_profit_pct,
            stop_loss_pct=self._stop_loss_pct,
            trend_days=self._trend_days,
        )

    # ------------------------------------------------------------------
    # Persistencia de posiciones
    # ------------------------------------------------------------------

    def _load_positions(self) -> None:
        if MOMENTUM_POSITIONS_FILE.exists():
            try:
                raw = json.loads(MOMENTUM_POSITIONS_FILE.read_text())
                self._positions = [
                    MomentumPosition.from_dict(p) for p in raw.get("positions", [])
                ]
                logger.info(
                    "Momentum: %d posiciones cargadas", len(self._positions),
                )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Error leyendo momentum_positions.json: %s", exc)
                self._positions = []
        else:
            logger.info("Momentum: sin posiciones previas.")

    def save_positions(self) -> None:
        data = {
            "positions": [p.to_dict() for p in self._positions],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        MOMENTUM_POSITIONS_FILE.write_text(json.dumps(data, indent=2))
        logger.info("Momentum: posiciones guardadas (%d)", len(self._positions))

    @property
    def positions(self) -> list[MomentumPosition]:
        return list(self._positions)

    # ------------------------------------------------------------------
    # Lógica principal
    # ------------------------------------------------------------------

    def evaluate(
        self,
        price_changes_24h: dict[str, float],
        current_prices: dict[str, float],
        daily_closes: dict[str, list[float]],
    ) -> list[MomentumAction]:
        """Evalúa el mercado y genera acciones de compra/venta.

        Args:
            price_changes_24h: {symbol: pct_change} Ej. {"BTCUSDT": 0.06}
            current_prices: {symbol: price} Ej. {"BTCUSDT": 70000.0}
            daily_closes: {symbol: [close_day1, ..., close_dayN]}
                Lista de precios de cierre diarios ordenados cronológicamente
                (el último elemento es el más reciente).

        Returns:
            Lista de acciones a ejecutar.
        """
        actions: list[MomentumAction] = []

        # 1. Revisar posiciones existentes -> SELL si take-profit o stop-loss
        actions.extend(self._check_exits(current_prices))

        # 2. Revisar si hay señales de momentum para comprar -> BUY
        actions.extend(
            self._check_momentum_buys(price_changes_24h, current_prices, daily_closes),
        )

        return actions

    def _check_exits(
        self, current_prices: dict[str, float],
    ) -> list[MomentumAction]:
        """Genera SELL para posiciones con take-profit o stop-loss."""
        sells: list[MomentumAction] = []
        for pos in self._positions:
            price = current_prices.get(pos.symbol)
            if price is None:
                continue

            policy = self._get_policy(pos.symbol)
            pnl_pct = (price - pos.entry_price) / pos.entry_price

            # Take-profit
            if pnl_pct >= policy.take_profit_pct:
                profit_usdt = pos.invested_usdt * pnl_pct
                sells.append(
                    MomentumAction(
                        action="SELL",
                        symbol=pos.symbol,
                        quote_qty=0,
                        base_qty=pos.quantity,
                        reason=(
                            f"Momentum take-profit: {pos.symbol} "
                            f"+{pnl_pct:.1%} (entrada: ${pos.entry_price:,.2f}, "
                            f"actual: ${price:,.2f}, ganancia: ${profit_usdt:,.2f})"
                        ),
                        entry_price=pos.entry_price,
                    )
                )
                logger.info(
                    "Momentum TP %s: +%.1f%% (entrada=$%.2f, actual=$%.2f)",
                    pos.symbol, pnl_pct * 100, pos.entry_price, price,
                )
            # Stop-loss
            elif pnl_pct <= policy.stop_loss_pct:
                loss_usdt = pos.invested_usdt * pnl_pct
                sells.append(
                    MomentumAction(
                        action="SELL",
                        symbol=pos.symbol,
                        quote_qty=0,
                        base_qty=pos.quantity,
                        reason=(
                            f"Momentum stop-loss: {pos.symbol} "
                            f"{pnl_pct:.1%} (entrada: ${pos.entry_price:,.2f}, "
                            f"actual: ${price:,.2f}, pérdida: ${loss_usdt:,.2f})"
                        ),
                        entry_price=pos.entry_price,
                    )
                )
                logger.info(
                    "Momentum SL %s: %.1f%% (entrada=$%.2f, actual=$%.2f)",
                    pos.symbol, pnl_pct * 100, pos.entry_price, price,
                )
        return sells

    def _check_momentum_buys(
        self,
        price_changes_24h: dict[str, float],
        current_prices: dict[str, float],
        daily_closes: dict[str, list[float]],
    ) -> list[MomentumAction]:
        """Genera BUY cuando un activo tiene momentum alcista fuerte."""
        buys: list[MomentumAction] = []

        # Calcular USDT libre (budget menos lo invertido en posiciones abiertas)
        invested = sum(p.invested_usdt for p in self._positions)
        free_usdt = self._budget - invested

        if free_usdt < MIN_ORDER_USDT:
            logger.info(
                "Momentum: sin presupuesto libre (budget=%.2f, invertido=%.2f, libre=%.2f)",
                self._budget, invested, free_usdt,
            )
            return buys

        # Buscar activos con señal de momentum
        momentum_assets: list[tuple[str, float]] = []
        for symbol in self._assets:
            change_24h = price_changes_24h.get(symbol)
            if change_24h is None:
                continue

            policy = self._get_policy(symbol)

            # Condición 1: subida en 24h superior al umbral
            if change_24h < policy.momentum_threshold:
                continue

            # Condición 2: tendencia alcista en los últimos N días
            closes = daily_closes.get(symbol)
            if not self._is_uptrend(closes, policy.trend_days):
                logger.info(
                    "Momentum: %s sube %.1f%% en 24h pero sin tendencia alcista en %dd",
                    symbol, change_24h * 100, policy.trend_days,
                )
                continue

            momentum_assets.append((symbol, change_24h))

        if not momentum_assets:
            logger.info(
                "Momentum: sin señales de momentum hoy (umbrales por moneda)",
            )
            return buys

        # Repartir el presupuesto entre los activos con señal
        per_buy = min(free_usdt / len(momentum_assets), free_usdt)

        for symbol, change in momentum_assets:
            buy_amount = per_buy
            if buy_amount < MIN_ORDER_USDT:
                logger.info(
                    "Momentum: insuficiente para %s (%.2f < %.2f min)",
                    symbol, buy_amount, MIN_ORDER_USDT,
                )
                continue

            price = current_prices.get(symbol)
            if price is None:
                continue

            buys.append(
                MomentumAction(
                    action="BUY",
                    symbol=symbol,
                    quote_qty=buy_amount,
                    base_qty=0,
                    reason=(
                        f"Momentum compra en tendencia: {symbol} "
                        f"+{change:.1%} en 24h (precio: ${price:,.2f})"
                    ),
                    entry_price=price,
                )
            )
            logger.info(
                "Momentum BUY %s: subida %.1f%% | $%.2f USDT | precio=$%.2f",
                symbol, change * 100, buy_amount, price,
            )

        return buys

    @staticmethod
    def _is_uptrend(closes: list[float] | None, trend_days: int) -> bool:
        """Verifica si los precios de cierre muestran tendencia alcista.

        Criterio: el precio del último día es mayor que el del primer día
        en la ventana de trend_days, y al menos la mitad de los días
        cerraron por encima del día anterior (momentum sostenido).
        """
        if not closes or len(closes) < 2:
            return False

        # Tomar solo los últimos trend_days (o lo que haya disponible)
        window = closes[-trend_days:] if len(closes) >= trend_days else closes

        # Condición A: precio final > precio inicial en la ventana
        if window[-1] <= window[0]:
            return False

        # Condición B: al menos la mitad de los días cerraron al alza
        up_days = sum(1 for i in range(1, len(window)) if window[i] > window[i - 1])
        total_days = len(window) - 1

        return up_days >= total_days / 2

    # ------------------------------------------------------------------
    # Gestión de posiciones post-ejecución
    # ------------------------------------------------------------------

    def record_buy(
        self,
        symbol: str,
        price: float,
        quantity: float,
        usdt_spent: float,
    ) -> None:
        """Registra una compra Momentum ejecutada."""
        # Verificar si ya tenemos posición en este symbol -> promediar
        existing = next((p for p in self._positions if p.symbol == symbol), None)
        if existing is not None:
            # Precio medio ponderado
            total_invested = existing.invested_usdt + usdt_spent
            total_qty = existing.quantity + quantity
            existing.entry_price = total_invested / total_qty if total_qty > 0 else price
            existing.quantity = total_qty
            existing.invested_usdt = total_invested
            logger.info(
                "Momentum: posición %s actualizada (qty=%.8f, avg=$%.2f, invertido=$%.2f)",
                symbol, total_qty, existing.entry_price, total_invested,
            )
        else:
            pos = MomentumPosition(
                symbol=symbol,
                entry_price=price,
                quantity=quantity,
                invested_usdt=usdt_spent,
                entry_date=datetime.now(timezone.utc).isoformat(),
            )
            self._positions.append(pos)
            logger.info(
                "Momentum: nueva posición %s (qty=%.8f, precio=$%.2f, invertido=$%.2f)",
                symbol, quantity, price, usdt_spent,
            )
        self.save_positions()

    def record_sell(self, symbol: str) -> float:
        """Registra una venta Momentum. Devuelve el USDT invertido que se libera."""
        pos = next((p for p in self._positions if p.symbol == symbol), None)
        if pos is None:
            logger.warning("Momentum: no se encontró posición para %s", symbol)
            return 0.0

        freed = pos.invested_usdt
        self._positions = [p for p in self._positions if p.symbol != symbol]
        self.save_positions()
        logger.info("Momentum: posición %s cerrada (liberados $%.2f)", symbol, freed)
        return freed

    def get_summary(self, current_prices: dict[str, float]) -> dict[str, Any]:
        """Resumen del estado Momentum para el reporte."""
        invested = sum(p.invested_usdt for p in self._positions)
        free_usdt = self._budget - invested

        positions_info = []
        total_pnl = 0.0
        for pos in self._positions:
            price = current_prices.get(pos.symbol, pos.entry_price)
            current_value = pos.quantity * price
            pnl = current_value - pos.invested_usdt
            pnl_pct = (pnl / pos.invested_usdt * 100) if pos.invested_usdt > 0 else 0
            total_pnl += pnl
            positions_info.append({
                "symbol": pos.symbol,
                "entry_price": pos.entry_price,
                "current_price": price,
                "quantity": pos.quantity,
                "invested": pos.invested_usdt,
                "current_value": round(current_value, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "entry_date": pos.entry_date,
            })

        return {
            "budget": self._budget,
            "invested": round(invested, 2),
            "free": round(free_usdt, 2),
            "positions": positions_info,
            "total_pnl": round(total_pnl, 2),
        }
