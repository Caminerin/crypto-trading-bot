"""Libro de posiciones de la estrategia Prediction.

Rastrea que posiciones pertenecen a la estrategia de prediccion ML,
aislando su inventario del resto de estrategias (DCA, Momentum).

Posiciones se persisten en data/prediction_positions.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PREDICTION_POSITIONS_FILE = DATA_DIR / "prediction_positions.json"
PENDING_LIMIT_ORDERS_FILE = DATA_DIR / "pending_limit_orders.json"


class PredictionPosition:
    """Representa una posicion abierta de la estrategia Prediction."""

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
    def from_dict(cls, data: dict[str, Any]) -> PredictionPosition:
        return cls(
            symbol=data["symbol"],
            entry_price=data["entry_price"],
            quantity=data["quantity"],
            invested_usdt=data["invested_usdt"],
            entry_date=data["entry_date"],
        )


class PendingLimitOrder:
    """Orden limit de compra pendiente de ejecucion."""

    def __init__(
        self,
        symbol: str,
        order_id: int,
        quote_qty: float,
        limit_price: float,
        created_at: str,
    ) -> None:
        self.symbol = symbol
        self.order_id = order_id
        self.quote_qty = quote_qty
        self.limit_price = limit_price
        self.created_at = created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "order_id": self.order_id,
            "quote_qty": self.quote_qty,
            "limit_price": self.limit_price,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingLimitOrder:
        return cls(
            symbol=data["symbol"],
            order_id=data["order_id"],
            quote_qty=data["quote_qty"],
            limit_price=data["limit_price"],
            created_at=data["created_at"],
        )


class PredictionBook:
    """Gestiona el inventario de la estrategia Prediction.

    Funciona igual que DCAStrategy y MomentumStrategy en cuanto a
    persistencia de posiciones, pero sin logica de señales (esa la
    maneja PricePredictor + PortfolioManager).
    """

    def __init__(self) -> None:
        self._positions: list[PredictionPosition] = []
        self._pending_orders: list[PendingLimitOrder] = []
        self._load_positions()
        self._load_pending_orders()

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def _load_positions(self) -> None:
        if PREDICTION_POSITIONS_FILE.exists():
            try:
                raw = json.loads(PREDICTION_POSITIONS_FILE.read_text())
                self._positions = [
                    PredictionPosition.from_dict(p)
                    for p in raw.get("positions", [])
                ]
                logger.info(
                    "Prediction book: %d posiciones cargadas",
                    len(self._positions),
                )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(
                    "Error leyendo prediction_positions.json: %s", exc,
                )
                self._positions = []
        else:
            logger.info("Prediction book: sin posiciones previas.")

    def _load_pending_orders(self) -> None:
        if PENDING_LIMIT_ORDERS_FILE.exists():
            try:
                raw = json.loads(PENDING_LIMIT_ORDERS_FILE.read_text())
                self._pending_orders = [
                    PendingLimitOrder.from_dict(o)
                    for o in raw.get("orders", [])
                ]
                if self._pending_orders:
                    logger.info(
                        "Pending limit orders: %d cargadas",
                        len(self._pending_orders),
                    )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(
                    "Error leyendo pending_limit_orders.json: %s", exc,
                )
                self._pending_orders = []

    def _save_pending_orders(self) -> None:
        data = {
            "orders": [o.to_dict() for o in self._pending_orders],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        PENDING_LIMIT_ORDERS_FILE.write_text(json.dumps(data, indent=2))

    @property
    def pending_orders(self) -> list[PendingLimitOrder]:
        return list(self._pending_orders)

    @property
    def pending_symbols(self) -> set[str]:
        """Simbolos con orden limit pendiente."""
        return {o.symbol for o in self._pending_orders}

    @property
    def pending_invested_usdt(self) -> float:
        """USDT reservado en ordenes limit pendientes."""
        return sum(o.quote_qty for o in self._pending_orders)

    def record_pending_order(
        self,
        symbol: str,
        order_id: int,
        quote_qty: float,
        limit_price: float,
    ) -> None:
        """Registra una orden limit de compra pendiente."""
        pending = PendingLimitOrder(
            symbol=symbol,
            order_id=order_id,
            quote_qty=quote_qty,
            limit_price=limit_price,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._pending_orders.append(pending)
        self._save_pending_orders()
        logger.info(
            "Pending limit: %s registrada (orderId=%d, price=%.8f, ~$%.2f)",
            symbol,
            order_id,
            limit_price,
            quote_qty,
        )

    def remove_pending_order(self, order_id: int) -> PendingLimitOrder | None:
        """Elimina una orden pendiente por ID. Devuelve la orden o None."""
        order = next(
            (o for o in self._pending_orders if o.order_id == order_id),
            None,
        )
        if order is not None:
            self._pending_orders = [
                o for o in self._pending_orders if o.order_id != order_id
            ]
            self._save_pending_orders()
        return order

    def save_positions(self) -> None:
        data = {
            "positions": [p.to_dict() for p in self._positions],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        PREDICTION_POSITIONS_FILE.write_text(json.dumps(data, indent=2))
        logger.info(
            "Prediction book: posiciones guardadas (%d)",
            len(self._positions),
        )

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------

    @property
    def positions(self) -> list[PredictionPosition]:
        return list(self._positions)

    @property
    def open_symbols(self) -> set[str]:
        """Simbolos con posicion abierta (ej. {'BTCUSDC', 'ETHUSDC'})."""
        return {p.symbol for p in self._positions}

    @property
    def invested_usdt(self) -> float:
        """Total de USDT invertido en posiciones abiertas."""
        return sum(p.invested_usdt for p in self._positions)

    def get_portfolio_dict(self, quote_asset: str) -> dict[str, float]:
        """Devuelve un dict {asset: quantity} solo con posiciones de Prediction.

        Util para pasarlo al PortfolioManager como ``current_portfolio``
        en vez de la cartera global de Binance.
        """
        portfolio: dict[str, float] = {}
        for pos in self._positions:
            asset = pos.symbol.replace(quote_asset, "")
            portfolio[asset] = portfolio.get(asset, 0.0) + pos.quantity
        return portfolio

    # Umbral en USD por debajo del cual un balance residual se
    # considera "polvo" (dust) y no una posición real.  Esto ocurre
    # cuando la OCO vende pero queda un resto por redondeo de stepSize.
    _DUST_THRESHOLD_USD = 1.0

    def reconcile(
        self,
        portfolio: dict[str, float],
        open_orders_by_symbol: dict[str, list],
        quote_asset: str,
    ) -> list[PredictionPosition]:
        """Reconcilia el libro con el estado real de Binance.

        Compara cada posición registrada con los balances reales y las
        órdenes abiertas.  Si una posición no tiene balance (o solo
        queda polvo residual < $1) NI órdenes abiertas, significa que
        la OCO (TP/SL) ya se ejecutó → se elimina del libro.

        Devuelve la lista de posiciones eliminadas (para logging).
        """
        closed: list[PredictionPosition] = []
        remaining: list[PredictionPosition] = []

        for pos in self._positions:
            base_asset = pos.symbol.replace(quote_asset, "")
            balance = portfolio.get(base_asset, 0.0)
            has_orders = len(open_orders_by_symbol.get(pos.symbol, [])) > 0

            balance_value = balance * pos.entry_price if balance > 0 else 0.0
            is_dust = balance_value < self._DUST_THRESHOLD_USD

            if (balance <= 0 or is_dust) and not has_orders:
                reason = (
                    "polvo residual"
                    if balance > 0
                    else "sin balance ni órdenes"
                )
                logger.info(
                    "Reconciliación: %s cerrada por OCO (%s, "
                    "balance=%.8f ~$%.4f). Liberando $%.2f del book.",
                    pos.symbol,
                    reason,
                    balance,
                    balance_value,
                    pos.invested_usdt,
                )
                closed.append(pos)
            else:
                remaining.append(pos)

        if closed:
            self._positions = remaining
            self.save_positions()
            logger.info(
                "Reconciliación: %d posiciones eliminadas, %d vigentes",
                len(closed),
                len(remaining),
            )
        else:
            logger.info("Reconciliación: todas las posiciones coinciden con Binance")

        return closed

    def get_expired_positions(self, horizon_hours: int) -> list[PredictionPosition]:
        """Devuelve posiciones cuya ventana temporal ha expirado.

        Una posición se considera expirada si han pasado más de
        *horizon_hours* horas desde su ``entry_date``.
        """
        now = datetime.now(timezone.utc)
        expired: list[PredictionPosition] = []
        for pos in self._positions:
            try:
                entry = datetime.fromisoformat(pos.entry_date)
                if entry.tzinfo is None:
                    entry = entry.replace(tzinfo=timezone.utc)
                elapsed_hours = (now - entry).total_seconds() / 3600
                if elapsed_hours >= horizon_hours:
                    expired.append(pos)
            except (ValueError, TypeError):
                logger.warning(
                    "Fecha inválida en posición %s: %s",
                    pos.symbol,
                    pos.entry_date,
                )
        return expired

    # ------------------------------------------------------------------
    # Registro de operaciones
    # ------------------------------------------------------------------

    def record_buy(
        self,
        symbol: str,
        price: float,
        quantity: float,
        usdt_spent: float,
    ) -> None:
        """Registra una compra Prediction ejecutada."""
        existing = next(
            (p for p in self._positions if p.symbol == symbol), None,
        )
        if existing is not None:
            total_invested = existing.invested_usdt + usdt_spent
            total_qty = existing.quantity + quantity
            existing.entry_price = (
                total_invested / total_qty if total_qty > 0 else price
            )
            existing.quantity = total_qty
            existing.invested_usdt = total_invested
            logger.info(
                "Prediction book: posicion %s actualizada "
                "(qty=%.8f, avg=$%.2f, invertido=$%.2f)",
                symbol, total_qty, existing.entry_price, total_invested,
            )
        else:
            pos = PredictionPosition(
                symbol=symbol,
                entry_price=price,
                quantity=quantity,
                invested_usdt=usdt_spent,
                entry_date=datetime.now(timezone.utc).isoformat(),
            )
            self._positions.append(pos)
            logger.info(
                "Prediction book: nueva posicion %s "
                "(qty=%.8f, precio=$%.2f, invertido=$%.2f)",
                symbol, quantity, price, usdt_spent,
            )
        self.save_positions()

    def record_sell(self, symbol: str) -> float:
        """Registra una venta Prediction. Devuelve USDT invertido liberado."""
        pos = next(
            (p for p in self._positions if p.symbol == symbol), None,
        )
        if pos is None:
            logger.warning(
                "Prediction book: no se encontro posicion para %s", symbol,
            )
            return 0.0

        freed = pos.invested_usdt
        self._positions = [p for p in self._positions if p.symbol != symbol]
        self.save_positions()
        logger.info(
            "Prediction book: posicion %s cerrada (liberados $%.2f)",
            symbol, freed,
        )
        return freed

    def get_summary(self, current_prices: dict[str, float]) -> dict[str, float]:
        """Resumen del estado Prediction para el reporte."""
        invested = self.invested_usdt
        positions_info = []
        total_pnl = 0.0
        for pos in self._positions:
            price = current_prices.get(pos.symbol, pos.entry_price)
            current_value = pos.quantity * price
            pnl = current_value - pos.invested_usdt
            pnl_pct = (
                (pnl / pos.invested_usdt * 100) if pos.invested_usdt > 0 else 0
            )
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
            "invested": round(invested, 2),
            "positions": positions_info,
            "total_pnl": round(total_pnl, 2),
        }
