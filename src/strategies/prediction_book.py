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


class PredictionBook:
    """Gestiona el inventario de la estrategia Prediction.

    Funciona igual que DCAStrategy y MomentumStrategy en cuanto a
    persistencia de posiciones, pero sin logica de señales (esa la
    maneja PricePredictor + PortfolioManager).
    """

    def __init__(self) -> None:
        self._positions: list[PredictionPosition] = []
        self._load_positions()

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
