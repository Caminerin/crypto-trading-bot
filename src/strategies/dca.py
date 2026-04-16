"""Estrategia DCA Inteligente (Dollar Cost Averaging).

Compra BTC, ETH y BNB cuando caen fuerte (umbral por moneda).
Vende automaticamente al take-profit o stop-loss (por moneda).
El dinero rota: compra barato -> vende caro -> repite.

Posiciones se persisten en data/dca_positions.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import _QUOTE, DEFAULT_ASSET_POLICIES, DCAAssetPolicy
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DCA_POSITIONS_FILE = DATA_DIR / "dca_positions.json"

# Monedas objetivo para DCA (usa el quote asset configurado)
DCA_ASSETS = [f"BTC{_QUOTE}", f"ETH{_QUOTE}", f"BNB{_QUOTE}"]

# Umbrales globales (fallback — se usan si no hay politica por moneda)
DIP_THRESHOLD = -0.05       # Comprar cuando cae mas del 5% en 24h
TAKE_PROFIT_PCT = 0.15      # Vender cuando sube 15% desde precio de compra
STOP_LOSS_PCT = -0.10       # Vender si cae 10% desde precio de compra
MIN_ORDER_USDT = 10.0       # Minimo por orden en Binance


@dataclass
class DCAAction:
    """Accion generada por la estrategia DCA."""

    action: str          # "BUY" o "SELL"
    symbol: str          # Ej. "BTCUSDT"
    quote_qty: float     # USDT a gastar (compra) o 0 (venta)
    base_qty: float      # Cantidad cripto a vender o 0 (compra)
    reason: str
    entry_price: float   # Precio de compra original (para ventas)


class DCAPosition:
    """Representa una posicion abierta de DCA."""

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
    def from_dict(cls, data: dict[str, Any]) -> DCAPosition:
        return cls(
            symbol=data["symbol"],
            entry_price=data["entry_price"],
            quantity=data["quantity"],
            invested_usdt=data["invested_usdt"],
            entry_date=data["entry_date"],
        )


class DCAStrategy:
    """Estrategia DCA Inteligente con politica por moneda."""

    def __init__(
        self,
        budget_usdt: float,
        dip_threshold: float = DIP_THRESHOLD,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        stop_loss_pct: float = STOP_LOSS_PCT,
        assets: list[str] | None = None,
        asset_policies: dict[str, DCAAssetPolicy] | None = None,
    ) -> None:
        self._budget = budget_usdt
        self._dip_threshold = dip_threshold
        self._take_profit_pct = take_profit_pct
        self._stop_loss_pct = stop_loss_pct
        self._assets = assets or DCA_ASSETS.copy()
        self._policies = asset_policies or DEFAULT_ASSET_POLICIES
        self._positions: list[DCAPosition] = []
        self._load_positions()

    def _get_policy(self, symbol: str) -> DCAAssetPolicy:
        """Devuelve la politica para un symbol, o la global por defecto."""
        if symbol in self._policies:
            return self._policies[symbol]
        return DCAAssetPolicy(
            dip_threshold=self._dip_threshold,
            take_profit_pct=self._take_profit_pct,
            stop_loss_pct=self._stop_loss_pct,
        )

    # ------------------------------------------------------------------
    # Persistencia de posiciones
    # ------------------------------------------------------------------

    def _load_positions(self) -> None:
        if DCA_POSITIONS_FILE.exists():
            try:
                raw = json.loads(DCA_POSITIONS_FILE.read_text())
                self._positions = [
                    DCAPosition.from_dict(p) for p in raw.get("positions", [])
                ]
                logger.info(
                    "DCA: %d posiciones cargadas", len(self._positions),
                )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Error leyendo dca_positions.json: %s", exc)
                self._positions = []
        else:
            logger.info("DCA: sin posiciones previas.")

    def save_positions(self) -> None:
        data = {
            "positions": [p.to_dict() for p in self._positions],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        DCA_POSITIONS_FILE.write_text(json.dumps(data, indent=2))
        logger.info("DCA: posiciones guardadas (%d)", len(self._positions))

    @property
    def positions(self) -> list[DCAPosition]:
        return list(self._positions)

    # ------------------------------------------------------------------
    # Logica principal
    # ------------------------------------------------------------------

    def evaluate(
        self,
        price_changes_24h: dict[str, float],
        current_prices: dict[str, float],
    ) -> list[DCAAction]:
        """Evalua el mercado y genera acciones de compra/venta.

        Args:
            price_changes_24h: {symbol: pct_change} Ej. {"BTCUSDT": -0.06}
            current_prices: {symbol: price} Ej. {"BTCUSDT": 60000.0}

        Returns:
            Lista de acciones a ejecutar.
        """
        actions: list[DCAAction] = []

        # 1. Revisar posiciones existentes -> SELL si take-profit o stop-loss
        actions.extend(self._check_exits(current_prices))

        # 2. Revisar si hay dips para comprar -> BUY
        actions.extend(self._check_dip_buys(price_changes_24h, current_prices))

        return actions

    def _check_exits(
        self, current_prices: dict[str, float],
    ) -> list[DCAAction]:
        """Genera SELL para posiciones con take-profit o stop-loss."""
        sells: list[DCAAction] = []
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
                    DCAAction(
                        action="SELL",
                        symbol=pos.symbol,
                        quote_qty=0,
                        base_qty=pos.quantity,
                        reason=(
                            f"DCA take-profit: {pos.symbol} "
                            f"+{pnl_pct:.1%} (entrada: ${pos.entry_price:,.2f}, "
                            f"actual: ${price:,.2f}, ganancia: ${profit_usdt:,.2f})"
                        ),
                        entry_price=pos.entry_price,
                    )
                )
                logger.info(
                    "DCA TP %s: +%.1f%% (entrada=$%.2f, actual=$%.2f)",
                    pos.symbol, pnl_pct * 100, pos.entry_price, price,
                )
            # Stop-loss
            elif pnl_pct <= policy.stop_loss_pct:
                loss_usdt = pos.invested_usdt * pnl_pct
                sells.append(
                    DCAAction(
                        action="SELL",
                        symbol=pos.symbol,
                        quote_qty=0,
                        base_qty=pos.quantity,
                        reason=(
                            f"DCA stop-loss: {pos.symbol} "
                            f"{pnl_pct:.1%} (entrada: ${pos.entry_price:,.2f}, "
                            f"actual: ${price:,.2f}, perdida: ${loss_usdt:,.2f})"
                        ),
                        entry_price=pos.entry_price,
                    )
                )
                logger.info(
                    "DCA SL %s: %.1f%% (entrada=$%.2f, actual=$%.2f)",
                    pos.symbol, pnl_pct * 100, pos.entry_price, price,
                )
        return sells

    def _check_dip_buys(
        self,
        price_changes_24h: dict[str, float],
        current_prices: dict[str, float],
    ) -> list[DCAAction]:
        """Genera BUY cuando un activo DCA cae mas del umbral."""
        buys: list[DCAAction] = []

        # Calcular USDT libre (budget menos lo invertido en posiciones abiertas)
        invested = sum(p.invested_usdt for p in self._positions)
        free_usdt = self._budget - invested

        if free_usdt < MIN_ORDER_USDT:
            logger.info(
                "DCA: sin presupuesto libre (budget=%.2f, invertido=%.2f, libre=%.2f)",
                self._budget, invested, free_usdt,
            )
            return buys

        # Buscar dips en los activos DCA (umbral por moneda)
        dip_assets: list[tuple[str, float]] = []
        for symbol in self._assets:
            change = price_changes_24h.get(symbol)
            if change is None:
                continue
            policy = self._get_policy(symbol)
            if change <= policy.dip_threshold:
                dip_assets.append((symbol, change))

        if not dip_assets:
            logger.info(
                "DCA: sin caidas significativas hoy (umbrales por moneda)",
            )
            return buys

        # Repartir el presupuesto entre los activos en dip
        per_buy = min(free_usdt / len(dip_assets), free_usdt)

        for symbol, change in dip_assets:
            buy_amount = per_buy
            if buy_amount < MIN_ORDER_USDT:
                logger.info(
                    "DCA: insuficiente para %s (%.2f < %.2f min)",
                    symbol, buy_amount, MIN_ORDER_USDT,
                )
                continue

            price = current_prices.get(symbol)
            if price is None:
                continue

            buys.append(
                DCAAction(
                    action="BUY",
                    symbol=symbol,
                    quote_qty=buy_amount,
                    base_qty=0,
                    reason=(
                        f"DCA compra en caida: {symbol} "
                        f"{change:.1%} en 24h (precio: ${price:,.2f})"
                    ),
                    entry_price=price,
                )
            )
            logger.info(
                "DCA BUY %s: caida %.1f%% | $%.2f USDT | precio=$%.2f",
                symbol, change * 100, buy_amount, price,
            )

        return buys

    # ------------------------------------------------------------------
    # Gestion de posiciones post-ejecucion
    # ------------------------------------------------------------------

    def record_buy(
        self,
        symbol: str,
        price: float,
        quantity: float,
        usdt_spent: float,
    ) -> None:
        """Registra una compra DCA ejecutada."""
        # Verificar si ya tenemos posicion en este symbol -> promediar
        existing = next((p for p in self._positions if p.symbol == symbol), None)
        if existing is not None:
            # Precio medio ponderado
            total_invested = existing.invested_usdt + usdt_spent
            total_qty = existing.quantity + quantity
            existing.entry_price = total_invested / total_qty if total_qty > 0 else price
            existing.quantity = total_qty
            existing.invested_usdt = total_invested
            logger.info(
                "DCA: posicion %s actualizada (qty=%.8f, avg=$%.2f, invertido=$%.2f)",
                symbol, total_qty, existing.entry_price, total_invested,
            )
        else:
            pos = DCAPosition(
                symbol=symbol,
                entry_price=price,
                quantity=quantity,
                invested_usdt=usdt_spent,
                entry_date=datetime.now(timezone.utc).isoformat(),
            )
            self._positions.append(pos)
            logger.info(
                "DCA: nueva posicion %s (qty=%.8f, precio=$%.2f, invertido=$%.2f)",
                symbol, quantity, price, usdt_spent,
            )
        self.save_positions()

    def record_sell(self, symbol: str) -> float:
        """Registra una venta DCA. Devuelve el USDT invertido que se libera."""
        pos = next((p for p in self._positions if p.symbol == symbol), None)
        if pos is None:
            logger.warning("DCA: no se encontro posicion para %s", symbol)
            return 0.0

        freed = pos.invested_usdt
        self._positions = [p for p in self._positions if p.symbol != symbol]
        self.save_positions()
        logger.info("DCA: posicion %s cerrada (liberados $%.2f)", symbol, freed)
        return freed

    def get_summary(self, current_prices: dict[str, float]) -> dict[str, Any]:
        """Resumen del estado DCA para el reporte."""
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
