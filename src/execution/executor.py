"""
Ejecutor de órdenes.

Recibe las acciones del PortfolioManager y las ejecuta contra Binance
(modo live) o las simula (modo paper trading).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.config import AppConfig
from src.data.binance_client import BinanceTradingClient
from src.portfolio.manager import TradeAction
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Resultado de la ejecución de una acción."""

    action: TradeAction
    success: bool
    executed_qty: float = 0.0
    executed_price: float = 0.0
    commission: float = 0.0
    error: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class OrderExecutor:
    """Ejecuta órdenes en modo live o paper trading."""

    def __init__(self, config: AppConfig, trading_client: BinanceTradingClient | None) -> None:
        self._config = config
        self._client = trading_client

    def execute(self, actions: list[TradeAction]) -> list[ExecutionResult]:
        """Ejecuta una lista de acciones y devuelve los resultados."""
        results: list[ExecutionResult] = []

        # Ejecutar ventas primero para liberar USDT
        sells = [a for a in actions if a.action == "SELL"]
        buys = [a for a in actions if a.action == "BUY"]

        for action in sells + buys:
            if self._config.is_paper_trading:
                result = self._execute_paper(action)
            else:
                result = self._execute_live(action)
            results.append(result)

        return results

    def _execute_paper(self, action: TradeAction) -> ExecutionResult:
        """Simula la ejecución sin tocar Binance."""
        logger.info(
            "[PAPER] %s %s | qty_quote=%.2f | qty_base=%.8f | reason=%s",
            action.action,
            action.symbol,
            action.quote_qty,
            action.base_qty,
            action.reason,
        )
        return ExecutionResult(
            action=action,
            success=True,
            executed_qty=action.base_qty if action.action == "SELL" else action.quote_qty,
            executed_price=0.0,
            commission=0.0,
            error="",
        )

    def _execute_live(self, action: TradeAction) -> ExecutionResult:
        """Ejecuta la orden de verdad en Binance."""
        if self._client is None:
            return ExecutionResult(
                action=action,
                success=False,
                error="Trading client no configurado",
            )

        try:
            if action.action == "SELL":
                # Cancelar órdenes abiertas antes de vender
                self._client.cancel_open_orders(action.symbol)
                order = self._client.place_market_sell(action.symbol, action.base_qty)
            else:
                order = self._client.place_market_buy(action.symbol, action.quote_qty)

            # Extraer datos del resultado
            fills = order.get("fills", [])
            total_qty = sum(float(f["qty"]) for f in fills)
            total_commission = sum(float(f["commission"]) for f in fills)
            avg_price = (
                sum(float(f["price"]) * float(f["qty"]) for f in fills) / total_qty
                if total_qty > 0
                else 0.0
            )

            result = ExecutionResult(
                action=action,
                success=True,
                executed_qty=total_qty,
                executed_price=avg_price,
                commission=total_commission,
            )

            # Colocar OCO (stop-loss + take-profit) para compras
            if action.action == "BUY" and total_qty > 0:
                self._client.place_oco_sell(
                    symbol=action.symbol,
                    quantity=total_qty,
                    entry_price=avg_price,
                )

            logger.info(
                "[LIVE] %s %s | precio=%.8f | qty=%.8f | comisión=%.8f",
                action.action,
                action.symbol,
                avg_price,
                total_qty,
                total_commission,
            )
            return result

        except Exception as exc:
            logger.error("[LIVE] Error ejecutando %s %s: %s", action.action, action.symbol, exc)
            return ExecutionResult(
                action=action,
                success=False,
                error=str(exc),
            )
