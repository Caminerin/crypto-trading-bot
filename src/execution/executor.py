"""
Ejecutor de órdenes.

Recibe las acciones del PortfolioManager y las ejecuta contra Binance
(modo live) o las simula (modo paper trading).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from binance.exceptions import BinanceAPIException

from src.config import AppConfig
from src.data.binance_client import BinanceDataClient, BinanceTradingClient
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

    def __init__(
        self,
        config: AppConfig,
        trading_client: BinanceTradingClient | None,
        data_client: BinanceDataClient | None = None,
    ) -> None:
        self._config = config
        self._client = trading_client
        self._data_client = data_client

    def execute(self, actions: list[TradeAction]) -> list[ExecutionResult]:
        """Ejecuta una lista de acciones y devuelve los resultados."""
        results: list[ExecutionResult] = []

        # Ejecutar ventas primero para liberar quote asset
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
        """Simula la ejecución sin tocar Binance.

        Intenta obtener el precio real del activo para que las
        cantidades simuladas sean realistas (qty base = quote / precio).
        """
        # Obtener precio real para simulación realista
        sim_price = 0.0
        if self._data_client is not None:
            try:
                sim_price = self._data_client.get_current_price(action.symbol)
            except Exception:
                pass

        if action.action == "BUY":
            if sim_price > 0:
                executed_qty = action.quote_qty / sim_price
                executed_price = sim_price
            else:
                executed_qty = action.quote_qty
                executed_price = 0.0
        else:
            executed_qty = action.base_qty
            executed_price = sim_price

        logger.info(
            "[PAPER] %s %s | qty_quote=%.2f | qty_base=%.8f | "
            "precio=%.8f | reason=%s",
            action.action,
            action.symbol,
            action.quote_qty,
            executed_qty,
            executed_price,
            action.reason,
        )
        return ExecutionResult(
            action=action,
            success=True,
            executed_qty=executed_qty,
            executed_price=executed_price,
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
                # Leer balance real de Binance (evita problemas por
                # comisiones que reducen la cantidad disponible).
                sell_qty = action.base_qty
                try:
                    portfolio = self._client.get_portfolio()
                    quote = self._config.portfolio.quote_asset
                    base_asset = action.symbol.replace(quote, "")
                    real_balance = portfolio.get(base_asset, 0.0)
                    if real_balance > 0:
                        sell_qty = real_balance
                        logger.info(
                            "SELL %s: usando balance real %.8f (teórico %.8f)",
                            action.symbol,
                            real_balance,
                            action.base_qty,
                        )
                except Exception as exc:
                    logger.warning(
                        "No se pudo leer balance real de %s: %s",
                        action.symbol,
                        exc,
                    )

                # Validar y ajustar cantidad al step_size de Binance
                adjusted_qty, error = self._client.validate_and_adjust_sell(
                    action.symbol,
                    sell_qty,
                )
                if error:
                    logger.warning(
                        "[SKIP] SELL %s omitida: %s",
                        action.symbol,
                        error,
                    )
                    return ExecutionResult(
                        action=action,
                        success=False,
                        error=error,
                    )
                # Cancelar órdenes abiertas antes de vender
                self._client.cancel_open_orders(action.symbol)
                order = self._client.place_market_sell(action.symbol, adjusted_qty)
            else:
                # Validar monto mínimo para compra
                buy_error = self._client.validate_buy(
                    action.symbol,
                    action.quote_qty,
                )
                if buy_error:
                    logger.warning(
                        "[SKIP] BUY %s omitida: %s",
                        action.symbol,
                        buy_error,
                    )
                    return ExecutionResult(
                        action=action,
                        success=False,
                        error=buy_error,
                    )
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
                # Leer balance real post-compra: Binance cobra comisión
                # en el activo comprado, así que el saldo disponible es
                # menor que total_qty de los fills.
                oco_qty = total_qty
                try:
                    portfolio = self._client.get_portfolio()
                    quote = self._config.portfolio.quote_asset
                    base_asset = action.symbol.replace(quote, "")
                    real_balance = portfolio.get(base_asset, 0.0)
                    if 0 < real_balance < total_qty:
                        oco_qty = real_balance
                        logger.info(
                            "OCO %s: usando balance real %.8f (fills %.8f, comisión %.8f)",
                            action.symbol,
                            real_balance,
                            total_qty,
                            total_qty - real_balance,
                        )
                except Exception as exc:
                    logger.warning(
                        "No se pudo leer balance real post-compra de %s: %s",
                        action.symbol,
                        exc,
                    )
                self._client.place_oco_sell(
                    symbol=action.symbol,
                    quantity=oco_qty,
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

        except BinanceAPIException as exc:
            # -2010 en SELL: intentar fallback via Convert API
            if exc.code == -2010 and action.action == "SELL":
                return self._try_convert_sell(action)
            # -1013: lot size / min notional no cumplido
            if exc.code in (-2010, -1013):
                logger.warning(
                    "[SKIP] %s %s omitida (Binance %d): %s",
                    action.action,
                    action.symbol,
                    exc.code,
                    exc.message,
                )
            else:
                logger.error(
                    "[LIVE] Error Binance %s %s: %s",
                    action.action,
                    action.symbol,
                    exc,
                )
            return ExecutionResult(
                action=action,
                success=False,
                error=str(exc),
            )
        except Exception as exc:
            logger.error("[LIVE] Error ejecutando %s %s: %s", action.action, action.symbol, exc)
            return ExecutionResult(
                action=action,
                success=False,
                error=str(exc),
            )

    def _try_convert_sell(self, action: TradeAction) -> ExecutionResult:
        """Fallback: vender via Convert API cuando Spot da -2010."""
        assert self._client is not None
        quote = self._config.portfolio.quote_asset  # "USDT"
        from_asset = action.symbol.replace(quote, "")

        logger.info(
            "[CONVERT FALLBACK] Intentando vender %s via Convert API...",
            action.symbol,
        )
        try:
            order = self._client.convert_sell(
                from_asset=from_asset,
                to_asset=quote,
                amount=action.base_qty,
            )
            fills = order.get("fills", [])
            total_qty = sum(float(f["qty"]) for f in fills)
            total_commission = sum(float(f["commission"]) for f in fills)
            avg_price = (
                sum(float(f["price"]) * float(f["qty"]) for f in fills) / total_qty
                if total_qty > 0
                else 0.0
            )
            logger.info(
                "[CONVERT] SELL %s OK | precio=%.8f | qty=%.8f",
                action.symbol,
                avg_price,
                total_qty,
            )
            return ExecutionResult(
                action=action,
                success=True,
                executed_qty=total_qty,
                executed_price=avg_price,
                commission=total_commission,
            )
        except Exception as convert_exc:
            logger.warning(
                "[CONVERT FALLBACK] También falló para %s: %s",
                action.symbol,
                convert_exc,
            )
            return ExecutionResult(
                action=action,
                success=False,
                error=f"Spot: -2010 | Convert: {convert_exc}",
            )
