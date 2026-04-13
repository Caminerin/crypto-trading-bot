"""Tests para el ejecutor de órdenes (modo paper trading)."""

import pytest

from src.config import AppConfig
from src.execution.executor import OrderExecutor
from src.portfolio.manager import TradeAction


@pytest.fixture
def paper_executor() -> OrderExecutor:
    config = AppConfig(trading_mode="paper")
    return OrderExecutor(config, trading_client=None)


class TestPaperTrading:
    def test_buy_succeeds(self, paper_executor: OrderExecutor) -> None:
        action = TradeAction(
            action="BUY",
            symbol="BTCUSDT",
            quote_qty=200.0,
            base_qty=0,
            reason="Test buy",
            probability=0.85,
        )
        results = paper_executor.execute([action])
        assert len(results) == 1
        assert results[0].success is True

    def test_sell_succeeds(self, paper_executor: OrderExecutor) -> None:
        action = TradeAction(
            action="SELL",
            symbol="ETHUSDT",
            quote_qty=0,
            base_qty=0.5,
            reason="Test sell",
            probability=0.0,
        )
        results = paper_executor.execute([action])
        assert len(results) == 1
        assert results[0].success is True

    def test_sells_execute_before_buys(self, paper_executor: OrderExecutor) -> None:
        """Las ventas deben ejecutarse antes que las compras."""
        buy = TradeAction("BUY", "BTCUSDT", 200, 0, "Buy", 0.8)
        sell = TradeAction("SELL", "ETHUSDT", 0, 1.0, "Sell", 0.0)

        # Pasamos compra primero, pero debe ejecutar venta antes
        results = paper_executor.execute([buy, sell])
        assert results[0].action.action == "SELL"
        assert results[1].action.action == "BUY"

    def test_multiple_actions(self, paper_executor: OrderExecutor) -> None:
        actions = [
            TradeAction("BUY", "BTCUSDT", 200, 0, "Buy BTC", 0.85),
            TradeAction("BUY", "ETHUSDT", 200, 0, "Buy ETH", 0.80),
            TradeAction("SELL", "SOLUSDT", 0, 10.0, "Sell SOL", 0.0),
        ]
        results = paper_executor.execute(actions)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_live_without_client_fails(self) -> None:
        config = AppConfig(trading_mode="live")
        executor = OrderExecutor(config, trading_client=None)
        action = TradeAction("BUY", "BTCUSDT", 200, 0, "Test", 0.8)
        results = executor.execute([action])
        assert results[0].success is False
        assert "no configurado" in results[0].error
