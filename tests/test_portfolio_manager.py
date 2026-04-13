"""Tests para el módulo de gestión de cartera."""

import pytest

from src.config import PortfolioConfig, RiskConfig
from src.portfolio.manager import PortfolioManager


@pytest.fixture
def manager() -> PortfolioManager:
    return PortfolioManager(
        portfolio_cfg=PortfolioConfig(),
        risk_cfg=RiskConfig(),
    )


class TestDecideActions:
    def test_sell_positions_not_recommended(self, manager: PortfolioManager) -> None:
        """Debe vender posiciones que ya no están recomendadas."""
        portfolio = {"USDT": 500.0, "BTC": 0.01, "ETH": 0.5}
        # Solo recomendar SOL, no BTC ni ETH
        recommendations = [("SOLUSDT", 0.85)]
        prices = {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0, "SOLUSDT": 150.0}

        actions = manager.decide_actions(portfolio, 2500.0, recommendations, prices)

        sell_symbols = [a.symbol for a in actions if a.action == "SELL"]
        assert "BTCUSDT" in sell_symbols
        assert "ETHUSDT" in sell_symbols

    def test_buy_recommended_coins(self, manager: PortfolioManager) -> None:
        """Debe comprar monedas recomendadas que no tiene."""
        portfolio = {"USDT": 1000.0}
        recommendations = [("BTCUSDT", 0.80), ("ETHUSDT", 0.75)]
        prices = {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}

        actions = manager.decide_actions(portfolio, 1000.0, recommendations, prices)

        buy_symbols = [a.symbol for a in actions if a.action == "BUY"]
        assert "BTCUSDT" in buy_symbols
        assert "ETHUSDT" in buy_symbols

    def test_respects_max_positions(self, manager: PortfolioManager) -> None:
        """No debe superar el máximo de 5 posiciones."""
        portfolio = {"USDT": 5000.0}
        recommendations = [
            (f"COIN{i}USDT", 0.90 - i * 0.01)
            for i in range(10)
        ]
        prices = {f"COIN{i}USDT": 100.0 for i in range(10)}

        actions = manager.decide_actions(portfolio, 5000.0, recommendations, prices)

        buy_count = sum(1 for a in actions if a.action == "BUY")
        assert buy_count <= 5

    def test_keeps_stablecoin_reserve(self, manager: PortfolioManager) -> None:
        """Debe mantener al menos un 10% en stablecoins."""
        portfolio = {"USDT": 100.0}  # Justo el 10% de 1000
        recommendations = [("BTCUSDT", 0.90)]
        prices = {"BTCUSDT": 50000.0}

        actions = manager.decide_actions(portfolio, 1000.0, recommendations, prices)

        # No debería comprar porque todo el USDT es reserva
        buy_actions = [a for a in actions if a.action == "BUY"]
        assert len(buy_actions) == 0

    def test_skips_small_orders(self, manager: PortfolioManager) -> None:
        """No compra si el importe por moneda es menor a 10 USDT."""
        portfolio = {"USDT": 15.0}
        recommendations = [("BTCUSDT", 0.90), ("ETHUSDT", 0.85)]
        prices = {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}

        actions = manager.decide_actions(portfolio, 150.0, recommendations, prices)

        buy_actions = [a for a in actions if a.action == "BUY"]
        # Con 15 USDT - 15 reserva (10%) = 0 para trading
        assert len(buy_actions) == 0

    def test_no_actions_when_no_recommendations(self, manager: PortfolioManager) -> None:
        """Sin recomendaciones, no hace nada (solo ventas si tiene posiciones)."""
        portfolio = {"USDT": 1000.0}
        actions = manager.decide_actions(portfolio, 1000.0, [], {})
        assert len(actions) == 0

    def test_keeps_existing_recommended_positions(self, manager: PortfolioManager) -> None:
        """No vende posiciones que siguen recomendadas."""
        portfolio = {"USDT": 500.0, "BTC": 0.01}
        recommendations = [("BTCUSDT", 0.85)]
        prices = {"BTCUSDT": 50000.0}

        actions = manager.decide_actions(portfolio, 1000.0, recommendations, prices)

        sell_symbols = [a.symbol for a in actions if a.action == "SELL"]
        assert "BTCUSDT" not in sell_symbols
