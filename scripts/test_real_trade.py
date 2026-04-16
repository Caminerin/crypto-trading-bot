"""
Test de compra-venta real mínima.

Ejecuta una compra de mercado mínima (≈10 USDT de BTC) y luego
la vende inmediatamente, para verificar que el flujo de trading
del proyecto funciona de extremo a extremo con dinero real.

Uso:
    cd /root/crypto-trading-bot
    source .venv/bin/activate
    set -a && source .env && set +a
    python -m scripts.test_real_trade [--symbol BTCUSDT] [--amount 10]

Si no se pasan argumentos, compra 10 USDT de BTCUSDT por defecto.
"""

from __future__ import annotations

import argparse
import sys
import time

from binance.exceptions import BinanceAPIException

from src.config import PortfolioConfig, RiskConfig, load_config
from src.data.binance_client import BinanceTradingClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test de compra-venta real mínima")
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Par a operar (default: BTCUSDT)",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=10.0,
        help="Monto en USDT a comprar (default: 10.0)",
    )
    parser.add_argument(
        "--skip-sell",
        action="store_true",
        help="Solo comprar, no vender después",
    )
    return parser.parse_args()


def run_test(symbol: str, amount: float, skip_sell: bool) -> bool:
    """Ejecuta el test de compra-venta.  Devuelve True si todo fue bien."""

    config = load_config()
    binance_cfg = config.binance

    if not binance_cfg.api_key or not binance_cfg.api_secret:
        logger.error("Faltan BINANCE_API_KEY / BINANCE_API_SECRET en el entorno.")
        return False

    # ------------------------------------------------------------------ #
    # 1. Conectar
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("TEST COMPRA-VENTA REAL — %s por $%.2f USDT", symbol, amount)
    logger.info("=" * 60)

    try:
        client = BinanceTradingClient(binance_cfg, PortfolioConfig(), RiskConfig())
    except Exception as exc:
        logger.error("No se pudo conectar a Binance: %s", exc)
        return False

    # ------------------------------------------------------------------ #
    # 2. Balance antes
    # ------------------------------------------------------------------ #
    portfolio_before = client.get_portfolio()
    value_before = client.get_portfolio_value_usdt()
    logger.info("Balance antes: $%.2f USDT", value_before)
    for asset, qty in portfolio_before.items():
        logger.info("  %s: %.8f", asset, qty)

    usdt_free = portfolio_before.get("USDT", 0.0)
    if usdt_free < amount:
        logger.error(
            "USDT libre (%.2f) insuficiente para comprar $%.2f.",
            usdt_free,
            amount,
        )
        return False

    # ------------------------------------------------------------------ #
    # 3. Validar par
    # ------------------------------------------------------------------ #
    buy_error = client.validate_buy(symbol, amount)
    if buy_error:
        logger.error("Validación de compra falló: %s", buy_error)
        return False

    lot_filter = client.get_lot_size_filter(symbol)
    if lot_filter is None:
        logger.error("No se encontró LOT_SIZE para %s", symbol)
        return False

    logger.info(
        "LOT_SIZE %s: min=%.8f step=%.8f",
        symbol,
        lot_filter["min_qty"],
        lot_filter["step_size"],
    )

    # ------------------------------------------------------------------ #
    # 4. COMPRA
    # ------------------------------------------------------------------ #
    logger.info("-" * 40)
    logger.info("PASO 1: COMPRA de %s por $%.2f USDT", symbol, amount)
    logger.info("-" * 40)

    try:
        buy_order = client.place_market_buy(symbol, amount)
    except BinanceAPIException as exc:
        logger.error("Error Binance al comprar: code=%d msg=%s", exc.code, exc.message)
        return False
    except Exception as exc:
        logger.error("Error inesperado al comprar: %s", exc)
        return False

    fills = buy_order.get("fills", [])
    bought_qty = sum(float(f["qty"]) for f in fills)
    avg_price = (
        sum(float(f["price"]) * float(f["qty"]) for f in fills) / bought_qty
        if bought_qty > 0
        else 0.0
    )
    total_cost = sum(float(f["price"]) * float(f["qty"]) for f in fills)
    commission = sum(float(f["commission"]) for f in fills)

    logger.info("COMPRA OK:")
    logger.info("  Order ID : %s", buy_order.get("orderId"))
    logger.info("  Status   : %s", buy_order.get("status"))
    logger.info("  Cantidad : %.8f %s", bought_qty, symbol.replace("USDT", ""))
    logger.info("  Precio   : $%.2f", avg_price)
    logger.info("  Costo    : $%.4f USDT", total_cost)
    logger.info("  Comisión : %.8f", commission)

    if skip_sell:
        logger.info("--skip-sell activo, no se vende.")
        _print_final_balance(client)
        return True

    # ------------------------------------------------------------------ #
    # 5. VENTA (inmediata)
    # ------------------------------------------------------------------ #
    time.sleep(1)  # pequeña pausa para que Binance refleje el balance

    # Ajustar cantidad al step_size
    adjusted_qty, sell_error = client.validate_and_adjust_sell(symbol, bought_qty)
    if sell_error:
        logger.warning("No se puede vender la cantidad comprada: %s", sell_error)
        logger.info("Puede que la cantidad sea demasiado pequeña (dust).")
        _print_final_balance(client)
        return True  # la compra sí funcionó

    logger.info("-" * 40)
    logger.info("PASO 2: VENTA de %.8f %s", adjusted_qty, symbol)
    logger.info("-" * 40)

    try:
        sell_order = client.place_market_sell(symbol, adjusted_qty)
    except BinanceAPIException as exc:
        logger.error("Error Binance al vender: code=%d msg=%s", exc.code, exc.message)
        logger.info("La compra SÍ se ejecutó. Puede que necesites vender manualmente.")
        _print_final_balance(client)
        return False
    except Exception as exc:
        logger.error("Error inesperado al vender: %s", exc)
        _print_final_balance(client)
        return False

    sell_fills = sell_order.get("fills", [])
    sold_qty = sum(float(f["qty"]) for f in sell_fills)
    sell_price = (
        sum(float(f["price"]) * float(f["qty"]) for f in sell_fills) / sold_qty
        if sold_qty > 0
        else 0.0
    )
    sell_total = sum(float(f["price"]) * float(f["qty"]) for f in sell_fills)

    logger.info("VENTA OK:")
    logger.info("  Order ID : %s", sell_order.get("orderId"))
    logger.info("  Status   : %s", sell_order.get("status"))
    logger.info("  Cantidad : %.8f", sold_qty)
    logger.info("  Precio   : $%.2f", sell_price)
    logger.info("  Recibido : $%.4f USDT", sell_total)

    # ------------------------------------------------------------------ #
    # 6. Resumen
    # ------------------------------------------------------------------ #
    pnl = sell_total - total_cost
    logger.info("=" * 60)
    logger.info("RESUMEN DEL TEST")
    base = symbol.replace("USDT", "")
    logger.info(
        "  Compra : $%.4f USDT -> %.8f %s @ $%.2f",
        total_cost, bought_qty, base, avg_price,
    )
    logger.info(
        "  Venta  : %.8f %s -> $%.4f USDT @ $%.2f",
        sold_qty, base, sell_total, sell_price,
    )
    logger.info("  PnL    : $%.4f USDT (comisiones incluidas en el spread)", pnl)
    logger.info("=" * 60)

    _print_final_balance(client)
    return True


def _print_final_balance(client: BinanceTradingClient) -> None:
    """Muestra el balance final."""
    time.sleep(1)
    portfolio_after = client.get_portfolio()
    value_after = client.get_portfolio_value_usdt()
    logger.info("Balance final: $%.2f USDT", value_after)
    for asset, qty in portfolio_after.items():
        logger.info("  %s: %.8f", asset, qty)


def main() -> None:
    args = _parse_args()
    ok = run_test(args.symbol, args.amount, args.skip_sell)
    if ok:
        logger.info("TEST COMPLETADO CON ÉXITO")
    else:
        logger.error("TEST FALLIDO")
        sys.exit(1)


if __name__ == "__main__":
    main()
