"""
Test de compra-venta real mínima.

Ejecuta una compra de mercado mínima (≈10 USDT/USDC de BTC) y luego
la vende inmediatamente, para verificar que el flujo de trading
del proyecto funciona de extremo a extremo con dinero real.

Auto-detecta si el balance está en USDT o USDC y usa el par
correspondiente (ej. BTCUSDT o BTCUSDC).

Uso:
    cd /root/crypto-trading-bot
    source .venv/bin/activate
    set -a && source .env && set +a
    python -m scripts.test_real_trade [--base BTC] [--amount 10]

Si no se pasan argumentos, compra 10 USD de BTC por defecto.
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

# Stablecoins soportadas, en orden de preferencia
_QUOTE_PREFERENCE = ("USDT", "USDC")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test de compra-venta real mínima",
    )
    parser.add_argument(
        "--base",
        default="BTC",
        help="Moneda base a comprar (default: BTC)",
    )
    parser.add_argument(
        "--quote",
        default="",
        help="Moneda quote (USDT/USDC). Si no se indica, se auto-detecta.",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=10.0,
        help="Monto en quote a gastar (default: 10.0)",
    )
    parser.add_argument(
        "--skip-sell",
        action="store_true",
        help="Solo comprar, no vender después",
    )
    parser.add_argument(
        "--sell-only",
        action="store_true",
        help="Solo vender el base asset existente, sin comprar.",
    )
    return parser.parse_args()


def _detect_quote(
    portfolio: dict[str, float],
    amount: float,
) -> str | None:
    """Elige la stablecoin con balance suficiente."""
    for quote in _QUOTE_PREFERENCE:
        free = portfolio.get(quote, 0.0)
        if free >= amount:
            return quote
    return None


def run_test(
    base: str,
    quote: str,
    amount: float,
    skip_sell: bool,
    *,
    sell_only: bool = False,
) -> bool:
    """Ejecuta el test de compra-venta.  Devuelve True si todo fue bien."""

    config = load_config()
    binance_cfg = config.binance

    if not binance_cfg.api_key or not binance_cfg.api_secret:
        logger.error(
            "Faltan BINANCE_API_KEY / BINANCE_API_SECRET en el entorno.",
        )
        return False

    # ------------------------------------------------------------------ #
    # 1. Conectar
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("TEST COMPRA-VENTA REAL")
    logger.info("=" * 60)

    try:
        client = BinanceTradingClient(
            binance_cfg, PortfolioConfig(), RiskConfig(),
        )
    except Exception as exc:
        logger.error("No se pudo conectar a Binance: %s", exc)
        return False

    # ------------------------------------------------------------------ #
    # 2. Balance antes
    # ------------------------------------------------------------------ #
    portfolio_before = client.get_portfolio()
    value_before = client.get_portfolio_value_usdt()
    logger.info("Balance antes: $%.2f (valoración USDT)", value_before)
    for asset, qty in portfolio_before.items():
        logger.info("  %s: %.8f", asset, qty)

    # ------------------------------------------------------------------ #
    # 3. Detectar o validar quote asset
    # ------------------------------------------------------------------ #
    if not quote:
        detected = _detect_quote(portfolio_before, amount)
        if detected is None:
            logger.error(
                "No hay stablecoin con balance >= $%.2f. "
                "Tienes: %s",
                amount,
                {q: portfolio_before.get(q, 0.0) for q in _QUOTE_PREFERENCE},
            )
            return False
        quote = detected
        logger.info("Auto-detectado quote asset: %s", quote)

    quote_free = portfolio_before.get(quote, 0.0)
    if quote_free < amount:
        logger.error(
            "%s libre (%.2f) insuficiente para comprar $%.2f.",
            quote,
            quote_free,
            amount,
        )
        return False

    symbol = f"{base}{quote}"

    # ------------------------------------------------------------------ #
    # Modo --sell-only: vender lo que haya sin comprar
    # ------------------------------------------------------------------ #
    if sell_only:
        return _sell_existing(client, base, quote, symbol, portfolio_before)

    logger.info(
        "Operando %s — gastando %.2f %s", symbol, amount, quote,
    )

    # ------------------------------------------------------------------ #
    # 4. Validar par
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
    # 5. COMPRA
    # ------------------------------------------------------------------ #
    logger.info("-" * 40)
    logger.info(
        "PASO 1: COMPRA de %s por %.2f %s", symbol, amount, quote,
    )
    logger.info("-" * 40)

    try:
        buy_order = client.place_market_buy(symbol, amount)
    except BinanceAPIException as exc:
        logger.error(
            "Error Binance al comprar: code=%d msg=%s",
            exc.code,
            exc.message,
        )
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
    total_cost = sum(
        float(f["price"]) * float(f["qty"]) for f in fills
    )
    commission = sum(float(f["commission"]) for f in fills)

    logger.info("COMPRA OK:")
    logger.info("  Order ID : %s", buy_order.get("orderId"))
    logger.info("  Status   : %s", buy_order.get("status"))
    logger.info("  Cantidad : %.8f %s", bought_qty, base)
    logger.info("  Precio   : %.2f %s", avg_price, quote)
    logger.info("  Costo    : %.4f %s", total_cost, quote)
    logger.info("  Comisión : %.8f", commission)

    if skip_sell:
        logger.info("--skip-sell activo, no se vende.")
        _print_final_balance(client)
        return True

    # ------------------------------------------------------------------ #
    # 6. VENTA (inmediata)
    # ------------------------------------------------------------------ #
    time.sleep(2)  # pausa para que Binance refleje el balance

    # Leer balance real (la comisión ya está descontada)
    portfolio_after_buy = client.get_portfolio()
    real_qty = portfolio_after_buy.get(base, 0.0)
    logger.info(
        "Balance real de %s tras compra: %.8f (comisión ya descontada)",
        base,
        real_qty,
    )

    adjusted_qty, sell_error = client.validate_and_adjust_sell(
        symbol, real_qty,
    )
    if sell_error:
        logger.warning(
            "No se puede vender la cantidad comprada: %s", sell_error,
        )
        logger.info(
            "Puede que la cantidad sea demasiado pequeña (dust).",
        )
        _print_final_balance(client)
        return True  # la compra sí funcionó

    logger.info("-" * 40)
    logger.info("PASO 2: VENTA de %.8f %s", adjusted_qty, symbol)
    logger.info("-" * 40)

    try:
        sell_order = client.place_market_sell(symbol, adjusted_qty)
    except BinanceAPIException as exc:
        logger.error(
            "Error Binance al vender: code=%d msg=%s",
            exc.code,
            exc.message,
        )
        logger.info(
            "La compra SÍ se ejecutó. "
            "Puede que necesites vender manualmente.",
        )
        _print_final_balance(client)
        return False
    except Exception as exc:
        logger.error("Error inesperado al vender: %s", exc)
        _print_final_balance(client)
        return False

    sell_fills = sell_order.get("fills", [])
    sold_qty = sum(float(f["qty"]) for f in sell_fills)
    sell_price = (
        sum(float(f["price"]) * float(f["qty"]) for f in sell_fills)
        / sold_qty
        if sold_qty > 0
        else 0.0
    )
    sell_total = sum(
        float(f["price"]) * float(f["qty"]) for f in sell_fills
    )

    logger.info("VENTA OK:")
    logger.info("  Order ID : %s", sell_order.get("orderId"))
    logger.info("  Status   : %s", sell_order.get("status"))
    logger.info("  Cantidad : %.8f", sold_qty)
    logger.info("  Precio   : %.2f %s", sell_price, quote)
    logger.info("  Recibido : %.4f %s", sell_total, quote)

    # ------------------------------------------------------------------ #
    # 7. Resumen
    # ------------------------------------------------------------------ #
    pnl = sell_total - total_cost
    logger.info("=" * 60)
    logger.info("RESUMEN DEL TEST")
    logger.info(
        "  Compra : %.4f %s -> %.8f %s @ %.2f",
        total_cost, quote, bought_qty, base, avg_price,
    )
    logger.info(
        "  Venta  : %.8f %s -> %.4f %s @ %.2f",
        sold_qty, base, sell_total, quote, sell_price,
    )
    logger.info(
        "  PnL    : %.4f %s (comisiones incluidas en spread)",
        pnl,
        quote,
    )
    logger.info("=" * 60)

    _print_final_balance(client)
    return True


def _sell_existing(
    client: BinanceTradingClient,
    base: str,
    quote: str,
    symbol: str,
    portfolio: dict[str, float],
) -> bool:
    """Vende todo el base asset disponible sin comprar primero."""
    real_qty = portfolio.get(base, 0.0)
    if real_qty <= 0:
        logger.error("No tienes %s para vender.", base)
        return False

    logger.info("="* 60)
    logger.info("MODO SELL-ONLY: vendiendo %.8f %s", real_qty, base)
    logger.info("="* 60)

    adjusted_qty, sell_error = client.validate_and_adjust_sell(
        symbol, real_qty,
    )
    if sell_error:
        logger.error(
            "No se puede vender %.8f %s: %s", real_qty, base, sell_error,
        )
        return False

    logger.info(
        "VENTA de %.8f %s (%s)", adjusted_qty, base, symbol,
    )

    try:
        sell_order = client.place_market_sell(symbol, adjusted_qty)
    except BinanceAPIException as exc:
        logger.error(
            "Error Binance al vender: code=%d msg=%s",
            exc.code,
            exc.message,
        )
        return False
    except Exception as exc:
        logger.error("Error inesperado al vender: %s", exc)
        return False

    sell_fills = sell_order.get("fills", [])
    sold_qty = sum(float(f["qty"]) for f in sell_fills)
    sell_price = (
        sum(float(f["price"]) * float(f["qty"]) for f in sell_fills)
        / sold_qty
        if sold_qty > 0
        else 0.0
    )
    sell_total = sum(
        float(f["price"]) * float(f["qty"]) for f in sell_fills
    )

    logger.info("VENTA OK:")
    logger.info("  Order ID : %s", sell_order.get("orderId"))
    logger.info("  Status   : %s", sell_order.get("status"))
    logger.info("  Cantidad : %.8f %s", sold_qty, base)
    logger.info("  Precio   : %.2f %s", sell_price, quote)
    logger.info("  Recibido : %.4f %s", sell_total, quote)

    _print_final_balance(client)
    return True


def _print_final_balance(client: BinanceTradingClient) -> None:
    """Muestra el balance final."""
    time.sleep(1)
    portfolio_after = client.get_portfolio()
    value_after = client.get_portfolio_value_usdt()
    logger.info("Balance final: $%.2f (valoración USDT)", value_after)
    for asset, qty in portfolio_after.items():
        logger.info("  %s: %.8f", asset, qty)


def main() -> None:
    args = _parse_args()
    ok = run_test(
        args.base, args.quote, args.amount, args.skip_sell,
        sell_only=args.sell_only,
    )
    if ok:
        logger.info("TEST COMPLETADO CON ÉXITO")
    else:
        logger.error("TEST FALLIDO")
        sys.exit(1)


if __name__ == "__main__":
    main()
