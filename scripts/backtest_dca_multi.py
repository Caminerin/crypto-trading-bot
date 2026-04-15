#!/usr/bin/env python3
"""
Backtesting multi-combinacion DCA Inteligente.

Ejecuta multiples configuraciones de parametros y muestra una tabla
comparativa para elegir la mejor combinacion.

Uso:
    python3 scripts/backtest_dca_multi.py [--days 365] [--budget 30]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from backtest_dca import (
    DCA_ASSETS,
    BacktestResult,
    download_daily_klines,
    run_backtest,
)

# ---------------------------------------------------------------------------
# Combinaciones a probar
# ---------------------------------------------------------------------------

# Formato: (nombre, {symbol: {dip_threshold, take_profit, stop_loss}})
# Regla base: TP = Nx umbral, SL = Mx umbral

COMBINATIONS: list[tuple[str, dict[str, dict[str, float]]]] = [
    # --- v2 original (referencia) ---
    (
        "v2-base (dip5/3, TP3x, SL2x)",
        {
            "BTCUSDT": {"dip_threshold": -0.03, "take_profit": 0.09, "stop_loss": -0.06},
            "ETHUSDT": {"dip_threshold": -0.05, "take_profit": 0.15, "stop_loss": -0.10},
            "SOLUSDT": {"dip_threshold": -0.05, "take_profit": 0.15, "stop_loss": -0.10},
            "BNBUSDT": {"dip_threshold": -0.03, "take_profit": 0.09, "stop_loss": -0.06},
            "XRPUSDT": {"dip_threshold": -0.05, "take_profit": 0.15, "stop_loss": -0.10},
        },
    ),
    # --- Altcoins -7%, SL 2x ---
    (
        "dip7/3, TP3x, SL2x",
        {
            "BTCUSDT": {"dip_threshold": -0.03, "take_profit": 0.09, "stop_loss": -0.06},
            "ETHUSDT": {"dip_threshold": -0.07, "take_profit": 0.21, "stop_loss": -0.14},
            "SOLUSDT": {"dip_threshold": -0.07, "take_profit": 0.21, "stop_loss": -0.14},
            "BNBUSDT": {"dip_threshold": -0.03, "take_profit": 0.09, "stop_loss": -0.06},
            "XRPUSDT": {"dip_threshold": -0.07, "take_profit": 0.21, "stop_loss": -0.14},
        },
    ),
    # --- Altcoins -7%, SL 2.5x ---
    (
        "dip7/3, TP3x, SL2.5x",
        {
            "BTCUSDT": {"dip_threshold": -0.03, "take_profit": 0.09, "stop_loss": -0.075},
            "ETHUSDT": {"dip_threshold": -0.07, "take_profit": 0.21, "stop_loss": -0.175},
            "SOLUSDT": {"dip_threshold": -0.07, "take_profit": 0.21, "stop_loss": -0.175},
            "BNBUSDT": {"dip_threshold": -0.03, "take_profit": 0.09, "stop_loss": -0.075},
            "XRPUSDT": {"dip_threshold": -0.07, "take_profit": 0.21, "stop_loss": -0.175},
        },
    ),
    # --- Altcoins -10%, SL 2x ---
    (
        "dip10/3, TP3x, SL2x",
        {
            "BTCUSDT": {"dip_threshold": -0.03, "take_profit": 0.09, "stop_loss": -0.06},
            "ETHUSDT": {"dip_threshold": -0.10, "take_profit": 0.30, "stop_loss": -0.20},
            "SOLUSDT": {"dip_threshold": -0.10, "take_profit": 0.30, "stop_loss": -0.20},
            "BNBUSDT": {"dip_threshold": -0.03, "take_profit": 0.09, "stop_loss": -0.06},
            "XRPUSDT": {"dip_threshold": -0.10, "take_profit": 0.30, "stop_loss": -0.20},
        },
    ),
    # --- Solo BTC+BNB (las ganadoras), dip3%, SL 2x ---
    (
        "solo BTC+BNB, dip3, TP3x, SL2x",
        {
            "BTCUSDT": {"dip_threshold": -0.03, "take_profit": 0.09, "stop_loss": -0.06},
            "BNBUSDT": {"dip_threshold": -0.03, "take_profit": 0.09, "stop_loss": -0.06},
        },
    ),
    # --- Todo -5%, TP 4x, SL 2x (TP mas alto) ---
    (
        "dip5/3, TP4x, SL2x",
        {
            "BTCUSDT": {"dip_threshold": -0.03, "take_profit": 0.12, "stop_loss": -0.06},
            "ETHUSDT": {"dip_threshold": -0.05, "take_profit": 0.20, "stop_loss": -0.10},
            "SOLUSDT": {"dip_threshold": -0.05, "take_profit": 0.20, "stop_loss": -0.10},
            "BNBUSDT": {"dip_threshold": -0.03, "take_profit": 0.12, "stop_loss": -0.06},
            "XRPUSDT": {"dip_threshold": -0.05, "take_profit": 0.20, "stop_loss": -0.10},
        },
    ),
    # --- Altcoins -7%, TP 4x, SL 2x ---
    (
        "dip7/3, TP4x, SL2x",
        {
            "BTCUSDT": {"dip_threshold": -0.03, "take_profit": 0.12, "stop_loss": -0.06},
            "ETHUSDT": {"dip_threshold": -0.07, "take_profit": 0.28, "stop_loss": -0.14},
            "SOLUSDT": {"dip_threshold": -0.07, "take_profit": 0.28, "stop_loss": -0.14},
            "BNBUSDT": {"dip_threshold": -0.03, "take_profit": 0.12, "stop_loss": -0.06},
            "XRPUSDT": {"dip_threshold": -0.07, "take_profit": 0.28, "stop_loss": -0.14},
        },
    ),
    # --- BTC -5%, Altcoins -7%, TP3x, SL2x ---
    (
        "BTC-5/alt-7, TP3x, SL2x",
        {
            "BTCUSDT": {"dip_threshold": -0.05, "take_profit": 0.15, "stop_loss": -0.10},
            "ETHUSDT": {"dip_threshold": -0.07, "take_profit": 0.21, "stop_loss": -0.14},
            "SOLUSDT": {"dip_threshold": -0.07, "take_profit": 0.21, "stop_loss": -0.14},
            "BNBUSDT": {"dip_threshold": -0.05, "take_profit": 0.15, "stop_loss": -0.10},
            "XRPUSDT": {"dip_threshold": -0.07, "take_profit": 0.21, "stop_loss": -0.14},
        },
    ),
    # --- TP 2x (vende antes, mas trades cerrados) ---
    (
        "dip5/3, TP2x, SL2x",
        {
            "BTCUSDT": {"dip_threshold": -0.03, "take_profit": 0.06, "stop_loss": -0.06},
            "ETHUSDT": {"dip_threshold": -0.05, "take_profit": 0.10, "stop_loss": -0.10},
            "SOLUSDT": {"dip_threshold": -0.05, "take_profit": 0.10, "stop_loss": -0.10},
            "BNBUSDT": {"dip_threshold": -0.03, "take_profit": 0.06, "stop_loss": -0.06},
            "XRPUSDT": {"dip_threshold": -0.05, "take_profit": 0.10, "stop_loss": -0.10},
        },
    ),
    # --- TP 2x con altcoins -7% ---
    (
        "dip7/3, TP2x, SL2x",
        {
            "BTCUSDT": {"dip_threshold": -0.03, "take_profit": 0.06, "stop_loss": -0.06},
            "ETHUSDT": {"dip_threshold": -0.07, "take_profit": 0.14, "stop_loss": -0.14},
            "SOLUSDT": {"dip_threshold": -0.07, "take_profit": 0.14, "stop_loss": -0.14},
            "BNBUSDT": {"dip_threshold": -0.03, "take_profit": 0.06, "stop_loss": -0.06},
            "XRPUSDT": {"dip_threshold": -0.07, "take_profit": 0.14, "stop_loss": -0.14},
        },
    ),
    # --- Solo BTC+BNB con TP 2x ---
    (
        "solo BTC+BNB, dip3, TP2x, SL2x",
        {
            "BTCUSDT": {"dip_threshold": -0.03, "take_profit": 0.06, "stop_loss": -0.06},
            "BNBUSDT": {"dip_threshold": -0.03, "take_profit": 0.06, "stop_loss": -0.06},
        },
    ),
    # --- TP 2.5x (punto medio entre 2x y 3x) ---
    (
        "dip5/3, TP2.5x, SL2x",
        {
            "BTCUSDT": {"dip_threshold": -0.03, "take_profit": 0.075, "stop_loss": -0.06},
            "ETHUSDT": {"dip_threshold": -0.05, "take_profit": 0.125, "stop_loss": -0.10},
            "SOLUSDT": {"dip_threshold": -0.05, "take_profit": 0.125, "stop_loss": -0.10},
            "BNBUSDT": {"dip_threshold": -0.03, "take_profit": 0.075, "stop_loss": -0.06},
            "XRPUSDT": {"dip_threshold": -0.05, "take_profit": 0.125, "stop_loss": -0.10},
        },
    ),
    # --- Solo BTC+BNB con TP 2.5x ---
    (
        "solo BTC+BNB, dip3, TP2.5x, SL2x",
        {
            "BTCUSDT": {"dip_threshold": -0.03, "take_profit": 0.075, "stop_loss": -0.06},
            "BNBUSDT": {"dip_threshold": -0.03, "take_profit": 0.075, "stop_loss": -0.06},
        },
    ),
]


# ---------------------------------------------------------------------------
# Ejecucion
# ---------------------------------------------------------------------------

def _fmt_sign(val: float) -> str:
    return "+" if val >= 0 else "-"


def run_all(
    prices: dict[str, pd.DataFrame], budget: float, days: int,
) -> str:
    """Ejecuta todas las combinaciones y devuelve tabla comparativa."""
    results: list[tuple[str, BacktestResult]] = []

    for name, cfgs in COMBINATIONS:
        # Filtrar precios a solo los simbolos de esta combinacion
        combo_prices = {s: prices[s] for s in cfgs if s in prices}
        r = run_backtest(combo_prices, budget, asset_configs=cfgs)
        results.append((name, r))

    # Construir tabla
    lines: list[str] = [
        "",
        "=" * 100,
        "  COMPARATIVA BACKTESTING DCA — MULTIPLES COMBINACIONES",
        "=" * 100,
        f"  Periodo: {days} dias | Presupuesto: ${budget:.2f} USDT",
        "",
        f"  {'Combinacion':<35s}"
        f" {'P&L':>9s}"
        f" {'%':>7s}"
        f" {'Trades':>7s}"
        f" {'TP':>4s}"
        f" {'SL':>4s}"
        f" {'Win%':>5s}"
        f" {'MaxDD':>7s}"
        f" {'HoldD':>6s}"
        f" {'Mejor':>7s}"
        f" {'Peor':>7s}",
        "  " + "-" * 96,
    ]

    for name, r in results:
        pnl_pct = (
            (r.final_value - r.initial_budget) / r.initial_budget * 100
            if r.initial_budget > 0 else 0
        )
        s = _fmt_sign(r.total_profit)
        sp = _fmt_sign(pnl_pct)
        wr = (
            f"{r.wins / r.sells * 100:.0f}%"
            if r.sells > 0 else "N/A"
        )
        lines.append(
            f"  {name:<35s}"
            f" {s}${abs(r.total_profit):>7.2f}"
            f" {sp}{abs(pnl_pct):>5.1f}%"
            f" {r.total_trades:>7d}"
            f" {r.take_profits:>4d}"
            f" {r.stop_losses:>4d}"
            f" {wr:>5s}"
            f" {-r.max_drawdown_pct:>6.1f}%"
            f" {r.avg_hold_days:>5.0f}d"
            f" +{r.best_trade_pct:>5.1f}%"
            f" {r.worst_trade_pct:>6.1f}%",
        )

    # Encontrar la mejor combinacion
    best_name, best_r = max(results, key=lambda x: x[1].final_value)
    best_pnl = best_r.final_value - best_r.initial_budget
    bs = _fmt_sign(best_pnl)
    best_pct = best_pnl / best_r.initial_budget * 100 if best_r.initial_budget > 0 else 0
    bsp = _fmt_sign(best_pct)

    lines.extend([
        "",
        "=" * 100,
        f"  GANADORA: {best_name}",
        f"  Resultado: {bs}${abs(best_pnl):.2f} ({bsp}{abs(best_pct):.1f}%)",
        "=" * 100,
        "",
    ])

    # Detalle por moneda de la ganadora
    lines.append("  P&L por moneda de la ganadora:")
    for sym, pnl in best_r.per_asset_pnl.items():
        nm = sym.replace("USDT", "")
        ps = _fmt_sign(pnl)
        lines.append(f"    {nm:4s}: {ps}${abs(pnl):.2f}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest DCA multi-combinacion",
    )
    parser.add_argument(
        "--days", type=int, default=365,
        help="Dias de historia (default: 365)",
    )
    parser.add_argument(
        "--budget", type=float, default=30.0,
        help="Presupuesto DCA en USDT (default: 30)",
    )
    args = parser.parse_args()

    # Descargar datos una sola vez (todas las monedas posibles)
    all_symbols = sorted(set(DCA_ASSETS))
    print(f"\nDescargando {args.days} dias de datos...")
    prices: dict[str, pd.DataFrame] = {}
    for symbol in all_symbols:
        nm = symbol.replace("USDT", "")
        print(f"  {nm}...", end=" ", flush=True)
        df = download_daily_klines(symbol, args.days)
        prices[symbol] = df
        print(f"{len(df)} velas")

    print(f"\nEjecutando {len(COMBINATIONS)} combinaciones...")
    report = run_all(prices, args.budget, args.days)
    print(report)

    # Guardar reporte
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    rpath = out_dir / "backtest_dca_multi_report.txt"
    rpath.write_text(report)
    print(f"Reporte guardado en: {rpath}")


if __name__ == "__main__":
    main()
