#!/usr/bin/env python3
"""
Backtesting DCA por moneda individual.

Prueba cada moneda por separado con todas las combinaciones de
dip/TP/SL para encontrar la politica optima de cada una.

Uso:
    python3 scripts/backtest_dca_per_coin.py [--days 365] [--budget 30]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd
from backtest_dca import (
    DCA_ASSETS,
    BacktestResult,
    download_daily_klines,
    run_backtest,
)

# ---------------------------------------------------------------------------
# Monedas y parametros
# ---------------------------------------------------------------------------

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

DIP_THRESHOLDS: list[float] = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
TP_MULTIPLIERS: list[float] = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
SL_MULTIPLIERS: list[float] = [1.5, 2.0, 2.5, 3.0]


@dataclass
class CoinBest:
    """Mejor resultado para una moneda."""

    coin: str
    config_name: str
    dip: float
    tp_mult: float
    sl_mult: float
    result: BacktestResult


def _fmt_sign(val: float) -> str:
    return "+" if val >= 0 else "-"


def _fmt_mult(val: float) -> str:
    return f"{val:.1f}".rstrip("0").rstrip(".")


def run_coin_matrix(
    coin: str,
    prices: dict[str, pd.DataFrame],
    budget: float,
    days: int,
) -> tuple[str, CoinBest]:
    """Prueba todas las combinaciones para una moneda.

    Returns (report_text, best_result).
    """
    results: list[tuple[str, float, float, float, BacktestResult]] = []

    for dip, tp_m, sl_m in product(
        DIP_THRESHOLDS, TP_MULTIPLIERS, SL_MULTIPLIERS,
    ):
        if sl_m >= tp_m:
            continue

        cfg = {
            coin: {
                "dip_threshold": -dip,
                "take_profit": dip * tp_m,
                "stop_loss": -(dip * sl_m),
            },
        }
        coin_prices = {coin: prices[coin]}
        r = run_backtest(coin_prices, budget, asset_configs=cfg)
        name = f"d{dip*100:.0f} TP{_fmt_mult(tp_m)}x SL{_fmt_mult(sl_m)}x"
        results.append((name, dip, tp_m, sl_m, r))

    # Ordenar por P&L
    results.sort(key=lambda x: x[4].final_value, reverse=True)

    nm = coin.replace("USDT", "")
    total = len(results)

    lines: list[str] = [
        "",
        "=" * 100,
        f"  {nm} — {total} combinaciones probadas",
        "=" * 100,
        "",
        f"  {'#':>3s}"
        f" {'Config':<28s}"
        f" {'Dip':>5s}"
        f" {'TP%':>6s}"
        f" {'SL%':>6s}"
        f" {'P&L':>9s}"
        f" {'%':>7s}"
        f" {'Trades':>7s}"
        f" {'TP':>4s}"
        f" {'SL':>4s}"
        f" {'Win%':>5s}"
        f" {'MaxDD':>7s}"
        f" {'HoldD':>6s}",
        "  " + "-" * 97,
    ]

    # TOP 15
    show_n = min(15, total)
    for rank, (name, dip, tp_m, sl_m, r) in enumerate(
        results[:show_n], 1,
    ):
        pnl = r.final_value - r.initial_budget
        pnl_pct = (
            pnl / r.initial_budget * 100
            if r.initial_budget > 0 else 0
        )
        s = _fmt_sign(pnl)
        sp = _fmt_sign(pnl_pct)
        wr = (
            f"{r.wins / r.sells * 100:.0f}%"
            if r.sells > 0 else "N/A"
        )
        tp_pct = dip * tp_m * 100
        sl_pct = dip * sl_m * 100
        lines.append(
            f"  {rank:>3d}"
            f" {name:<28s}"
            f" {dip*100:>4.0f}%"
            f" +{tp_pct:>4.1f}%"
            f" -{sl_pct:>4.1f}%"
            f" {s}${abs(pnl):>7.2f}"
            f" {sp}{abs(pnl_pct):>5.1f}%"
            f" {r.total_trades:>7d}"
            f" {r.take_profits:>4d}"
            f" {r.stop_losses:>4d}"
            f" {wr:>5s}"
            f" {-r.max_drawdown_pct:>6.1f}%"
            f" {r.avg_hold_days:>5.0f}d",
        )

    # Stats
    profitable = sum(
        1 for *_, r in results
        if r.final_value > r.initial_budget
    )
    pct_ok = profitable * 100 // total if total > 0 else 0
    lines.extend([
        "",
        f"  Rentables: {profitable}/{total} ({pct_ok}%)",
        "",
    ])

    # Mejor resultado
    b_name, b_dip, b_tp, b_sl, b_r = results[0]
    best = CoinBest(
        coin=coin,
        config_name=b_name,
        dip=b_dip,
        tp_mult=b_tp,
        sl_mult=b_sl,
        result=b_r,
    )

    return "\n".join(lines), best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest DCA por moneda individual",
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

    # Descargar datos
    all_symbols = sorted(set(DCA_ASSETS))
    print(f"\nDescargando {args.days} dias de datos...")
    prices: dict[str, pd.DataFrame] = {}
    for symbol in all_symbols:
        nm = symbol.replace("USDT", "")
        print(f"  {nm}...", end=" ", flush=True)
        df = download_daily_klines(symbol, args.days)
        prices[symbol] = df
        print(f"{len(df)} velas")

    # Ejecutar por moneda
    full_report: list[str] = [
        "",
        "#" * 100,
        "  BACKTESTING DCA POR MONEDA — POLITICA OPTIMA INDIVIDUAL",
        "#" * 100,
        f"  Periodo: {args.days} dias"
        f" | Presupuesto por moneda: ${args.budget:.2f} USDT",
    ]

    bests: list[CoinBest] = []

    for coin in COINS:
        nm = coin.replace("USDT", "")
        print(f"\nAnalizando {nm}...", flush=True)
        report, best = run_coin_matrix(
            coin, prices, args.budget, args.days,
        )
        full_report.append(report)
        bests.append(best)

    # Resumen final
    full_report.extend([
        "",
        "#" * 100,
        "  RESUMEN: MEJOR POLITICA POR MONEDA",
        "#" * 100,
        "",
        f"  {'Moneda':<6s}"
        f" {'Config':<28s}"
        f" {'Dip':>5s}"
        f" {'TP%':>6s}"
        f" {'SL%':>6s}"
        f" {'P&L':>9s}"
        f" {'%':>7s}"
        f" {'Win%':>5s}"
        f" {'MaxDD':>7s}",
        "  " + "-" * 80,
    ])

    for b in bests:
        nm = b.coin.replace("USDT", "")
        pnl = b.result.final_value - b.result.initial_budget
        pnl_pct = (
            pnl / b.result.initial_budget * 100
            if b.result.initial_budget > 0 else 0
        )
        s = _fmt_sign(pnl)
        sp = _fmt_sign(pnl_pct)
        wr = (
            f"{b.result.wins / b.result.sells * 100:.0f}%"
            if b.result.sells > 0 else "N/A"
        )
        tp_pct = b.dip * b.tp_mult * 100
        sl_pct = b.dip * b.sl_mult * 100
        full_report.append(
            f"  {nm:<6s}"
            f" {b.config_name:<28s}"
            f" {b.dip*100:>4.0f}%"
            f" +{tp_pct:>4.1f}%"
            f" -{sl_pct:>4.1f}%"
            f" {s}${abs(pnl):>7.2f}"
            f" {sp}{abs(pnl_pct):>5.1f}%"
            f" {wr:>5s}"
            f" {-b.result.max_drawdown_pct:>6.1f}%",
        )

    full_report.append("")
    report_text = "\n".join(full_report)
    print(report_text)

    # Guardar
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    rpath = out_dir / "backtest_dca_per_coin_report.txt"
    rpath.write_text(report_text)
    print(f"\nReporte guardado en: {rpath}")


if __name__ == "__main__":
    main()
