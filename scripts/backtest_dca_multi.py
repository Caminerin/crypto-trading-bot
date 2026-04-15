#!/usr/bin/env python3
"""
Backtesting multi-combinacion DCA Inteligente — bateria ampliada.

Genera combinaciones de forma programatica variando:
  - Grupo de monedas (BTC, BNB, BTC+BNB, +ETH, +SOL, 5 monedas)
  - Umbral de compra (dip)
  - Multiplicador Take-Profit
  - Multiplicador Stop-Loss

Uso:
    python3 scripts/backtest_dca_multi.py [--days 365] [--budget 30]
"""

from __future__ import annotations

import argparse
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
# Generador de combinaciones
# ---------------------------------------------------------------------------

# Grupos de monedas a probar
COIN_GROUPS: dict[str, list[str]] = {
    "BTC":         ["BTCUSDT"],
    "BNB":         ["BNBUSDT"],
    "BTC+BNB":     ["BTCUSDT", "BNBUSDT"],
    "BTC+BNB+ETH": ["BTCUSDT", "BNBUSDT", "ETHUSDT"],
    "BTC+BNB+SOL": ["BTCUSDT", "BNBUSDT", "SOLUSDT"],
    "5monedas":    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
}

# Clasificacion de monedas
STABLE_COINS = {"BTCUSDT", "BNBUSDT"}  # dip mas bajo
ALT_COINS = {"ETHUSDT", "SOLUSDT", "XRPUSDT"}  # dip mas alto

# Parametros a variar
# (dip_stable%, dip_alt%)
DIP_COMBOS: list[tuple[float, float]] = [
    (0.02, 0.05),   # 2%/5%  — muy sensible
    (0.03, 0.05),   # 3%/5%  — base original
    (0.03, 0.07),   # 3%/7%  — alt mas selectivo
    (0.04, 0.07),   # 4%/7%  — ambos selectivos
    (0.04, 0.10),   # 4%/10% — muy selectivo
    (0.05, 0.07),   # 5%/7%  — BTC selectivo
    (0.05, 0.10),   # 5%/10% — muy selectivo ambos
    (0.02, 0.03),   # 2%/3%  — todo muy sensible
    (0.03, 0.03),   # 3%/3%  — igual para todos
]

TP_MULTIPLIERS: list[float] = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
SL_MULTIPLIERS: list[float] = [1.5, 2.0, 2.5, 3.0]


def _build_config(
    coins: list[str],
    dip_stable: float,
    dip_alt: float,
    tp_mult: float,
    sl_mult: float,
) -> dict[str, dict[str, float]]:
    """Construye config por moneda dados los parametros."""
    cfg: dict[str, dict[str, float]] = {}
    for coin in coins:
        dip = dip_stable if coin in STABLE_COINS else dip_alt
        cfg[coin] = {
            "dip_threshold": -dip,
            "take_profit": dip * tp_mult,
            "stop_loss": -(dip * sl_mult),
        }
    return cfg


def _make_name(
    group: str,
    dip_stable: float,
    dip_alt: float,
    tp_mult: float,
    sl_mult: float,
) -> str:
    """Genera nombre corto para la combinacion."""
    ds = f"{dip_stable*100:.0f}"
    da = f"{dip_alt*100:.0f}"
    tp = f"{tp_mult:.1f}".rstrip("0").rstrip(".")
    sl = f"{sl_mult:.1f}".rstrip("0").rstrip(".")
    return f"{group} d{ds}/{da} TP{tp}x SL{sl}x"


def generate_combinations() -> list[tuple[str, dict[str, dict[str, float]]]]:
    """Genera todas las combinaciones validas."""
    combos: list[tuple[str, dict[str, dict[str, float]]]] = []
    seen: set[str] = set()

    for group_name, coins in COIN_GROUPS.items():
        has_alt = any(c in ALT_COINS for c in coins)

        for (dip_s, dip_a), tp_m, sl_m in product(
            DIP_COMBOS, TP_MULTIPLIERS, SL_MULTIPLIERS,
        ):
            # Filtro: SL no puede ser >= TP (no tiene sentido)
            if sl_m >= tp_m:
                continue

            # Filtro: si no hay altcoins, dip_a no importa — dedup
            if not has_alt:
                key = f"{group_name}|{dip_s}|TP{tp_m}|SL{sl_m}"
            else:
                key = f"{group_name}|{dip_s}/{dip_a}|TP{tp_m}|SL{sl_m}"

            if key in seen:
                continue
            seen.add(key)

            # Filtro: si solo hay stables, usar dip_s para todo
            effective_dip_a = dip_a if has_alt else dip_s

            name = _make_name(
                group_name, dip_s, effective_dip_a, tp_m, sl_m,
            )
            cfg = _build_config(
                coins, dip_s, effective_dip_a, tp_m, sl_m,
            )
            combos.append((name, cfg))

    return combos


# ---------------------------------------------------------------------------
# Ejecucion y tabla
# ---------------------------------------------------------------------------

def _fmt_sign(val: float) -> str:
    return "+" if val >= 0 else "-"


def run_all(
    prices: dict[str, pd.DataFrame],
    budget: float,
    days: int,
    combinations: list[tuple[str, dict[str, dict[str, float]]]],
) -> str:
    """Ejecuta todas las combinaciones y devuelve tabla comparativa."""
    results: list[tuple[str, BacktestResult]] = []

    total = len(combinations)
    for idx, (name, cfgs) in enumerate(combinations, 1):
        if idx % 50 == 0 or idx == total:
            print(f"  [{idx}/{total}]", flush=True)
        combo_prices = {s: prices[s] for s in cfgs if s in prices}
        r = run_backtest(combo_prices, budget, asset_configs=cfgs)
        results.append((name, r))

    # Ordenar por resultado (mejor primero)
    results.sort(key=lambda x: x[1].final_value, reverse=True)

    # Construir tabla
    lines: list[str] = [
        "",
        "=" * 115,
        "  COMPARATIVA BACKTESTING DCA — BATERIA AMPLIADA",
        "=" * 115,
        f"  Periodo: {days} dias | Presupuesto: ${budget:.2f} USDT"
        f" | {total} combinaciones probadas",
        "",
    ]

    # --- TOP 20 ---
    top_n = min(20, len(results))
    lines.extend([
        f"  TOP {top_n} MEJORES:",
        "",
        f"  {'#':>3s}"
        f" {'Combinacion':<38s}"
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
        "  " + "-" * 112,
    ])

    for rank, (name, r) in enumerate(results[:top_n], 1):
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
            f"  {rank:>3d}"
            f" {name:<38s}"
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

    # --- BOTTOM 5 ---
    lines.extend([
        "",
        "  PEORES 5:",
        "  " + "-" * 112,
    ])
    for rank, (name, r) in enumerate(
        results[-5:], len(results) - 4,
    ):
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
            f"  {rank:>3d}"
            f" {name:<38s}"
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

    # --- Estadisticas ---
    profitable = sum(
        1 for _, r in results
        if r.final_value > r.initial_budget
    )
    breakeven = sum(
        1 for _, r in results
        if r.final_value == r.initial_budget
    )
    negative = total - profitable - breakeven
    pnl_vals = [
        r.final_value - r.initial_budget for _, r in results
    ]
    avg_pnl = sum(pnl_vals) / len(pnl_vals) if pnl_vals else 0
    median_pnl = sorted(pnl_vals)[len(pnl_vals) // 2] if pnl_vals else 0

    pct_profit = profitable * 100 // total if total > 0 else 0
    pct_loss = negative * 100 // total if total > 0 else 0

    lines.extend([
        "",
        "=" * 115,
        "  ESTADISTICAS GLOBALES",
        "=" * 115,
        f"  Combinaciones probadas: {total}",
        f"  Rentables (>0%):        {profitable} ({pct_profit}%)",
        f"  En perdida (<0%):       {negative} ({pct_loss}%)",
        f"  P&L promedio:           ${avg_pnl:+.2f}",
        f"  P&L mediana:            ${median_pnl:+.2f}",
        "",
    ])

    # --- Analisis por grupo de monedas ---
    lines.extend([
        "  MEJOR POR GRUPO DE MONEDAS:",
        "  " + "-" * 60,
    ])
    group_best: dict[str, tuple[str, BacktestResult]] = {}
    for name, r in results:
        # Extraer grupo del nombre
        group = name.split(" d")[0]
        if group not in group_best:
            group_best[group] = (name, r)

    for group, (name, r) in group_best.items():
        pnl = r.final_value - r.initial_budget
        pnl_pct = (
            pnl / r.initial_budget * 100
            if r.initial_budget > 0 else 0
        )
        s = _fmt_sign(pnl)
        sp = _fmt_sign(pnl_pct)
        lines.append(
            f"    {group:<16s}: {s}${abs(pnl):.2f}"
            f" ({sp}{abs(pnl_pct):.1f}%)"
            f"  -- {name}",
        )
    lines.append("")

    # --- Ganadora absoluta ---
    best_name, best_r = results[0]
    best_pnl = best_r.final_value - best_r.initial_budget
    bs = _fmt_sign(best_pnl)
    best_pct = (
        best_pnl / best_r.initial_budget * 100
        if best_r.initial_budget > 0 else 0
    )
    bsp = _fmt_sign(best_pct)

    lines.extend([
        "=" * 115,
        f"  GANADORA ABSOLUTA: {best_name}",
        f"  Resultado: {bs}${abs(best_pnl):.2f}"
        f" ({bsp}{abs(best_pct):.1f}%)",
        "=" * 115,
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
        description="Backtest DCA multi-combinacion (bateria ampliada)",
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

    # Generar combinaciones
    combinations = generate_combinations()
    print(f"\nGeneradas {len(combinations)} combinaciones")

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

    print(f"\nEjecutando {len(combinations)} combinaciones...")
    report = run_all(prices, args.budget, args.days, combinations)
    print(report)

    # Guardar reporte
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    rpath = out_dir / "backtest_dca_multi_report.txt"
    rpath.write_text(report)
    print(f"Reporte guardado en: {rpath}")


if __name__ == "__main__":
    main()
