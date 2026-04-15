#!/usr/bin/env python3
"""
Backtesting multi-combinacion Momentum — bateria ampliada.

Genera combinaciones de forma programatica variando:
  - Grupo de monedas (BTC, BNB, BTC+BNB, +ETH, +SOL, 5 monedas)
  - Umbral de momentum (subida 24h minima para comprar)
  - Take-Profit
  - Stop-Loss
  - Dias de tendencia (trend_days)

Uso:
    python3 scripts/backtest_momentum_multi.py [--days 365] [--budget 30]
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd
from backtest_momentum import (
    MOMENTUM_ASSETS,
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
    "ETH":         ["ETHUSDT"],
    "BNB":         ["BNBUSDT"],
    "SOL":         ["SOLUSDT"],
    "XRP":         ["XRPUSDT"],
    "BTC+ETH":     ["BTCUSDT", "ETHUSDT"],
    "BTC+BNB":     ["BTCUSDT", "BNBUSDT"],
    "BTC+ETH+BNB": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    "5monedas":    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
}

# Clasificacion de monedas (BTC menos volatil => umbral mas bajo)
STABLE_COINS = {"BTCUSDT", "BNBUSDT"}
ALT_COINS = {"ETHUSDT", "SOLUSDT", "XRPUSDT"}

# Parametros a variar
# (threshold_stable%, threshold_alt%)
THRESHOLD_COMBOS: list[tuple[float, float]] = [
    (0.03, 0.05),   # 3%/5%  — muy sensible
    (0.03, 0.07),   # 3%/7%  — base
    (0.05, 0.07),   # 5%/7%  — moderado
    (0.05, 0.10),   # 5%/10% — selectivo
    (0.07, 0.10),   # 7%/10% — selectivo alto
    (0.07, 0.15),   # 7%/15% — muy selectivo
    (0.03, 0.03),   # 3%/3%  — igual para todos
    (0.05, 0.05),   # 5%/5%  — igual para todos
    (0.10, 0.10),   # 10%/10% — solo grandes subidas
    (0.10, 0.15),   # 10%/15% — extremo
]

TP_VALUES: list[float] = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
SL_VALUES: list[float] = [-0.03, -0.05, -0.07, -0.10]
TREND_DAYS_VALUES: list[int] = [3, 5, 7, 14]


def _build_config(
    coins: list[str],
    thr_stable: float,
    thr_alt: float,
    tp: float,
    sl: float,
    trend_d: int,
) -> dict[str, dict[str, float | int]]:
    """Construye config por moneda dados los parametros."""
    cfg: dict[str, dict[str, float | int]] = {}
    for coin in coins:
        thr = thr_stable if coin in STABLE_COINS else thr_alt
        cfg[coin] = {
            "momentum_threshold": thr,
            "take_profit": tp,
            "stop_loss": sl,
            "trend_days": trend_d,
        }
    return cfg


def _make_name(
    group: str,
    thr_stable: float,
    thr_alt: float,
    tp: float,
    sl: float,
    trend_d: int,
) -> str:
    """Genera nombre corto para la combinacion."""
    ts = f"{thr_stable*100:.0f}"
    ta = f"{thr_alt*100:.0f}"
    tp_s = f"{tp*100:.0f}"
    sl_s = f"{abs(sl)*100:.0f}"
    return f"{group} t{ts}/{ta} TP{tp_s} SL{sl_s} T{trend_d}d"


def generate_combinations(
) -> list[tuple[str, dict[str, dict[str, float | int]]]]:
    """Genera todas las combinaciones validas."""
    combos: list[tuple[str, dict[str, dict[str, float | int]]]] = []
    seen: set[str] = set()

    for group_name, coins in COIN_GROUPS.items():
        has_alt = any(c in ALT_COINS for c in coins)

        for (thr_s, thr_a), tp, sl, trend_d in product(
            THRESHOLD_COMBOS, TP_VALUES, SL_VALUES, TREND_DAYS_VALUES,
        ):
            # Filtro: SL mas profundo que TP no tiene sentido
            if abs(sl) >= tp:
                continue

            # Filtro: dedup si no hay altcoins
            if not has_alt:
                key = f"{group_name}|{thr_s}|TP{tp}|SL{sl}|T{trend_d}"
            else:
                key = (
                    f"{group_name}|{thr_s}/{thr_a}|TP{tp}"
                    f"|SL{sl}|T{trend_d}"
                )

            if key in seen:
                continue
            seen.add(key)

            effective_thr_a = thr_a if has_alt else thr_s

            name = _make_name(
                group_name, thr_s, effective_thr_a, tp, sl, trend_d,
            )
            cfg = _build_config(
                coins, thr_s, effective_thr_a, tp, sl, trend_d,
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
    combinations: list[tuple[str, dict[str, dict[str, float | int]]]],
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
        "=" * 120,
        "  COMPARATIVA BACKTESTING MOMENTUM — BATERIA AMPLIADA",
        "=" * 120,
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
        "  " + "-" * 117,
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
        "  " + "-" * 117,
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
        "=" * 120,
        "  ESTADISTICAS GLOBALES",
        "=" * 120,
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
        group = name.split(" t")[0]
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

    # --- Mejor por moneda individual ---
    lines.extend([
        "  MEJOR POR MONEDA INDIVIDUAL:",
        "  " + "-" * 60,
    ])
    single_coins = ["BTC", "ETH", "BNB", "SOL", "XRP"]
    for coin in single_coins:
        if coin in group_best:
            name, r = group_best[coin]
            pnl = r.final_value - r.initial_budget
            pnl_pct = (
                pnl / r.initial_budget * 100
                if r.initial_budget > 0 else 0
            )
            s = _fmt_sign(pnl)
            sp = _fmt_sign(pnl_pct)
            # Extraer parametros del nombre
            lines.append(
                f"    {coin:4s}: {s}${abs(pnl):.2f}"
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
        "=" * 120,
        f"  GANADORA ABSOLUTA: {best_name}",
        f"  Resultado: {bs}${abs(best_pnl):.2f}"
        f" ({bsp}{abs(best_pct):.1f}%)",
        "=" * 120,
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


# ---------------------------------------------------------------------------
# Backtesting por moneda individual
# ---------------------------------------------------------------------------

def run_per_coin(
    prices: dict[str, pd.DataFrame],
    budget: float,
    days: int,
) -> str:
    """Busca la mejor configuracion para CADA moneda por separado."""
    coins = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

    # Rangos individuales mas amplios
    thresholds = [0.03, 0.05, 0.07, 0.10, 0.15]
    tps = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
    sls = [-0.03, -0.05, -0.07, -0.10, -0.15]
    trends = [3, 5, 7, 14]

    lines: list[str] = [
        "",
        "=" * 80,
        "  MEJOR CONFIGURACION MOMENTUM POR MONEDA",
        "=" * 80,
        f"  Periodo: {days} dias | Presupuesto: ${budget:.2f} USDT",
        "",
    ]

    for coin in coins:
        nm = coin.replace("USDT", "")
        if coin not in prices:
            lines.append(f"  {nm}: SIN DATOS")
            continue

        coin_prices = {coin: prices[coin]}
        best_pnl = -999999.0
        best_cfg: dict[str, dict[str, float | int]] = {}
        best_result: BacktestResult | None = None
        combos_tested = 0

        for thr, tp, sl, td in product(thresholds, tps, sls, trends):
            if abs(sl) >= tp:
                continue
            cfg = {
                coin: {
                    "momentum_threshold": thr,
                    "take_profit": tp,
                    "stop_loss": sl,
                    "trend_days": td,
                }
            }
            r = run_backtest(coin_prices, budget, asset_configs=cfg)
            combos_tested += 1
            pnl = r.final_value - r.initial_budget
            if pnl > best_pnl:
                best_pnl = pnl
                best_cfg = cfg
                best_result = r

        if best_result is None:
            lines.append(f"  {nm}: SIN RESULTADOS")
            continue

        c = best_cfg[coin]
        pnl_pct = (
            best_pnl / budget * 100 if budget > 0 else 0
        )
        s = _fmt_sign(best_pnl)
        sp = _fmt_sign(pnl_pct)
        wr = (
            f"{best_result.wins / best_result.sells * 100:.0f}%"
            if best_result.sells > 0 else "N/A"
        )
        lines.extend([
            f"  {nm}:",
            f"    Umbral:     +{float(c['momentum_threshold'])*100:.0f}%",
            f"    TP:         +{float(c['take_profit'])*100:.0f}%",
            f"    SL:         {float(c['stop_loss'])*100:.0f}%",
            f"    Trend:      {int(c['trend_days'])}d",
            f"    P&L:        {s}${abs(best_pnl):.2f}"
            f" ({sp}{abs(pnl_pct):.1f}%)",
            f"    Trades:     {best_result.total_trades}"
            f" (TP: {best_result.take_profits},"
            f" SL: {best_result.stop_losses})",
            f"    Win rate:   {wr}",
            f"    Max DD:     -{best_result.max_drawdown_pct:.1f}%",
            f"    Hold prom:  {best_result.avg_hold_days:.0f}d",
            f"    Combos:     {combos_tested} probadas",
            "",
        ])

    lines.extend(["=" * 80, ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest Momentum multi-combinacion (bateria ampliada)",
    )
    parser.add_argument(
        "--days", type=int, default=365,
        help="Dias de historia (default: 365)",
    )
    parser.add_argument(
        "--budget", type=float, default=30.0,
        help="Presupuesto momentum en USDT (default: 30)",
    )
    args = parser.parse_args()

    # Generar combinaciones
    combinations = generate_combinations()
    print(f"\nGeneradas {len(combinations)} combinaciones")

    # Descargar datos una sola vez
    all_symbols = sorted(set(MOMENTUM_ASSETS))
    print(f"\nDescargando {args.days} dias de datos...")
    prices: dict[str, pd.DataFrame] = {}
    for symbol in all_symbols:
        nm = symbol.replace("USDT", "")
        print(f"  {nm}...", end=" ", flush=True)
        df = download_daily_klines(symbol, args.days)
        prices[symbol] = df
        print(f"{len(df)} velas")

    # Parte 1: Multi-combo (grupos de monedas)
    print(f"\nEjecutando {len(combinations)} combinaciones...")
    report_multi = run_all(prices, args.budget, args.days, combinations)
    print(report_multi)

    # Parte 2: Mejor por moneda individual
    print("\nBuscando mejor configuracion por moneda individual...")
    report_per_coin = run_per_coin(prices, args.budget, args.days)
    print(report_per_coin)

    # Guardar reportes
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    full_report = report_multi + "\n\n" + report_per_coin

    rpath = out_dir / "backtest_momentum_multi_report.txt"
    rpath.write_text(full_report)
    print(f"\nReporte guardado en: {rpath}")

    rpath_coin = out_dir / "backtest_momentum_per_coin_report.txt"
    rpath_coin.write_text(report_per_coin)
    print(f"Reporte por moneda guardado en: {rpath_coin}")


if __name__ == "__main__":
    main()
