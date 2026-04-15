#!/usr/bin/env python3
"""
Auto-optimizador mensual de estrategias DCA y Momentum.

Ejecuta backtesting con datos de los ultimos 365 dias, encuentra la mejor
configuracion por moneda que cumpla los filtros de calidad, y actualiza
src/config.py automaticamente.

Filtros de calidad:
  - P&L > 5%
  - Trades > 5
  - Win rate > 40%
  - Max drawdown < 20%

Si ninguna configuracion cumple los filtros, se mantiene la actual.

Uso:
    python3 scripts/auto_optimize.py [--days 365] [--budget 30] [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from itertools import product
from pathlib import Path

import pandas as pd

# Importar funciones de los scripts de backtesting existentes
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_dca import (  # noqa: E402
    BacktestResult as DCAResult,
)
from backtest_dca import (
    download_daily_klines,
)
from backtest_dca import (
    run_backtest as run_dca_backtest,
)
from backtest_momentum import (  # noqa: E402
    BacktestResult as MomentumResult,
)
from backtest_momentum import (
    run_backtest as run_momentum_backtest,
)

# ---------------------------------------------------------------------------
# Filtros de calidad
# ---------------------------------------------------------------------------
MIN_PNL_PCT = 5.0       # P&L minimo >5%
MIN_TRADES = 5           # Minimo 5 trades cerrados
MIN_WIN_RATE = 40.0      # Win rate minimo 40%
MAX_DRAWDOWN = 20.0      # Max drawdown maximo 20%

# Monedas
DCA_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
MOMENTUM_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

# ---------------------------------------------------------------------------
# Rangos de parametros para optimizacion
# ---------------------------------------------------------------------------

# DCA: dip_threshold, take_profit, stop_loss
DCA_DIPS = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
DCA_TPS = [0.08, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25]
DCA_SLS = [0.045, 0.06, 0.075, 0.08, 0.10, 0.105, 0.12]

# Momentum: threshold, take_profit, stop_loss, trend_days
MOM_THRESHOLDS = [0.03, 0.05, 0.07, 0.10, 0.15]
MOM_TPS = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
MOM_SLS = [0.03, 0.05, 0.07, 0.10, 0.15]
MOM_TRENDS = [3, 5, 7, 14]


# ---------------------------------------------------------------------------
# Resultado candidato
# ---------------------------------------------------------------------------

class CandidateConfig:
    """Almacena la mejor configuracion encontrada para una moneda."""

    def __init__(
        self,
        coin: str,
        strategy: str,
        params: dict[str, float],
        pnl: float,
        pnl_pct: float,
        trades: int,
        win_rate: float,
        max_dd: float,
        combos_tested: int,
    ) -> None:
        self.coin = coin
        self.strategy = strategy
        self.params = params
        self.pnl = pnl
        self.pnl_pct = pnl_pct
        self.trades = trades
        self.win_rate = win_rate
        self.max_dd = max_dd
        self.combos_tested = combos_tested

    def passes_filters(self) -> bool:
        return (
            self.pnl_pct > MIN_PNL_PCT
            and self.trades >= MIN_TRADES
            and self.win_rate >= MIN_WIN_RATE
            and self.max_dd <= MAX_DRAWDOWN
        )


# ---------------------------------------------------------------------------
# Optimizacion DCA por moneda
# ---------------------------------------------------------------------------

def optimize_dca_coin(
    coin: str,
    prices: dict[str, pd.DataFrame],
    budget: float,
) -> CandidateConfig | None:
    """Busca la mejor configuracion DCA para una moneda."""
    nm = coin.replace("USDT", "")
    print(f"  Optimizando DCA {nm}...", end=" ", flush=True)

    if coin not in prices:
        print("SIN DATOS")
        return None

    coin_prices = {coin: prices[coin]}
    best_pnl = -999999.0
    best_cfg: dict[str, float] = {}
    best_result: DCAResult | None = None
    combos = 0

    for dip, tp, sl in product(DCA_DIPS, DCA_TPS, DCA_SLS):
        # SL no puede ser >= TP
        if sl >= tp:
            continue
        cfg = {
            coin: {
                "dip_threshold": -dip,
                "take_profit": tp,
                "stop_loss": -sl,
            }
        }
        r = run_dca_backtest(coin_prices, budget, asset_configs=cfg)
        combos += 1
        pnl = r.final_value - r.initial_budget
        if pnl > best_pnl:
            best_pnl = pnl
            best_cfg = {
                "dip_threshold": -dip,
                "take_profit_pct": tp,
                "stop_loss_pct": -sl,
            }
            best_result = r

    if best_result is None:
        print("SIN RESULTADOS")
        return None

    pnl_pct = (
        best_pnl / budget * 100 if budget > 0 else 0
    )
    wr = (
        best_result.wins / best_result.sells * 100
        if best_result.sells > 0 else 0
    )
    print(
        f"{combos} combos | P&L: {best_pnl:+.2f}$ ({pnl_pct:+.1f}%)"
        f" | Trades: {best_result.sells}"
        f" | Win: {wr:.0f}% | DD: {best_result.max_drawdown_pct:.1f}%"
    )

    return CandidateConfig(
        coin=coin,
        strategy="DCA",
        params=best_cfg,
        pnl=best_pnl,
        pnl_pct=pnl_pct,
        trades=best_result.sells,
        win_rate=wr,
        max_dd=best_result.max_drawdown_pct,
        combos_tested=combos,
    )


# ---------------------------------------------------------------------------
# Optimizacion Momentum por moneda
# ---------------------------------------------------------------------------

def optimize_momentum_coin(
    coin: str,
    prices: dict[str, pd.DataFrame],
    budget: float,
) -> CandidateConfig | None:
    """Busca la mejor configuracion Momentum para una moneda."""
    nm = coin.replace("USDT", "")
    print(f"  Optimizando Momentum {nm}...", end=" ", flush=True)

    if coin not in prices:
        print("SIN DATOS")
        return None

    coin_prices = {coin: prices[coin]}
    best_pnl = -999999.0
    best_cfg: dict[str, float] = {}
    best_result: MomentumResult | None = None
    combos = 0

    for thr, tp, sl, td in product(
        MOM_THRESHOLDS, MOM_TPS, MOM_SLS, MOM_TRENDS,
    ):
        # SL no puede ser >= TP
        if sl >= tp:
            continue
        cfg = {
            coin: {
                "momentum_threshold": thr,
                "take_profit": tp,
                "stop_loss": -sl,
                "trend_days": td,
            }
        }
        r = run_momentum_backtest(coin_prices, budget, asset_configs=cfg)
        combos += 1
        pnl = r.final_value - r.initial_budget
        if pnl > best_pnl:
            best_pnl = pnl
            best_cfg = {
                "momentum_threshold": thr,
                "take_profit_pct": tp,
                "stop_loss_pct": -sl,
                "trend_days": float(td),
            }
            best_result = r

    if best_result is None:
        print("SIN RESULTADOS")
        return None

    pnl_pct = (
        best_pnl / budget * 100 if budget > 0 else 0
    )
    wr = (
        best_result.wins / best_result.sells * 100
        if best_result.sells > 0 else 0
    )
    print(
        f"{combos} combos | P&L: {best_pnl:+.2f}$ ({pnl_pct:+.1f}%)"
        f" | Trades: {best_result.sells}"
        f" | Win: {wr:.0f}% | DD: {best_result.max_drawdown_pct:.1f}%"
    )

    return CandidateConfig(
        coin=coin,
        strategy="Momentum",
        params=best_cfg,
        pnl=best_pnl,
        pnl_pct=pnl_pct,
        trades=best_result.sells,
        win_rate=wr,
        max_dd=best_result.max_drawdown_pct,
        combos_tested=combos,
    )


# ---------------------------------------------------------------------------
# Actualizacion de src/config.py
# ---------------------------------------------------------------------------

def _update_dca_policies(
    config_text: str,
    candidates: dict[str, CandidateConfig],
) -> str:
    """Reemplaza DEFAULT_ASSET_POLICIES en el texto de config.py."""
    lines = []
    for coin in DCA_COINS:
        c = candidates[coin]
        lines.append(f'    "{coin}": DCAAssetPolicy(')
        lines.append(
            f"        dip_threshold={c.params['dip_threshold']},",
        )
        lines.append(
            f"        take_profit_pct={c.params['take_profit_pct']},",
        )
        lines.append(
            f"        stop_loss_pct={c.params['stop_loss_pct']},",
        )
        lines.append("    ),")

    new_block = "\n".join(lines)

    pattern = (
        r"(# Política óptima por moneda.*?\n"
        r"DEFAULT_ASSET_POLICIES: dict\[str, DCAAssetPolicy\] = \{)\n"
        r"(.*?)"
        r"(\})"
    )
    replacement = rf"\1\n{new_block}\n\3"
    result = re.sub(pattern, replacement, config_text, flags=re.DOTALL)
    return result


def _update_momentum_policies(
    config_text: str,
    candidates: dict[str, CandidateConfig],
) -> str:
    """Reemplaza DEFAULT_MOMENTUM_POLICIES en el texto de config.py."""
    lines = []
    for coin in MOMENTUM_COINS:
        c = candidates[coin]
        td = int(c.params["trend_days"])
        lines.append(f'    "{coin}": MomentumAssetPolicy(')
        lines.append(
            f"        momentum_threshold="
            f"{c.params['momentum_threshold']},",
        )
        lines.append(
            f"        take_profit_pct={c.params['take_profit_pct']},",
        )
        lines.append(
            f"        stop_loss_pct={c.params['stop_loss_pct']},",
        )
        if td != 7:
            lines.append(f"        trend_days={td},")
        lines.append("    ),")

    new_block = "\n".join(lines)

    pattern = (
        r"(# Política óptima momentum por moneda.*?\n"
        r"DEFAULT_MOMENTUM_POLICIES: dict\[str, MomentumAssetPolicy\]"
        r" = \{)\n"
        r"(.*?)"
        r"(\})"
    )
    replacement = rf"\1\n{new_block}\n\3"
    result = re.sub(pattern, replacement, config_text, flags=re.DOTALL)
    return result


def update_config_file(
    dca_candidates: dict[str, CandidateConfig],
    momentum_candidates: dict[str, CandidateConfig],
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Actualiza src/config.py con las mejores configuraciones.

    Returns:
        (changed, report): si se hizo algun cambio y el reporte de cambios.
    """
    config_path = (
        Path(__file__).resolve().parent.parent / "src" / "config.py"
    )
    original = config_path.read_text()
    updated = original

    report_lines: list[str] = []
    changed = False

    # Actualizar DCA si hay candidatos validos
    if dca_candidates:
        updated = _update_dca_policies(updated, dca_candidates)
        if updated != original:
            changed = True
            report_lines.append("DCA actualizado:")
            for coin, c in dca_candidates.items():
                nm = coin.replace("USDT", "")
                report_lines.append(
                    f"  {nm}: dip={c.params['dip_threshold']:.0%}"
                    f" TP={c.params['take_profit_pct']:.1%}"
                    f" SL={c.params['stop_loss_pct']:.1%}"
                    f" (P&L: {c.pnl_pct:+.1f}%,"
                    f" Win: {c.win_rate:.0f}%,"
                    f" DD: {c.max_dd:.1f}%)"
                )

    after_dca = updated

    # Actualizar Momentum si hay candidatos validos
    if momentum_candidates:
        updated = _update_momentum_policies(updated, momentum_candidates)
        if updated != after_dca:
            changed = True
            report_lines.append("Momentum actualizado:")
            for coin, c in momentum_candidates.items():
                nm = coin.replace("USDT", "")
                td = int(c.params["trend_days"])
                report_lines.append(
                    f"  {nm}: thr={c.params['momentum_threshold']:.0%}"
                    f" TP={c.params['take_profit_pct']:.1%}"
                    f" SL={c.params['stop_loss_pct']:.1%}"
                    f" Trend={td}d"
                    f" (P&L: {c.pnl_pct:+.1f}%,"
                    f" Win: {c.win_rate:.0f}%,"
                    f" DD: {c.max_dd:.1f}%)"
                )

    report = "\n".join(report_lines) if report_lines else "Sin cambios."

    if changed and not dry_run:
        config_path.write_text(updated)
        print(f"\n  config.py actualizado en {config_path}")
    elif dry_run and changed:
        print(f"\n  [DRY-RUN] Se habria actualizado {config_path}")

    return changed, report


# ---------------------------------------------------------------------------
# Reporte completo
# ---------------------------------------------------------------------------

def build_report(
    days: int,
    budget: float,
    dca_results: dict[str, CandidateConfig | None],
    momentum_results: dict[str, CandidateConfig | None],
    dca_applied: dict[str, CandidateConfig],
    momentum_applied: dict[str, CandidateConfig],
    config_changed: bool,
) -> str:
    """Genera reporte completo de la optimizacion."""
    lines = [
        "",
        "=" * 70,
        "  AUTO-OPTIMIZACION MENSUAL — REPORTE",
        "=" * 70,
        f"  Periodo: {days} dias | Presupuesto: ${budget:.2f}",
        f"  Filtros: P&L>{MIN_PNL_PCT}%,"
        f" Trades>={MIN_TRADES},"
        f" WinRate>={MIN_WIN_RATE}%,"
        f" MaxDD<{MAX_DRAWDOWN}%",
        "",
        "-" * 70,
        "  RESULTADOS DCA POR MONEDA",
        "-" * 70,
    ]

    for coin in DCA_COINS:
        nm = coin.replace("USDT", "")
        c = dca_results.get(coin)
        if c is None:
            lines.append(f"  {nm}: SIN DATOS")
            continue
        passed = "SI" if c.passes_filters() else "NO"
        applied = "APLICADO" if coin in dca_applied else "MANTENIDO"
        lines.append(
            f"  {nm}:"
            f" dip={c.params['dip_threshold']:.0%}"
            f" TP={c.params['take_profit_pct']:.1%}"
            f" SL={c.params['stop_loss_pct']:.1%}"
        )
        lines.append(
            f"      P&L: {c.pnl_pct:+.1f}%"
            f" | Trades: {c.trades}"
            f" | Win: {c.win_rate:.0f}%"
            f" | DD: {c.max_dd:.1f}%"
            f" | Filtro: {passed}"
            f" | {applied}"
        )

    lines.extend([
        "",
        "-" * 70,
        "  RESULTADOS MOMENTUM POR MONEDA",
        "-" * 70,
    ])

    for coin in MOMENTUM_COINS:
        nm = coin.replace("USDT", "")
        c = momentum_results.get(coin)
        if c is None:
            lines.append(f"  {nm}: SIN DATOS")
            continue
        passed = "SI" if c.passes_filters() else "NO"
        applied = "APLICADO" if coin in momentum_applied else "MANTENIDO"
        td = int(c.params["trend_days"])
        lines.append(
            f"  {nm}:"
            f" thr={c.params['momentum_threshold']:.0%}"
            f" TP={c.params['take_profit_pct']:.1%}"
            f" SL={c.params['stop_loss_pct']:.1%}"
            f" Trend={td}d"
        )
        lines.append(
            f"      P&L: {c.pnl_pct:+.1f}%"
            f" | Trades: {c.trades}"
            f" | Win: {c.win_rate:.0f}%"
            f" | DD: {c.max_dd:.1f}%"
            f" | Filtro: {passed}"
            f" | {applied}"
        )

    lines.extend([
        "",
        "=" * 70,
        f"  CONFIG ACTUALIZADO: {'SI' if config_changed else 'NO'}",
        "=" * 70,
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-optimizador mensual DCA + Momentum",
    )
    parser.add_argument(
        "--days", type=int, default=365,
        help="Dias de historia (default: 365)",
    )
    parser.add_argument(
        "--budget", type=float, default=30.0,
        help="Presupuesto por estrategia en USDT (default: 30)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Solo mostrar resultados, no actualizar config.py",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  AUTO-OPTIMIZACION MENSUAL")
    print("=" * 70)

    # Descargar datos una sola vez
    all_coins = sorted(
        set(DCA_COINS) | set(MOMENTUM_COINS),
    )
    print(f"\nDescargando {args.days} dias de datos...")
    prices: dict[str, pd.DataFrame] = {}
    for symbol in all_coins:
        nm = symbol.replace("USDT", "")
        print(f"  {nm}...", end=" ", flush=True)
        df = download_daily_klines(symbol, args.days)
        prices[symbol] = df
        print(f"{len(df)} velas")

    # Optimizar DCA por moneda
    print(f"\n--- OPTIMIZACION DCA ({len(DCA_COINS)} monedas) ---")
    dca_results: dict[str, CandidateConfig | None] = {}
    for coin in DCA_COINS:
        dca_results[coin] = optimize_dca_coin(coin, prices, args.budget)

    # Optimizar Momentum por moneda
    print(f"\n--- OPTIMIZACION MOMENTUM ({len(MOMENTUM_COINS)} monedas) ---")
    momentum_results: dict[str, CandidateConfig | None] = {}
    for coin in MOMENTUM_COINS:
        momentum_results[coin] = optimize_momentum_coin(
            coin, prices, args.budget,
        )

    # Filtrar: solo aplicar los que pasan todos los filtros
    dca_applied: dict[str, CandidateConfig] = {}
    for coin, c in dca_results.items():
        if c is not None and c.passes_filters():
            dca_applied[coin] = c
            nm = coin.replace("USDT", "")
            print(f"  DCA {nm}: PASA filtros -> se aplicara")
        elif c is not None:
            nm = coin.replace("USDT", "")
            reasons = []
            if c.pnl_pct <= MIN_PNL_PCT:
                reasons.append(f"P&L {c.pnl_pct:.1f}%<={MIN_PNL_PCT}%")
            if c.trades < MIN_TRADES:
                reasons.append(f"Trades {c.trades}<{MIN_TRADES}")
            if c.win_rate < MIN_WIN_RATE:
                reasons.append(f"Win {c.win_rate:.0f}%<{MIN_WIN_RATE}%")
            if c.max_dd > MAX_DRAWDOWN:
                reasons.append(f"DD {c.max_dd:.1f}%>{MAX_DRAWDOWN}%")
            print(f"  DCA {nm}: NO pasa ({', '.join(reasons)})")

    momentum_applied: dict[str, CandidateConfig] = {}
    for coin, c in momentum_results.items():
        if c is not None and c.passes_filters():
            momentum_applied[coin] = c
            nm = coin.replace("USDT", "")
            print(f"  Momentum {nm}: PASA filtros -> se aplicara")
        elif c is not None:
            nm = coin.replace("USDT", "")
            reasons = []
            if c.pnl_pct <= MIN_PNL_PCT:
                reasons.append(f"P&L {c.pnl_pct:.1f}%<={MIN_PNL_PCT}%")
            if c.trades < MIN_TRADES:
                reasons.append(f"Trades {c.trades}<{MIN_TRADES}")
            if c.win_rate < MIN_WIN_RATE:
                reasons.append(f"Win {c.win_rate:.0f}%<{MIN_WIN_RATE}%")
            if c.max_dd > MAX_DRAWDOWN:
                reasons.append(f"DD {c.max_dd:.1f}%>{MAX_DRAWDOWN}%")
            print(f"  Momentum {nm}: NO pasa ({', '.join(reasons)})")

    # Actualizar config.py
    print("\n--- ACTUALIZACION DE CONFIG ---")
    if not dca_applied and not momentum_applied:
        print("  Ninguna configuracion nueva pasa los filtros.")
        print("  Se mantiene la configuracion actual.")
        config_changed = False
    else:
        config_changed, change_report = update_config_file(
            dca_applied, momentum_applied, dry_run=args.dry_run,
        )
        print(f"  {change_report}")

    # Generar y guardar reporte
    report = build_report(
        args.days, args.budget,
        dca_results, momentum_results,
        dca_applied, momentum_applied,
        config_changed,
    )
    print(report)

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    rpath = out_dir / "auto_optimize_report.txt"
    rpath.write_text(report)
    print(f"Reporte guardado en: {rpath}")

    # Exit code: 0 si se actualizo, 1 si no hubo cambios
    sys.exit(0 if config_changed else 1)


if __name__ == "__main__":
    main()
