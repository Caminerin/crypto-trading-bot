#!/usr/bin/env python3
"""
Backtesting de la estrategia DCA Inteligente v2.

Descarga datos historicos de BTC, ETH, SOL, BNB y XRP desde Binance
y simula la estrategia DCA con parametros personalizados por moneda.

Regla: TP = 3x umbral de compra, SL = 1.5x umbral de compra.

Uso:
    python3 scripts/backtest_dca.py [--days 365] [--budget 30]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuracion por moneda
# Regla: TP = 3x umbral, SL = 2x umbral
# ---------------------------------------------------------------------------

ASSET_CONFIGS: dict[str, dict[str, float]] = {
    "BTCUSDT": {
        "dip_threshold": -0.03,   # Compra si cae >3%
        "take_profit":    0.09,   # Vende al +9%   (3x de 3%)
        "stop_loss":     -0.045,  # Corta al -4.5% (1.5x de 3%)
    },
    "ETHUSDT": {
        "dip_threshold": -0.07,   # Compra si cae >7%
        "take_profit":    0.21,   # Vende al +21%   (3x de 7%)
        "stop_loss":     -0.105,  # Corta al -10.5% (1.5x de 7%)
    },
    "SOLUSDT": {
        "dip_threshold": -0.07,   # Compra si cae >7%
        "take_profit":    0.21,   # Vende al +21%   (3x de 7%)
        "stop_loss":     -0.105,  # Corta al -10.5% (1.5x de 7%)
    },
    "BNBUSDT": {
        "dip_threshold": -0.03,   # Compra si cae >3%
        "take_profit":    0.09,   # Vende al +9%   (3x de 3%)
        "stop_loss":     -0.045,  # Corta al -4.5% (1.5x de 3%)
    },
    "XRPUSDT": {
        "dip_threshold": -0.07,   # Compra si cae >7%
        "take_profit":    0.21,   # Vende al +21%   (3x de 7%)
        "stop_loss":     -0.105,  # Corta al -10.5% (1.5x de 7%)
    },
}

DCA_ASSETS = list(ASSET_CONFIGS.keys())
MIN_ORDER_USDT = 10.0


# ---------------------------------------------------------------------------
# Descarga de datos
# ---------------------------------------------------------------------------

def download_daily_klines(symbol: str, days: int) -> pd.DataFrame:
    """Descarga velas diarias de Binance (API publica, sin auth)."""
    url = "https://data-api.binance.vision/api/v3/klines"
    all_data: list[list] = []
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    current = start_time
    while current < end_time:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": current,
            "limit": 1000,
        }
        resp = requests.get(
            url, params=params, timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_data.extend(batch)
        if len(batch) < 1000:
            break
        current = batch[-1][0] + 1

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(all_data, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("open_time")
    return df


# ---------------------------------------------------------------------------
# Simulacion
# ---------------------------------------------------------------------------

@dataclass
class BTPosition:
    symbol: str
    entry_price: float
    quantity: float
    invested: float
    entry_date: str


@dataclass
class BacktestResult:
    initial_budget: float = 0.0
    final_value: float = 0.0
    total_trades: int = 0
    buys: int = 0
    sells: int = 0
    wins: int = 0
    losses: int = 0
    stop_losses: int = 0
    take_profits: int = 0
    total_profit: float = 0.0
    max_drawdown_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    avg_hold_days: float = 0.0
    trades_log: list[dict] = field(default_factory=list)
    open_positions: list[dict] = field(default_factory=list)
    daily_equity: list[dict] = field(default_factory=list)
    per_asset_pnl: dict[str, float] = field(default_factory=dict)


def run_backtest(
    prices: dict[str, pd.DataFrame],
    budget: float,
) -> BacktestResult:
    """Simula la estrategia DCA con params por moneda (TP+SL)."""

    result = BacktestResult(initial_budget=budget)
    positions: list[BTPosition] = []
    cash = budget
    peak_equity = budget
    max_dd = 0.0
    hold_days_list: list[float] = []
    trade_returns: list[float] = []
    asset_pnl: dict[str, float] = {sym: 0.0 for sym in DCA_ASSETS}

    # Construir timeline unificado
    all_dates: set[pd.Timestamp] = set()
    for df in prices.values():
        all_dates.update(df.index)
    sorted_dates = sorted(all_dates)

    if len(sorted_dates) < 2:
        print("Error: datos insuficientes para backtesting.")
        return result

    for i, date in enumerate(sorted_dates):
        if i == 0:
            continue

        prev_date = sorted_dates[i - 1]

        # Precios actuales y cambios 24h
        current_prices: dict[str, float] = {}
        changes_24h: dict[str, float] = {}
        for symbol, df in prices.items():
            if date in df.index and prev_date in df.index:
                current_prices[symbol] = df.loc[date, "close"]
                prev_close = df.loc[prev_date, "close"]
                changes_24h[symbol] = (
                    (current_prices[symbol] - prev_close) / prev_close
                )

        # 1. Check take-profits y stop-losses
        sells_to_remove: list[str] = []
        for pos in positions:
            price = current_prices.get(pos.symbol)
            if price is None:
                continue

            cfg = ASSET_CONFIGS[pos.symbol]
            pnl_pct = (price - pos.entry_price) / pos.entry_price

            sell_reason = ""
            if pnl_pct >= cfg["take_profit"]:
                sell_reason = "TP"
                result.take_profits += 1
            elif pnl_pct <= cfg["stop_loss"]:
                sell_reason = "SL"
                result.stop_losses += 1

            if sell_reason:
                sell_value = pos.quantity * price
                profit = sell_value - pos.invested
                cash += sell_value
                result.sells += 1
                result.total_trades += 1
                hold_days = (
                    date - pd.Timestamp(pos.entry_date)
                ).days
                hold_days_list.append(hold_days)
                trade_returns.append(pnl_pct)
                asset_pnl[pos.symbol] += profit
                if profit >= 0:
                    result.wins += 1
                else:
                    result.losses += 1
                result.trades_log.append({
                    "date": str(date.date()),
                    "action": f"SELL({sell_reason})",
                    "symbol": pos.symbol,
                    "price": round(price, 2),
                    "entry_price": round(pos.entry_price, 2),
                    "invested": round(pos.invested, 2),
                    "sell_value": round(sell_value, 2),
                    "profit": round(profit, 2),
                    "pnl_pct": round(pnl_pct * 100, 1),
                    "hold_days": hold_days,
                })
                sells_to_remove.append(pos.symbol)

        positions = [
            p for p in positions
            if p.symbol not in sells_to_remove
        ]

        # 2. Check dip buys (umbral propio por moneda)
        dip_assets: list[tuple[str, float]] = []
        for symbol in DCA_ASSETS:
            change = changes_24h.get(symbol)
            if change is None:
                continue
            cfg = ASSET_CONFIGS[symbol]
            if change <= cfg["dip_threshold"]:
                already_in = any(
                    p.symbol == symbol for p in positions
                )
                if not already_in:
                    dip_assets.append((symbol, change))

        if dip_assets and cash >= MIN_ORDER_USDT:
            per_buy = cash / len(dip_assets)
            for symbol, change in dip_assets:
                buy_amount = min(per_buy, cash)
                if buy_amount < MIN_ORDER_USDT:
                    continue
                price = current_prices[symbol]
                quantity = buy_amount / price

                positions.append(BTPosition(
                    symbol=symbol,
                    entry_price=price,
                    quantity=quantity,
                    invested=buy_amount,
                    entry_date=str(date),
                ))

                cash -= buy_amount
                result.buys += 1
                result.total_trades += 1
                cfg = ASSET_CONFIGS[symbol]
                tp = round(
                    price * (1 + cfg["take_profit"]), 2,
                )
                sl = round(
                    price * (1 + cfg["stop_loss"]), 2,
                )
                result.trades_log.append({
                    "date": str(date.date()),
                    "action": "BUY",
                    "symbol": symbol,
                    "price": round(price, 2),
                    "amount_usdt": round(buy_amount, 2),
                    "change_24h": round(change * 100, 1),
                    "tp_target": tp,
                    "sl_target": sl,
                })

        # Calcular equity diaria
        pos_value = sum(
            p.quantity
            * current_prices.get(p.symbol, p.entry_price)
            for p in positions
        )
        equity = cash + pos_value
        result.daily_equity.append({
            "date": str(date.date()),
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "invested": round(pos_value, 2),
        })

        # Drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd = (
            (peak_equity - equity) / peak_equity
            if peak_equity > 0 else 0
        )
        if dd > max_dd:
            max_dd = dd

    # Resultados finales
    final_pos_value = 0.0
    last_prices: dict[str, float] = {}
    for symbol, df in prices.items():
        if len(df) > 0:
            last_prices[symbol] = df.iloc[-1]["close"]

    for pos in positions:
        price = last_prices.get(pos.symbol, pos.entry_price)
        pv = pos.quantity * price
        final_pos_value += pv
        pnl_pct = (price - pos.entry_price) / pos.entry_price
        result.open_positions.append({
            "symbol": pos.symbol,
            "entry_price": round(pos.entry_price, 2),
            "current_price": round(price, 2),
            "invested": round(pos.invested, 2),
            "current_value": round(pv, 2),
            "pnl_pct": round(pnl_pct * 100, 1),
        })

    result.final_value = round(cash + final_pos_value, 2)
    result.total_profit = round(
        result.final_value - budget, 2,
    )
    result.max_drawdown_pct = round(max_dd * 100, 2)
    if trade_returns:
        result.best_trade_pct = round(
            max(trade_returns) * 100, 1,
        )
        result.worst_trade_pct = round(
            min(trade_returns) * 100, 1,
        )
    if hold_days_list:
        result.avg_hold_days = round(
            sum(hold_days_list) / len(hold_days_list), 1,
        )
    result.per_asset_pnl = {
        k: round(v, 2) for k, v in asset_pnl.items()
    }

    return result


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def _fmt_sign(val: float) -> str:
    return "+" if val >= 0 else ""


def print_report(result: BacktestResult, days: int) -> str:
    """Genera reporte legible del backtesting."""
    pnl_pct = (
        (result.final_value - result.initial_budget)
        / result.initial_budget * 100
        if result.initial_budget > 0 else 0
    )
    s = _fmt_sign(result.total_profit)

    lines = [
        "",
        "=" * 65,
        "  BACKTESTING DCA INTELIGENTE v3 - RESULTADOS",
        "=" * 65,
        "",
        f"  Periodo:           Ultimos {days} dias",
        f"  Presupuesto:       ${result.initial_budget:,.2f} USDT",
        "",
        "  Configuracion por moneda (Regla: TP=3x, SL=1.5x):",
    ]

    for symbol, cfg in ASSET_CONFIGS.items():
        nm = symbol.replace("USDT", "")
        dip = abs(cfg["dip_threshold"]) * 100
        tp = cfg["take_profit"] * 100
        sl_val = cfg["stop_loss"] * 100
        lines.append(
            f"    {nm:4s} | Compra: >{dip:.0f}%"
            f" | TP: +{tp:.0f}%"
            f" | SL: {sl_val:.0f}%"
        )

    lines.extend([
        "",
        "-" * 65,
        "  RENDIMIENTO",
        "-" * 65,
        f"  Valor final:       ${result.final_value:,.2f} USDT",
        (
            f"  Ganancia/Perdida:  {s}${result.total_profit:,.2f}"
            f" ({s}{pnl_pct:.1f}%)"
        ),
        f"  Max drawdown:      -{result.max_drawdown_pct:.1f}%",
        "",
        "-" * 65,
        "  OPERACIONES",
        "-" * 65,
        f"  Total trades:      {result.total_trades}",
        f"  Compras:           {result.buys}",
        (
            f"  Ventas:            {result.sells}"
            f"  (TP: {result.take_profits}"
            f" | SL: {result.stop_losses})"
        ),
        f"  Ganadoras:         {result.wins}",
        f"  Perdedoras:        {result.losses}",
    ])

    if result.sells > 0:
        wr = result.wins / result.sells * 100
        lines.append(f"  Win rate:          {wr:.0f}%")
    else:
        lines.append(
            "  Win rate:          N/A (sin ventas aun)"
        )

    lines.extend([
        f"  Mejor trade:       +{result.best_trade_pct:.1f}%",
        f"  Peor trade:        {result.worst_trade_pct:.1f}%",
        f"  Hold promedio:     {result.avg_hold_days:.0f} dias",
        "",
    ])

    # P&L por moneda
    has_pnl = any(
        v != 0 for v in result.per_asset_pnl.values()
    )
    if has_pnl:
        lines.extend([
            "-" * 65,
            "  P&L POR MONEDA (solo trades cerrados)",
            "-" * 65,
        ])
        for sym, pnl in result.per_asset_pnl.items():
            nm = sym.replace("USDT", "")
            ps = _fmt_sign(pnl)
            lines.append(f"    {nm:4s}: {ps}${pnl:,.2f}")
        lines.append("")

    if result.open_positions:
        lines.extend([
            "-" * 65,
            "  POSICIONES ABIERTAS (no vendidas)",
            "-" * 65,
        ])
        for p in result.open_positions:
            ps = _fmt_sign(p["pnl_pct"])
            nm = p["symbol"].replace("USDT", "")
            lines.append(
                f"  {nm:4s}"
                f" | Compra: ${p['entry_price']:>10,.2f}"
                f" | Actual: ${p['current_price']:>10,.2f}"
                f" | {ps}{p['pnl_pct']:.1f}%"
                f" | Inv: ${p['invested']:,.2f}"
            )
        lines.append("")

    if result.trades_log:
        lines.extend([
            "-" * 65,
            "  HISTORIAL DE OPERACIONES",
            "-" * 65,
        ])
        for t in result.trades_log:
            nm = t["symbol"].replace("USDT", "")
            if t["action"] == "BUY":
                lines.append(
                    f"  {t['date']}"
                    f" | COMPRA {nm:4s}"
                    f" | ${t['amount_usdt']:>8,.2f}"
                    f" @ ${t['price']:>10,.2f}"
                    f" ({t['change_24h']:.1f}%)"
                    f" [TP=${t['tp_target']:,.2f}"
                    f" SL=${t['sl_target']:,.2f}]"
                )
            else:
                ps = _fmt_sign(t["profit"])
                lines.append(
                    f"  {t['date']}"
                    f" | {t['action']:8s} {nm:4s}"
                    f" | ${t['sell_value']:>8,.2f}"
                    f" @ ${t['price']:>10,.2f}"
                    f" ({ps}${t['profit']:,.2f},"
                    f" {ps}{t['pnl_pct']:.1f}%,"
                    f" {t['hold_days']}d)"
                )
        lines.append("")

    lines.extend([
        "=" * 65,
        (
            f"  VEREDICTO: {s}${result.total_profit:,.2f}"
            f" ({s}{pnl_pct:.1f}%) en {days} dias"
        ),
        "=" * 65,
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest DCA v2 (5 monedas, SL+TP)",
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

    print(f"\nDescargando {args.days} dias de datos...")
    prices: dict[str, pd.DataFrame] = {}
    for symbol in DCA_ASSETS:
        nm = symbol.replace("USDT", "")
        print(f"  {nm}...", end=" ", flush=True)
        df = download_daily_klines(symbol, args.days)
        prices[symbol] = df
        print(f"{len(df)} velas")

    print("\nEjecutando simulacion...")
    result = run_backtest(prices, budget=args.budget)

    report = print_report(result, args.days)
    print(report)

    # Guardar reporte
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    rpath = out_dir / "backtest_dca_report.txt"
    rpath.write_text(report)
    print(f"Reporte guardado en: {rpath}")


if __name__ == "__main__":
    main()
