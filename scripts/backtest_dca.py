#!/usr/bin/env python3
"""
Backtesting de la estrategia DCA Inteligente.

Descarga datos historicos de BTC y ETH desde Binance y simula
la estrategia DCA: compra en caidas >5% diarias, vende al +15%.

Uso:
    python scripts/backtest_dca.py [--days 365] [--budget 30]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

DCA_ASSETS = ["BTCUSDT", "ETHUSDT"]
DIP_THRESHOLD = -0.05      # Compra cuando cae >5% en 24h
TAKE_PROFIT_PCT = 0.15     # Vende cuando sube +15%
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
    total_profit: float = 0.0
    max_drawdown_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    avg_hold_days: float = 0.0
    trades_log: list[dict] = field(default_factory=list)
    open_positions: list[dict] = field(default_factory=list)
    daily_equity: list[dict] = field(default_factory=list)


def run_backtest(
    prices: dict[str, pd.DataFrame],
    budget: float,
    dip_threshold: float = DIP_THRESHOLD,
    take_profit_pct: float = TAKE_PROFIT_PCT,
) -> BacktestResult:
    """Simula la estrategia DCA sobre datos historicos."""

    result = BacktestResult(initial_budget=budget)
    positions: list[BTPosition] = []
    cash = budget
    peak_equity = budget
    max_dd = 0.0
    hold_days_list: list[float] = []
    trade_returns: list[float] = []

    # Construir timeline unificado (dias que tenemos datos de todos los assets)
    all_dates: set[pd.Timestamp] = set()
    for df in prices.values():
        all_dates.update(df.index)
    sorted_dates = sorted(all_dates)

    if len(sorted_dates) < 2:
        print("Error: datos insuficientes para backtesting.")
        return result

    for i, date in enumerate(sorted_dates):
        if i == 0:
            continue  # Necesitamos dia anterior para calcular cambio 24h

        prev_date = sorted_dates[i - 1]

        # Precios actuales
        current_prices: dict[str, float] = {}
        changes_24h: dict[str, float] = {}
        for symbol, df in prices.items():
            if date in df.index and prev_date in df.index:
                current_prices[symbol] = df.loc[date, "close"]
                prev_close = df.loc[prev_date, "close"]
                changes_24h[symbol] = (
                    (current_prices[symbol] - prev_close) / prev_close
                )

        # 1. Check take-profits
        sells_to_remove: list[str] = []
        for pos in positions:
            price = current_prices.get(pos.symbol)
            if price is None:
                continue
            pnl_pct = (price - pos.entry_price) / pos.entry_price
            if pnl_pct >= take_profit_pct:
                sell_value = pos.quantity * price
                profit = sell_value - pos.invested
                cash += sell_value
                result.sells += 1
                result.total_trades += 1
                hold_days = (date - pd.Timestamp(pos.entry_date)).days
                hold_days_list.append(hold_days)
                trade_returns.append(pnl_pct)
                if pnl_pct >= 0:
                    result.wins += 1
                else:
                    result.losses += 1
                result.trades_log.append({
                    "date": str(date.date()),
                    "action": "SELL",
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

        positions = [p for p in positions if p.symbol not in sells_to_remove]

        # 2. Check dip buys
        dip_assets = [
            (sym, chg) for sym, chg in changes_24h.items()
            if chg <= dip_threshold and sym in DCA_ASSETS
        ]

        if dip_assets and cash >= MIN_ORDER_USDT:
            per_buy = cash / len(dip_assets)
            for symbol, change in dip_assets:
                buy_amount = min(per_buy, cash)
                if buy_amount < MIN_ORDER_USDT:
                    continue
                price = current_prices[symbol]
                quantity = buy_amount / price

                # Promediar si ya tenemos posicion
                existing = next(
                    (p for p in positions if p.symbol == symbol), None,
                )
                if existing is not None:
                    total_inv = existing.invested + buy_amount
                    total_qty = existing.quantity + quantity
                    existing.entry_price = total_inv / total_qty
                    existing.quantity = total_qty
                    existing.invested = total_inv
                else:
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
                result.trades_log.append({
                    "date": str(date.date()),
                    "action": "BUY",
                    "symbol": symbol,
                    "price": round(price, 2),
                    "amount_usdt": round(buy_amount, 2),
                    "change_24h": round(change * 100, 1),
                })

        # Calcular equity diaria
        pos_value = sum(
            p.quantity * current_prices.get(p.symbol, p.entry_price)
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
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Resultados finales
    final_pos_value = 0.0
    for pos in positions:
        last_prices: dict[str, float] = {}
        for symbol, df in prices.items():
            if len(df) > 0:
                last_prices[symbol] = df.iloc[-1]["close"]
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
    result.total_profit = round(result.final_value - budget, 2)
    result.max_drawdown_pct = round(max_dd * 100, 2)
    result.best_trade_pct = round(
        max(trade_returns) * 100, 1,
    ) if trade_returns else 0.0
    result.worst_trade_pct = round(
        min(trade_returns) * 100, 1,
    ) if trade_returns else 0.0
    result.avg_hold_days = round(
        sum(hold_days_list) / len(hold_days_list), 1,
    ) if hold_days_list else 0.0

    return result


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def print_report(result: BacktestResult, days: int) -> str:
    """Genera reporte legible del backtesting."""
    pnl_pct = (
        (result.final_value - result.initial_budget) / result.initial_budget * 100
        if result.initial_budget > 0 else 0
    )
    pnl_sign = "+" if result.total_profit >= 0 else ""

    lines = [
        "",
        "=" * 60,
        "  BACKTESTING DCA INTELIGENTE - RESULTADOS",
        "=" * 60,
        "",
        f"  Periodo:           Ultimos {days} dias",
        f"  Activos:           {', '.join(DCA_ASSETS)}",
        f"  Umbral de compra:  Caida >{abs(DIP_THRESHOLD)*100:.0f}% en 24h",
        f"  Take-profit:       +{TAKE_PROFIT_PCT*100:.0f}%",
        f"  Presupuesto:       ${result.initial_budget:,.2f} USDT",
        "",
        "-" * 60,
        "  RENDIMIENTO",
        "-" * 60,
        f"  Valor final:       ${result.final_value:,.2f} USDT",
        f"  Ganancia/Perdida:  {pnl_sign}${result.total_profit:,.2f}"
        f" ({pnl_sign}{pnl_pct:.1f}%)",
        f"  Max drawdown:      -{result.max_drawdown_pct:.1f}%",
        "",
        "-" * 60,
        "  OPERACIONES",
        "-" * 60,
        f"  Total trades:      {result.total_trades}",
        f"  Compras:           {result.buys}",
        f"  Ventas:            {result.sells}",
        f"  Ganadoras:         {result.wins}",
        f"  Perdedoras:        {result.losses}",
        f"  Win rate:          "
        f"{result.wins / result.sells * 100:.0f}%" if result.sells > 0
        else "  Win rate:          N/A (sin ventas aun)",
        f"  Mejor trade:       +{result.best_trade_pct:.1f}%",
        f"  Peor trade:        {result.worst_trade_pct:.1f}%",
        f"  Dias hold promedio:{result.avg_hold_days:.0f} dias",
        "",
    ]

    if result.open_positions:
        lines.extend([
            "-" * 60,
            "  POSICIONES ABIERTAS",
            "-" * 60,
        ])
        for p in result.open_positions:
            sign = "+" if p["pnl_pct"] >= 0 else ""
            lines.append(
                f"  {p['symbol']:10s} | Compra: ${p['entry_price']:>10,.2f}"
                f" | Actual: ${p['current_price']:>10,.2f}"
                f" | {sign}{p['pnl_pct']:.1f}%"
            )
        lines.append("")

    if result.trades_log:
        lines.extend([
            "-" * 60,
            "  HISTORIAL DE OPERACIONES",
            "-" * 60,
        ])
        for t in result.trades_log:
            if t["action"] == "BUY":
                lines.append(
                    f"  {t['date']} | COMPRA {t['symbol']:10s}"
                    f" | ${t['amount_usdt']:>8,.2f}"
                    f" @ ${t['price']:>10,.2f}"
                    f" (caida {t['change_24h']:.1f}%)"
                )
            else:
                sign = "+" if t["profit"] >= 0 else ""
                lines.append(
                    f"  {t['date']} | VENTA  {t['symbol']:10s}"
                    f" | ${t['sell_value']:>8,.2f}"
                    f" @ ${t['price']:>10,.2f}"
                    f" ({sign}${t['profit']:,.2f},"
                    f" +{t['pnl_pct']:.1f}%,"
                    f" {t['hold_days']}d)"
                )
        lines.append("")

    lines.extend([
        "=" * 60,
        f"  VEREDICTO: {pnl_sign}${result.total_profit:,.2f}"
        f" ({pnl_sign}{pnl_pct:.1f}%) en {days} dias",
        "=" * 60,
        "",
    ])

    report = "\n".join(lines)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest DCA Inteligente")
    parser.add_argument(
        "--days", type=int, default=365,
        help="Dias de historia para simular (default: 365)",
    )
    parser.add_argument(
        "--budget", type=float, default=30.0,
        help="Presupuesto DCA en USDT (default: 30 = 40%% de 75)",
    )
    args = parser.parse_args()

    print(f"\nDescargando {args.days} dias de datos historicos...")
    prices: dict[str, pd.DataFrame] = {}
    for symbol in DCA_ASSETS:
        print(f"  {symbol}...", end=" ", flush=True)
        df = download_daily_klines(symbol, args.days)
        prices[symbol] = df
        print(f"{len(df)} velas descargadas")

    print("\nEjecutando simulacion...")
    result = run_backtest(prices, budget=args.budget)

    report = print_report(result, args.days)
    print(report)

    # Guardar reporte
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "backtest_dca_report.txt"
    report_path.write_text(report)
    print(f"Reporte guardado en: {report_path}")


if __name__ == "__main__":
    main()
