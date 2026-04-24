#!/usr/bin/env python3
"""
Backtesting de la estrategia de Prediccion ML.

Simula el comportamiento del bot predictivo sobre datos historicos:
1. Descarga velas horarias para N monedas.
2. Entrena el modelo con la primera mitad de los datos.
3. Recorre la segunda mitad dia a dia, generando predicciones.
4. Compra cuando la probabilidad calibrada >= threshold.
5. Vende por take-profit (+3%) o stop-loss (-5%).
6. Reporta resultados.

Uso:
    python -m scripts.backtest_prediction [--days 30] [--budget 100] [--threshold 0.65]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import load_config
from src.model.predictor import PricePredictor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
BEST_TPSL_FILE = DATA_DIR / "best_tpsl.json"

# ---------------------------------------------------------------------------
# Descarga de datos
# ---------------------------------------------------------------------------

def download_hourly_klines(symbol: str, days: int) -> pd.DataFrame:
    """Descarga velas horarias de Binance (API publica, sin auth)."""
    url = "https://data-api.binance.vision/api/v3/klines"
    all_data: list[list] = []
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    current = start_time
    while current < end_time:
        params = {
            "symbol": symbol,
            "interval": "1h",
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
        time.sleep(0.1)

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(all_data, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("open_time")
    return df


# ---------------------------------------------------------------------------
# Costes de trading (realistas para Binance spot)
# ---------------------------------------------------------------------------
COMMISSION_RATE = 0.001   # 0.1% por operacion (taker fee)
SLIPPAGE_RATE = 0.0005    # 0.05% slippage estimado

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
    tp_price: float
    sl_price: float


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
    avg_hold_hours: float = 0.0
    recommendations_made: int = 0
    avg_probability: float = 0.0
    trades_log: list[dict] = field(default_factory=list)
    open_positions: list[dict] = field(default_factory=list)


def run_backtest(
    klines_by_symbol: dict[str, pd.DataFrame],
    predictor: PricePredictor,
    budget: float,
    threshold: float,
    tp_pct: float,
    sl_pct: float,
    max_positions: int,
    quote: str,
    *,
    skip_train: bool = False,
) -> BacktestResult:
    """Simula la estrategia de prediccion sobre datos historicos.

    Divide los datos: primera mitad para entrenar, segunda mitad para simular.
    Cada 24 horas (simulando ejecucion 2x/dia), genera predicciones
    y compra si la probabilidad supera el threshold.

    Si *skip_train* es True, se asume que el predictor ya esta entrenado
    y se salta el paso de entrenamiento (util para sweep de thresholds).
    """
    result = BacktestResult(initial_budget=budget)
    positions: list[BTPosition] = []
    cash = budget
    peak_equity = budget
    max_dd = 0.0
    hold_hours_list: list[float] = []
    trade_returns: list[float] = []
    all_probs: list[float] = []

    # Dividir datos: primera mitad entrena, segunda mitad simula
    # Encontrar el punto medio temporal
    all_timestamps: set[pd.Timestamp] = set()
    for df in klines_by_symbol.values():
        all_timestamps.update(df.index)
    sorted_ts = sorted(all_timestamps)

    if len(sorted_ts) < 48:
        print("Error: datos insuficientes para backtesting.")
        return result

    mid_point = sorted_ts[len(sorted_ts) // 2]
    print(f"\n  Datos totales: {len(sorted_ts)} horas")
    print(f"  Entrenamiento: hasta {mid_point}")
    print(f"  Simulacion: desde {mid_point}")

    # Preparar datos de entrenamiento (primera mitad)
    train_klines: dict[str, pd.DataFrame] = {}
    for symbol, df in klines_by_symbol.items():
        train_df = df[df.index <= mid_point]
        if len(train_df) >= 48:
            train_klines[symbol] = train_df

    if not skip_train:
        if len(train_klines) < 5:
            print("Error: muy pocas monedas con datos suficientes para entrenar.")
            return result

        # Entrenar modelo
        print(f"\n  Entrenando modelo con {len(train_klines)} monedas...")
        metrics = predictor.train(train_klines)
        print(f"  AUC CV: {metrics['mean_auc']:.4f}")
        print(f"  Precision: {metrics.get('cv_precision_1', 0):.1%}")
        print(f"  Recall: {metrics.get('cv_recall_1', 0):.1%}")

    # Simular dia a dia sobre la segunda mitad
    sim_timestamps = [ts for ts in sorted_ts if ts > mid_point]

    # Simular cada 12 horas (como el cron de 7:00 y 19:00)
    sim_steps = sim_timestamps[::12]
    if not sim_steps:
        sim_steps = sim_timestamps[::6]

    print(f"\n  Simulando {len(sim_steps)} pasos de prediccion...")
    print(f"  Threshold: {threshold:.0%} | TP: +{tp_pct:.0%} | SL: -{sl_pct:.0%}")
    print(f"  Max posiciones: {max_positions}")
    print()

    for step_idx, current_ts in enumerate(sim_steps):
        # Preparar ventana de datos para prediccion (ultimas 120 horas)
        lookback_start = current_ts - pd.Timedelta(hours=120)
        pred_klines: dict[str, pd.DataFrame] = {}
        for symbol, df in klines_by_symbol.items():
            window = df[(df.index >= lookback_start) & (df.index <= current_ts)]
            if len(window) >= 48:
                pred_klines[symbol] = window

        if len(pred_klines) < 5:
            continue

        # 1. Revisar TP/SL de posiciones abiertas.
        #    Recorremos CADA vela horaria entre el step anterior y el actual
        #    para simular como funcionan las OCO reales (trigger inmediato).
        prev_ts = sim_steps[step_idx - 1] if step_idx > 0 else mid_point
        sells_to_remove: list[int] = []
        for idx, pos in enumerate(positions):
            if pos.symbol not in klines_by_symbol:
                continue
            sym_df = klines_by_symbol[pos.symbol]
            entry_ts = pd.Timestamp(pos.entry_date)
            check_start = max(entry_ts, prev_ts)
            candles = sym_df[(sym_df.index > check_start) & (sym_df.index <= current_ts)]

            sell_reason = ""
            sell_price = 0.0
            sell_ts = current_ts

            for candle_ts, candle in candles.iterrows():
                # Comprobar SL primero (consistente con create_labels)
                if candle["low"] <= pos.sl_price:
                    sell_reason = "SL"
                    sell_price = pos.sl_price
                    sell_ts = candle_ts
                    result.stop_losses += 1
                    break
                if candle["high"] >= pos.tp_price:
                    sell_reason = "TP"
                    sell_price = pos.tp_price
                    sell_ts = candle_ts
                    result.take_profits += 1
                    break

            if sell_reason:
                # Aplicar slippage al precio de venta (recibimos un poco menos)
                eff_sell = sell_price * (1 - SLIPPAGE_RATE)
                sell_value = pos.quantity * eff_sell
                sell_commission = sell_value * COMMISSION_RATE
                sell_value -= sell_commission
                pnl_pct = (eff_sell - pos.entry_price) / pos.entry_price
                profit = sell_value - pos.invested
                cash += sell_value
                result.sells += 1
                result.total_trades += 1
                hold_h = (sell_ts - entry_ts).total_seconds() / 3600
                hold_hours_list.append(hold_h)
                trade_returns.append(pnl_pct)
                if profit >= 0:
                    result.wins += 1
                else:
                    result.losses += 1
                result.trades_log.append({
                    "date": str(sell_ts),
                    "action": f"SELL({sell_reason})",
                    "symbol": pos.symbol,
                    "price": round(sell_price, 6),
                    "entry_price": round(pos.entry_price, 6),
                    "invested": round(pos.invested, 2),
                    "sell_value": round(sell_value, 2),
                    "profit": round(profit, 2),
                    "pnl_pct": round(pnl_pct * 100, 1),
                    "hold_hours": round(hold_h, 1),
                })
                sells_to_remove.append(idx)

        positions = [p for i, p in enumerate(positions) if i not in sells_to_remove]

        # 2. Generar predicciones
        try:
            predictions = predictor.predict(pred_klines)
        except Exception as exc:
            print(f"  Error prediciendo en {current_ts}: {exc}")
            continue

        recommendations = predictor.get_recommendations(predictions, threshold)

        if recommendations:
            all_probs.extend([prob for _, prob in recommendations])
            result.recommendations_made += len(recommendations)

        # 3. Comprar las recomendadas (si hay espacio)
        held_symbols = {p.symbol for p in positions}
        for symbol, prob in recommendations:
            if len(positions) >= max_positions:
                break
            if symbol in held_symbols:
                continue
            if symbol not in pred_klines:
                continue

            # Calcular cuanto invertir (reparto equitativo del cash disponible)
            available_slots = max_positions - len(positions)
            per_buy = cash / max(available_slots, 1)
            buy_amount = min(per_buy, cash * 0.20)  # max 20% por posicion
            if buy_amount < 10.0:
                continue

            current_price = pred_klines[symbol]["close"].iloc[-1]
            # Aplicar slippage al precio de compra (pagamos un poco mas)
            fill_price = current_price * (1 + SLIPPAGE_RATE)
            commission = buy_amount * COMMISSION_RATE
            net_invest = buy_amount - commission
            quantity = net_invest / fill_price
            tp_price = fill_price * (1 + tp_pct)
            sl_price = fill_price * (1 - sl_pct)

            positions.append(BTPosition(
                symbol=symbol,
                entry_price=fill_price,
                quantity=quantity,
                invested=buy_amount,
                entry_date=str(current_ts),
                tp_price=tp_price,
                sl_price=sl_price,
            ))
            held_symbols.add(symbol)
            cash -= buy_amount
            result.buys += 1
            result.total_trades += 1
            result.trades_log.append({
                "date": str(current_ts),
                "action": "BUY",
                "symbol": symbol,
                "price": round(current_price, 6),
                "amount": round(buy_amount, 2),
                "probability": round(prob * 100, 1),
                "tp_target": round(tp_price, 6),
                "sl_target": round(sl_price, 6),
            })

        # Calcular equity
        pos_value = 0.0
        for pos in positions:
            if pos.symbol in pred_klines:
                price = pred_klines[pos.symbol]["close"].iloc[-1]
            else:
                price = pos.entry_price
            pos_value += pos.quantity * price

        equity = cash + pos_value
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Cerrar posiciones abiertas al precio final (con costes)
    final_pos_value = 0.0
    for pos in positions:
        if pos.symbol in klines_by_symbol:
            last_price = klines_by_symbol[pos.symbol]["close"].iloc[-1]
        else:
            last_price = pos.entry_price
        eff_close = last_price * (1 - SLIPPAGE_RATE)
        pv = pos.quantity * eff_close * (1 - COMMISSION_RATE)
        final_pos_value += pv
        pnl_pct = (eff_close - pos.entry_price) / pos.entry_price
        result.open_positions.append({
            "symbol": pos.symbol,
            "entry_price": round(pos.entry_price, 6),
            "current_price": round(last_price, 6),
            "invested": round(pos.invested, 2),
            "current_value": round(pv, 2),
            "pnl_pct": round(pnl_pct * 100, 1),
        })

    result.final_value = round(cash + final_pos_value, 2)
    result.total_profit = round(result.final_value - budget, 2)
    result.max_drawdown_pct = round(max_dd * 100, 2)
    if trade_returns:
        result.best_trade_pct = round(max(trade_returns) * 100, 1)
        result.worst_trade_pct = round(min(trade_returns) * 100, 1)
    if hold_hours_list:
        result.avg_hold_hours = round(sum(hold_hours_list) / len(hold_hours_list), 1)
    if all_probs:
        result.avg_probability = round(float(np.mean(all_probs)) * 100, 1)

    return result


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def _fmt_sign(val: float) -> str:
    return "+" if val >= 0 else ""


def print_report(result: BacktestResult, days: int, threshold: float) -> str:
    """Genera reporte legible del backtesting de prediccion."""
    pnl_pct = (
        (result.final_value - result.initial_budget)
        / result.initial_budget * 100
        if result.initial_budget > 0 else 0
    )
    s = _fmt_sign(result.total_profit)

    lines = [
        "",
        "=" * 65,
        "  BACKTESTING ESTRATEGIA PREDICCION ML - RESULTADOS",
        "=" * 65,
        "",
        f"  Periodo total:     {days} dias ({days // 2} entrenamiento + {days // 2} simulacion)",
        f"  Presupuesto:       ${result.initial_budget:,.2f}",
        f"  Threshold:         {threshold:.0%}",
        "",
        "-" * 65,
        "  RENDIMIENTO",
        "-" * 65,
        f"  Valor final:       ${result.final_value:,.2f}",
        f"  Ganancia/Perdida:  {s}${result.total_profit:,.2f} ({s}{pnl_pct:.1f}%)",
        f"  Max drawdown:      -{result.max_drawdown_pct:.1f}%",
        "",
        "-" * 65,
        "  PREDICCIONES",
        "-" * 65,
        f"  Recomendaciones:   {result.recommendations_made}",
        f"  Prob. media reco.: {result.avg_probability:.1f}%",
        "",
        "-" * 65,
        "  OPERACIONES",
        "-" * 65,
        f"  Total trades:      {result.total_trades}",
        f"  Compras:           {result.buys}",
        f"  Ventas:            {result.sells}"
        f" (TP: {result.take_profits} | SL: {result.stop_losses})",
    ]

    if result.sells > 0:
        wr = result.wins / result.sells * 100
        lines.append(f"  Win rate:          {wr:.0f}%")
    else:
        lines.append("  Win rate:          N/A (sin ventas cerradas)")

    lines.extend([
        f"  Ganadoras:         {result.wins}",
        f"  Perdedoras:        {result.losses}",
        f"  Mejor trade:       +{result.best_trade_pct:.1f}%",
        f"  Peor trade:        {result.worst_trade_pct:.1f}%",
        f"  Hold promedio:     {result.avg_hold_hours:.0f}h",
    ])

    if result.open_positions:
        lines.extend([
            "",
            "-" * 65,
            "  POSICIONES ABIERTAS (no vendidas al final)",
            "-" * 65,
        ])
        for p in result.open_positions:
            ps = _fmt_sign(p["pnl_pct"])
            lines.append(
                f"  {p['symbol']:12s}"
                f" | Compra: {p['entry_price']:>12.6f}"
                f" | Actual: {p['current_price']:>12.6f}"
                f" | {ps}{p['pnl_pct']:.1f}%"
            )

    if result.trades_log:
        lines.extend([
            "",
            "-" * 65,
            "  HISTORIAL DE OPERACIONES",
            "-" * 65,
        ])
        for t in result.trades_log:
            if t["action"] == "BUY":
                lines.append(
                    f"  {t['date'][:16]} | BUY  {t['symbol']:12s}"
                    f" | ${t['amount']:.2f}"
                    f" @ {t['price']:.6f}"
                    f" | prob={t['probability']:.1f}%"
                )
            else:
                ps = _fmt_sign(t["pnl_pct"])
                lines.append(
                    f"  {t['date'][:16]} | {t['action']:8s} {t['symbol']:12s}"
                    f" | ${t['sell_value']:.2f}"
                    f" @ {t['price']:.6f}"
                    f" | {ps}{t['pnl_pct']:.1f}%"
                    f" | {t['hold_hours']:.0f}h"
                )

    lines.extend(["", "=" * 65, ""])
    report = "\n".join(lines)
    print(report)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_sweep_table(sweep_results: list[dict]) -> None:
    """Imprime tabla comparativa de multiples thresholds."""
    print()
    print("=" * 80)
    print("  COMPARATIVA DE THRESHOLDS")
    print("=" * 80)
    print()
    header = (
        f"  {'Threshold':>10s} | {'Trades':>7s} | {'Wins':>5s} | "
        f"{'Losses':>7s} | {'Win%':>5s} | {'P&L':>10s} | "
        f"{'P&L%':>7s} | {'MaxDD':>6s}"
    )
    print(header)
    print("  " + "-" * 76)
    for row in sweep_results:
        wr = (
            f"{row['wins'] / row['sells'] * 100:.0f}%"
            if row["sells"] > 0 else "N/A"
        )
        s = "+" if row["pnl"] >= 0 else ""
        print(
            f"  {row['threshold']:>9.0%} | {row['trades']:>7d} | "
            f"{row['wins']:>5d} | {row['losses']:>7d} | "
            f"{wr:>5s} | {s}${row['pnl']:>8.2f} | "
            f"{s}{row['pnl_pct']:>6.1f}% | "
            f"-{row['max_dd']:.1f}%"
        )
    print()
    print("=" * 80)
    # Highlight best threshold
    profitable = [r for r in sweep_results if r["trades"] > 0]
    if profitable:
        best = max(profitable, key=lambda r: r["pnl"])
        print(
            f"  >> Mejor resultado: threshold={best['threshold']:.0%}"
            f" con {best['trades']} trades,"
            f" P&L={'+' if best['pnl'] >= 0 else ''}${best['pnl']:.2f}"
            f" ({'+' if best['pnl_pct'] >= 0 else ''}{best['pnl_pct']:.1f}%)"
        )
    else:
        print("  >> Ning\u00fan threshold produjo trades.")
    print()


def _generate_tpsl_combinations() -> list[tuple[float, float]]:
    """Genera combinaciones de TP y SL donde TP >= SL."""
    tp_values = [0.03, 0.04, 0.05, 0.06, 0.08]
    sl_values = [0.03, 0.04, 0.05, 0.06, 0.08]
    combos: list[tuple[float, float]] = []
    for tp in tp_values:
        for sl in sl_values:
            if tp >= sl:
                combos.append((tp, sl))
    return combos


def print_tpsl_sweep_table(rows: list[dict]) -> None:
    """Imprime tabla comparativa de combinaciones TP-SL."""
    print()
    print("=" * 95)
    print("  COMPARATIVA TP-SL (TP \u2265 SL)")
    print("=" * 95)
    print()
    header = (
        f"  {'TP%':>5s} | {'SL%':>5s} | {'Trades':>7s} | "
        f"{'Win%':>5s} | {'TP':>4s} | {'SL':>4s} | "
        f"{'P&L medio':>10s} | {'P&L comp.':>10s} | "
        f"{'MaxDD':>6s}"
    )
    print(header)
    print("  " + "-" * 91)
    for r in rows:
        wr = f"{r['win_pct']:.0f}%" if r["sells"] > 0 else "N/A"
        s_avg = "+" if r["avg_pnl_per_trade"] >= 0 else ""
        s_comp = "+" if r["compound_pnl"] >= 0 else ""
        print(
            f"  {r['tp_pct']:>4.0f}% | {r['sl_pct']:>4.0f}% | "
            f"{r['trades']:>7d} | {wr:>5s} | "
            f"{r['take_profits']:>4d} | {r['stop_losses']:>4d} | "
            f"{s_avg}${r['avg_pnl_per_trade']:>8.2f} | "
            f"{s_comp}{r['compound_pnl']:>8.1f}% | "
            f"-{r['max_dd']:.1f}%"
        )
    print()
    print("=" * 95)

    with_trades = [r for r in rows if r["trades"] > 0]
    if with_trades:
        best_comp = max(with_trades, key=lambda r: r["compound_pnl"])
        best_avg = max(with_trades, key=lambda r: r["avg_pnl_per_trade"])
        print(
            f"  >> Mejor P&L compuesto: TP={best_comp['tp_pct']:.0f}% / "
            f"SL={best_comp['sl_pct']:.0f}% \u2192 "
            f"{'+' if best_comp['compound_pnl'] >= 0 else ''}"
            f"{best_comp['compound_pnl']:.1f}%"
        )
        print(
            f"  >> Mejor P&L medio/trade: TP={best_avg['tp_pct']:.0f}% / "
            f"SL={best_avg['sl_pct']:.0f}% \u2192 "
            f"{'+' if best_avg['avg_pnl_per_trade'] >= 0 else ''}"
            f"${best_avg['avg_pnl_per_trade']:.2f}"
        )
    else:
        print("  >> Ninguna combinaci\u00f3n produjo trades.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest estrategia prediccion ML")
    parser.add_argument(
        "--days", type=int, default=30,
        help="Dias totales (mitad entrena, mitad simula)",
    )
    parser.add_argument(
        "--budget", type=float, default=100.0,
        help="Presupuesto inicial en USDC/USDT",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="Threshold de confianza (0-1)",
    )
    parser.add_argument(
        "--coins", type=int, default=30,
        help="Numero de monedas a analizar",
    )
    parser.add_argument(
        "--tp", type=float, default=0.05,
        help="Take-profit (default 5%%)",
    )
    parser.add_argument(
        "--sl", type=float, default=0.05,
        help="Stop-loss (default 5%%)",
    )
    parser.add_argument(
        "--max-positions", type=int, default=5,
        help="Maximo posiciones simultaneas",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Ejecutar barrido de thresholds (solo 65%%)",
    )
    parser.add_argument(
        "--sweep-tpsl", action="store_true",
        help="Barrido de combinaciones TP-SL (siempre TP >= SL)",
    )
    args = parser.parse_args()

    config = load_config()
    quote = config.portfolio.quote_asset

    print(f"\n  Descargando datos de las top {args.coins} monedas ({args.days} dias)...")
    print(f"  Quote asset: {quote}")

    # Obtener top monedas por volumen (via API publica)
    url = "https://data-api.binance.vision/api/v3/ticker/24hr"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    tickers = resp.json()

    # Filtrar por quote asset y ordenar por volumen
    _STABLECOIN_BASES = (
        "USDC", "USDT", "BUSD", "TUSD", "FDUSD", "DAI",
        "UST", "USDP", "GUSD", "FRAX", "LUSD", "PYUSD",
        "USDD", "EURC", "EURT",
    )
    _FIAT_BASES = ("EUR", "GBP", "TRY", "BRL")
    quote_tickers = [
        t for t in tickers
        if t["symbol"].endswith(quote)
        and not t["symbol"].startswith(_STABLECOIN_BASES + _FIAT_BASES)
    ]
    quote_tickers.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
    top_symbols = [t["symbol"] for t in quote_tickers[:args.coins]]

    print(f"  Top {len(top_symbols)} monedas: {', '.join(top_symbols[:5])}...")

    # Descargar klines horarias
    klines_by_symbol: dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(top_symbols):
        try:
            df = download_hourly_klines(symbol, args.days)
            if len(df) >= 48:
                klines_by_symbol[symbol] = df
            if (i + 1) % 10 == 0:
                print(f"  Descargadas {i + 1}/{len(top_symbols)} monedas...")
        except Exception as exc:
            print(f"  Error descargando {symbol}: {exc}")

    print(f"  {len(klines_by_symbol)} monedas con datos suficientes")

    if len(klines_by_symbol) < 5:
        print("Error: muy pocas monedas con datos. Abortando.")
        sys.exit(1)

    # Crear predictor y ejecutar backtest
    predictor = PricePredictor(config.model)

    if args.sweep_tpsl:
        # ---------------------------------------------------------
        # Barrido TP-SL: entrena una vez, recorre combinaciones
        # ---------------------------------------------------------
        combos = _generate_tpsl_combinations()
        thr = args.threshold
        print(f"\n  Barrido TP-SL: {len(combos)} combinaciones (threshold={thr:.0%})")
        print("  Entrenando modelo una sola vez...")

        # Primera ejecuci\u00f3n entrena el modelo
        first_tp, first_sl = combos[0]
        first_result = run_backtest(
            klines_by_symbol=klines_by_symbol,
            predictor=predictor,
            budget=args.budget,
            threshold=thr,
            tp_pct=first_tp,
            sl_pct=first_sl,
            max_positions=args.max_positions,
            quote=quote,
            skip_train=False,
        )

        tpsl_rows: list[dict] = []

        def _build_tpsl_row(
            r: BacktestResult, tp: float, sl: float,
        ) -> dict:
            sells = r.sells
            win_pct = (r.wins / sells * 100) if sells > 0 else 0.0
            # P&L medio por operaci\u00f3n (realista)
            avg_pnl = r.total_profit / sells if sells > 0 else 0.0
            # P&L compuesto (simulando reinversi\u00f3n desde budget inicial)
            compound = (
                (r.final_value / r.initial_budget - 1) * 100
                if r.initial_budget > 0 else 0.0
            )
            return {
                "tp_pct": tp * 100,
                "sl_pct": sl * 100,
                "trades": r.total_trades,
                "sells": sells,
                "wins": r.wins,
                "losses": r.losses,
                "take_profits": r.take_profits,
                "stop_losses": r.stop_losses,
                "win_pct": win_pct,
                "avg_pnl_per_trade": avg_pnl,
                "compound_pnl": compound,
                "max_dd": r.max_drawdown_pct,
            }

        tpsl_rows.append(_build_tpsl_row(first_result, first_tp, first_sl))
        print(
            f"  [1/{len(combos)}] TP={first_tp:.0%} SL={first_sl:.0%} \u2192 "
            f"{first_result.total_trades} trades"
        )

        # Resto de combinaciones reutilizando modelo ya entrenado
        for idx, (tp, sl) in enumerate(combos[1:], start=2):
            r = run_backtest(
                klines_by_symbol=klines_by_symbol,
                predictor=predictor,
                budget=args.budget,
                threshold=thr,
                tp_pct=tp,
                sl_pct=sl,
                max_positions=args.max_positions,
                quote=quote,
                skip_train=True,
            )
            tpsl_rows.append(_build_tpsl_row(r, tp, sl))
            print(
                f"  [{idx}/{len(combos)}] TP={tp:.0%} SL={sl:.0%} \u2192 "
                f"{r.total_trades} trades"
            )

        # Ordenar por P&L compuesto descendente
        tpsl_rows.sort(key=lambda x: x["compound_pnl"], reverse=True)
        print_tpsl_sweep_table(tpsl_rows)

        # Guardar la mejor combinación en data/best_tpsl.json
        with_trades = [r for r in tpsl_rows if r["trades"] > 0]
        if with_trades:
            best = with_trades[0]  # ya ordenado por compound_pnl desc
            best_data = {
                "take_profit_pct": best["tp_pct"] / 100,
                "stop_loss_pct": best["sl_pct"] / 100,
                "compound_pnl_pct": round(best["compound_pnl"], 2),
                "avg_pnl_per_trade": round(best["avg_pnl_per_trade"], 4),
                "win_pct": round(best["win_pct"], 1),
                "trades": best["trades"],
                "max_drawdown_pct": best["max_dd"],
                "sweep_date": datetime.now(timezone.utc).isoformat(),
                "sweep_days": args.days,
                "sweep_threshold": thr,
            }
            BEST_TPSL_FILE.write_text(json.dumps(best_data, indent=2))
            print(
                f"  \u2192 Guardado en {BEST_TPSL_FILE}: "
                f"TP={best_data['take_profit_pct']:.0%} / "
                f"SL={best_data['stop_loss_pct']:.0%}"
            )
            print(
                "  El bot usar\u00e1 estos valores autom\u00e1ticamente "
                "en la pr\u00f3xima ejecuci\u00f3n."
            )

    elif args.sweep:
        # Backtest con threshold unico (65%)
        thr = 0.65
        result = run_backtest(
            klines_by_symbol=klines_by_symbol,
            predictor=predictor,
            budget=args.budget,
            threshold=thr,
            tp_pct=args.tp,
            sl_pct=args.sl,
            max_positions=args.max_positions,
            quote=quote,
        )
        pnl_pct = (
            (result.final_value - result.initial_budget)
            / result.initial_budget * 100
            if result.initial_budget > 0 else 0
        )
        sweep_results: list[dict] = [{
            "threshold": thr,
            "trades": result.total_trades,
            "buys": result.buys,
            "sells": result.sells,
            "wins": result.wins,
            "losses": result.losses,
            "pnl": result.total_profit,
            "pnl_pct": pnl_pct,
            "max_dd": result.max_drawdown_pct,
            "recos": result.recommendations_made,
        }]
        print(
            f"  Threshold {thr:.0%}: "
            f"{result.total_trades} trades, "
            f"P&L={'+'if result.total_profit>=0 else ''}"
            f"${result.total_profit:.2f}"
        )

        print_sweep_table(sweep_results)
    else:
        result = run_backtest(
            klines_by_symbol=klines_by_symbol,
            predictor=predictor,
            budget=args.budget,
            threshold=args.threshold,
            tp_pct=args.tp,
            sl_pct=args.sl,
            max_positions=args.max_positions,
            quote=quote,
        )

        print_report(result, args.days, args.threshold)


if __name__ == "__main__":
    main()
