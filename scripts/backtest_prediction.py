#!/usr/bin/env python3
"""
Backtesting de la estrategia de Prediccion ML.

Simula el comportamiento del bot predictivo sobre datos historicos:
1. Descarga velas horarias para N monedas.
2. Entrena el modelo con la primera mitad de los datos.
3. Recorre la segunda mitad dia a dia, generando predicciones.
4. Compra cuando la probabilidad calibrada >= threshold.
5. Vende por take-profit (+5%) o stop-loss (-3%).
6. Reporta resultados.

Uso:
    python -m scripts.backtest_prediction [--days 30] [--budget 100] [--threshold 0.65]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from src.config import load_config
from src.model.predictor import PricePredictor

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
) -> BacktestResult:
    """Simula la estrategia de prediccion sobre datos historicos.

    Divide los datos: primera mitad para entrenar, segunda mitad para simular.
    Cada 24 horas (simulando ejecucion 2x/dia), genera predicciones
    y compra si la probabilidad supera el threshold.
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

        # 1. Revisar TP/SL de posiciones abiertas
        sells_to_remove: list[int] = []
        for idx, pos in enumerate(positions):
            if pos.symbol not in pred_klines:
                continue
            current_price = pred_klines[pos.symbol]["close"].iloc[-1]

            pnl_pct = (current_price - pos.entry_price) / pos.entry_price
            sell_reason = ""

            if current_price >= pos.tp_price:
                sell_reason = "TP"
                result.take_profits += 1
            elif current_price <= pos.sl_price:
                sell_reason = "SL"
                result.stop_losses += 1

            if sell_reason:
                sell_value = pos.quantity * current_price
                profit = sell_value - pos.invested
                cash += sell_value
                result.sells += 1
                result.total_trades += 1
                hold_h = (current_ts - pd.Timestamp(pos.entry_date)).total_seconds() / 3600
                hold_hours_list.append(hold_h)
                trade_returns.append(pnl_pct)
                if profit >= 0:
                    result.wins += 1
                else:
                    result.losses += 1
                result.trades_log.append({
                    "date": str(current_ts),
                    "action": f"SELL({sell_reason})",
                    "symbol": pos.symbol,
                    "price": round(current_price, 6),
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
            quantity = buy_amount / current_price
            tp_price = current_price * (1 + tp_pct)
            sl_price = current_price * (1 - sl_pct)

            positions.append(BTPosition(
                symbol=symbol,
                entry_price=current_price,
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

    # Cerrar posiciones abiertas al precio final
    final_pos_value = 0.0
    for pos in positions:
        if pos.symbol in klines_by_symbol:
            last_price = klines_by_symbol[pos.symbol]["close"].iloc[-1]
        else:
            last_price = pos.entry_price
        pv = pos.quantity * last_price
        final_pos_value += pv
        pnl_pct = (last_price - pos.entry_price) / pos.entry_price
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
        "--sl", type=float, default=0.03,
        help="Stop-loss (default 3%%)",
    )
    parser.add_argument(
        "--max-positions", type=int, default=5,
        help="Maximo posiciones simultaneas",
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
    quote_tickers = [
        t for t in tickers
        if t["symbol"].endswith(quote)
        and not any(
            t["symbol"].startswith(s)
            for s in ("USDC", "USDT", "BUSD", "DAI", "TUSD", "FDUSD", "EUR", "GBP", "TRY", "BRL")
        )
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
