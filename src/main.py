"""
Orquestador principal del bot de trading.

Este es el punto de entrada.  Coordina todos los módulos:
1. Carga configuración.
2. Conecta con Binance.
3. Inicializa el asignador de cartera (virtual wallets).
4. Ejecuta la estrategia de PREDICCION (35% del balance).
5. Ejecuta la estrategia DCA Inteligente (20% del balance).
6. Ejecuta la estrategia Momentum (35% del balance).
7. Envía el reporte por email (incluye todas las estrategias).
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from src.allocation.allocator import PortfolioAllocator
from src.config import (
    BEST_TPSL_FILE,
    DEFAULT_ASSET_POLICIES,
    DEFAULT_MOMENTUM_POLICIES,
    MODELS_DIR,
    AppConfig,
    load_config,
)
from src.data.binance_client import BinanceDataClient, BinanceTradingClient
from src.execution.executor import ExecutionResult, OrderExecutor
from src.market.regime import MarketRegimeResult, evaluate_market_regime
from src.model.predictor import PricePredictor
from src.notifications.email_report import send_daily_report
from src.portfolio.manager import PortfolioManager, TradeAction
from src.strategies.dca import DCAAction, DCAStrategy
from src.strategies.momentum import MomentumAction, MomentumStrategy
from src.strategies.prediction_book import PredictionBook
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_FILE = MODELS_DIR / "predictor.joblib"


def run_daily(config: AppConfig | None = None) -> None:
    """Ejecución diaria completa del bot (predicción + DCA)."""
    config = config or load_config()
    logger.info("=" * 60)
    now = datetime.now(timezone.utc).isoformat()
    logger.info("INICIO — %s — modo=%s", now, config.trading_mode)
    logger.info("=" * 60)

    # Mostrar TP/SL activos (auto-seleccionados o por defecto)
    tp = config.risk.take_profit_pct
    sl = config.risk.stop_loss_pct
    src = "best_tpsl.json" if BEST_TPSL_FILE.exists() else "defaults"
    logger.info("TP=%.1f%% / SL=%.1f%% (fuente: %s)", tp * 100, sl * 100, src)

    # ------------------------------------------------------------------
    # 1. Inicializar clientes
    # ------------------------------------------------------------------
    data_client = BinanceDataClient(config.binance)

    trading_client: BinanceTradingClient | None = None
    is_paper = config.is_paper_trading

    if not is_paper:
        try:
            trading_client = BinanceTradingClient(config.binance, config.portfolio, config.risk)
        except Exception as exc:
            logger.warning(
                "No se pudo crear BinanceTradingClient: %s. Cayendo a modo paper.",
                exc,
            )
            is_paper = True

    executor = OrderExecutor(config, trading_client, data_client)
    portfolio_mgr = PortfolioManager(config.portfolio, config.risk)
    predictor = PricePredictor(config.model)

    # ------------------------------------------------------------------
    # 2. Obtener estado actual de la cartera
    # ------------------------------------------------------------------
    if is_paper:
        quote = config.portfolio.quote_asset
        portfolio_before: dict[str, float] = {quote: 1000.0}
        total_value_before = 1000.0
        logger.info("[PAPER] Cartera simulada: 1000 %s", quote)
    else:
        assert trading_client is not None
        try:
            portfolio_before = trading_client.get_portfolio()
            total_value_before = trading_client.get_portfolio_value_usdt()
            logger.info(
                "Cartera actual: %s | Valor: $%.2f",
                portfolio_before,
                total_value_before,
            )
        except Exception as exc:
            logger.warning(
                "Error conectando con Binance para leer cartera: %s. Cayendo a modo paper.",
                exc,
            )
            is_paper = True
            trading_client = None
            executor = OrderExecutor(config, None)
            portfolio_before = {config.portfolio.quote_asset: 1000.0}
            total_value_before = 1000.0

    # ------------------------------------------------------------------
    # 3. Inicializar asignador de cartera (virtual wallets)
    # ------------------------------------------------------------------
    alloc_pcts = {
        "prediction": config.allocation.prediction_pct,
        "dca": config.allocation.dca_pct,
        "momentum": config.allocation.momentum_pct,
        "reserve": config.allocation.reserve_pct,
    }
    allocator = PortfolioAllocator(alloc_pcts)
    if not allocator.is_initialized:
        allocator.initialize(total_value_before)

    # Siempre rebalancear al inicio con el balance real de Binance.
    # Esto corrige desviaciones causadas por OCOs ejecutadas entre
    # ejecuciones, depósitos, retiradas, etc.
    if not is_paper:
        allocator.rebalance(total_value_before)
        logger.info(
            "Allocator rebalanceado con balance real $%.2f",
            total_value_before,
        )

    budgets = allocator.get_all_budgets()
    logger.info(
        "Asignacion: prediccion=$%.2f | dca=$%.2f | momentum=$%.2f | reserva=$%.2f",
        budgets.get("prediction", 0),
        budgets.get("dca", 0),
        budgets.get("momentum", 0),
        budgets.get("reserve", 0),
    )

    # ------------------------------------------------------------------
    # 4. Obtener top monedas y descargar datos
    # ------------------------------------------------------------------
    top_symbols: list[str] = []
    klines: dict[str, pd.DataFrame] = {}

    try:
        logger.info(
            "Obteniendo top %d monedas por volumen...",
            config.model.top_n_coins,
        )
        top_symbols = data_client.get_top_coins_by_volume(
            config.model.top_n_coins,
            quote=config.portfolio.quote_asset,
        )
        logger.info("Top monedas obtenidas: %d", len(top_symbols))

        logger.info(
            "Descargando velas (%dh lookback)...",
            config.model.lookback_hours,
        )
        klines = data_client.get_klines_batch(
            top_symbols,
            interval=config.model.candle_interval,
            lookback_hours=config.model.lookback_hours,
        )
        logger.info("Velas descargadas para %d monedas", len(klines))
    except Exception as exc:
        logger.warning("Error obteniendo datos de Binance: %s", exc)
        if not is_paper:
            raise
        logger.info("[PAPER] Continuando sin datos reales (primera ejecución)")

    # ------------------------------------------------------------------
    # 5. Entrenar o cargar modelo
    # ------------------------------------------------------------------
    model_ready = False
    if klines and _should_retrain(config):
        logger.info("Entrenando modelo con datos extendidos...")
        try:
            training_klines = data_client.get_klines_batch(
                top_symbols,
                interval=config.model.candle_interval,
                lookback_hours=config.model.training_days * 24,
            )
            metrics = predictor.train(training_klines)
            predictor.save()
            model_ready = True
            logger.info("Modelo entrenado — métricas: %s", metrics)
        except Exception as exc:
            logger.warning("Error entrenando modelo: %s", exc)
    elif MODEL_FILE.exists():
        try:
            logger.info("Cargando modelo existente...")
            predictor.load()
            model_ready = True
        except Exception as exc:
            logger.warning("Error cargando modelo: %s", exc)
    else:
        logger.info(
            "No hay modelo entrenado ni datos para entrenar. "
            "Se necesita al menos una ejecución con conexión a Binance."
        )

    # ------------------------------------------------------------------
    # 6. ESTRATEGIA 1: Predicciones (usa budget "prediction")
    # ------------------------------------------------------------------
    predictions: dict[str, float] = {}
    recommendations: list[tuple[str, float]] = []

    if model_ready and klines:
        logger.info("Ejecutando predicciones...")
        predictions = predictor.predict(klines)
        recommendations = predictor.get_recommendations(predictions)
        logger.info(
            "Predicciones: %d monedas analizadas | %d recomendadas (umbral=%.0f%%)",
            len(predictions),
            len(recommendations),
            config.model.confidence_threshold * 100,
        )
        for sym, prob in recommendations[:10]:
            logger.info("  %s — prob=%.1f%%", sym, prob * 100)
    else:
        logger.info(
            "Sin predicciones disponibles (modelo=%s, datos=%d monedas)",
            "listo" if model_ready else "no disponible",
            len(klines),
        )

    # Inicializar libro de posiciones de prediccion (inventario aislado)
    pred_book = PredictionBook()
    prediction_budget = allocator.get_budget("prediction")
    quote = config.portfolio.quote_asset

    # ----------------------------------------------------------
    # 6a-pre. Reconciliar PredictionBook con Binance
    # ----------------------------------------------------------
    # Si una OCO (TP/SL) se ejecutó entre ejecuciones del bot,
    # el JSON queda desincronizado.  Reconciliamos leyendo balances
    # reales (free + locked) y órdenes abiertas.
    if not is_paper and trading_client is not None and pred_book.positions:
        logger.info(
            "Reconciliando PredictionBook (%d posiciones)...",
            len(pred_book.positions),
        )
        try:
            real_portfolio = trading_client.get_portfolio(
                include_locked=True,
            )
            orders_by_symbol: dict[str, list] = {}
            for pos in pred_book.positions:
                orders_by_symbol[pos.symbol] = (
                    trading_client.get_open_orders(pos.symbol)
                )
            closed = pred_book.reconcile(
                real_portfolio, orders_by_symbol, quote,
            )
            if closed:
                logger.info(
                    "Reconciliación: %d posiciones cerradas por OCO",
                    len(closed),
                )
                # Recalcular budget tras reconciliación
                prediction_budget = allocator.get_budget("prediction")
        except Exception as exc:
            logger.warning(
                "Error en reconciliación de PredictionBook: %s", exc,
            )

    # ----------------------------------------------------------
    # 6a-post. Gestionar ordenes limit pendientes de la ejecucion anterior
    # ----------------------------------------------------------
    # Si en la ejecucion anterior se colocaron limit buys que no se
    # ejecutaron al instante, comprobamos su estado:
    #   - FILLED → registrar en el book y colocar OCO
    #   - Cualquier otro estado → cancelar la orden
    if not is_paper and trading_client is not None and pred_book.pending_orders:
        logger.info(
            "Comprobando %d ordenes limit pendientes...",
            len(pred_book.pending_orders),
        )
        for pending in list(pred_book.pending_orders):
            try:
                order_info = trading_client.get_order(
                    pending.symbol, pending.order_id,
                )
                status = order_info.get("status", "UNKNOWN")
                logger.info(
                    "Pending %s (orderId=%d): status=%s",
                    pending.symbol,
                    pending.order_id,
                    status,
                )
                if status == "FILLED":
                    # Extraer fills y registrar en el book
                    fills = order_info.get("fills", [])
                    if fills:
                        total_qty = sum(float(f["qty"]) for f in fills)
                        avg_price = (
                            sum(
                                float(f["price"]) * float(f["qty"])
                                for f in fills
                            ) / total_qty
                            if total_qty > 0
                            else pending.limit_price
                        )
                    else:
                        total_qty = float(
                            order_info.get("executedQty", 0),
                        )
                        avg_price = float(
                            order_info.get("price", pending.limit_price),
                        )
                    if total_qty > 0:
                        pred_book.record_buy(
                            symbol=pending.symbol,
                            price=avg_price,
                            quantity=total_qty,
                            usdt_spent=pending.quote_qty,
                        )
                        # Colocar OCO sobre el precio de entrada real
                        oco_qty = total_qty
                        try:
                            real_bal = trading_client.get_portfolio()
                            base = pending.symbol.replace(quote, "")
                            rb = real_bal.get(base, 0.0)
                            if 0 < rb < total_qty:
                                oco_qty = rb
                        except Exception:
                            pass
                        trading_client.place_oco_sell(
                            symbol=pending.symbol,
                            quantity=oco_qty,
                            entry_price=avg_price,
                        )
                        logger.info(
                            "Pending %s ejecutada: qty=%.8f price=%.8f "
                            "→ OCO colocada sobre precio de entrada",
                            pending.symbol,
                            total_qty,
                            avg_price,
                        )
                    pred_book.remove_pending_order(pending.order_id)
                else:
                    # No ejecutada → cancelar
                    trading_client.cancel_open_orders(pending.symbol)
                    pred_book.remove_pending_order(pending.order_id)
                    logger.info(
                        "Pending %s cancelada (status=%s)",
                        pending.symbol,
                        status,
                    )
            except Exception as exc:
                logger.warning(
                    "Error comprobando pending %s (orderId=%d): %s",
                    pending.symbol,
                    pending.order_id,
                    exc,
                )
                # En caso de error, intentar cancelar por seguridad
                try:
                    trading_client.cancel_open_orders(pending.symbol)
                except Exception:
                    pass
                pred_book.remove_pending_order(pending.order_id)

        # Recalcular budget tras procesar pending orders
        prediction_budget = allocator.get_budget("prediction")

    # ----------------------------------------------------------
    # 6a. Vender posiciones expiradas (ventana temporal cumplida)
    # ----------------------------------------------------------
    # Las posiciones se cierran por 3 vías:
    #   1. TP — OCO en Binance (automático)
    #   2. SL — OCO en Binance (automático)
    #   3. Expiración — si pasan target_horizon_hours sin que
    #      salte ni TP ni SL, vendemos a mercado.
    horizon = config.model.target_horizon_hours
    expired = pred_book.get_expired_positions(horizon)
    expired_results: list[ExecutionResult] = []

    if expired:
        logger.info(
            "Posiciones expiradas (>%dh): %d",
            horizon,
            len(expired),
        )
    for pos in expired:
        logger.info(
            "  Cerrando %s (entrada %s, >%dh)",
            pos.symbol,
            pos.entry_date,
            horizon,
        )
        if is_paper:
            logger.info("[PAPER] Venta por expiración de %s", pos.symbol)
            pred_book.record_sell(pos.symbol)
            continue

        assert trading_client is not None
        # Verificar si la posición aún tiene saldo (la OCO pudo cerrarla)
        try:
            portfolio_now = trading_client.get_portfolio()
            base_asset = pos.symbol.replace(quote, "")
            real_balance = portfolio_now.get(base_asset, 0.0)
        except Exception:
            real_balance = pos.quantity

        balance_value = real_balance * pos.entry_price
        if real_balance <= 0 or balance_value < 1.0:
            logger.info(
                "  %s ya sin balance real (OCO ejecutada, "
                "residuo=%.8f ~$%.4f)",
                pos.symbol,
                real_balance,
                balance_value,
            )
            pred_book.record_sell(pos.symbol)
            continue

        # El executor ya cancela órdenes abiertas (OCO) antes de vender
        sell_action = TradeAction(
            action="SELL",
            symbol=pos.symbol,
            quote_qty=0,
            base_qty=real_balance,
            reason=f"Ventana de {horizon}h expirada",
            probability=0.0,
        )
        result = executor.execute([sell_action])[0]
        expired_results.append(result)

        freed = pred_book.record_sell(pos.symbol)
        if result.success and result.executed_qty > 0:
            sell_value = result.executed_qty * result.executed_price
            profit = sell_value - freed
            allocator.add_profit("prediction", profit)

    # ----------------------------------------------------------
    # 6b. Filtro de régimen de mercado
    # ----------------------------------------------------------
    # Evalúa BTC ROC 24h, BTC RSI 14 y amplitud de mercado.
    # Si el mercado es adverso, se saltan las compras nuevas
    # pero se mantiene la gestión de posiciones (TP/SL/expiración).
    market_regime: MarketRegimeResult | None = None
    btc_symbol = f"BTC{quote}"
    btc_klines = klines.get(btc_symbol)

    if btc_klines is not None and len(btc_klines) >= 25:
        market_regime = evaluate_market_regime(btc_klines, klines)
    else:
        logger.warning(
            "Sin datos de BTC suficientes para filtro de mercado — "
            "se permite operar por defecto."
        )

    buys_blocked = market_regime is not None and not market_regime.allow_buys

    if buys_blocked:
        logger.warning(
            "COMPRAS BLOQUEADAS por filtro de mercado. "
            "Las posiciones existentes (TP/SL/expiración) siguen activas."
        )

    # ----------------------------------------------------------
    # 6c. Compras nuevas (decide_actions ya solo genera compras)
    # ----------------------------------------------------------
    # Recalcular presupuesto tras posibles ventas por expiración
    # Descontar tambien USDT reservado en ordenes limit pendientes
    pred_quote_available = max(
        0.0,
        prediction_budget - pred_book.invested_usdt - pred_book.pending_invested_usdt,
    )
    pred_portfolio = pred_book.get_portfolio_dict(quote)

    # Incluir simbolos con ordenes limit pendientes como slots ocupados
    # para que decide_actions no intente comprar la misma moneda.
    for sym in pred_book.pending_symbols:
        base = sym.replace(quote, "")
        if base not in pred_portfolio:
            pred_portfolio[base] = 0.01  # placeholder

    current_prices: dict[str, float] = {}
    for sym, _ in recommendations:
        try:
            current_prices[sym] = data_client.get_current_price(sym)
        except Exception:
            pass

    # Obtener precios de posiciones abiertas
    for symbol in pred_book.open_symbols:
        if symbol not in current_prices:
            try:
                current_prices[symbol] = data_client.get_current_price(symbol)
            except Exception:
                pass

    pred_results: list[ExecutionResult] = []

    if buys_blocked:
        actions: list[TradeAction] = []
        logger.info(
            "Compras nuevas omitidas — mercado adverso. "
            "Recomendaciones ignoradas: %d",
            len(recommendations),
        )
    else:
        actions = portfolio_mgr.decide_actions(
            current_portfolio=pred_portfolio,
            total_value_usdt=prediction_budget,
            recommendations=recommendations,
            current_prices=current_prices,
            strategy_quote_available=pred_quote_available,
        )

        # Asignar limit_pct a las compras de prediccion
        buy_discount = config.model.buy_limit_discount
        if buy_discount > 0:
            for a in actions:
                if a.action == "BUY":
                    a.limit_pct = buy_discount

        logger.info("Acciones predicción: %d", len(actions))
        for a in actions:
            extra = f" (limit -{a.limit_pct:.0%})" if a.limit_pct > 0 else ""
            logger.info("  %s %s%s | %s", a.action, a.symbol, extra, a.reason)

        # Ejecutar órdenes de predicción
        pred_results = executor.execute(actions)
        successful = sum(1 for r in pred_results if r.success)
        logger.info(
            "Órdenes predicción: %d/%d exitosas",
            successful,
            len(pred_results),
        )

        # Registrar compras en el libro de prediction
        for result in pred_results:
            if not result.success:
                continue
            action = result.action
            if action.action == "BUY":
                if result.order_status == "FILLED" and result.executed_qty > 0:
                    pred_book.record_buy(
                        symbol=action.symbol,
                        price=result.executed_price,
                        quantity=result.executed_qty,
                        usdt_spent=action.quote_qty,
                    )
                elif result.order_id > 0 and result.order_status != "FILLED":
                    pred_book.record_pending_order(
                        symbol=action.symbol,
                        order_id=result.order_id,
                        quote_qty=action.quote_qty,
                        limit_price=result.executed_price or 0.0,
                    )

    # ------------------------------------------------------------------
    # 7. ESTRATEGIA 2: DCA Inteligente (usa budget "dca")
    # ------------------------------------------------------------------
    dca_summary: dict = {}
    dca_actions_today: list[DCAAction] = []

    if config.dca.enabled:
        logger.info("=" * 40)
        logger.info("ESTRATEGIA DCA INTELIGENTE")
        logger.info("=" * 40)

        dca_budget = allocator.get_budget("dca")
        dca_strategy = DCAStrategy(
            budget_usdt=dca_budget,
            dip_threshold=config.dca.dip_threshold,
            take_profit_pct=config.dca.take_profit_pct,
            stop_loss_pct=config.dca.stop_loss_pct,
            assets=list(config.dca.assets),
            asset_policies=DEFAULT_ASSET_POLICIES,
        )

        # Reconciliar DCA con Binance
        if not is_paper and trading_client is not None and dca_strategy.positions:
            try:
                dca_portfolio = trading_client.get_portfolio()
                dca_strategy.reconcile(dca_portfolio, quote)
            except Exception as exc:
                logger.warning("Error en reconciliación DCA: %s", exc)

        # Obtener cambios de precio 24h y precios actuales para activos DCA
        price_changes_24h: dict[str, float] = {}
        dca_prices: dict[str, float] = {}
        try:
            dca_prices = _get_dca_prices(data_client, list(config.dca.assets))
            price_changes_24h = _get_24h_changes(data_client, list(config.dca.assets))
        except Exception as exc:
            logger.warning("Error obteniendo datos DCA: %s", exc)

        # Evaluar y generar acciones DCA
        dca_actions = dca_strategy.evaluate(price_changes_24h, dca_prices)
        dca_actions_today = list(dca_actions)
        logger.info("Acciones DCA: %d", len(dca_actions))

        # Ejecutar acciones DCA
        for dca_action in dca_actions:
            if is_paper:
                logger.info(
                    "[PAPER DCA] %s %s | $%.2f | reason=%s",
                    dca_action.action,
                    dca_action.symbol,
                    dca_action.quote_qty,
                    dca_action.reason,
                )
                if dca_action.action == "BUY":
                    price = dca_prices.get(dca_action.symbol, 0)
                    if price > 0:
                        qty = dca_action.quote_qty / price
                        dca_strategy.record_buy(
                            dca_action.symbol,
                            price,
                            qty,
                            dca_action.quote_qty,
                        )
                elif dca_action.action == "SELL":
                    dca_strategy.record_sell(dca_action.symbol)
            else:
                _execute_dca_live(
                    dca_action,
                    executor,
                    dca_strategy,
                    allocator,
                    dca_prices,
                )

        # Resumen DCA para el email
        dca_summary = dca_strategy.get_summary(dca_prices)
        logger.info("DCA resumen: %s", dca_summary)
    else:
        logger.info("DCA deshabilitado en configuracion.")

    # ------------------------------------------------------------------
    # 8. ESTRATEGIA 3: Momentum (usa budget "momentum")
    # ------------------------------------------------------------------
    momentum_summary: dict = {}

    if config.momentum.enabled:
        logger.info("=" * 40)
        logger.info("ESTRATEGIA MOMENTUM")
        logger.info("=" * 40)

        momentum_budget = allocator.get_budget("momentum")
        momentum_strategy = MomentumStrategy(
            budget_usdt=momentum_budget,
            momentum_threshold=config.momentum.momentum_threshold,
            take_profit_pct=config.momentum.take_profit_pct,
            stop_loss_pct=config.momentum.stop_loss_pct,
            trend_days=config.momentum.trend_days,
            assets=list(config.momentum.assets),
            asset_policies=DEFAULT_MOMENTUM_POLICIES,
        )

        # Reconciliar Momentum con Binance
        if not is_paper and trading_client is not None and momentum_strategy.positions:
            try:
                mom_portfolio = trading_client.get_portfolio()
                momentum_strategy.reconcile(mom_portfolio, quote)
            except Exception as exc:
                logger.warning("Error en reconciliación Momentum: %s", exc)

        # Obtener datos para momentum
        momentum_prices: dict[str, float] = {}
        momentum_changes_24h: dict[str, float] = {}
        daily_closes: dict[str, list[float]] = {}
        try:
            momentum_prices = _get_dca_prices(
                data_client,
                list(config.momentum.assets),
            )
            momentum_changes_24h = _get_24h_changes(
                data_client,
                list(config.momentum.assets),
            )
            daily_closes = _get_daily_closes(
                data_client,
                list(config.momentum.assets),
                days=14,
            )
        except Exception as exc:
            logger.warning("Error obteniendo datos Momentum: %s", exc)

        # Evaluar y generar acciones Momentum
        momentum_actions = momentum_strategy.evaluate(
            momentum_changes_24h,
            momentum_prices,
            daily_closes,
        )
        logger.info("Acciones Momentum: %d", len(momentum_actions))

        # Ejecutar acciones Momentum
        for m_action in momentum_actions:
            if is_paper:
                logger.info(
                    "[PAPER MOMENTUM] %s %s | $%.2f | reason=%s",
                    m_action.action,
                    m_action.symbol,
                    m_action.quote_qty,
                    m_action.reason,
                )
                if m_action.action == "BUY":
                    price = momentum_prices.get(m_action.symbol, 0)
                    if price > 0:
                        qty = m_action.quote_qty / price
                        momentum_strategy.record_buy(
                            m_action.symbol,
                            price,
                            qty,
                            m_action.quote_qty,
                        )
                elif m_action.action == "SELL":
                    momentum_strategy.record_sell(m_action.symbol)
            else:
                _execute_momentum_live(
                    m_action,
                    executor,
                    momentum_strategy,
                    allocator,
                    momentum_prices,
                )

        # Resumen Momentum para el email
        momentum_summary = momentum_strategy.get_summary(momentum_prices)
        logger.info("Momentum resumen: %s", momentum_summary)
    else:
        logger.info("Momentum deshabilitado en configuracion.")

    # ------------------------------------------------------------------
    # 9. Estado final de la cartera
    # ------------------------------------------------------------------
    if is_paper:
        portfolio_after = portfolio_before.copy()
        total_value_after = total_value_before
    else:
        assert trading_client is not None
        time.sleep(2)  # Esperar a que las órdenes se reflejen
        portfolio_after = trading_client.get_portfolio()
        total_value_after = trading_client.get_portfolio_value_usdt()

    logger.info("Balance final: $%.2f", total_value_after)

    # Rebalancear con el valor final real para que los budgets
    # reflejen el capital actual de cara a la próxima ejecución.
    if not is_paper:
        allocator.rebalance(total_value_after)
        logger.info(
            "Allocator rebalanceado con balance final $%.2f",
            total_value_after,
        )

    # ------------------------------------------------------------------
    # 9. Enviar reporte
    # ------------------------------------------------------------------
    model_info = _get_model_info(config)

    email_sent = send_daily_report(
        config=config.email,
        portfolio_before=portfolio_before,
        portfolio_after=portfolio_after,
        total_value_before=total_value_before,
        total_value_after=total_value_after,
        results=expired_results + pred_results,
        predictions=predictions,
        is_paper=is_paper,
        dca_summary=dca_summary,
        allocation_budgets=allocator.get_all_budgets(),
        dca_actions=dca_actions_today,
        momentum_summary=momentum_summary,
        model_info=model_info,
        market_regime=market_regime,
    )
    if email_sent:
        logger.info("Reporte enviado por email")
    else:
        logger.warning("No se pudo enviar el reporte por email")

    logger.info("=" * 60)
    logger.info("FIN — %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)


def _get_dca_prices(
    data_client: BinanceDataClient,
    symbols: list[str],
) -> dict[str, float]:
    """Obtiene precios actuales para los activos DCA."""
    prices: dict[str, float] = {}
    for symbol in symbols:
        try:
            prices[symbol] = data_client.get_current_price(symbol)
        except Exception as exc:
            logger.warning("Error obteniendo precio de %s: %s", symbol, exc)
    return prices


def _get_24h_changes(
    data_client: BinanceDataClient,
    symbols: list[str],
) -> dict[str, float]:
    """Calcula el cambio porcentual de precio en las ultimas 24h."""
    changes: dict[str, float] = {}
    for symbol in symbols:
        try:
            klines = data_client.get_klines(symbol, interval="1h", lookback_hours=25)
            if len(klines) >= 2:
                price_24h_ago = float(klines.iloc[0]["close"])
                price_now = float(klines.iloc[-1]["close"])
                if price_24h_ago > 0:
                    changes[symbol] = (price_now - price_24h_ago) / price_24h_ago
                    logger.info(
                        "DCA %s cambio 24h: %.2f%% ($%.2f -> $%.2f)",
                        symbol,
                        changes[symbol] * 100,
                        price_24h_ago,
                        price_now,
                    )
        except Exception as exc:
            logger.warning("Error calculando cambio 24h de %s: %s", symbol, exc)
    return changes


def _get_daily_closes(
    data_client: BinanceDataClient,
    symbols: list[str],
    days: int = 14,
) -> dict[str, list[float]]:
    """Obtiene precios de cierre diarios para los últimos N días."""
    closes: dict[str, list[float]] = {}
    for symbol in symbols:
        try:
            klines = data_client.get_klines(
                symbol,
                interval="1d",
                lookback_hours=days * 24,
            )
            if len(klines) >= 2:
                closes[symbol] = [float(row["close"]) for _, row in klines.iterrows()]
                logger.info(
                    "Momentum %s: %d cierres diarios obtenidos",
                    symbol,
                    len(closes[symbol]),
                )
        except Exception as exc:
            logger.warning("Error obteniendo cierres diarios de %s: %s", symbol, exc)
    return closes


def _execute_momentum_live(
    m_action: MomentumAction,
    executor: OrderExecutor,
    momentum_strategy: MomentumStrategy,
    allocator: PortfolioAllocator,
    momentum_prices: dict[str, float],
) -> None:
    """Ejecuta una accion Momentum en modo live."""
    trade = TradeAction(
        action=m_action.action,
        symbol=m_action.symbol,
        quote_qty=m_action.quote_qty,
        base_qty=m_action.base_qty,
        reason=m_action.reason,
        probability=0.0,
    )
    results = executor.execute([trade])

    if results and results[0].success:
        result = results[0]
        if m_action.action == "BUY":
            momentum_strategy.record_buy(
                symbol=m_action.symbol,
                price=result.executed_price,
                quantity=result.executed_qty,
                usdt_spent=m_action.quote_qty,
            )
        elif m_action.action == "SELL":
            freed = momentum_strategy.record_sell(m_action.symbol)
            if result.executed_qty > 0:
                sell_value = result.executed_qty * result.executed_price
                profit = sell_value - freed
                allocator.add_profit("momentum", profit)
    else:
        error = results[0].error if results else "Sin resultado"
        logger.error(
            "Momentum %s %s fallo: %s",
            m_action.action,
            m_action.symbol,
            error,
        )


def _execute_dca_live(
    dca_action: DCAAction,
    executor: OrderExecutor,
    dca_strategy: DCAStrategy,
    allocator: PortfolioAllocator,
    dca_prices: dict[str, float],
) -> None:
    """Ejecuta una accion DCA en modo live."""
    # Convertir DCAAction en TradeAction para reutilizar el executor
    trade = TradeAction(
        action=dca_action.action,
        symbol=dca_action.symbol,
        quote_qty=dca_action.quote_qty,
        base_qty=dca_action.base_qty,
        reason=dca_action.reason,
        probability=0.0,
    )
    results = executor.execute([trade])

    if results and results[0].success:
        result = results[0]
        if dca_action.action == "BUY":
            dca_strategy.record_buy(
                symbol=dca_action.symbol,
                price=result.executed_price,
                quantity=result.executed_qty,
                usdt_spent=dca_action.quote_qty,
            )
        elif dca_action.action == "SELL":
            freed = dca_strategy.record_sell(dca_action.symbol)
            price = dca_prices.get(dca_action.symbol, 0)
            if price > 0 and result.executed_qty > 0:
                sell_value = result.executed_qty * result.executed_price
                profit = sell_value - freed
                allocator.add_profit("dca", profit)
    else:
        error = results[0].error if results else "Sin resultado"
        logger.error(
            "DCA %s %s fallo: %s",
            dca_action.action,
            dca_action.symbol,
            error,
        )


def run_train_only(config: AppConfig | None = None) -> None:
    """Solo entrena el modelo (no opera). Útil para primer setup."""
    config = config or load_config()
    logger.info("Modo ENTRENAMIENTO — solo entrena el modelo, no opera")
    logger.info(
        "BINANCE_API_KEY configurada: %s (len=%d)",
        bool(config.binance.api_key),
        len(config.binance.api_key),
    )

    data_client = BinanceDataClient(config.binance)
    if not data_client.is_connected:
        logger.error(
            "No se pudo conectar a Binance tras probar multiples endpoints. "
            "Revisa los logs anteriores para mas detalle."
        )
        sys.exit(1)

    predictor = PricePredictor(config.model)

    logger.info(
        "Obteniendo top %d monedas...",
        config.model.top_n_coins,
    )
    top_symbols = data_client.get_top_coins_by_volume(
        config.model.top_n_coins,
        quote=config.portfolio.quote_asset,
    )

    logger.info(
        "Descargando datos de entrenamiento (%d días)...",
        config.model.training_days,
    )
    training_klines = data_client.get_klines_batch(
        top_symbols,
        interval=config.model.candle_interval,
        lookback_hours=config.model.training_days * 24,
    )

    metrics = predictor.train(training_klines)
    predictor.save()
    logger.info("Modelo entrenado y guardado — métricas: %s", metrics)

    # Mostrar importancia de features
    importance = predictor.feature_importance()
    logger.info(
        "Top 10 features:\n%s",
        importance.head(10).to_string(),
    )


def _get_model_info(config: AppConfig) -> dict[str, object]:
    """Obtiene informacion sobre el modelo para el reporte."""
    info: dict[str, object] = {
        "retrain_interval_days": config.model.retrain_interval_days,
    }
    if not MODEL_FILE.exists():
        info["trained_at"] = "No existe"
        info["age_days"] = -1
        info["status"] = "missing"
        return info

    mtime = datetime.fromtimestamp(
        os.path.getmtime(MODEL_FILE), tz=timezone.utc,
    )
    age_days = (datetime.now(timezone.utc) - mtime).days
    info["trained_at"] = mtime.strftime("%Y-%m-%d %H:%M UTC")
    info["age_days"] = age_days
    info["status"] = (
        "ok" if age_days < config.model.retrain_interval_days else "stale"
    )
    return info


def _should_retrain(config: AppConfig) -> bool:
    """Decide si hay que re-entrenar el modelo."""
    if not MODEL_FILE.exists():
        return True

    mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_FILE), tz=timezone.utc)
    age_days = (datetime.now(timezone.utc) - mtime).days
    return age_days >= config.model.retrain_interval_days


def main() -> None:
    """Punto de entrada CLI."""
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        run_train_only()
    else:
        run_daily()


if __name__ == "__main__":
    main()
