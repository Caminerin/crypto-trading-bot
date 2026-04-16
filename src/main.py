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

import sys
import time
from datetime import datetime, timezone

import pandas as pd

from src.allocation.allocator import PortfolioAllocator
from src.config import (
    DEFAULT_ASSET_POLICIES,
    DEFAULT_MOMENTUM_POLICIES,
    MODELS_DIR,
    AppConfig,
    load_config,
)
from src.data.binance_client import BinanceDataClient, BinanceTradingClient
from src.execution.executor import OrderExecutor, load_restricted_symbols
from src.model.predictor import PricePredictor
from src.notifications.email_report import send_daily_report
from src.portfolio.manager import PortfolioManager, TradeAction
from src.strategies.dca import DCAAction, DCAStrategy
from src.strategies.momentum import MomentumAction, MomentumStrategy
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

    executor = OrderExecutor(config, trading_client)
    portfolio_mgr = PortfolioManager(config.portfolio, config.risk)
    predictor = PricePredictor(config.model)

    # ------------------------------------------------------------------
    # 2. Obtener estado actual de la cartera
    # ------------------------------------------------------------------
    if is_paper:
        portfolio_before: dict[str, float] = {"USDT": 1000.0}
        total_value_before = 1000.0
        logger.info("[PAPER] Cartera simulada: 1000 USDT")
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
            portfolio_before = {"USDT": 1000.0}
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

    # Decidir acciones de prediccion (limitadas al budget de prediccion)
    prediction_budget = allocator.get_budget("prediction")
    current_prices: dict[str, float] = {}
    for sym, _ in recommendations:
        try:
            current_prices[sym] = data_client.get_current_price(sym)
        except Exception:
            pass

    # Obtener precios de posiciones actuales (para filtrar dust)
    if not is_paper:
        stablecoins = {"USDT", "USDC", "BUSD", "FDUSD", "DAI", "TUSD"}
        for asset in portfolio_before:
            if asset not in stablecoins:
                symbol = f"{asset}USDT"
                if symbol not in current_prices:
                    try:
                        current_prices[symbol] = data_client.get_current_price(symbol)
                    except Exception:
                        pass

    # Filtrar activos con restricción de cuenta (blacklist persistente)
    restricted = load_restricted_symbols()
    portfolio_for_actions = portfolio_before
    if restricted and not is_paper:
        portfolio_for_actions = {
            asset: qty
            for asset, qty in portfolio_before.items()
            if f"{asset}USDT" not in restricted
            and asset not in ("USDT", "USDC", "BUSD", "FDUSD")
            or asset in ("USDT", "USDC", "BUSD", "FDUSD")
        }
        skipped = set(portfolio_before) - set(portfolio_for_actions)
        if skipped:
            logger.info("Activos en blacklist (omitidos): %s", skipped)

    actions = portfolio_mgr.decide_actions(
        current_portfolio=portfolio_for_actions,
        total_value_usdt=prediction_budget,
        recommendations=recommendations,
        current_prices=current_prices,
    )
    logger.info("Acciones prediccion: %d", len(actions))
    for a in actions:
        logger.info("  %s %s | %s", a.action, a.symbol, a.reason)

    # Ejecutar ordenes de prediccion
    pred_results = executor.execute(actions)
    successful = sum(1 for r in pred_results if r.success)
    logger.info(
        "Ordenes prediccion: %d/%d exitosas",
        successful,
        len(pred_results),
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

    # Actualizar budgets del allocator con el valor real
    if not is_paper and total_value_after != total_value_before:
        pnl = total_value_after - total_value_before
        active_pct = (
            config.allocation.prediction_pct
            + config.allocation.dca_pct
            + config.allocation.momentum_pct
        )
        if active_pct > 0:
            allocator.add_profit(
                "prediction",
                pnl * config.allocation.prediction_pct / active_pct,
            )
            allocator.add_profit(
                "dca",
                pnl * config.allocation.dca_pct / active_pct,
            )
            allocator.add_profit(
                "momentum",
                pnl * config.allocation.momentum_pct / active_pct,
            )

    # ------------------------------------------------------------------
    # 9. Enviar reporte
    # ------------------------------------------------------------------
    email_sent = send_daily_report(
        config=config.email,
        portfolio_before=portfolio_before,
        portfolio_after=portfolio_after,
        total_value_before=total_value_before,
        total_value_after=total_value_after,
        results=pred_results,
        predictions=predictions,
        is_paper=is_paper,
        dca_summary=dca_summary,
        allocation_budgets=allocator.get_all_budgets(),
        dca_actions=dca_actions_today,
        momentum_summary=momentum_summary,
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


def _should_retrain(config: AppConfig) -> bool:
    """Decide si hay que re-entrenar el modelo."""
    if not MODEL_FILE.exists():
        return True

    import os

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
