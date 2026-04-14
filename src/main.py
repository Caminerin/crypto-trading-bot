"""
Orquestador principal del bot de trading.

Este es el punto de entrada.  Coordina todos los módulos:
1. Carga configuración.
2. Conecta con Binance.
3. Obtiene datos de las top 50 monedas por liquidez.
4. Ejecuta (o entrena) el modelo predictivo.
5. Decide qué comprar/vender.
6. Ejecuta las órdenes.
7. Envía el reporte por email.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone

import pandas as pd

from src.config import MODELS_DIR, AppConfig, load_config
from src.data.binance_client import BinanceDataClient, BinanceTradingClient
from src.execution.executor import OrderExecutor
from src.model.predictor import PricePredictor
from src.notifications.email_report import send_daily_report
from src.portfolio.manager import PortfolioManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_FILE = MODELS_DIR / "predictor.joblib"


def run_daily(config: AppConfig | None = None) -> None:
    """Ejecución diaria completa del bot."""
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
            trading_client = BinanceTradingClient(
                config.binance, config.portfolio, config.risk
            )
        except Exception as exc:
            logger.warning(
                "No se pudo crear BinanceTradingClient: %s. "
                "Cayendo a modo paper.", exc,
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
                portfolio_before, total_value_before,
            )
        except Exception as exc:
            logger.warning(
                "Error conectando con Binance para leer cartera: %s. "
                "Cayendo a modo paper.", exc,
            )
            is_paper = True
            trading_client = None
            executor = OrderExecutor(config, None)
            portfolio_before = {"USDT": 1000.0}
            total_value_before = 1000.0

    # ------------------------------------------------------------------
    # 3. Obtener top monedas y descargar datos
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
        logger.info(
            "[PAPER] Continuando sin datos reales (primera ejecución)"
        )

    # ------------------------------------------------------------------
    # 4. Entrenar o cargar modelo
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
    # 5. Predecir
    # ------------------------------------------------------------------
    predictions: dict[str, float] = {}
    recommendations: list[tuple[str, float]] = []

    if model_ready and klines:
        logger.info("Ejecutando predicciones...")
        predictions = predictor.predict(klines)
        recommendations = predictor.get_recommendations(predictions)
        logger.info(
            "Predicciones: %d monedas analizadas "
            "| %d recomendadas (umbral=%.0f%%)",
            len(predictions),
            len(recommendations),
            config.model.confidence_threshold * 100,
        )
        for sym, prob in recommendations[:10]:
            logger.info("  %s — prob=%.1f%%", sym, prob * 100)
    else:
        logger.info(
            "Sin predicciones disponibles "
            "(modelo=%s, datos=%d monedas)",
            "listo" if model_ready else "no disponible",
            len(klines),
        )

    # ------------------------------------------------------------------
    # 6. Decidir acciones
    # ------------------------------------------------------------------
    current_prices: dict[str, float] = {}
    for sym, _ in recommendations:
        try:
            current_prices[sym] = data_client.get_current_price(sym)
        except Exception:
            pass

    actions = portfolio_mgr.decide_actions(
        current_portfolio=portfolio_before,
        total_value_usdt=total_value_before,
        recommendations=recommendations,
        current_prices=current_prices,
    )
    logger.info("Acciones decididas: %d", len(actions))
    for a in actions:
        logger.info("  %s %s | %s", a.action, a.symbol, a.reason)

    # ------------------------------------------------------------------
    # 7. Ejecutar órdenes
    # ------------------------------------------------------------------
    results = executor.execute(actions)
    successful = sum(1 for r in results if r.success)
    logger.info(
        "Órdenes ejecutadas: %d/%d exitosas", successful, len(results),
    )

    # ------------------------------------------------------------------
    # 8. Estado final de la cartera
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

    # ------------------------------------------------------------------
    # 9. Enviar reporte
    # ------------------------------------------------------------------
    email_sent = send_daily_report(
        config=config.email,
        portfolio_before=portfolio_before,
        portfolio_after=portfolio_after,
        total_value_before=total_value_before,
        total_value_after=total_value_after,
        results=results,
        predictions=predictions,
        is_paper=is_paper,
    )
    if email_sent:
        logger.info("Reporte enviado por email")
    else:
        logger.warning("No se pudo enviar el reporte por email")

    logger.info("=" * 60)
    logger.info("FIN — %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)


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
        "Obteniendo top %d monedas...", config.model.top_n_coins,
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
        "Top 10 features:\n%s", importance.head(10).to_string(),
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
