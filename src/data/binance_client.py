"""
Módulo de conexión a la API de Binance.

Responsabilidades:
- Obtener las top N monedas por volumen (pares USDT).
- Descargar velas históricas (klines).
- Consultar balances de la cartera.
- Ejecutar órdenes de mercado (market orders).
- Colocar órdenes OCO (stop-loss + take-profit).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests as _requests
from binance.client import Client
from binance.exceptions import BinanceAPIException

from src.config import BinanceConfig, PortfolioConfig, RiskConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Endpoints alternativos de Binance (por si el principal falla).
_BINANCE_ENDPOINTS = [
    "https://api.binance.com/api",
    "https://api1.binance.com/api",
    "https://api2.binance.com/api",
    "https://api3.binance.com/api",
]


def _find_working_endpoint(timeout: int = 10) -> str | None:
    """Prueba endpoints de Binance y devuelve el primero que responda."""
    for endpoint in _BINANCE_ENDPOINTS:
        try:
            resp = _requests.get(f"{endpoint}/v3/ping", timeout=timeout)
            if resp.status_code == 200:
                logger.info("Endpoint accesible: %s", endpoint)
                return endpoint
        except Exception as exc:
            logger.debug("Endpoint %s no accesible: %s", endpoint, exc)
    return None


class BinanceDataClient:
    """Lectura de datos de mercado desde Binance."""

    def __init__(self, config: BinanceConfig) -> None:
        self._client: Client | None = None

        # 1. Comprobar conectividad con Binance antes de crear el Client.
        working_endpoint = _find_working_endpoint()
        if working_endpoint is None:
            logger.warning(
                "No se pudo conectar a ningun endpoint de Binance. "
                "Posible problema de red."
            )
            return

        # 2. Crear el Client. El constructor de python-binance hace ping
        #    al endpoint por defecto; si falla, capturamos y reintentamos
        #    sin credenciales (API publica — suficiente para datos de mercado).
        api_key = config.api_key
        api_secret = config.api_secret

        for label, key, secret in [
            ("auth", api_key, api_secret),
            ("public", "", ""),
        ]:
            try:
                client = Client(key, secret, {"timeout": 20})
                # Asegurar que usamos el endpoint que ya sabemos que funciona.
                client.API_URL = working_endpoint
                client.ping()
                self._client = client
                logger.info(
                    "Conectado a Binance (%s) via %s",
                    label,
                    working_endpoint,
                )
                return
            except Exception as exc:
                logger.warning(
                    "Client(%s) fallo: %s: %s",
                    label,
                    type(exc).__name__,
                    exc,
                )

        logger.error("No se pudo crear un Client de Binance funcional.")

    @property
    def is_connected(self) -> bool:
        """Indica si la conexión con Binance está activa."""
        return self._client is not None

    def _ensure_connected(self) -> Client:
        """Devuelve el cliente o lanza ConnectionError."""
        if self._client is None:
            raise ConnectionError("No hay conexión con Binance.")
        return self._client

    # ------------------------------------------------------------------
    # Datos de mercado
    # ------------------------------------------------------------------

    def get_top_coins_by_volume(self, top_n: int, quote: str = "USDT") -> list[str]:
        """Devuelve los símbolos de las *top_n* monedas con mayor volumen 24 h en pares *quote*."""
        if not self.is_connected:
            raise ConnectionError("No hay conexión con Binance.")
        tickers = self._client.get_ticker()
        usdt_tickers = [
            t
            for t in tickers
            if t["symbol"].endswith(quote)
            and not t["symbol"].startswith(("USDC", "BUSD", "TUSD", "FDUSD", "DAI"))
        ]
        sorted_tickers = sorted(
            usdt_tickers, key=lambda t: float(t["quoteVolume"]), reverse=True
        )
        return [t["symbol"] for t in sorted_tickers[:top_n]]

    def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        lookback_hours: int = 72,
    ) -> pd.DataFrame:
        """Descarga velas históricas y las devuelve como DataFrame."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        start_ms = int(start_time.timestamp() * 1000)

        raw = self._client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_ms,
            limit=1000,
        )

        df = pd.DataFrame(
            raw,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df = df.set_index("open_time")
        return df

    def get_klines_batch(
        self,
        symbols: list[str],
        interval: str = "1h",
        lookback_hours: int = 72,
    ) -> dict[str, pd.DataFrame]:
        """Descarga velas para múltiples símbolos."""
        result: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                result[symbol] = self.get_klines(symbol, interval, lookback_hours)
            except BinanceAPIException as exc:
                logger.warning("No se pudieron obtener klines de %s: %s", symbol, exc)
        return result

    def get_current_price(self, symbol: str) -> float:
        """Precio actual de un símbolo."""
        if not self.is_connected:
            raise ConnectionError("No hay conexión con Binance.")
        tick = self._client.get_symbol_ticker(symbol=symbol)
        return float(tick["price"])


class BinanceTradingClient:
    """Operaciones de trading: lectura de cartera y ejecución de órdenes."""

    def __init__(
        self,
        config: BinanceConfig,
        portfolio_cfg: PortfolioConfig,
        risk_cfg: RiskConfig,
    ) -> None:
        self._client = Client(config.api_key, config.api_secret)
        self._portfolio_cfg = portfolio_cfg
        self._risk_cfg = risk_cfg

    # ------------------------------------------------------------------
    # Cartera
    # ------------------------------------------------------------------

    def get_portfolio(self) -> dict[str, float]:
        """Devuelve {asset: free_balance} para activos con saldo > 0."""
        account = self._client.get_account()
        balances: dict[str, float] = {}
        for b in account["balances"]:
            free = float(b["free"])
            if free > 0:
                balances[b["asset"]] = free
        return balances

    def get_portfolio_value_usdt(self) -> float:
        """Valor total de la cartera en USDT."""
        portfolio = self.get_portfolio()
        total = 0.0
        for asset, qty in portfolio.items():
            if asset in ("USDT", "USDC", "BUSD", "FDUSD"):
                total += qty
            else:
                try:
                    price = float(
                        self._client.get_symbol_ticker(symbol=f"{asset}USDT")["price"]
                    )
                    total += qty * price
                except BinanceAPIException:
                    logger.warning("No se pudo valorar %s en USDT", asset)
        return total

    # ------------------------------------------------------------------
    # Órdenes
    # ------------------------------------------------------------------

    def place_market_buy(self, symbol: str, quote_qty: float) -> dict[str, Any]:
        """Compra a mercado gastando *quote_qty* USDT."""
        logger.info("MARKET BUY %s por %.2f USDT", symbol, quote_qty)
        order = self._client.order_market_buy(
            symbol=symbol,
            quoteOrderQty=f"{quote_qty:.2f}",
        )
        return order

    def place_market_sell(self, symbol: str, quantity: float) -> dict[str, Any]:
        """Vende a mercado la cantidad indicada."""
        logger.info("MARKET SELL %s qty=%.8f", symbol, quantity)
        order = self._client.order_market_sell(
            symbol=symbol,
            quantity=f"{quantity:.8f}",
        )
        return order

    def place_oco_sell(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
    ) -> dict[str, Any] | None:
        """Coloca una orden OCO de venta (stop-loss + take-profit)."""
        take_profit_price = round(entry_price * (1 + self._risk_cfg.take_profit_pct), 8)
        stop_price = round(entry_price * (1 - self._risk_cfg.stop_loss_pct), 8)
        stop_limit_price = round(stop_price * 0.995, 8)

        logger.info(
            "OCO SELL %s qty=%.8f | TP=%.8f | SL=%.8f",
            symbol,
            quantity,
            take_profit_price,
            stop_price,
        )
        try:
            order = self._client.create_oco_order(
                symbol=symbol,
                side="SELL",
                quantity=f"{quantity:.8f}",
                price=f"{take_profit_price:.8f}",
                stopPrice=f"{stop_price:.8f}",
                stopLimitPrice=f"{stop_limit_price:.8f}",
                stopLimitTimeInForce="GTC",
            )
            return order
        except BinanceAPIException as exc:
            logger.error("Error al colocar OCO para %s: %s", symbol, exc)
            return None

    def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Devuelve la info de un símbolo (filtros de precio, cantidad, etc.)."""
        return self._client.get_symbol_info(symbol)

    def cancel_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Cancela todas las órdenes abiertas de un símbolo."""
        try:
            orders = self._client.get_open_orders(symbol=symbol)
            cancelled = []
            for order in orders:
                result = self._client.cancel_order(
                    symbol=symbol, orderId=order["orderId"]
                )
                cancelled.append(result)
            return cancelled
        except BinanceAPIException as exc:
            logger.warning("Error cancelando órdenes de %s: %s", symbol, exc)
            return []
