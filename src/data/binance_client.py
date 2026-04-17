"""
Módulo de conexión a la API de Binance.

Responsabilidades:
- Obtener las top N monedas por volumen (pares USDT).
- Descargar velas históricas (klines).
- Consultar balances de la cartera.
- Ejecutar órdenes de mercado (market orders).
- Colocar órdenes OCO (stop-loss + take-profit).

Estrategia de conexión (para datos de mercado):
1. Intenta usar python-binance Client (api.binance.com, etc.)
2. Si falla (geo-bloqueo 451 desde GitHub Actions / US), usa
   peticiones HTTP directas a data-api.binance.vision (API pública
   sin geo-restricción).
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests as _requests
from binance.client import Client
from binance.exceptions import BinanceAPIException

from src.config import BinanceConfig, PortfolioConfig, RiskConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Endpoints para python-binance Client (requiere que no haya geo-bloqueo).
_CLIENT_ENDPOINTS = [
    "https://api.binance.com/api",
    "https://api1.binance.com/api",
    "https://api2.binance.com/api",
    "https://api3.binance.com/api",
]

# Endpoints para HTTP directo (fallback cuando Client falla por geo-bloqueo).
_HTTP_BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
]

_KLINE_COLUMNS = [
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
]

_NUMERIC_COLS = ["open", "high", "low", "close", "volume", "quote_volume"]


def _find_working_client_endpoint(timeout: int = 10) -> str | None:
    """Prueba endpoints de python-binance y devuelve el primero que responda."""
    for endpoint in _CLIENT_ENDPOINTS:
        try:
            resp = _requests.get(f"{endpoint}/v3/ping", timeout=timeout)
            if resp.status_code == 200:
                logger.info("Client endpoint accesible: %s", endpoint)
                return endpoint
        except Exception as exc:
            logger.debug("Client endpoint %s no accesible: %s", endpoint, exc)
    return None


def _find_working_http_base(timeout: int = 10) -> str | None:
    """Prueba base URLs para HTTP directo y devuelve el primero que responda."""
    for base_url in _HTTP_BASE_URLS:
        try:
            resp = _requests.get(
                f"{base_url}/api/v3/ping",
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if resp.status_code == 200:
                logger.info("HTTP base URL accesible: %s", base_url)
                return base_url
        except Exception as exc:
            logger.debug("HTTP base %s no accesible: %s", base_url, exc)
    return None


def _klines_to_dataframe(raw: list[list[Any]]) -> pd.DataFrame:
    """Convierte la respuesta cruda de klines en un DataFrame limpio."""
    df = pd.DataFrame(raw, columns=_KLINE_COLUMNS)
    df[_NUMERIC_COLS] = df[_NUMERIC_COLS].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    return df


class BinanceDataClient:
    """Lectura de datos de mercado desde Binance.

    Usa python-binance Client si es posible; si no, hace HTTP directo
    a la API pública de Binance (data-api.binance.vision).
    """

    def __init__(self, config: BinanceConfig) -> None:
        self._client: Client | None = None
        self._http_base: str | None = None

        # 1. Intentar crear python-binance Client (mejor experiencia).
        client_endpoint = _find_working_client_endpoint()
        if client_endpoint is not None:
            for label, key, secret in [
                ("auth", config.api_key, config.api_secret),
                ("public", "", ""),
            ]:
                try:
                    client = Client(key, secret, {"timeout": 20})
                    client.API_URL = client_endpoint
                    client.ping()
                    self._client = client
                    logger.info(
                        "Conectado a Binance via Client (%s) en %s",
                        label,
                        client_endpoint,
                    )
                    return
                except Exception as exc:
                    logger.warning(
                        "Client(%s) fallo: %s: %s",
                        label,
                        type(exc).__name__,
                        exc,
                    )

        # 2. Fallback: HTTP directo (funciona desde GitHub Actions / US).
        self._http_base = _find_working_http_base()
        if self._http_base is not None:
            logger.info(
                "Conectado a Binance via HTTP directo en %s",
                self._http_base,
            )
            return

        logger.error("No se pudo conectar a Binance por ningún método (Client ni HTTP directo).")

    @property
    def is_connected(self) -> bool:
        """Indica si la conexión con Binance está activa."""
        return self._client is not None or self._http_base is not None

    @property
    def _uses_http(self) -> bool:
        """True si estamos usando HTTP directo en vez de Client."""
        return self._client is None and self._http_base is not None

    def _http_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET a la API REST de Binance vía HTTP directo."""
        url = f"{self._http_base}/api/v3/{path}"
        resp = _requests.get(
            url,
            params=params,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Datos de mercado
    # ------------------------------------------------------------------

    def get_top_coins_by_volume(self, top_n: int, quote: str = "USDT") -> list[str]:
        """Devuelve los símbolos de las *top_n* monedas con mayor volumen 24 h."""
        if not self.is_connected:
            raise ConnectionError("No hay conexión con Binance.")

        if self._uses_http:
            tickers = self._http_get("ticker/24hr")
        else:
            tickers = self._client.get_ticker()

        usdt_tickers = [
            t
            for t in tickers
            if t["symbol"].endswith(quote)
            and not t["symbol"].startswith(("USDC", "BUSD", "TUSD", "FDUSD", "DAI"))
        ]
        sorted_tickers = sorted(usdt_tickers, key=lambda t: float(t["quoteVolume"]), reverse=True)
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

        # Paginar para obtener todas las velas solicitadas (máx 1000 por llamada)
        all_raw: list[list[Any]] = []
        current_start = start_ms
        while True:
            if self._uses_http:
                raw = self._http_get(
                    "klines",
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "startTime": current_start,
                        "limit": 1000,
                    },
                )
            else:
                raw = self._client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=current_start,
                    limit=1000,
                )
            if not raw:
                break
            all_raw.extend(raw)
            if len(raw) < 1000:
                break
            current_start = raw[-1][0] + 1
            if self._uses_http:
                time.sleep(0.1)
        raw = all_raw

        if not raw:
            return pd.DataFrame(columns=_KLINE_COLUMNS).set_index("open_time")

        return _klines_to_dataframe(raw)

    def get_klines_batch(
        self,
        symbols: list[str],
        interval: str = "1h",
        lookback_hours: int = 72,
    ) -> dict[str, pd.DataFrame]:
        """Descarga velas para múltiples símbolos."""
        result: dict[str, pd.DataFrame] = {}
        for i, symbol in enumerate(symbols):
            try:
                result[symbol] = self.get_klines(symbol, interval, lookback_hours)
                if self._uses_http and i % 50 == 49:
                    logger.info("  Descargadas %d/%d monedas...", i + 1, len(symbols))
            except (BinanceAPIException, _requests.RequestException) as exc:
                logger.warning("No se pudieron obtener klines de %s: %s", symbol, exc)
        return result

    def get_current_price(self, symbol: str) -> float:
        """Precio actual de un símbolo."""
        if not self.is_connected:
            raise ConnectionError("No hay conexión con Binance.")

        if self._uses_http:
            data = self._http_get("ticker/price", {"symbol": symbol})
            return float(data["price"])

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

    # Activos fiat que no tienen par USDT en Binance
    _FIAT_ASSETS = frozenset(
        {
            "EUR",
            "USD",
            "GBP",
            "TRY",
            "BRL",
            "ARS",
            "RUB",
            "UAH",
            "NGN",
            "AUD",
            "JPY",
            "KRW",
            "INR",
            "PLN",
            "RON",
            "CZK",
            "HUF",
            "BGN",
            "SEK",
            "NOK",
            "DKK",
            "CHF",
            "CAD",
            "NZD",
            "MXN",
            "ZAR",
            "SGD",
            "HKD",
            "TWD",
            "THB",
            "IDR",
            "PHP",
            "VND",
            "MYR",
            "PKR",
            "BDT",
            "EGP",
            "CLP",
            "COP",
            "PEN",
        }
    )

    # Tasas aproximadas fiat → USDT para valoración de cartera.
    # Se usan solo cuando Binance no tiene par directo XXXUSDT.
    _FIAT_TO_USDT: dict[str, float] = {
        "EUR": 1.08,
        "GBP": 1.26,
        "CHF": 1.12,
        "AUD": 0.64,
        "CAD": 0.73,
        "JPY": 0.0067,
    }

    def get_portfolio_value_usdt(self) -> float:
        """Valor total de la cartera en USD (usando USDT o USDC como referencia)."""
        portfolio = self.get_portfolio()
        quote = self._portfolio_cfg.quote_asset
        total = 0.0
        for asset, qty in portfolio.items():
            if asset in ("USDT", "USDC", "BUSD", "FDUSD"):
                total += qty
            elif asset in self._FIAT_ASSETS:
                rate = self._FIAT_TO_USDT.get(asset, 0)
                if rate > 0:
                    value = qty * rate
                    total += value
                    logger.info(
                        "Fiat %s: %.2f x %.4f = $%.2f (tasa aprox.)",
                        asset,
                        qty,
                        rate,
                        value,
                    )
                else:
                    logger.info(
                        "Fiat %s (%.4f) sin tasa conocida, no incluido",
                        asset,
                        qty,
                    )
            else:
                try:
                    price = float(
                        self._client.get_symbol_ticker(
                            symbol=f"{asset}{quote}",
                        )["price"],
                    )
                    total += qty * price
                except BinanceAPIException:
                    # Fallback: intentar con USDT si quote es otro
                    if quote != "USDT":
                        try:
                            price = float(
                                self._client.get_symbol_ticker(
                                    symbol=f"{asset}USDT",
                                )["price"],
                            )
                            total += qty * price
                            continue
                        except BinanceAPIException:
                            pass
                    logger.warning("No se pudo valorar %s", asset)
        return total

    # ------------------------------------------------------------------
    # Órdenes
    # ------------------------------------------------------------------

    def place_market_buy(self, symbol: str, quote_qty: float) -> dict[str, Any]:
        """Compra a mercado gastando *quote_qty* del quote asset."""
        logger.info("MARKET BUY %s por %.2f %s", symbol, quote_qty, self._portfolio_cfg.quote_asset)
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
        """Coloca una orden OCO de venta (stop-loss + take-profit).

        Usa la API nueva de Binance (orderList/oco) con parámetros
        aboveType/belowType.

        Above = take-profit (LIMIT_MAKER): vende cuando sube.
        Below = stop-loss (STOP_LOSS_LIMIT): vende cuando baja.

        Redondea precios a tickSize y cantidad a stepSize para cumplir
        los filtros PRICE_FILTER y LOT_SIZE de Binance.
        """
        # Obtener filtros del símbolo para redondear correctamente
        price_filter = self.get_price_filter(symbol)
        lot_filter = self.get_lot_size_filter(symbol)

        # Calcular precios
        take_profit_price = entry_price * (1 + self._risk_cfg.take_profit_pct)
        stop_price = entry_price * (1 - self._risk_cfg.stop_loss_pct)
        stop_limit_price = stop_price * 0.995

        # Redondear precios a tick_size
        if price_filter:
            tick_size = price_filter["tick_size"]
            take_profit_price = self.round_to_tick_size(take_profit_price, tick_size)
            stop_price = self.round_to_tick_size(stop_price, tick_size)
            stop_limit_price = self.round_to_tick_size(stop_limit_price, tick_size)
            tick_precision = max(0, int(round(-math.log10(tick_size)))) if tick_size > 0 else 8
        else:
            tick_precision = 8
            logger.warning("No se encontró PRICE_FILTER para %s, usando 8 decimales", symbol)

        # Redondear cantidad a step_size
        if lot_filter:
            step_size = lot_filter["step_size"]
            quantity = self.round_to_step_size(quantity, step_size)
            step_precision = max(0, int(round(-math.log10(step_size)))) if step_size > 0 else 8
        else:
            step_precision = 8
            logger.warning("No se encontró LOT_SIZE para %s, usando 8 decimales", symbol)

        logger.info(
            "OCO SELL %s qty=%s | TP=%s | SL=%s",
            symbol,
            f"{quantity:.{step_precision}f}",
            f"{take_profit_price:.{tick_precision}f}",
            f"{stop_price:.{tick_precision}f}",
        )
        try:
            order = self._client.create_oco_order(
                symbol=symbol,
                side="SELL",
                quantity=f"{quantity:.{step_precision}f}",
                aboveType="LIMIT_MAKER",
                abovePrice=f"{take_profit_price:.{tick_precision}f}",
                belowType="STOP_LOSS_LIMIT",
                belowPrice=f"{stop_limit_price:.{tick_precision}f}",
                belowStopPrice=f"{stop_price:.{tick_precision}f}",
                belowTimeInForce="GTC",
            )
            return order
        except BinanceAPIException as exc:
            logger.error("Error al colocar OCO para %s: %s", symbol, exc)
            return None

    def get_symbol_info(self, symbol: str) -> dict[str, Any] | None:
        """Devuelve la info de un símbolo (filtros de precio, cantidad, etc.)."""
        return self._client.get_symbol_info(symbol)

    def get_lot_size_filter(self, symbol: str) -> dict[str, float] | None:
        """Devuelve min_qty, max_qty y step_size de un símbolo.

        Retorna None si el símbolo no existe.
        """
        info = self.get_symbol_info(symbol)
        if not info:
            return None
        for f in info.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                return {
                    "min_qty": float(f["minQty"]),
                    "max_qty": float(f["maxQty"]),
                    "step_size": float(f["stepSize"]),
                }
        return None

    def get_price_filter(self, symbol: str) -> dict[str, float] | None:
        """Devuelve min_price, max_price y tick_size de un símbolo.

        Retorna None si el símbolo no existe.
        """
        info = self.get_symbol_info(symbol)
        if not info:
            return None
        for f in info.get("filters", []):
            if f["filterType"] == "PRICE_FILTER":
                return {
                    "min_price": float(f["minPrice"]),
                    "max_price": float(f["maxPrice"]),
                    "tick_size": float(f["tickSize"]),
                }
        return None

    def round_to_tick_size(self, price: float, tick_size: float) -> float:
        """Redondea el precio al tick_size más cercano (hacia abajo)."""
        if tick_size <= 0:
            return price
        precision = max(0, int(round(-math.log10(tick_size))))
        rounded = math.floor(price / tick_size) * tick_size
        return round(rounded, precision)

    def get_min_notional(self, symbol: str) -> float:
        """Devuelve el valor mínimo en USDT para una orden (MIN_NOTIONAL)."""
        info = self.get_symbol_info(symbol)
        if not info:
            return 10.0  # fallback conservador
        for f in info.get("filters", []):
            if f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL"):
                return float(f.get("minNotional", 10.0))
        return 10.0

    def round_to_step_size(self, quantity: float, step_size: float) -> float:
        """Redondea la cantidad hacia abajo al step_size más cercano."""
        if step_size <= 0:
            return quantity
        precision = max(0, int(round(-math.log10(step_size))))
        rounded = math.floor(quantity / step_size) * step_size
        return round(rounded, precision)

    def validate_and_adjust_sell(
        self,
        symbol: str,
        quantity: float,
    ) -> tuple[float, str]:
        """Valida y ajusta una orden de venta.

        Retorna (adjusted_qty, error_msg).
        Si error_msg no está vacío, la orden no debe ejecutarse.
        """
        lot_filter = self.get_lot_size_filter(symbol)
        if lot_filter is None:
            return 0.0, f"Symbol {symbol} no encontrado en Binance"

        step_size = lot_filter["step_size"]
        min_qty = lot_filter["min_qty"]

        adjusted = self.round_to_step_size(quantity, step_size)

        if adjusted < min_qty:
            return 0.0, (f"Cantidad {quantity:.8f} < mínimo {min_qty:.8f} para {symbol}")

        return adjusted, ""

    def validate_buy(
        self,
        symbol: str,
        quote_qty: float,
    ) -> str:
        """Valida una orden de compra.

        Retorna cadena vacía si es válida, o el mensaje de error.
        """
        info = self.get_symbol_info(symbol)
        if info is None:
            return f"Symbol {symbol} no encontrado en Binance"

        min_notional = self.get_min_notional(symbol)
        if quote_qty < min_notional:
            return (
                f"Monto {quote_qty:.2f} {self._portfolio_cfg.quote_asset}"
                f" < mínimo {min_notional:.2f} para {symbol}"
            )

        return ""

    def convert_sell(self, from_asset: str, to_asset: str, amount: float) -> dict[str, Any]:
        """Vende via Convert API (fallback cuando Spot da -2010).

        Flujo: pedir quote → aceptar quote.
        Devuelve dict con claves compatibles con el resultado de market sell.
        """
        logger.info(
            "CONVERT SELL %s → %s qty=%.8f",
            from_asset,
            to_asset,
            amount,
        )
        # 1. Pedir cotización
        quote = self._client.convert_request_quote(
            fromAsset=from_asset,
            toAsset=to_asset,
            fromAmount=f"{amount:.8f}",
        )
        quote_id = quote["quoteId"]
        ratio = float(quote.get("ratio", 0))
        logger.info(
            "Convert quote recibida: id=%s ratio=%.8f",
            quote_id,
            ratio,
        )

        # 2. Aceptar cotización
        result = self._client.convert_accept_quote(quoteId=quote_id)
        logger.info("Convert aceptada: status=%s", result.get("orderStatus", "?"))

        # 3. Devolver resultado normalizado (compatible con market sell)
        to_amount = float(result.get("toAmount", 0))
        from_amount_result = float(result.get("fromAmount", amount))
        price = to_amount / from_amount_result if from_amount_result > 0 else 0

        return {
            "fills": [
                {
                    "qty": str(from_amount_result),
                    "price": str(price),
                    "commission": "0",
                }
            ],
            "via": "convert",
        }

    def get_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Devuelve las órdenes abiertas de un símbolo."""
        try:
            return self._client.get_open_orders(symbol=symbol)
        except BinanceAPIException as exc:
            logger.warning("Error leyendo órdenes abiertas de %s: %s", symbol, exc)
            return []

    def cancel_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Cancela todas las órdenes abiertas de un símbolo."""
        try:
            orders = self._client.get_open_orders(symbol=symbol)
            cancelled = []
            for order in orders:
                result = self._client.cancel_order(symbol=symbol, orderId=order["orderId"])
                cancelled.append(result)
            return cancelled
        except BinanceAPIException as exc:
            logger.warning("Error cancelando órdenes de %s: %s", symbol, exc)
            return []
