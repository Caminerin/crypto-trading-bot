"""
Configuración central del bot.

Carga variables de entorno y define constantes del sistema.
Cada parámetro tiene un valor por defecto sensato para facilitar el arranque.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class BinanceConfig:
    api_key: str = os.getenv("BINANCE_API_KEY", "")
    api_secret: str = os.getenv("BINANCE_API_SECRET", "")


@dataclass(frozen=True)
class ModelConfig:
    """Parámetros del modelo predictivo."""

    top_n_coins: int = 75
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
    target_pct_change: float = 0.05  # 5% TP
    stop_loss_pct: float = 0.05  # 5% SL para labels (ratio 1:1)
    target_horizon_hours: int = 48
    lookback_hours: int = 120
    candle_interval: str = "1h"
    retrain_interval_days: int = 7
    training_days: int = 30


@dataclass(frozen=True)
class PortfolioConfig:
    """Reglas de gestión de cartera."""

    max_positions: int = 5
    max_pct_per_coin: float = 0.20  # 20%
    min_stablecoin_reserve: float = 0.0  # 0% (la reserva ya la gestiona AllocationConfig)
    quote_asset: str = os.getenv("QUOTE_ASSET", "USDT")


@dataclass(frozen=True)
class RiskConfig:
    """Parámetros de gestión de riesgo."""

    stop_loss_pct: float = 0.05  # 5%
    take_profit_pct: float = 0.05  # 5% (ratio 1:1 — esperanza mas robusta)


@dataclass(frozen=True)
class DCAAssetPolicy:
    """Politica individual por moneda."""

    dip_threshold: float   # Ej. -0.07 = compra si cae >7%
    take_profit_pct: float # Ej. 0.175 = vende si sube 17.5%
    stop_loss_pct: float   # Ej. -0.105 = vende si cae 10.5%


@dataclass(frozen=True)
class MomentumAssetPolicy:
    """Política individual por moneda para la estrategia momentum."""

    momentum_threshold: float  # Ej. 0.05 = compra si sube >5% en 24h
    take_profit_pct: float     # Ej. 0.10 = vende si sube 10%
    stop_loss_pct: float       # Ej. -0.05 = vende si cae 5%
    trend_days: int = 7        # Días para confirmar tendencia alcista


# Quote asset configurado (USDT o USDC según el .env)
_QUOTE = os.getenv("QUOTE_ASSET", "USDT")

# Política óptima momentum por moneda (backtested 730d, 520 combos/moneda)
DEFAULT_MOMENTUM_POLICIES: dict[str, MomentumAssetPolicy] = {
    f"BTC{_QUOTE}": MomentumAssetPolicy(
        momentum_threshold=0.10,
        take_profit_pct=0.30,
        stop_loss_pct=-0.15,
        trend_days=3,
    ),
    f"ETH{_QUOTE}": MomentumAssetPolicy(
        momentum_threshold=0.10,
        take_profit_pct=0.30,
        stop_loss_pct=-0.15,
        trend_days=14,
    ),
    f"BNB{_QUOTE}": MomentumAssetPolicy(
        momentum_threshold=0.03,
        take_profit_pct=0.25,
        stop_loss_pct=-0.15,
        trend_days=3,
    ),
    f"SOL{_QUOTE}": MomentumAssetPolicy(
        momentum_threshold=0.07,
        take_profit_pct=0.20,
        stop_loss_pct=-0.15,
        trend_days=3,
    ),
    f"XRP{_QUOTE}": MomentumAssetPolicy(
        momentum_threshold=0.05,
        take_profit_pct=0.25,
        stop_loss_pct=-0.03,
        trend_days=14,
    ),
}


# Política óptima por moneda (backtested 365d, 84 combos/moneda)
DEFAULT_ASSET_POLICIES: dict[str, DCAAssetPolicy] = {
    f"BTC{_QUOTE}": DCAAssetPolicy(
        dip_threshold=-0.07,
        take_profit_pct=0.175,
        stop_loss_pct=-0.105,
    ),
    f"ETH{_QUOTE}": DCAAssetPolicy(
        dip_threshold=-0.05,
        take_profit_pct=0.10,
        stop_loss_pct=-0.075,
    ),
    f"BNB{_QUOTE}": DCAAssetPolicy(
        dip_threshold=-0.04,
        take_profit_pct=0.20,
        stop_loss_pct=-0.08,
    ),
}


@dataclass(frozen=True)
class DCAConfig:
    """Parametros de la estrategia DCA Inteligente."""

    enabled: bool = True
    assets: tuple[str, ...] = (
        f"BTC{_QUOTE}", f"ETH{_QUOTE}", f"BNB{_QUOTE}",
    )
    # Parametros globales (fallback si no hay politica por moneda)
    dip_threshold: float = -0.05
    take_profit_pct: float = 0.15
    stop_loss_pct: float = -0.10
    min_order_usdt: float = 10.0


@dataclass(frozen=True)
class MomentumConfig:
    """Parámetros de la estrategia Momentum."""

    enabled: bool = True
    assets: tuple[str, ...] = (
        f"BTC{_QUOTE}", f"ETH{_QUOTE}", f"BNB{_QUOTE}",
        f"SOL{_QUOTE}", f"XRP{_QUOTE}",
    )
    # Parámetros globales (fallback si no hay política por moneda)
    momentum_threshold: float = 0.05
    take_profit_pct: float = 0.10
    stop_loss_pct: float = -0.05
    trend_days: int = 7
    min_order_usdt: float = 10.0


@dataclass(frozen=True)
class AllocationConfig:
    """Reparto del balance entre estrategias (virtual wallets)."""

    prediction_pct: float = 0.60   # 60% para bot de prediccion ML
    dca_pct: float = 0.20          # 20% para DCA inteligente
    momentum_pct: float = 0.20     # 20% para momentum
    reserve_pct: float = 0.00      # 0% reserva


@dataclass(frozen=True)
class EmailConfig:
    mailjet_api_key: str = os.getenv("MAILJET_API_KEY", "")
    mailjet_api_secret: str = os.getenv("MAILJET_API_SECRET", "")
    email_from: str = os.getenv("EMAIL_FROM", "")
    email_to: str = os.getenv("EMAIL_TO", "")


@dataclass(frozen=True)
class AppConfig:
    """Configuración raíz que agrupa todas las secciones."""

    trading_mode: str = os.getenv("TRADING_MODE", "paper")
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    dca: DCAConfig = field(default_factory=DCAConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)

    @property
    def is_paper_trading(self) -> bool:
        return self.trading_mode == "paper"


def load_config() -> AppConfig:
    """Crea y devuelve la configuración de la aplicación."""
    return AppConfig()
