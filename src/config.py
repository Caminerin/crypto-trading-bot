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

    top_n_coins: int = 50
    confidence_threshold: float = 0.70
    target_pct_change: float = 0.03  # 3%
    target_horizon_hours: int = 48
    lookback_hours: int = 120
    candle_interval: str = "1h"
    retrain_interval_days: int = 7
    training_days: int = 90


@dataclass(frozen=True)
class PortfolioConfig:
    """Reglas de gestión de cartera."""

    max_positions: int = 5
    max_pct_per_coin: float = 0.20  # 20%
    min_stablecoin_reserve: float = 0.10  # 10%
    quote_asset: str = "USDT"


@dataclass(frozen=True)
class RiskConfig:
    """Parámetros de gestión de riesgo."""

    stop_loss_pct: float = 0.03  # 3%
    take_profit_pct: float = 0.05  # 5%


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

    @property
    def is_paper_trading(self) -> bool:
        return self.trading_mode == "paper"


def load_config() -> AppConfig:
    """Crea y devuelve la configuración de la aplicación."""
    return AppConfig()
