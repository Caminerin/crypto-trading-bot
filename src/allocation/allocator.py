"""
Asignador de cartera por estrategias (virtual wallets).

Divide el balance total de Binance en "monederos virtuales",
cada uno asignado a una estrategia distinta.  La asignacion se
persiste en un fichero JSON para que sobreviva entre ejecuciones.

Ejemplo con 70 EUR (~75 USDT):
  - prediction: 50 % -> 37.5 USDT
  - dca:        40 % -> 30.0 USDT
  - reserve:    10 % ->  7.5 USDT
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
ALLOCATION_FILE = DATA_DIR / "allocation.json"

# Porcentajes por defecto
DEFAULT_ALLOCATION = {
    "prediction": 0.50,
    "dca": 0.40,
    "reserve": 0.10,
}


class PortfolioAllocator:
    """Gestiona la asignacion virtual del balance entre estrategias."""

    def __init__(
        self,
        allocation_pcts: dict[str, float] | None = None,
    ) -> None:
        self._pcts = allocation_pcts or DEFAULT_ALLOCATION.copy()
        self._wallets: dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Carga la asignacion desde disco (si existe)."""
        if ALLOCATION_FILE.exists():
            try:
                raw = json.loads(ALLOCATION_FILE.read_text())
                self._wallets = raw.get("wallets", {})
                logger.info("Asignacion cargada: %s", self._wallets)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Error leyendo allocation.json: %s", exc)
                self._wallets = {}
        else:
            logger.info("Sin asignacion previa — se inicializara al conocer el balance.")

    def save(self) -> None:
        """Persiste la asignacion actual a disco."""
        data: dict[str, Any] = {
            "percentages": self._pcts,
            "wallets": self._wallets,
        }
        ALLOCATION_FILE.write_text(json.dumps(data, indent=2))
        logger.info("Asignacion guardada: %s", self._wallets)

    # ------------------------------------------------------------------
    # Inicializacion y consulta
    # ------------------------------------------------------------------

    def initialize(self, total_usdt: float) -> None:
        """Primera asignacion cuando no hay datos previos."""
        if self._wallets:
            logger.info("Ya existe asignacion previa — no se reinicializa.")
            return
        for strategy, pct in self._pcts.items():
            self._wallets[strategy] = round(total_usdt * pct, 2)
        self.save()
        logger.info(
            "Asignacion inicial (total=%.2f USDT): %s",
            total_usdt, self._wallets,
        )

    @property
    def is_initialized(self) -> bool:
        return bool(self._wallets)

    def get_budget(self, strategy: str) -> float:
        """Devuelve el USDT disponible para una estrategia."""
        return self._wallets.get(strategy, 0.0)

    def get_all_budgets(self) -> dict[str, float]:
        """Devuelve copia del estado de todos los monederos."""
        return self._wallets.copy()

    # ------------------------------------------------------------------
    # Actualizaciones
    # ------------------------------------------------------------------

    def update_budget(self, strategy: str, new_value: float) -> None:
        """Actualiza el saldo virtual de una estrategia."""
        old = self._wallets.get(strategy, 0.0)
        self._wallets[strategy] = round(new_value, 2)
        self.save()
        logger.info(
            "Budget [%s]: %.2f -> %.2f", strategy, old, new_value,
        )

    def add_profit(self, strategy: str, profit: float) -> None:
        """Suma (o resta) beneficio al monedero de una estrategia."""
        current = self._wallets.get(strategy, 0.0)
        self._wallets[strategy] = round(current + profit, 2)
        self.save()
        logger.info(
            "Profit [%s]: %+.2f (nuevo saldo: %.2f)",
            strategy, profit, self._wallets[strategy],
        )

    def rebalance(self, total_usdt: float) -> None:
        """Re-asigna todo el balance segun los porcentajes configurados.

        Util si el usuario quiere resetear la distribucion.
        """
        for strategy, pct in self._pcts.items():
            self._wallets[strategy] = round(total_usdt * pct, 2)
        self.save()
        logger.info("Rebalance (total=%.2f): %s", total_usdt, self._wallets)
