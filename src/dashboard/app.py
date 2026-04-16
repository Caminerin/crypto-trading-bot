"""
Dashboard web ligero para monitorizar el bot de trading.

Solo lectura — nunca modifica datos del bot.
Usa Flask + Jinja2 + Pico CSS (CDN).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template

from src.config import _QUOTE, DEFAULT_ASSET_POLICIES, AllocationConfig

# ---------------------------------------------------------------------------
# Rutas a ficheros de datos
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DCA_POSITIONS_FILE = DATA_DIR / "dca_positions.json"
PORTFOLIO_STATE_FILE = DATA_DIR / "portfolio_state.json"
ALLOCATION_FILE = DATA_DIR / "allocation.json"

# ---------------------------------------------------------------------------
# App Flask
# ---------------------------------------------------------------------------
app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers de lectura (solo lectura, tolerante a ficheros ausentes)
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    """Lee un fichero JSON de forma segura. Devuelve {} si no existe o falla."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError):
        return {}


def _get_dca_positions() -> list[dict[str, Any]]:
    """Devuelve la lista de posiciones DCA abiertas."""
    data = _read_json(DCA_POSITIONS_FILE)
    return data.get("positions", [])  # type: ignore[no-any-return]


def _get_dca_updated_at() -> str | None:
    """Devuelve la fecha de última actualización del fichero DCA."""
    data = _read_json(DCA_POSITIONS_FILE)
    return data.get("updated_at")  # type: ignore[no-any-return]


def _get_allocation() -> dict[str, Any]:
    """Devuelve el estado del asignador de cartera."""
    return _read_json(ALLOCATION_FILE)


def _model_exists() -> bool:
    return (MODELS_DIR / "predictor.joblib").exists()


def _model_modified_time() -> str | None:
    model_path = MODELS_DIR / "predictor.joblib"
    if not model_path.exists():
        return None
    mtime = os.path.getmtime(model_path)
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def _find_daily_reports() -> list[Path]:
    """Busca ficheros de reporte diario en data/."""
    if not DATA_DIR.exists():
        return []
    reports = sorted(DATA_DIR.glob("daily_report_*.json"), reverse=True)
    return reports[:10]  # últimos 10


def _last_execution_time() -> str | None:
    """Estima la última ejecución mirando el fichero DCA o allocation."""
    updated = _get_dca_updated_at()
    if updated:
        return updated
    alloc = _get_allocation()
    if alloc:
        # allocation.json no tiene timestamp, pero su mtime sí
        if ALLOCATION_FILE.exists():
            mtime = os.path.getmtime(ALLOCATION_FILE)
            return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    return None


def _format_datetime_es(iso_str: str | None) -> str:
    """Formatea un ISO datetime a formato legible en español."""
    if not iso_str:
        return "Nunca ejecutado"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%d/%m/%Y %H:%M UTC")
    except (ValueError, TypeError):
        return iso_str


def _compute_portfolio_value(
    positions: list[dict[str, Any]],
    allocation: dict[str, Any],
) -> float:
    """Calcula el valor total aproximado del portafolio."""
    wallets = allocation.get("wallets", {})
    total = sum(wallets.values()) if wallets else 0.0

    # Sumar el valor actual de posiciones DCA (ya incluido en el budget, pero
    # la diferencia P&L se suma aparte)
    for pos in positions:
        pnl = pos.get("pnl", 0.0) if "pnl" not in pos else 0.0
        # pnl ya está contabilizado en get_summary, lo ignoramos aquí
        _ = pnl

    return total


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------


@app.route("/")
def home() -> str:
    """Página principal: resumen del bot."""
    allocation = _get_allocation()
    wallets = allocation.get("wallets", {})
    percentages = allocation.get("percentages", {})

    positions = _get_dca_positions()
    last_exec = _last_execution_time()

    # Valor total
    total_value = sum(wallets.values()) if wallets else 0.0

    # Info del modelo
    model_ready = _model_exists()
    model_time = _model_modified_time()

    # Reportes recientes
    reports = _find_daily_reports()
    reports_info = []
    for rp in reports[:5]:
        rdata = _read_json(rp)
        reports_info.append({
            "filename": rp.name,
            "date": rdata.get("date", rp.stem.replace("daily_report_", "")),
        })

    alloc_config = AllocationConfig()

    return render_template(
        "home.html",
        last_exec=_format_datetime_es(last_exec),
        next_exec="07:00 / 19:00 UTC (diario)",
        total_value=round(total_value, 2),
        wallets=wallets,
        percentages=percentages,
        alloc_config=alloc_config,
        model_ready=model_ready,
        model_time=_format_datetime_es(model_time),
        n_positions=len(positions),
        reports=reports_info,
        now=datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC"),
    )


@app.route("/dca")
def dca() -> str:
    """Página DCA: posiciones abiertas y política por activo."""
    positions = _get_dca_positions()
    updated_at = _get_dca_updated_at()

    allocation = _get_allocation()
    wallets = allocation.get("wallets", {})
    dca_budget = wallets.get("dca", 0.0)

    # Calcular P&L para cada posición (sin precio en vivo, usamos los datos guardados)
    total_invested = 0.0
    total_current = 0.0
    for pos in positions:
        invested = pos.get("invested_usdt", 0.0)
        current_value = pos.get("current_value", invested)
        total_invested += invested
        total_current += current_value

    total_pnl = total_current - total_invested
    free_usdt = dca_budget - total_invested

    # Política por activo
    policies = []
    for symbol, policy in DEFAULT_ASSET_POLICIES.items():
        clean_name = symbol.replace(_QUOTE, "")
        policies.append({
            "symbol": symbol,
            "name": clean_name,
            "dip_threshold": policy.dip_threshold,
            "take_profit_pct": policy.take_profit_pct,
            "stop_loss_pct": policy.stop_loss_pct,
        })

    return render_template(
        "dca.html",
        positions=positions,
        updated_at=_format_datetime_es(updated_at),
        dca_budget=round(dca_budget, 2),
        total_invested=round(total_invested, 2),
        total_pnl=round(total_pnl, 2),
        free_usdt=round(free_usdt, 2),
        policies=policies,
        now=datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC"),
    )


@app.route("/api/status")
def api_status() -> tuple[Any, int]:
    """Endpoint JSON con el estado completo del bot."""
    allocation = _get_allocation()
    wallets = allocation.get("wallets", {})
    positions = _get_dca_positions()
    last_exec = _last_execution_time()

    total_value = sum(wallets.values()) if wallets else 0.0

    # P&L de posiciones DCA
    total_invested = sum(p.get("invested_usdt", 0.0) for p in positions)
    total_current = sum(p.get("current_value", p.get("invested_usdt", 0.0)) for p in positions)

    policies_dict: dict[str, Any] = {}
    for symbol, policy in DEFAULT_ASSET_POLICIES.items():
        policies_dict[symbol] = {
            "dip_threshold": policy.dip_threshold,
            "take_profit_pct": policy.take_profit_pct,
            "stop_loss_pct": policy.stop_loss_pct,
        }

    status = {
        "bot": {
            "última_ejecución": last_exec,
            "próxima_ejecución": "07:00 / 19:00 UTC (diario)",
            "modelo_entrenado": _model_exists(),
            "modelo_fecha": _model_modified_time(),
        },
        "portafolio": {
            "valor_total_usdt": round(total_value, 2),
            "asignación": wallets,
            "porcentajes": allocation.get("percentages", {}),
        },
        "dca": {
            "posiciones_abiertas": len(positions),
            "posiciones": positions,
            "invertido_usdt": round(total_invested, 2),
            "valor_actual_usdt": round(total_current, 2),
            "pnl_usdt": round(total_current - total_invested, 2),
            "políticas": policies_dict,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify(status), 200


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
