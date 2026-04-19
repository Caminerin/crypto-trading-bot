"""Generación y envío de reportes por email vía Mailjet.

Construye un email HTML con:
- Resumen de cartera (antes y después).
- Posiciones de predicción abiertas con P&L.
- Operaciones de predicción ejecutadas (ventas expiradas + compras).
- Política DCA por moneda (umbrales, TP, SL).
- Acciones DCA ejecutadas hoy con estado.
- Posiciones DCA abiertas con días en cartera.
- Posiciones y acciones Momentum.
- Top predicciones del modelo.
- P&L estimado.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from src.config import _QUOTE, DEFAULT_ASSET_POLICIES, DEFAULT_MOMENTUM_POLICIES, EmailConfig
from src.execution.executor import ExecutionResult
from src.strategies.dca import DCAAction
from src.strategies.momentum import MomentumAction
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Formateo inteligente de precios
# ------------------------------------------------------------------

def _fmt_price(price: float) -> str:
    """Formatea un precio con los decimales necesarios.

    Monedas caras (>=1$): 2 decimales.
    Monedas baratas (<1$): muestra hasta el primer dígito significativo + 2 más.
    """
    if price <= 0:
        return "$0"
    if price >= 1:
        return f"${price:,.2f}"
    if price >= 0.01:
        return f"${price:,.4f}"
    decimals = max(2, int(math.ceil(-math.log10(price))) + 2)
    return f"${price:,.{decimals}f}"


# ------------------------------------------------------------------
# Función principal de envío
# ------------------------------------------------------------------

def send_daily_report(
    config: EmailConfig,
    portfolio_before: dict[str, float],
    portfolio_after: dict[str, float],
    total_value_before: float,
    total_value_after: float,
    results: list[ExecutionResult],
    predictions: dict[str, float],
    is_paper: bool,
    dca_summary: dict[str, object] | None = None,
    allocation_budgets: dict[str, float] | None = None,
    dca_actions: list[DCAAction] | None = None,
    momentum_summary: dict[str, object] | None = None,
    prediction_summary: dict[str, object] | None = None,
    dca_results: list[ExecutionResult] | None = None,
    momentum_results: list[ExecutionResult] | None = None,
    momentum_actions: list[MomentumAction] | None = None,
) -> bool:
    """Envía el reporte diario por email.

    Devuelve True si el envío fue exitoso.
"""
    if not config.mailjet_api_key or not config.mailjet_api_secret:
        logger.warning("Mailjet API key/secret no configuradas. No se envía email.")
        return False

    try:
        from mailjet_rest import Client as MailjetClient
    except ImportError:
        logger.error("mailjet-rest no está instalado. Ejecuta: pip install mailjet-rest")
        return False

    subject = _build_subject(total_value_before, total_value_after, is_paper)
    html_body = _build_html_body(
        portfolio_before=portfolio_before,
        portfolio_after=portfolio_after,
        total_value_before=total_value_before,
        total_value_after=total_value_after,
        results=results,
        predictions=predictions,
        is_paper=is_paper,
        dca_summary=dca_summary or {},
        allocation_budgets=allocation_budgets or {},
        dca_actions=dca_actions or [],
        momentum_summary=momentum_summary or {},
        prediction_summary=prediction_summary or {},
        dca_results=dca_results or [],
        momentum_results=momentum_results or [],
        momentum_actions=momentum_actions or [],
    )

    mailjet = MailjetClient(
        auth=(config.mailjet_api_key, config.mailjet_api_secret),
        version="v3.1",
    )

    data = {
        "Messages": [
            {
                "From": {"Email": config.email_from, "Name": "Crypto Trading Bot"},
                "To": [
                    {"Email": addr.strip()}
                    for addr in config.email_to.split(",")
                    if addr.strip()
                ],
                "Subject": subject,
                "HTMLPart": html_body,
            }
        ]
    }

    try:
        response = mailjet.send.create(data=data)
        status = response.status_code
        logger.info("Email enviado — status=%d", status)
        if status != 200:
            logger.error("Mailjet respuesta: %s", response.json())
        return status == 200
    except Exception as exc:
        logger.error("Error enviando email: %s", exc)
        return False


def _build_subject(
    value_before: float, value_after: float, is_paper: bool
) -> str:
    mode = "[PAPER] " if is_paper else ""
    pnl = value_after - value_before
    pnl_pct = (pnl / value_before * 100) if value_before > 0 else 0
    sign = "+" if pnl >= 0 else ""
    return (
        f"{mode}Crypto Bot — "
        f"Balance: ${value_after:,.2f} "
        f"({sign}{pnl_pct:.2f}%)"
    )


def _days_held(entry_date: str) -> int:
    """Calcula días transcurridos desde la fecha de entrada."""
    try:
        entry = datetime.fromisoformat(entry_date)
        if entry.tzinfo is None:
            entry = entry.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - entry
        return max(delta.days, 0)
    except (ValueError, TypeError):
        return 0


def _build_dca_policy_table() -> str:
    """Genera la tabla HTML con la política DCA por moneda."""
    _cell = 'style="padding:6px;border:1px solid #ddd;text-align:center"'
    rows = ""
    for symbol, policy in sorted(DEFAULT_ASSET_POLICIES.items()):
        coin = symbol.replace(_QUOTE, "")
        rows += (
            "<tr>"
            f"<td {_cell}><strong>{coin}</strong></td>"
            f"<td {_cell}>{policy.dip_threshold:.0%}</td>"
            f"<td {_cell}>+{policy.take_profit_pct:.1%}</td>"
            f"<td {_cell}>{policy.stop_loss_pct:.1%}</td>"
            "</tr>"
        )
    return (
        '<h3 style="margin-top:15px">Política DCA por Moneda</h3>'
        '<table style="border-collapse:collapse;width:100%">'
        '<tr style="background:#f8f9fa">'
        '<th style="padding:6px;border:1px solid #ddd">Moneda</th>'
        '<th style="padding:6px;border:1px solid #ddd">Umbral compra</th>'
        '<th style="padding:6px;border:1px solid #ddd">Take-Profit</th>'
        '<th style="padding:6px;border:1px solid #ddd">Stop-Loss</th>'
        "</tr>"
        f"{rows}"
        "</table>"
    )


def _build_dca_actions_section(
    dca_actions: list[DCAAction],
    dca_results: list[ExecutionResult],
) -> str:
    """Genera la sección HTML de acciones DCA ejecutadas hoy (con estado)."""
    if not dca_actions:
        return (
            '<h3 style="margin-top:15px">Acciones DCA Hoy</h3>'
            "<p>Sin acciones DCA hoy. Mercado estable.</p>"
        )

    result_map: dict[str, ExecutionResult] = {}
    for r in dca_results:
        key = f"{r.action.action}_{r.action.symbol}"
        result_map[key] = r

    _cell = 'style="padding:6px;border:1px solid #ddd"'
    rows = ""
    for a in dca_actions:
        color = "#28a745" if a.action == "BUY" else "#dc3545"
        icon = "COMPRA" if a.action == "BUY" else "VENTA"
        coin = a.symbol.replace(_QUOTE, "")
        amount = f"${a.quote_qty:,.2f}" if a.action == "BUY" else f"{a.base_qty:.6f}"

        key = f"{a.action}_{a.symbol}"
        result = result_map.get(key)
        if result is not None:
            if result.success:
                status = '<span style="color:#28a745;font-weight:bold">OK</span>'
            else:
                err = result.error
                status = (
                    '<span style="color:#dc3545;'
                    f'font-weight:bold">ERROR: {err}</span>'
                )
        else:
            status = '<span style="color:#999">\u2014</span>'

        rows += (
            "<tr>"
            f'<td {_cell} style="color:{color};'
            f'font-weight:bold">{icon}</td>'
            f"<td {_cell}>{coin}</td>"
            f"<td {_cell}>{amount}</td>"
            f"<td {_cell}>{status}</td>"
            f"<td {_cell}>{a.reason}</td>"
            "</tr>"
        )

    return (
        '<h3 style="margin-top:15px">'
        f'Acciones DCA Hoy ({len(dca_actions)})</h3>'
        '<table style="border-collapse:collapse;width:100%">'
        '<tr style="background:#f8f9fa">'
        f'<th {_cell}>Tipo</th>'
        f'<th {_cell}>Moneda</th>'
        f'<th {_cell}>Cantidad</th>'
        f'<th {_cell}>Estado</th>'
        f'<th {_cell}>Motivo</th>'
        "</tr>"
        f"{rows}"
        "</table>"
    )


def _build_dca_section(
    dca_summary: dict[str, object],
    dca_actions: list[DCAAction] | None = None,
    dca_results: list[ExecutionResult] | None = None,
) -> str:
    """Genera la sección HTML del DCA Inteligente para el email."""
    if not dca_summary:
        return ""

    budget = dca_summary.get("budget", 0)
    invested = dca_summary.get("invested", 0)
    free = dca_summary.get("free", 0)
    total_pnl = dca_summary.get("total_pnl", 0)
    positions = dca_summary.get("positions", [])

    pnl_color = "#28a745" if float(total_pnl) >= 0 else "#dc3545"
    pnl_sign = "+" if float(total_pnl) >= 0 else ""

    pos_rows = ""
    for p in positions:
        p_pnl = p.get("pnl", 0)
        p_pnl_pct = p.get("pnl_pct", 0)
        p_color = "#28a745" if p_pnl >= 0 else "#dc3545"
        p_sign = "+" if p_pnl >= 0 else ""
        days = _days_held(p.get("entry_date", ""))
        days_label = f"{days}d" if days > 0 else "hoy"
        entry_price = p.get("entry_price", 0)
        current_price = p.get("current_price", 0)
        pos_rows += (
            "<tr>"
            f'<td style="padding:6px;border:1px solid #ddd">'
            f'{p.get("symbol", "").replace(_QUOTE, "")}</td>'
            f'<td style="padding:6px;border:1px solid #ddd">{_fmt_price(entry_price)}</td>'
            f'<td style="padding:6px;border:1px solid #ddd">{_fmt_price(current_price)}</td>'
            f'<td style="padding:6px;border:1px solid #ddd">${p.get("invested", 0):,.2f}</td>'
            f'<td style="padding:6px;border:1px solid #ddd">${p.get("current_value", 0):,.2f}</td>'
            f'<td style="padding:6px;border:1px solid #ddd;color:{p_color};font-weight:bold">'
            f'{p_sign}${p_pnl:,.2f} ({p_sign}{p_pnl_pct:.1f}%)</td>'
            f'<td style="padding:6px;border:1px solid #ddd;text-align:center">{days_label}</td>'
            "</tr>"
        )

    if not positions:
        pos_table = "<p>Sin posiciones DCA abiertas. Esperando caídas para comprar.</p>"
    else:
        pos_table = (
            '<table style="border-collapse:collapse;width:100%">'
            '<tr style="background:#f8f9fa">'
            '<th style="padding:6px;border:1px solid #ddd">Moneda</th>'
            '<th style="padding:6px;border:1px solid #ddd">Precio compra</th>'
            '<th style="padding:6px;border:1px solid #ddd">Precio actual</th>'
            '<th style="padding:6px;border:1px solid #ddd">Invertido</th>'
            '<th style="padding:6px;border:1px solid #ddd">Valor actual</th>'
            '<th style="padding:6px;border:1px solid #ddd">P&amp;L</th>'
            '<th style="padding:6px;border:1px solid #ddd">Días</th>'
            "</tr>"
            f"{pos_rows}"
            "</table>"
        )

    return (
        '<h2>DCA Inteligente (BTC + ETH + BNB)</h2>'
        '<table style="border-collapse:collapse;width:100%;margin-bottom:10px">'
        "<tr>"
        f'<td style="padding:6px;border:1px solid #ddd"><strong>Presupuesto DCA</strong></td>'
        f'<td style="padding:6px;border:1px solid #ddd">${float(budget):,.2f}</td>'
        "</tr><tr>"
        f'<td style="padding:6px;border:1px solid #ddd"><strong>Invertido</strong></td>'
        f'<td style="padding:6px;border:1px solid #ddd">${float(invested):,.2f}</td>'
        "</tr><tr>"
        f'<td style="padding:6px;border:1px solid #ddd"><strong>Disponible</strong></td>'
        f'<td style="padding:6px;border:1px solid #ddd">${float(free):,.2f}</td>'
        "</tr><tr>"
        f'<td style="padding:6px;border:1px solid #ddd"><strong>P&amp;L DCA</strong></td>'
        f'<td style="padding:6px;border:1px solid #ddd;color:{pnl_color};font-weight:bold">'
        f'{pnl_sign}${float(total_pnl):,.2f}</td>'
        "</tr>"
        "</table>"
        f"{pos_table}"
        f"{_build_dca_policy_table()}"
        f"{_build_dca_actions_section(dca_actions or [], dca_results or [])}"
    )


def _build_momentum_policy_table() -> str:
    """Genera la tabla HTML con la política Momentum por moneda."""
    _cell = 'style="padding:6px;border:1px solid #ddd;text-align:center"'
    rows = ""
    for symbol, policy in sorted(DEFAULT_MOMENTUM_POLICIES.items()):
        coin = symbol.replace(_QUOTE, "")
        rows += (
            "<tr>"
            f"<td {_cell}><strong>{coin}</strong></td>"
            f"<td {_cell}>+{policy.momentum_threshold:.0%}</td>"
            f"<td {_cell}>+{policy.take_profit_pct:.0%}</td>"
            f"<td {_cell}>{policy.stop_loss_pct:.0%}</td>"
            f"<td {_cell}>{policy.trend_days}d</td>"
            "</tr>"
        )
    return (
        '<h3 style="margin-top:15px">Política Momentum por Moneda</h3>'
        '<table style="border-collapse:collapse;width:100%">'
        '<tr style="background:#f8f9fa">'
        '<th style="padding:6px;border:1px solid #ddd">Moneda</th>'
        '<th style="padding:6px;border:1px solid #ddd">Umbral</th>'
        '<th style="padding:6px;border:1px solid #ddd">Take-Profit</th>'
        '<th style="padding:6px;border:1px solid #ddd">Stop-Loss</th>'
        '<th style="padding:6px;border:1px solid #ddd">Días tendencia</th>'
        "</tr>"
        f"{rows}"
        "</table>"
    )


def _build_momentum_actions_section(
    momentum_actions: list[MomentumAction],
    momentum_results: list[ExecutionResult],
) -> str:
    """Genera la sección HTML de acciones Momentum ejecutadas hoy."""
    if not momentum_actions:
        return (
            '<h3 style="margin-top:15px">Acciones Momentum Hoy</h3>'
            "<p>Sin acciones Momentum hoy.</p>"
        )

    result_map: dict[str, ExecutionResult] = {}
    for r in momentum_results:
        key = f"{r.action.action}_{r.action.symbol}"
        result_map[key] = r

    _cell = 'style="padding:6px;border:1px solid #ddd"'
    rows = ""
    for a in momentum_actions:
        color = "#28a745" if a.action == "BUY" else "#dc3545"
        icon = "COMPRA" if a.action == "BUY" else "VENTA"
        coin = a.symbol.replace(_QUOTE, "")
        amount = f"${a.quote_qty:,.2f}" if a.action == "BUY" else f"{a.base_qty:.6f}"

        key = f"{a.action}_{a.symbol}"
        result = result_map.get(key)
        if result is not None:
            if result.success:
                status = '<span style="color:#28a745;font-weight:bold">OK</span>'
            else:
                err = result.error
                status = (
                    '<span style="color:#dc3545;'
                    f'font-weight:bold">ERROR: {err}</span>'
                )
        else:
            status = '<span style="color:#999">\u2014</span>'

        rows += (
            "<tr>"
            f'<td {_cell} style="color:{color};'
            f'font-weight:bold">{icon}</td>'
            f"<td {_cell}>{coin}</td>"
            f"<td {_cell}>{amount}</td>"
            f"<td {_cell}>{status}</td>"
            f"<td {_cell}>{a.reason}</td>"
            "</tr>"
        )

    return (
        '<h3 style="margin-top:15px">'
        f'Acciones Momentum Hoy ({len(momentum_actions)})</h3>'
        '<table style="border-collapse:collapse;width:100%">'
        '<tr style="background:#f8f9fa">'
        f'<th {_cell}>Tipo</th>'
        f'<th {_cell}>Moneda</th>'
        f'<th {_cell}>Cantidad</th>'
        f'<th {_cell}>Estado</th>'
        f'<th {_cell}>Motivo</th>'
        "</tr>"
        f"{rows}"
        "</table>"
    )


def _build_momentum_section(
    momentum_summary: dict[str, object],
    momentum_actions: list[MomentumAction] | None = None,
    momentum_results: list[ExecutionResult] | None = None,
) -> str:
    """Genera la sección HTML de Momentum para el email."""
    if not momentum_summary:
        return ""

    budget = momentum_summary.get("budget", 0)
    invested = momentum_summary.get("invested", 0)
    free = momentum_summary.get("free", 0)
    total_pnl = momentum_summary.get("total_pnl", 0)
    positions = momentum_summary.get("positions", [])

    pnl_color = "#28a745" if float(total_pnl) >= 0 else "#dc3545"
    pnl_sign = "+" if float(total_pnl) >= 0 else ""

    pos_rows = ""
    for p in positions:
        p_pnl = p.get("pnl", 0)
        p_pnl_pct = p.get("pnl_pct", 0)
        p_color = "#28a745" if p_pnl >= 0 else "#dc3545"
        p_sign = "+" if p_pnl >= 0 else ""
        days = _days_held(p.get("entry_date", ""))
        days_label = f"{days}d" if days > 0 else "hoy"
        entry_price = p.get("entry_price", 0)
        current_price = p.get("current_price", 0)
        pos_rows += (
            "<tr>"
            f'<td style="padding:6px;border:1px solid #ddd">'
            f'{p.get("symbol", "").replace(_QUOTE, "")}</td>'
            f'<td style="padding:6px;border:1px solid #ddd">{_fmt_price(entry_price)}</td>'
            f'<td style="padding:6px;border:1px solid #ddd">{_fmt_price(current_price)}</td>'
            f'<td style="padding:6px;border:1px solid #ddd">${p.get("invested", 0):,.2f}</td>'
            f'<td style="padding:6px;border:1px solid #ddd">${p.get("current_value", 0):,.2f}</td>'
            f'<td style="padding:6px;border:1px solid #ddd;color:{p_color};font-weight:bold">'
            f'{p_sign}${p_pnl:,.2f} ({p_sign}{p_pnl_pct:.1f}%)</td>'
            f'<td style="padding:6px;border:1px solid #ddd;text-align:center">{days_label}</td>'
            "</tr>"
        )

    if not positions:
        pos_table = "<p>Sin posiciones Momentum abiertas. Esperando subidas fuertes.</p>"
    else:
        pos_table = (
            '<table style="border-collapse:collapse;width:100%">'
            '<tr style="background:#f8f9fa">'
            '<th style="padding:6px;border:1px solid #ddd">Moneda</th>'
            '<th style="padding:6px;border:1px solid #ddd">Precio compra</th>'
            '<th style="padding:6px;border:1px solid #ddd">Precio actual</th>'
            '<th style="padding:6px;border:1px solid #ddd">Invertido</th>'
            '<th style="padding:6px;border:1px solid #ddd">Valor actual</th>'
            '<th style="padding:6px;border:1px solid #ddd">P&amp;L</th>'
            '<th style="padding:6px;border:1px solid #ddd">Días</th>'
            "</tr>"
            f"{pos_rows}"
            "</table>"
        )

    return (
        '<h2>Momentum (BTC + ETH + BNB + SOL + XRP)</h2>'
        '<table style="border-collapse:collapse;width:100%;margin-bottom:10px">'
        "<tr>"
        f'<td style="padding:6px;border:1px solid #ddd"><strong>Presupuesto Momentum</strong></td>'
        f'<td style="padding:6px;border:1px solid #ddd">${float(budget):,.2f}</td>'
        "</tr><tr>"
        f'<td style="padding:6px;border:1px solid #ddd"><strong>Invertido</strong></td>'
        f'<td style="padding:6px;border:1px solid #ddd">${float(invested):,.2f}</td>'
        "</tr><tr>"
        f'<td style="padding:6px;border:1px solid #ddd"><strong>Disponible</strong></td>'
        f'<td style="padding:6px;border:1px solid #ddd">${float(free):,.2f}</td>'
        "</tr><tr>"
        f'<td style="padding:6px;border:1px solid #ddd"><strong>P&amp;L Momentum</strong></td>'
        f'<td style="padding:6px;border:1px solid #ddd;color:{pnl_color};font-weight:bold">'
        f'{pnl_sign}${float(total_pnl):,.2f}</td>'
        "</tr>"
        "</table>"
        f"{pos_table}"
        f"{_build_momentum_policy_table()}"
        f"{_build_momentum_actions_section(momentum_actions or [], momentum_results or [])}"
    )


def _build_allocation_section(budgets: dict[str, float]) -> str:
    """Genera la sección HTML de asignación de cartera para el email."""
    if not budgets:
        return ""

    total = sum(budgets.values())
    rows = ""
    colors = {
        "prediction": "#007bff",
        "dca": "#28a745",
        "momentum": "#fd7e14",
        "reserve": "#6c757d",
    }
    labels = {
        "prediction": "Predicción (35%)",
        "dca": "DCA Inteligente (20%)",
        "momentum": "Momentum (35%)",
        "reserve": "Reserva (10%)",
    }

    for strategy, amount in budgets.items():
        pct = (amount / total * 100) if total > 0 else 0
        color = colors.get(strategy, "#333")
        label = labels.get(strategy, strategy)
        rows += (
            "<tr>"
            f'<td style="padding:6px;border:1px solid #ddd">{label}</td>'
            f'<td style="padding:6px;border:1px solid #ddd">${amount:,.2f}</td>'
            f'<td style="padding:6px;border:1px solid #ddd">'
            f'<div style="background:#eee;border-radius:4px;overflow:hidden;width:150px">'
            f'<div style="background:{color};height:14px;width:{pct:.0f}%"></div>'
            f'</div></td>'
            "</tr>"
        )

    return (
        '<h2>Distribución de Cartera</h2>'
        '<table style="border-collapse:collapse;width:100%">'
        '<tr style="background:#f8f9fa">'
        '<th style="padding:6px;border:1px solid #ddd">Estrategia</th>'
        '<th style="padding:6px;border:1px solid #ddd">Asignado</th>'
        '<th style="padding:6px;border:1px solid #ddd">%</th>'
        "</tr>"
        f"{rows}"
        f'<tr style="font-weight:bold">'
        f'<td style="padding:6px;border:1px solid #ddd">Total</td>'
        f'<td style="padding:6px;border:1px solid #ddd">${total:,.2f}</td>'
        f'<td style="padding:6px;border:1px solid #ddd">100%</td>'
        "</tr>"
        "</table>"
    )


def _build_prediction_section(
    prediction_summary: dict[str, object],
    results: list[ExecutionResult],
) -> str:
    """Genera la sección HTML de la estrategia Predicción."""
    if not prediction_summary:
        return ""

    budget = prediction_summary.get("budget", 0)
    invested = prediction_summary.get("invested", 0)
    total_pnl = prediction_summary.get("total_pnl", 0)
    positions = prediction_summary.get("positions", [])
    free = float(budget) - float(invested) if budget else 0

    pnl_color = "#28a745" if float(total_pnl) >= 0 else "#dc3545"
    pnl_sign = "+" if float(total_pnl) >= 0 else ""

    _cell = 'style="padding:6px;border:1px solid #ddd"'
    pos_rows = ""
    for p in positions:
        p_pnl = p.get("pnl", 0)
        p_pnl_pct = p.get("pnl_pct", 0)
        p_color = "#28a745" if p_pnl >= 0 else "#dc3545"
        p_sign = "+" if p_pnl >= 0 else ""
        days = _days_held(p.get("entry_date", ""))
        days_label = f"{days}d" if days > 0 else "hoy"
        entry_price = p.get("entry_price", 0)
        current_price = p.get("current_price", 0)
        pos_rows += (
            "<tr>"
            f'<td {_cell}>{p.get("symbol", "").replace(_QUOTE, "")}</td>'
            f"<td {_cell}>{_fmt_price(entry_price)}</td>"
            f"<td {_cell}>{_fmt_price(current_price)}</td>"
            f'<td {_cell}>${p.get("invested", 0):,.2f}</td>'
            f'<td {_cell}>${p.get("current_value", 0):,.2f}</td>'
            f'<td {_cell} style="color:{p_color};'
            f'font-weight:bold">'
            f'{p_sign}${p_pnl:,.2f} ({p_sign}{p_pnl_pct:.1f}%)</td>'
            f'<td {_cell} style="text-align:center">'
            f'{days_label}</td>'
            "</tr>"
        )

    if not positions:
        pos_table = "<p>Sin posiciones de predicción abiertas.</p>"
    else:
        pos_table = (
            '<table style="border-collapse:collapse;width:100%">'
            '<tr style="background:#f8f9fa">'
            f'<th {_cell}>Moneda</th>'
            f'<th {_cell}>Precio compra</th>'
            f'<th {_cell}>Precio actual</th>'
            f'<th {_cell}>Invertido</th>'
            f'<th {_cell}>Valor actual</th>'
            f'<th {_cell}>P&amp;L</th>'
            f'<th {_cell}>Días</th>'
            "</tr>"
            f"{pos_rows}"
            "</table>"
        )

    ops_html = _build_prediction_ops_section(results)

    return (
        '<h2>Predicción ML (Altcoins)</h2>'
        '<table style="border-collapse:collapse;width:100%;margin-bottom:10px">'
        "<tr>"
        f'<td {_cell}><strong>Presupuesto Predicción</strong></td>'
        f'<td {_cell}>${float(budget):,.2f}</td>'
        "</tr><tr>"
        f'<td {_cell}><strong>Invertido</strong></td>'
        f'<td {_cell}>${float(invested):,.2f}</td>'
        "</tr><tr>"
        f'<td {_cell}><strong>Disponible</strong></td>'
        f'<td {_cell}>${free:,.2f}</td>'
        "</tr><tr>"
        f'<td {_cell}><strong>P&amp;L Predicción</strong></td>'
        f'<td {_cell} style="padding:6px;border:1px solid #ddd;color:{pnl_color};font-weight:bold">'
        f'{pnl_sign}${float(total_pnl):,.2f}</td>'
        "</tr>"
        "</table>"
        f"{pos_table}"
        f"{ops_html}"
    )


def _build_prediction_ops_section(results: list[ExecutionResult]) -> str:
    """Genera la sección de operaciones de predicción, separando ventas de compras."""
    if not results:
        return (
            '<h3 style="margin-top:15px">Operaciones Predicción Hoy</h3>'
            "<p>No se ejecutaron operaciones de predicción hoy.</p>"
        )

    sells = [r for r in results if r.action.action == "SELL"]
    buys = [r for r in results if r.action.action == "BUY"]

    html = f'<h3 style="margin-top:15px">Operaciones Predicción Hoy ({len(results)})</h3>'

    if sells:
        html += (
            '<h4 style="margin-top:10px;color:#dc3545">'
            f'Ventas por expiración ({len(sells)})</h4>'
        )
        html += _build_ops_table(sells)

    if buys:
        html += f'<h4 style="margin-top:10px;color:#28a745">Compras nuevas ({len(buys)})</h4>'
        html += _build_ops_table(buys)

    return html


def _build_ops_table(results: list[ExecutionResult]) -> str:
    """Genera una tabla HTML para una lista de resultados de ejecución."""
    _cell = 'style="padding:6px;border:1px solid #ddd"'
    rows = ""
    for r in results:
        status = "OK" if r.success else f"ERROR: {r.error}"
        color = "#28a745" if r.success else "#dc3545"
        action_label = "COMPRA" if r.action.action == "BUY" else "VENTA"
        coin = r.action.symbol.replace(_QUOTE, "")
        rows += (
            "<tr>"
            f'<td {_cell} style="color:{color};font-weight:bold">{action_label}</td>'
            f"<td {_cell}>{coin}</td>"
            f"<td {_cell}>{r.executed_qty:.6f}</td>"
            f"<td {_cell}>{_fmt_price(r.executed_price)}</td>"
            f'<td {_cell} style="color:{color}">{status}</td>'
            f"<td {_cell}>{r.action.reason}</td>"
            "</tr>"
        )

    return (
        '<table style="border-collapse:collapse;width:100%">'
        '<tr style="background:#f8f9fa">'
        f'<th {_cell}>Tipo</th>'
        f'<th {_cell}>Moneda</th>'
        f'<th {_cell}>Cantidad</th>'
        f'<th {_cell}>Precio</th>'
        f'<th {_cell}>Estado</th>'
        f'<th {_cell}>Razón</th>'
        "</tr>"
        f"{rows}"
        "</table>"
    )


def _build_html_body(
    portfolio_before: dict[str, float],
    portfolio_after: dict[str, float],
    total_value_before: float,
    total_value_after: float,
    results: list[ExecutionResult],
    predictions: dict[str, float],
    is_paper: bool,
    dca_summary: dict[str, object] | None = None,
    allocation_budgets: dict[str, float] | None = None,
    dca_actions: list[DCAAction] | None = None,
    momentum_summary: dict[str, object] | None = None,
    prediction_summary: dict[str, object] | None = None,
    dca_results: list[ExecutionResult] | None = None,
    momentum_results: list[ExecutionResult] | None = None,
    momentum_actions: list[MomentumAction] | None = None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pnl = total_value_after - total_value_before
    pnl_pct = (pnl / total_value_before * 100) if total_value_before > 0 else 0
    pnl_color = "#28a745" if pnl >= 0 else "#dc3545"

    # Top predicciones
    top_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:20]
    pred_rows = ""
    for symbol, prob in top_preds:
        bar_width = int(prob * 100)
        bar_color = "#28a745" if prob >= 0.70 else "#ffc107" if prob >= 0.50 else "#dc3545"
        coin = symbol.replace(_QUOTE, "")
        pred_rows += (
            "<tr>"
            f"<td>{coin}</td>"
            f"<td>{prob:.1%}</td>"
            "<td>"
            '<div style="background:#eee;border-radius:4px;overflow:hidden;width:200px">'
            f'<div style="background:{bar_color};height:16px;width:{bar_width}%"></div>'
            "</div>"
            "</td>"
            "</tr>"
        )

    mode_badge = (
        '<span style="background:#ffc107;color:#000;padding:2px 8px;border-radius:4px;'
        'font-weight:bold">PAPER TRADING</span>'
        if is_paper
        else '<span style="background:#28a745;color:#fff;padding:2px 8px;border-radius:4px;'
        'font-weight:bold">LIVE</span>'
    )

    no_preds_msg = "<p>Sin predicciones disponibles (modelo no entrenado o sin datos).</p>"

    if predictions:
        preds_section = (
            '<table style="border-collapse:collapse;width:100%">'
            '<tr style="background:#f8f9fa">'
            '<th style="padding:8px;border:1px solid #ddd">Moneda</th>'
            '<th style="padding:8px;border:1px solid #ddd">Prob. subida &gt;5%</th>'
            '<th style="padding:8px;border:1px solid #ddd">Confianza</th>'
            "</tr>"
            f"{pred_rows}"
            "</table>"
        )
    else:
        preds_section = no_preds_msg

    portfolio_rows = "".join(
        f'<tr><td style="padding:8px;border:1px solid #ddd">{asset}</td>'
        f'<td style="padding:8px;border:1px solid #ddd">{qty:.8f}</td></tr>'
        for asset, qty in sorted(portfolio_after.items())
        if qty > 0
    )

    pnl_sign = "+" if pnl >= 0 else ""

    pred_html = _build_prediction_section(
        prediction_summary or dict(), results,
    )
    dca_html = _build_dca_section(
        dca_summary or dict(), dca_actions or [], dca_results or [],
    )
    mom_html = _build_momentum_section(
        momentum_summary or dict(),
        momentum_actions or [],
        momentum_results or [],
    )
    alloc_html = _build_allocation_section(allocation_budgets or dict())

    return (
        "<html>"
        '<body style="font-family:Arial,sans-serif;max-width:800px;'
        'margin:0 auto;padding:20px">'
        "<h1>Crypto Trading Bot \u2014 Reporte Diario</h1>"
        f"<p>{now} | {mode_badge}</p>"
        "<h2>Resumen</h2>"
        '<table style="border-collapse:collapse;width:100%">'
        "<tr>"
        '<td style="padding:8px;border:1px solid #ddd">'
        "<strong>Balance anterior</strong></td>"
        f'<td style="padding:8px;border:1px solid #ddd">'
        f"${total_value_before:,.2f}</td>"
        "</tr>"
        "<tr>"
        '<td style="padding:8px;border:1px solid #ddd">'
        "<strong>Balance actual</strong></td>"
        f'<td style="padding:8px;border:1px solid #ddd">'
        f"${total_value_after:,.2f}</td>"
        "</tr>"
        "<tr>"
        '<td style="padding:8px;border:1px solid #ddd">'
        "<strong>P&amp;L</strong></td>"
        f'<td style="padding:8px;border:1px solid #ddd;'
        f'color:{pnl_color};font-weight:bold">'
        f"{pnl_sign}${pnl:,.2f} ({pnl_pct:+.2f}%)"
        "</td>"
        "</tr>"
        "</table>"
        f"{pred_html}"
        f"{dca_html}"
        f"{mom_html}"
        f"{alloc_html}"
        "<h2>Top 20 Predicciones</h2>"
        f"{preds_section}"
        "<h2>Cartera Actual</h2>"
        '<table style="border-collapse:collapse;width:100%">'
        '<tr style="background:#f8f9fa">'
        '<th style="padding:8px;border:1px solid #ddd">Activo</th>'
        '<th style="padding:8px;border:1px solid #ddd">Cantidad</th>'
        "</tr>"
        f"{portfolio_rows}"
        "</table>"
        "<hr>"
        '<p style="color:#999;font-size:12px">'
        "Generado automáticamente por Crypto Trading Bot. "
        "Esto no es asesoramiento financiero."
        "</p>"
        "</body>"
        "</html>"
    )
