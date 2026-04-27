"""Generacion y envio de reportes por email via Mailjet.

Construye un email HTML con:
- Resumen de cartera (antes y despues).
- Politica DCA por moneda (umbrales, TP, SL).
- Acciones DCA ejecutadas hoy.
- Posiciones DCA abiertas con dias en cartera.
- Operaciones de prediccion ejecutadas.
- Top predicciones del modelo.
- P&L estimado.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.config import _QUOTE, DEFAULT_ASSET_POLICIES, DEFAULT_MOMENTUM_POLICIES, EmailConfig
from src.execution.executor import ExecutionResult
from src.market.regime import MarketRegimeResult
from src.strategies.dca import DCAAction
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
    model_info: dict[str, object] | None = None,
    market_regime: MarketRegimeResult | None = None,
) -> bool:
    """Envia el reporte diario por email.

    Devuelve True si el envio fue exitoso.
    """
    if not config.mailjet_api_key or not config.mailjet_api_secret:
        logger.warning("Mailjet API key/secret no configuradas. No se envia email.")
        return False

    try:
        from mailjet_rest import Client as MailjetClient
    except ImportError:
        logger.error("mailjet-rest no esta instalado. Ejecuta: pip install mailjet-rest")
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
        model_info=model_info or {},
        market_regime=market_regime,
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
    """Calcula dias transcurridos desde la fecha de entrada."""
    try:
        entry = datetime.fromisoformat(entry_date)
        if entry.tzinfo is None:
            entry = entry.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - entry
        return max(delta.days, 0)
    except (ValueError, TypeError):
        return 0


def _build_dca_policy_table() -> str:
    """Genera la tabla HTML con la politica DCA por moneda."""
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
        '<h3 style="margin-top:15px">Politica DCA por Moneda</h3>'
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


def _build_dca_actions_section(dca_actions: list[DCAAction]) -> str:
    """Genera la seccion HTML de acciones DCA ejecutadas hoy."""
    if not dca_actions:
        return (
            '<h3 style="margin-top:15px">Acciones DCA Hoy</h3>'
            "<p>Sin acciones DCA hoy. Mercado estable.</p>"
        )

    _cell = 'style="padding:6px;border:1px solid #ddd"'
    rows = ""
    for a in dca_actions:
        color = "#28a745" if a.action == "BUY" else "#dc3545"
        icon = "COMPRA" if a.action == "BUY" else "VENTA"
        coin = a.symbol.replace(_QUOTE, "")
        amount = f"${a.quote_qty:,.2f}" if a.action == "BUY" else f"{a.base_qty:.6f}"
        rows += (
            "<tr>"
            f'<td {_cell} style="padding:6px;border:1px solid #ddd;color:{color};'
            f'font-weight:bold">{icon}</td>'
            f"<td {_cell}>{coin}</td>"
            f"<td {_cell}>{amount}</td>"
            f"<td {_cell}>{a.reason}</td>"
            "</tr>"
        )

    return (
        f'<h3 style="margin-top:15px">Acciones DCA Hoy ({len(dca_actions)})</h3>'
        '<table style="border-collapse:collapse;width:100%">'
        '<tr style="background:#f8f9fa">'
        f'<th {_cell}>Tipo</th>'
        f'<th {_cell}>Moneda</th>'
        f'<th {_cell}>Cantidad</th>'
        f'<th {_cell}>Motivo</th>'
        "</tr>"
        f"{rows}"
        "</table>"
    )


def _build_dca_section(
    dca_summary: dict[str, object],
    dca_actions: list[DCAAction] | None = None,
) -> str:
    """Genera la seccion HTML del DCA Inteligente para el email."""
    if not dca_summary:
        return ""

    budget = dca_summary.get("budget", 0)
    invested = dca_summary.get("invested", 0)
    free = dca_summary.get("free", 0)
    total_pnl = dca_summary.get("total_pnl", 0)
    positions = dca_summary.get("positions", [])

    pnl_color = "#28a745" if float(str(total_pnl)) >= 0 else "#dc3545"
    pnl_sign = "+" if float(str(total_pnl)) >= 0 else ""

    pos_rows = ""
    for p in positions:
        p_pnl = p.get("pnl", 0)
        p_pnl_pct = p.get("pnl_pct", 0)
        p_color = "#28a745" if p_pnl >= 0 else "#dc3545"
        p_sign = "+" if p_pnl >= 0 else ""
        days = _days_held(p.get("entry_date", ""))
        days_label = f"{days}d" if days > 0 else "hoy"
        pos_rows += (
            "<tr>"
            f'<td style="padding:6px;border:1px solid #ddd">'
            f'{p.get("symbol", "").replace(_QUOTE, "")}</td>'
            f'<td style="padding:6px;border:1px solid #ddd">${p.get("entry_price", 0):,.2f}</td>'
            f'<td style="padding:6px;border:1px solid #ddd">${p.get("current_price", 0):,.2f}</td>'
            f'<td style="padding:6px;border:1px solid #ddd">${p.get("invested", 0):,.2f}</td>'
            f'<td style="padding:6px;border:1px solid #ddd">${p.get("current_value", 0):,.2f}</td>'
            f'<td style="padding:6px;border:1px solid #ddd;color:{p_color};font-weight:bold">'
            f'{p_sign}${p_pnl:,.2f} ({p_sign}{p_pnl_pct:.1f}%)</td>'
            f'<td style="padding:6px;border:1px solid #ddd;text-align:center">{days_label}</td>'
            "</tr>"
        )

    if not positions:
        pos_table = "<p>Sin posiciones DCA abiertas. Esperando caidas para comprar.</p>"
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
            '<th style="padding:6px;border:1px solid #ddd">Dias</th>'
            "</tr>"
            f"{pos_rows}"
            "</table>"
        )

    return (
        '<h2>DCA Inteligente (BTC + ETH + BNB)</h2>'
        '<table style="border-collapse:collapse;width:100%;margin-bottom:10px">'
        "<tr>"
        f'<td style="padding:6px;border:1px solid #ddd"><strong>Presupuesto DCA</strong></td>'
        f'<td style="padding:6px;border:1px solid #ddd">${float(str(budget)):,.2f}</td>'
        "</tr><tr>"
        f'<td style="padding:6px;border:1px solid #ddd"><strong>Invertido</strong></td>'
        f'<td style="padding:6px;border:1px solid #ddd">${float(str(invested)):,.2f}</td>'
        "</tr><tr>"
        f'<td style="padding:6px;border:1px solid #ddd"><strong>Disponible</strong></td>'
        f'<td style="padding:6px;border:1px solid #ddd">${float(str(free)):,.2f}</td>'
        "</tr><tr>"
        f'<td style="padding:6px;border:1px solid #ddd"><strong>P&amp;L DCA</strong></td>'
        f'<td style="padding:6px;border:1px solid #ddd;color:{pnl_color};font-weight:bold">'
        f'{pnl_sign}${float(str(total_pnl)):,.2f}</td>'
        "</tr>"
        "</table>"
        f"{pos_table}"
        f"{_build_dca_policy_table()}"
        f"{_build_dca_actions_section(dca_actions or [])}"
    )


def _build_momentum_policy_table() -> str:
    """Genera la tabla HTML con la politica Momentum por moneda."""
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
        '<h3 style="margin-top:15px">Politica Momentum por Moneda</h3>'
        '<table style="border-collapse:collapse;width:100%">'
        '<tr style="background:#f8f9fa">'
        '<th style="padding:6px;border:1px solid #ddd">Moneda</th>'
        '<th style="padding:6px;border:1px solid #ddd">Umbral</th>'
        '<th style="padding:6px;border:1px solid #ddd">Take-Profit</th>'
        '<th style="padding:6px;border:1px solid #ddd">Stop-Loss</th>'
        '<th style="padding:6px;border:1px solid #ddd">Trend Days</th>'
        "</tr>"
        f"{rows}"
        "</table>"
    )


def _build_momentum_section(momentum_summary: dict[str, object]) -> str:
    """Genera la seccion HTML de Momentum para el email."""
    if not momentum_summary:
        return ""

    budget = momentum_summary.get("budget", 0)
    invested = momentum_summary.get("invested", 0)
    free = momentum_summary.get("free", 0)
    total_pnl = momentum_summary.get("total_pnl", 0)
    positions = momentum_summary.get("positions", [])

    pnl_color = "#28a745" if float(str(total_pnl)) >= 0 else "#dc3545"
    pnl_sign = "+" if float(str(total_pnl)) >= 0 else ""

    pos_rows = ""
    for p in positions:
        p_pnl = p.get("pnl", 0)
        p_pnl_pct = p.get("pnl_pct", 0)
        p_color = "#28a745" if p_pnl >= 0 else "#dc3545"
        p_sign = "+" if p_pnl >= 0 else ""
        days = _days_held(p.get("entry_date", ""))
        days_label = f"{days}d" if days > 0 else "hoy"
        pos_rows += (
            "<tr>"
            f'<td style="padding:6px;border:1px solid #ddd">'
            f'{p.get("symbol", "").replace(_QUOTE, "")}</td>'
            f'<td style="padding:6px;border:1px solid #ddd">${p.get("entry_price", 0):,.2f}</td>'
            f'<td style="padding:6px;border:1px solid #ddd">${p.get("current_price", 0):,.2f}</td>'
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
            '<th style="padding:6px;border:1px solid #ddd">Dias</th>'
            "</tr>"
            f"{pos_rows}"
            "</table>"
        )

    return (
        '<h2>Momentum (BTC + ETH + BNB + SOL + XRP)</h2>'
        '<table style="border-collapse:collapse;width:100%;margin-bottom:10px">'
        "<tr>"
        f'<td style="padding:6px;border:1px solid #ddd"><strong>Presupuesto Momentum</strong></td>'
        f'<td style="padding:6px;border:1px solid #ddd">${float(str(budget)):,.2f}</td>'
        "</tr><tr>"
        f'<td style="padding:6px;border:1px solid #ddd"><strong>Invertido</strong></td>'
        f'<td style="padding:6px;border:1px solid #ddd">${float(str(invested)):,.2f}</td>'
        "</tr><tr>"
        f'<td style="padding:6px;border:1px solid #ddd"><strong>Disponible</strong></td>'
        f'<td style="padding:6px;border:1px solid #ddd">${float(str(free)):,.2f}</td>'
        "</tr><tr>"
        f'<td style="padding:6px;border:1px solid #ddd"><strong>P&amp;L Momentum</strong></td>'
        f'<td style="padding:6px;border:1px solid #ddd;color:{pnl_color};font-weight:bold">'
        f'{pnl_sign}${float(str(total_pnl)):,.2f}</td>'
        "</tr>"
        "</table>"
        f"{pos_table}"
        f"{_build_momentum_policy_table()}"
    )


def _build_allocation_section(budgets: dict[str, float]) -> str:
    """Genera la seccion HTML de asignacion de cartera para el email."""
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
        "prediction": "Prediccion (35%)",
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
        '<h2>Distribucion de Cartera</h2>'
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


def _build_model_info_section(model_info: dict[str, object]) -> str:
    """Genera la seccion HTML con informacion del modelo ML."""
    if not model_info:
        return ""

    trained_at = model_info.get("trained_at", "Desconocido")
    age_days = model_info.get("age_days", -1)
    retrain_interval = model_info.get("retrain_interval_days", 7)
    status = model_info.get("status", "desconocido")

    if age_days < 0:
        age_label = "Sin modelo"
        age_color = "#dc3545"
    elif age_days == 0:
        age_label = "Hoy"
        age_color = "#28a745"
    elif age_days <= retrain_interval:
        age_label = f"Hace {age_days} dia{'s' if age_days != 1 else ''}"
        age_color = "#28a745"
    else:
        age_label = f"Hace {age_days} dias (ATRASADO)"
        age_color = "#dc3545"

    status_color = "#28a745" if status == "ok" else "#dc3545"
    status_label = "Al dia" if status == "ok" else "Necesita re-entrenamiento"

    _cell = 'style="padding:6px;border:1px solid #ddd"'
    return (
        '<h2>Modelo ML</h2>'
        '<table style="border-collapse:collapse;width:100%">'
        '<tr>'
        f'<td {_cell}><strong>Ultimo entrenamiento</strong></td>'
        f'<td {_cell}>{trained_at}</td>'
        '</tr><tr>'
        f'<td {_cell}><strong>Antigueedad</strong></td>'
        f'<td {_cell} style="color:{age_color};font-weight:bold">{age_label}</td>'
        '</tr><tr>'
        f'<td {_cell}><strong>Re-entrenamiento cada</strong></td>'
        f'<td {_cell}>{retrain_interval} dias</td>'
        '</tr><tr>'
        f'<td {_cell}><strong>Estado</strong></td>'
        f'<td {_cell} style="color:{status_color};font-weight:bold">{status_label}</td>'
        '</tr>'
        '</table>'
    )


def send_training_report(
    config: EmailConfig,
    metrics: dict[str, object],
    top_features: list[tuple[str, float]] | None = None,
    best_tpsl: dict[str, object] | None = None,
) -> bool:
    """Envía un email con las métricas del entrenamiento del modelo.

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

    subject = _build_training_subject(metrics)
    html_body = _build_training_html(metrics, top_features, best_tpsl)

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
        logger.info("Email de entrenamiento enviado — status=%d", status)
        if status != 200:
            logger.error("Mailjet respuesta: %s", response.json())
        return status == 200
    except Exception as exc:
        logger.error("Error enviando email de entrenamiento: %s", exc)
        return False


def _build_training_subject(metrics: dict[str, object]) -> str:
    auc = metrics.get("mean_auc", 0)
    precision = metrics.get("cv_precision_1", 0)
    recall = metrics.get("cv_recall_1", 0)
    return (
        f"Modelo ML Entrenado — "
        f"AUC: {float(auc):.3f} | "
        f"Precisión: {float(precision):.0%} | "
        f"Recall: {float(recall):.0%}"
    )


def _build_training_html(
    metrics: dict[str, object],
    top_features: list[tuple[str, float]] | None = None,
    best_tpsl: dict[str, object] | None = None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    mean_auc = float(metrics.get("mean_auc", 0))
    accuracy = float(metrics.get("cv_accuracy", 0))
    precision = float(metrics.get("cv_precision_1", 0))
    recall = float(metrics.get("cv_recall_1", 0))
    f1 = float(metrics.get("cv_f1_1", 0))
    samples = int(metrics.get("samples", 0))
    positive_rate = float(metrics.get("positive_rate", 0))
    n_feat_orig = int(metrics.get("n_features_original", 0))
    n_feat_sel = int(metrics.get("n_features_selected", 0))

    # AUC por modelo base
    auc_lgbm = float(metrics.get("auc_lgbm", 0))
    auc_xgb = float(metrics.get("auc_xgb", 0))
    auc_rf = float(metrics.get("auc_rf", 0))
    auc_et = float(metrics.get("auc_et", 0))

    # Color del AUC
    if mean_auc >= 0.70:
        auc_color = "#28a745"
        auc_label = "BUENO"
    elif mean_auc >= 0.60:
        auc_color = "#ffc107"
        auc_label = "ACEPTABLE"
    else:
        auc_color = "#dc3545"
        auc_label = "BAJO"

    _cell = 'style="padding:6px;border:1px solid #ddd"'

    # Sección de métricas principales
    metrics_html = (
        '<h2>Métricas del Modelo</h2>'
        f'<div style="background:{"#d4edda" if mean_auc >= 0.65 else "#fff3cd"};'
        f'padding:10px;border-radius:6px;margin-bottom:10px">'
        f'<strong style="color:{auc_color};font-size:18px">'
        f'AUC: {mean_auc:.4f} — {auc_label}</strong>'
        '</div>'
        '<table style="border-collapse:collapse;width:100%">'
        '<tr>'
        f'<td {_cell}><strong>AUC medio (CV)</strong></td>'
        f'<td {_cell} style="font-weight:bold;color:{auc_color}">{mean_auc:.4f}</td>'
        '</tr><tr>'
        f'<td {_cell}><strong>Accuracy</strong></td>'
        f'<td {_cell}>{accuracy:.1%}</td>'
        '</tr><tr>'
        f'<td {_cell}><strong>Precisión (clase 1)</strong></td>'
        f'<td {_cell}>{precision:.1%}</td>'
        '</tr><tr>'
        f'<td {_cell}><strong>Recall (clase 1)</strong></td>'
        f'<td {_cell}>{recall:.1%}</td>'
        '</tr><tr>'
        f'<td {_cell}><strong>F1 (clase 1)</strong></td>'
        f'<td {_cell}>{f1:.1%}</td>'
        '</tr><tr>'
        f'<td {_cell}><strong>Muestras de entrenamiento</strong></td>'
        f'<td {_cell}>{samples:,}</td>'
        '</tr><tr>'
        f'<td {_cell}><strong>Tasa de positivos</strong></td>'
        f'<td {_cell}>{positive_rate:.1%}</td>'
        '</tr><tr>'
        f'<td {_cell}><strong>Features originales</strong></td>'
        f'<td {_cell}>{n_feat_orig}</td>'
        '</tr><tr>'
        f'<td {_cell}><strong>Features seleccionadas</strong></td>'
        f'<td {_cell}>{n_feat_sel}</td>'
        '</tr>'
        '</table>'
    )

    # Sección AUC por modelo
    models_html = (
        '<h2>AUC por Modelo Base</h2>'
        '<table style="border-collapse:collapse;width:100%">'
        '<tr style="background:#f8f9fa">'
        f'<th {_cell}>Modelo</th>'
        f'<th {_cell}>AUC</th>'
        f'<th {_cell}>Rendimiento</th>'
        '</tr>'
    )
    for name, auc_val in [
        ("LightGBM", auc_lgbm),
        ("XGBoost", auc_xgb),
        ("RandomForest", auc_rf),
        ("ExtraTrees", auc_et),
    ]:
        bar_w = int(auc_val * 100)
        bar_c = "#28a745" if auc_val >= 0.65 else "#ffc107" if auc_val >= 0.55 else "#dc3545"
        models_html += (
            '<tr>'
            f'<td {_cell}><strong>{name}</strong></td>'
            f'<td {_cell}>{auc_val:.4f}</td>'
            f'<td {_cell}>'
            f'<div style="background:#eee;border-radius:4px;overflow:hidden;width:150px">'
            f'<div style="background:{bar_c};height:14px;width:{bar_w}%"></div>'
            f'</div></td>'
            '</tr>'
        )
    models_html += '</table>'

    # Sección top features
    features_html = ""
    if top_features:
        feat_rows = ""
        for feat_name, importance in top_features[:15]:
            bar_w = int(importance * 100 * 5)  # escala x5 para mejor visual
            bar_w = min(bar_w, 100)
            feat_rows += (
                '<tr>'
                f'<td {_cell}>{feat_name}</td>'
                f'<td {_cell}>{importance:.4f}</td>'
                f'<td {_cell}>'
                f'<div style="background:#eee;border-radius:4px;overflow:hidden;width:150px">'
                f'<div style="background:#007bff;height:14px;width:{bar_w}%"></div>'
                f'</div></td>'
                '</tr>'
            )
        features_html = (
            '<h2>Top 15 Features Más Importantes</h2>'
            '<table style="border-collapse:collapse;width:100%">'
            '<tr style="background:#f8f9fa">'
            f'<th {_cell}>Feature</th>'
            f'<th {_cell}>Importancia</th>'
            f'<th {_cell}>Peso relativo</th>'
            '</tr>'
            f'{feat_rows}'
            '</table>'
        )

    # Sección TP/SL del sweep
    tpsl_html = ""
    if best_tpsl:
        tp = float(best_tpsl.get("take_profit_pct", 0))
        sl = float(best_tpsl.get("stop_loss_pct", 0))
        comp_pnl = float(best_tpsl.get("compound_pnl_pct", 0))
        win_pct = float(best_tpsl.get("win_pct", 0))
        trades = int(best_tpsl.get("trades", 0))
        max_dd = float(best_tpsl.get("max_drawdown_pct", 0))
        avg_pnl = float(best_tpsl.get("avg_pnl_per_trade", 0))
        sweep_days = int(best_tpsl.get("sweep_days", 30))

        comp_color = "#28a745" if comp_pnl >= 0 else "#dc3545"
        comp_sign = "+" if comp_pnl >= 0 else ""

        tpsl_html = (
            '<h2>TP/SL Seleccionado (Sweep)</h2>'
            f'<div style="background:#e7f1ff;padding:10px;border-radius:6px;margin-bottom:10px">'
            f'<strong style="color:#007bff;font-size:16px">'
            f'TP={tp:.0%} / SL={sl:.0%}</strong>'
            '</div>'
            '<table style="border-collapse:collapse;width:100%">'
            '<tr>'
            f'<td {_cell}><strong>P&amp;L compuesto ({sweep_days}d)</strong></td>'
            f'<td {_cell} style="color:{comp_color};font-weight:bold">'
            f'{comp_sign}{comp_pnl:.1f}%</td>'
            '</tr><tr>'
            f'<td {_cell}><strong>P&amp;L medio/trade</strong></td>'
            f'<td {_cell}>${avg_pnl:.4f}</td>'
            '</tr><tr>'
            f'<td {_cell}><strong>Win rate</strong></td>'
            f'<td {_cell}>{win_pct:.0f}%</td>'
            '</tr><tr>'
            f'<td {_cell}><strong>Trades simulados</strong></td>'
            f'<td {_cell}>{trades}</td>'
            '</tr><tr>'
            f'<td {_cell}><strong>Max drawdown</strong></td>'
            f'<td {_cell}>-{max_dd:.1f}%</td>'
            '</tr>'
            '</table>'
        )

    return (
        "<html>"
        '<body style="font-family:Arial,sans-serif;max-width:800px;'
        'margin:0 auto;padding:20px">'
        "<h1>Crypto Trading Bot — Entrenamiento del Modelo</h1>"
        f"<p>{now}</p>"
        f"{metrics_html}"
        f"{models_html}"
        f"{features_html}"
        f"{tpsl_html}"
        "<hr>"
        '<p style="color:#999;font-size:12px">'
        "Generado automáticamente por Crypto Trading Bot. "
        "Esto no es asesoramiento financiero."
        "</p>"
        "</body>"
        "</html>"
    )


def _build_market_regime_section(
    market_regime: MarketRegimeResult | None,
) -> str:
    """Genera la sección HTML del filtro de régimen de mercado."""
    if market_regime is None:
        return (
            '<h2>Régimen de Mercado</h2>'
            '<p style="color:#999">Sin datos de BTC suficientes para evaluar.</p>'
        )

    if market_regime.allow_buys:
        status_label = "FAVORABLE"
        status_color = "#28a745"
        status_bg = "#d4edda"
    else:
        status_label = "ADVERSO — compras bloqueadas"
        status_color = "#dc3545"
        status_bg = "#f8d7da"

    _cell = 'style="padding:6px;border:1px solid #ddd"'

    reasons_html = ""
    if market_regime.reasons:
        items = "".join(f"<li>{r}</li>" for r in market_regime.reasons)
        reasons_html = (
            '<tr>'
            f'<td {_cell}><strong>Razones de bloqueo</strong></td>'
            f'<td {_cell} style="color:#dc3545">{items}</td>'
            '</tr>'
        )

    return (
        '<h2>Régimen de Mercado</h2>'
        f'<div style="background:{status_bg};padding:10px;border-radius:6px;'
        f'margin-bottom:10px">'
        f'<strong style="color:{status_color};font-size:16px">{status_label}</strong>'
        '</div>'
        '<table style="border-collapse:collapse;width:100%">'
        '<tr>'
        f'<td {_cell}><strong>BTC ROC 24h</strong></td>'
        f'<td {_cell}>{market_regime.btc_roc_24h:+.1%}</td>'
        '</tr><tr>'
        f'<td {_cell}><strong>BTC RSI 14</strong></td>'
        f'<td {_cell}>{market_regime.btc_rsi_14:.1f}</td>'
        '</tr><tr>'
        f'<td {_cell}><strong>Monedas subiendo 24h</strong></td>'
        f'<td {_cell}>{market_regime.pct_coins_up_24h:.0%}</td>'
        '</tr>'
        f'{reasons_html}'
        '</table>'
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
    model_info: dict[str, object] | None = None,
    market_regime: MarketRegimeResult | None = None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pnl = total_value_after - total_value_before
    pnl_pct = (pnl / total_value_before * 100) if total_value_before > 0 else 0
    pnl_color = "#28a745" if pnl >= 0 else "#dc3545"

    # Tabla de operaciones
    ops_rows = ""
    for r in results:
        status = "OK" if r.success else f"ERROR: {r.error}"
        color = "#28a745" if r.success else "#dc3545"
        ops_rows += f"""
        <tr>
            <td>{r.action.action}</td>
            <td>{r.action.symbol}</td>
            <td>{r.executed_qty:.6f}</td>
            <td>${r.executed_price:.4f}</td>
            <td>{r.action.probability:.1%}</td>
            <td style="color:{color}">{status}</td>
            <td>{r.action.reason}</td>
        </tr>"""

    # Top predicciones
    top_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:20]
    pred_rows = ""
    for symbol, prob in top_preds:
        bar_width = int(prob * 100)
        bar_color = "#28a745" if prob >= 0.70 else "#ffc107" if prob >= 0.50 else "#dc3545"
        pred_rows += f"""
        <tr>
            <td>{symbol}</td>
            <td>{prob:.1%}</td>
            <td>
                <div style="background:#eee;border-radius:4px;overflow:hidden;width:200px">
                    <div style="background:{bar_color};height:16px;width:{bar_width}%"></div>
                </div>
            </td>
        </tr>"""

    mode_badge = (
        '<span style="background:#ffc107;color:#000;padding:2px 8px;border-radius:4px;'
        'font-weight:bold">PAPER TRADING</span>'
        if is_paper
        else '<span style="background:#28a745;color:#fff;padding:2px 8px;border-radius:4px;'
        'font-weight:bold">LIVE</span>'
    )

    no_ops_msg = "<p>No se ejecutaron operaciones hoy.</p>"
    no_preds_msg = "<p>Sin predicciones disponibles (modelo no entrenado o sin datos).</p>"

    if results:
        ops_section = (
            '<table style="border-collapse:collapse;width:100%">'
            '<tr style="background:#f8f9fa">'
            '<th style="padding:8px;border:1px solid #ddd">Accion</th>'
            '<th style="padding:8px;border:1px solid #ddd">Moneda</th>'
            '<th style="padding:8px;border:1px solid #ddd">Cantidad</th>'
            '<th style="padding:8px;border:1px solid #ddd">Precio</th>'
            '<th style="padding:8px;border:1px solid #ddd">Prob.</th>'
            '<th style="padding:8px;border:1px solid #ddd">Estado</th>'
            '<th style="padding:8px;border:1px solid #ddd">Razon</th>'
            "</tr>"
            f"{ops_rows}"
            "</table>"
        )
    else:
        ops_section = no_ops_msg

    if predictions:
        preds_section = (
            '<table style="border-collapse:collapse;width:100%">'
            '<tr style="background:#f8f9fa">'
            '<th style="padding:8px;border:1px solid #ddd">Moneda</th>'
            '<th style="padding:8px;border:1px solid #ddd">Prob. subida &gt;2%</th>'
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

    return (
        "<html>"
        '<body style="font-family:Arial,sans-serif;max-width:800px;margin:0 auto;padding:20px">'
        "<h1>Crypto Trading Bot — Reporte Diario</h1>"
        f"<p>{now} | {mode_badge}</p>"
        "<h2>Resumen</h2>"
        '<table style="border-collapse:collapse;width:100%">'
        "<tr>"
        '<td style="padding:8px;border:1px solid #ddd"><strong>Balance anterior</strong></td>'
        f'<td style="padding:8px;border:1px solid #ddd">${total_value_before:,.2f}</td>'
        "</tr>"
        "<tr>"
        '<td style="padding:8px;border:1px solid #ddd"><strong>Balance actual</strong></td>'
        f'<td style="padding:8px;border:1px solid #ddd">${total_value_after:,.2f}</td>'
        "</tr>"
        "<tr>"
        '<td style="padding:8px;border:1px solid #ddd"><strong>P&amp;L</strong></td>'
        f'<td style="padding:8px;border:1px solid #ddd;color:{pnl_color};font-weight:bold">'
        f"{pnl_sign}${pnl:,.2f} ({pnl_pct:+.2f}%)"
        "</td>"
        "</tr>"
        "</table>"
        f"<h2>Operaciones Ejecutadas ({len(results)})</h2>"
        f"{ops_section}"
        f"{_build_dca_section(dca_summary or {}, dca_actions or [])}"
        f"{_build_momentum_section(momentum_summary or {})}"
        f"{_build_allocation_section(allocation_budgets or {})}"
        f"{_build_market_regime_section(market_regime)}"
        f"{_build_model_info_section(model_info or {})}"
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
        "Generado automaticamente por Crypto Trading Bot. "
        "Esto no es asesoramiento financiero."
        "</p>"
        "</body>"
        "</html>"
    )
