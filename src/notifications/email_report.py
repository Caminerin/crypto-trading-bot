"""
Generación y envío de reportes por email vía SendGrid.

Construye un email HTML con:
- Resumen de cartera (antes y después).
- Operaciones ejecutadas.
- Predicciones del modelo.
- P&L estimado.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.config import EmailConfig
from src.execution.executor import ExecutionResult
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
) -> bool:
    """Envía el reporte diario por email.

    Devuelve True si el envío fue exitoso.
    """
    if not config.sendgrid_api_key:
        logger.warning("SendGrid API key no configurada. No se envía email.")
        return False

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Content, Mail
    except ImportError:
        logger.error("sendgrid no está instalado. Ejecuta: pip install sendgrid")
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
    )

    message = Mail(
        from_email=config.email_from,
        to_emails=config.email_to,
        subject=subject,
        html_content=Content("text/html", html_body),
    )

    try:
        sg = SendGridAPIClient(config.sendgrid_api_key)
        response = sg.send(message)
        logger.info("Email enviado — status=%d", response.status_code)
        return response.status_code in (200, 201, 202)
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


def _build_html_body(
    portfolio_before: dict[str, float],
    portfolio_after: dict[str, float],
    total_value_before: float,
    total_value_after: float,
    results: list[ExecutionResult],
    predictions: dict[str, float],
    is_paper: bool,
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

    return f"""
    <html>
    <body style="font-family:Arial,sans-serif;max-width:800px;margin:0 auto;padding:20px">
        <h1>Crypto Trading Bot — Reporte Diario</h1>
        <p>{now} | {mode_badge}</p>

        <h2>Resumen</h2>
        <table style="border-collapse:collapse;width:100%">
            <tr>
                <td style="padding:8px;border:1px solid #ddd"><strong>Balance anterior</strong></td>
                <td style="padding:8px;border:1px solid #ddd">${total_value_before:,.2f}</td>
            </tr>
            <tr>
                <td style="padding:8px;border:1px solid #ddd"><strong>Balance actual</strong></td>
                <td style="padding:8px;border:1px solid #ddd">${total_value_after:,.2f}</td>
            </tr>
            <tr>
                <td style="padding:8px;border:1px solid #ddd"><strong>P&L</strong></td>
                <td style="padding:8px;border:1px solid #ddd;color:{pnl_color};font-weight:bold">
                    {"+" if pnl >= 0 else ""}${pnl:,.2f} ({pnl_pct:+.2f}%)
                </td>
            </tr>
        </table>

        <h2>Operaciones Ejecutadas ({len(results)})</h2>
        {"<p>No se ejecutaron operaciones hoy.</p>" if not results else f'''
        <table style="border-collapse:collapse;width:100%">
            <tr style="background:#f8f9fa">
                <th style="padding:8px;border:1px solid #ddd">Accion</th>
                <th style="padding:8px;border:1px solid #ddd">Moneda</th>
                <th style="padding:8px;border:1px solid #ddd">Cantidad</th>
                <th style="padding:8px;border:1px solid #ddd">Precio</th>
                <th style="padding:8px;border:1px solid #ddd">Prob.</th>
                <th style="padding:8px;border:1px solid #ddd">Estado</th>
                <th style="padding:8px;border:1px solid #ddd">Razon</th>
            </tr>
            {ops_rows}
        </table>
        '''}

        <h2>Top 20 Predicciones</h2>
        <table style="border-collapse:collapse;width:100%">
            <tr style="background:#f8f9fa">
                <th style="padding:8px;border:1px solid #ddd">Moneda</th>
                <th style="padding:8px;border:1px solid #ddd">Prob. subida &gt;2%</th>
                <th style="padding:8px;border:1px solid #ddd">Confianza</th>
            </tr>
            {pred_rows}
        </table>

        <h2>Cartera Actual</h2>
        <table style="border-collapse:collapse;width:100%">
            <tr style="background:#f8f9fa">
                <th style="padding:8px;border:1px solid #ddd">Activo</th>
                <th style="padding:8px;border:1px solid #ddd">Cantidad</th>
            </tr>
            {"".join(
                f'<tr><td style="padding:8px;border:1px solid #ddd">{asset}</td>'
                f'<td style="padding:8px;border:1px solid #ddd">{qty:.8f}</td></tr>'
                for asset, qty in sorted(portfolio_after.items())
                if qty > 0
            )}
        </table>

        <hr>
        <p style="color:#999;font-size:12px">
            Generado automaticamente por Crypto Trading Bot.
            Esto no es asesoramiento financiero.
        </p>
    </body>
    </html>
    """
