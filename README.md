# Crypto Trading Bot

Bot de trading automático para Binance Spot que cada mañana analiza las 200 monedas con mayor volumen, predice cuáles subirán >2% en 24h, y optimiza tu cartera.

## Cómo funciona

1. **Datos**: Descarga velas de las top 200 monedas por volumen (pares USDT) de Binance.
2. **Modelo**: Un modelo de Machine Learning (LightGBM) calcula la probabilidad de que cada moneda suba >2% en las próximas 24h.
3. **Cartera**: Si alguna moneda supera el 70% de confianza, el bot ajusta tu cartera (vende lo que ya no cumple, compra lo nuevo).
4. **Riesgo**: Cada compra lleva un stop-loss (3%) y take-profit (5%) automáticos.
5. **Reporte**: Te envía un email con todo lo que hizo.

## Reglas de la cartera

| Parámetro | Valor |
|---|---|
| Máx. posiciones simultáneas | 5 |
| Máx. % por moneda | 20% |
| Reserva mínima en USDT | 10% |
| Stop-loss | 5% |
| Take-profit | 3% |
| Tipo de orden | Market |

## Requisitos

- Python 3.11+
- Cuenta de Binance con API keys (solo lectura + spot trading)
- Cuenta de SendGrid (gratis) para emails

## Instalación

```bash
# Clonar el repo
git clone https://github.com/TU_USUARIO/crypto-trading-bot.git
cd crypto-trading-bot

# Instalar dependencias
pip install .

# Copiar configuración
cp .env.example .env
# Editar .env con tus API keys
```

## Configuración

Edita el fichero `.env` con tus datos:

```
BINANCE_API_KEY=tu_api_key
BINANCE_API_SECRET=tu_api_secret
SENDGRID_API_KEY=tu_sendgrid_key
EMAIL_FROM=bot@tudominio.com
EMAIL_TO=tu@email.com
TRADING_MODE=paper
```

> **IMPORTANTE**: Empieza siempre con `TRADING_MODE=paper` para probar sin dinero real.

## Uso manual

```bash
# Ejecutar el bot (usa el modo de .env)
python -m src.main

# Solo entrenar el modelo (no opera)
python -m src.main train
```

## Ejecución automática (GitHub Actions)

El bot se ejecuta solo gracias a GitHub Actions:

- **Diario a las 07:00 UTC**: Analiza el mercado y opera.
- **Domingos a las 06:00 UTC**: Re-entrena el modelo con datos nuevos.

### Configurar los secrets en GitHub

Ve a tu repositorio > Settings > Secrets and variables > Actions y añade:

| Secret | Descripción |
|---|---|
| `BINANCE_API_KEY` | Tu API key de Binance |
| `BINANCE_API_SECRET` | Tu API secret de Binance |
| `SENDGRID_API_KEY` | Tu API key de SendGrid |
| `EMAIL_FROM` | Email remitente (configurado en SendGrid) |
| `EMAIL_TO` | Tu email personal (donde recibir reportes) |
| `TRADING_MODE` | `paper` para simulación, `live` para real |

## Estructura del proyecto

```
crypto-trading-bot/
  src/
    config.py           # Configuración central
    main.py             # Orquestador principal
    data/
      binance_client.py # Conexión a Binance API
      features.py       # Cálculo de indicadores técnicos
    model/
      predictor.py      # Modelo predictivo (LightGBM)
    portfolio/
      manager.py        # Gestión de cartera
    execution/
      executor.py       # Ejecución de órdenes
    notifications/
      email_report.py   # Reportes por email
    utils/
      logger.py         # Logging
  .github/workflows/
    daily_trading.yml   # Cron diario
    retrain_weekly.yml  # Re-entrenamiento semanal
  tests/
```

## Modo Paper Trading

El bot arranca en modo simulación por defecto. En este modo:
- NO ejecuta órdenes reales en Binance.
- SÍ analiza el mercado y genera predicciones.
- SÍ envía el reporte por email (para que veas qué haría).
- Usa una cartera simulada de 1000 USDT.

Cuando estés satisfecho con los resultados, cambia `TRADING_MODE=live`.

## Disclaimer

Esto no es asesoramiento financiero. El trading de criptomonedas conlleva riesgos significativos. Usa este bot bajo tu propia responsabilidad.
