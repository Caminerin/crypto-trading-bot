# Crypto Trading Bot

Bot de trading automático para Binance Spot que cada 12 horas analiza las 75 monedas con mayor volumen, predice cuáles subirán >5% en 48h, y gestiona tu cartera con 3 estrategias.

## Cómo funciona

1. **Datos**: Descarga velas de las top 75 monedas por volumen (pares USDC) de Binance.
2. **Modelo**: Un ensemble de Machine Learning (LightGBM + XGBoost + RandomForest + ExtraTrees con meta-learner LogisticRegression) calcula la probabilidad de que cada moneda suba >5% en las próximas 48h.
3. **Cartera**: Si alguna moneda supera el 65% de confianza, el bot compra y coloca una OCO (TP/SL automáticos).
4. **Riesgo**: Cada compra lleva un stop-loss (5%) y take-profit (5%) automáticos vía orden OCO.
5. **Expiración**: Si pasan 48h sin que salte TP ni SL, el bot vende a mercado.
6. **Reporte**: Te envía un email con todo lo que hizo.

## Estrategias

| Estrategia | Descripción | Asignación |
|---|---|---|
| **Prediction (ML)** | Compra altcoins que el modelo predice que subirán >5% en 48h | 35% |
| **DCA Inteligente** | Compra BTC/ETH/BNB en caídas fuertes | 20% |
| **Momentum** | Compra BTC/ETH/BNB/SOL/XRP en tendencia alcista | 35% |
| **Reserva** | Colchón intocable | 10% |

## Reglas de la cartera (Prediction)

| Parámetro | Valor |
|---|---|
| Máx. posiciones simultáneas | 5 |
| Máx. % por moneda | 20% |
| Stop-loss | 5% |
| Take-profit | 5% |
| Expiración | 48h |
| Tipo de orden | Market + OCO |

## Requisitos

- Python 3.11+
- Cuenta de Binance con API keys (solo lectura + spot trading)
- Cuenta de Mailjet para emails

## Instalación

```bash
# Clonar el repo
git clone https://github.com/Caminerin/crypto-trading-bot.git
cd crypto-trading-bot

# Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate
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
MAILJET_API_KEY=tu_mailjet_key
MAILJET_API_SECRET=tu_mailjet_secret
EMAIL_FROM=tu@email.com
EMAIL_TO=tu@email.com
QUOTE_ASSET=USDC
TRADING_MODE=paper
CONFIDENCE_THRESHOLD=0.65
```

> **IMPORTANTE**: Empieza siempre con `TRADING_MODE=paper` para probar sin dinero real.

## Uso manual

```bash
# Ejecutar el bot (usa el modo de .env)
python -m src.main

# Solo entrenar el modelo (no opera)
python -m src.main train
```

## Ejecución automática (cron en servidor)

El bot se ejecuta vía cron en un VPS:

- **07:00 y 19:00 hora Madrid**: Analiza el mercado y opera.
- **06:00 domingos hora Madrid**: Re-entrena el modelo con datos nuevos.

### Ejemplo de crontab

```
# Bot trading (07:00 y 19:00 Madrid)
0 5,17 * * * cd /root/crypto-trading-bot && source .venv/bin/activate && set -a && source .env && set +a && python -m src.main >> logs/cron.log 2>&1

# Re-entrenamiento semanal (domingos 06:00 Madrid)
0 4 * * 0 cd /root/crypto-trading-bot && source .venv/bin/activate && set -a && source .env && set +a && python -m src.main train >> logs/retrain.log 2>&1
```

## Estructura del proyecto

```
crypto-trading-bot/
  src/
    config.py              # Configuración central
    main.py                # Orquestador principal
    data/
      binance_client.py    # Conexión a Binance API
      features.py          # Cálculo de indicadores técnicos
    model/
      predictor.py         # Modelo predictivo (Stacking Ensemble)
    portfolio/
      manager.py           # Gestión de cartera
    allocation/
      allocator.py         # Reparto entre estrategias
    execution/
      executor.py          # Ejecución de órdenes
    strategies/
      prediction_book.py   # Inventario aislado de Prediction
      dca.py               # Estrategia DCA Inteligente
      momentum.py          # Estrategia Momentum
    notifications/
      email_report.py      # Reportes por email (Mailjet)
    utils/
      logger.py            # Logging
  scripts/
    backtest_prediction.py # Backtest del modelo
  data/                    # JSONs de posiciones y asignación
  models/                  # Modelo entrenado (.joblib)
  logs/                    # Logs de ejecución
  tests/
```

## Modo Paper Trading

El bot arranca en modo simulación por defecto. En este modo:
- NO ejecuta órdenes reales en Binance.
- SÍ analiza el mercado y genera predicciones.
- SÍ envía el reporte por email (para que veas qué haría).
- Usa precios reales de Binance para simular ejecuciones realistas.

Cuando estés satisfecho con los resultados, cambia `TRADING_MODE=live`.

## Disclaimer

Esto no es asesoramiento financiero. El trading de criptomonedas conlleva riesgos significativos. Usa este bot bajo tu propia responsabilidad.
