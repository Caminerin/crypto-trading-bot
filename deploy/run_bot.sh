#!/usr/bin/env bash
# Script ejecutado por cron para correr el bot
set -euo pipefail

APP_DIR="$HOME/crypto-trading-bot"
cd "$APP_DIR"

# Cargar variables de entorno
set -a
source "$APP_DIR/.env"
set +a

# Activar entorno virtual y ejecutar
source "$APP_DIR/.venv/bin/activate"

# Actualizar codigo (por si hay cambios)
git pull origin main --quiet 2>/dev/null || true

# Ejecutar el bot
python -m src.main >> "$APP_DIR/logs/cron_$(date +%Y%m%d_%H%M).log" 2>&1
