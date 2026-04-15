#!/usr/bin/env bash
# Arranca el dashboard web del bot de trading.
# Uso: bash deploy/run_dashboard.sh
#
# El dashboard es de solo lectura — no modifica datos del bot.
# Accesible en http://<IP_DEL_SERVIDOR>:8080

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activar entorno virtual si existe
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Cargar variables de entorno si existe .env
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

export FLASK_APP=src.dashboard.app
export FLASK_ENV=production

echo "Iniciando dashboard en http://0.0.0.0:8080 ..."
flask run --host=0.0.0.0 --port=8080
