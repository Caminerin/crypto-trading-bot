#!/usr/bin/env bash
# =============================================================
# Setup script para Oracle Cloud Always Free VM (Ubuntu)
# Configura el crypto-trading-bot para ejecucion automatica.
#
# Uso:
#   chmod +x deploy/setup_oracle.sh
#   ./deploy/setup_oracle.sh
# =============================================================
set -euo pipefail

REPO_URL="https://github.com/Caminerin/crypto-trading-bot.git"
APP_DIR="$HOME/crypto-trading-bot"
PYTHON_VERSION="3.11"

echo "========================================"
echo " Crypto Trading Bot — Setup Oracle Cloud"
echo "========================================"

# ------------------------------------------------------------------
# 1. Instalar dependencias del sistema
# ------------------------------------------------------------------
echo ""
echo "[1/6] Instalando dependencias del sistema..."
sudo apt-get update -qq
sudo apt-get install -y -qq python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev git cron

# ------------------------------------------------------------------
# 2. Clonar o actualizar el repositorio
# ------------------------------------------------------------------
echo ""
echo "[2/6] Clonando repositorio..."
if [ -d "$APP_DIR" ]; then
    echo "  Repositorio ya existe, actualizando..."
    cd "$APP_DIR"
    git pull origin main
else
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

# ------------------------------------------------------------------
# 3. Crear entorno virtual e instalar dependencias
# ------------------------------------------------------------------
echo ""
echo "[3/6] Creando entorno virtual e instalando dependencias..."
python${PYTHON_VERSION} -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install . -q
echo "  Dependencias instaladas correctamente."

# ------------------------------------------------------------------
# 4. Configurar variables de entorno
# ------------------------------------------------------------------
echo ""
echo "[4/6] Configurando variables de entorno..."
ENV_FILE="$APP_DIR/.env"

if [ -f "$ENV_FILE" ]; then
    echo "  .env ya existe. Si quieres reconfigurarlo, edita: $ENV_FILE"
else
    echo "  Necesito tus credenciales para configurar el bot."
    echo ""

    read -rp "  BINANCE_API_KEY: " BINANCE_API_KEY
    read -rsp "  BINANCE_API_SECRET: " BINANCE_API_SECRET
    echo ""
    read -rp "  MAILJET_API_KEY: " MAILJET_API_KEY
    read -rsp "  MAILJET_API_SECRET: " MAILJET_API_SECRET
    echo ""
    read -rp "  EMAIL_FROM (ej: caminerin@gmail.com): " EMAIL_FROM
    read -rp "  EMAIL_TO (ej: caminerin@gmail.com): " EMAIL_TO
    read -rp "  TRADING_MODE (paper/live) [paper]: " TRADING_MODE
    TRADING_MODE=${TRADING_MODE:-paper}

    cat > "$ENV_FILE" <<EOF
# Binance API
BINANCE_API_KEY=${BINANCE_API_KEY}
BINANCE_API_SECRET=${BINANCE_API_SECRET}

# Mailjet
MAILJET_API_KEY=${MAILJET_API_KEY}
MAILJET_API_SECRET=${MAILJET_API_SECRET}
EMAIL_FROM=${EMAIL_FROM}
EMAIL_TO=${EMAIL_TO}

# Modo de ejecucion
TRADING_MODE=${TRADING_MODE}
EOF

    chmod 600 "$ENV_FILE"
    echo "  .env creado y protegido (solo tu usuario puede leerlo)."
fi

# ------------------------------------------------------------------
# 5. Descargar modelo entrenado (si existe en el repo como artifact)
# ------------------------------------------------------------------
echo ""
echo "[5/6] Preparando directorio de modelos..."
mkdir -p "$APP_DIR/models"
mkdir -p "$APP_DIR/logs"

# ------------------------------------------------------------------
# 6. Configurar cron jobs
# ------------------------------------------------------------------
echo ""
echo "[6/6] Configurando ejecucion automatica (cron)..."

CRON_SCRIPT="$APP_DIR/deploy/run_bot.sh"

# Crear script wrapper que carga el entorno
cat > "$CRON_SCRIPT" <<'SCRIPT'
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
SCRIPT

chmod +x "$CRON_SCRIPT"

# Instalar cron jobs (07:00 y 19:00 UTC)
CRON_LINE_07="0 7 * * * $CRON_SCRIPT"
CRON_LINE_19="0 19 * * * $CRON_SCRIPT"

# Agregar al crontab sin duplicar
(crontab -l 2>/dev/null | grep -v "$CRON_SCRIPT"; echo "$CRON_LINE_07"; echo "$CRON_LINE_19") | crontab -
echo "  Cron configurado:"
echo "    - 07:00 UTC (09:00 hora Espana)"
echo "    - 19:00 UTC (21:00 hora Espana)"

echo ""
echo "========================================"
echo " SETUP COMPLETADO"
echo "========================================"
echo ""
echo " El bot se ejecutara automaticamente a las:"
echo "   - 07:00 UTC (09:00 hora Espana)"
echo "   - 19:00 UTC (21:00 hora Espana)"
echo ""
echo " Comandos utiles:"
echo "   - Ejecutar ahora:    cd $APP_DIR && source .venv/bin/activate && source .env && python -m src.main"
echo "   - Ver logs:          ls $APP_DIR/logs/"
echo "   - Editar config:     nano $APP_DIR/.env"
echo "   - Ver cron jobs:     crontab -l"
echo "   - Actualizar codigo: cd $APP_DIR && git pull origin main"
echo ""
echo " IMPORTANTE: Para pasar a live trading, edita .env y cambia TRADING_MODE=live"
echo ""
