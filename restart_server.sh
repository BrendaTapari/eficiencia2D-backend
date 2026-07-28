#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reinicio del backend Eficiencia2D tras un despliegue (lo invoca el workflow
# .github/workflows/deploy.yml por SSH, después del git pull).
#
# NOTA SOBRE "COMPILAR": el backend es Python/FastAPI, no hay binario que
# compilar. El equivalente es actualizar las dependencias del entorno virtual y
# pre-compilar el bytecode (.pyc) para que el primer request no pague ese costo.
# ---------------------------------------------------------------------------
set -euo pipefail

# Ruta del proyecto = carpeta donde vive este script (no depende del CWD).
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

SERVICE="${SERVICE_NAME:-eficiencia2d}"
VENV="${VENV_DIR:-$PROJECT_DIR/venv}"

echo "==> Desplegando en $PROJECT_DIR (servicio: $SERVICE)"

# 1. APAGAR ------------------------------------------------------------------
# Se detiene antes de tocar dependencias para no cambiarle el entorno a un
# proceso en ejecución. `|| true`: si aún no está instalado/activo, no aborta.
echo "==> Deteniendo el servicio..."
sudo systemctl stop "$SERVICE" || true

# 2. "COMPILAR" (dependencias + bytecode) ------------------------------------
if [[ ! -d "$VENV" ]]; then
  echo "==> Creando entorno virtual en $VENV"
  python3 -m venv "$VENV"
fi

echo "==> Instalando dependencias..."
"$VENV/bin/pip" install --upgrade pip --quiet
"$VENV/bin/pip" install -r requirements-server.txt --quiet

echo "==> Pre-compilando bytecode..."
# -q: silencioso. Falla el build si hay un error de sintaxis, así se detecta acá
# y no con el servicio ya arrancando.
"$VENV/bin/python" -m compileall -q api core database utils main.py

# 3. ARRANCAR ----------------------------------------------------------------
echo "==> Arrancando el servicio..."
sudo systemctl daemon-reload
sudo systemctl start "$SERVICE"

# 4. VERIFICAR ---------------------------------------------------------------
# Margen para que uvicorn levante antes de consultar el estado.
sleep 3
if sudo systemctl is-active --quiet "$SERVICE"; then
  echo "==> OK: $SERVICE está corriendo"
  sudo systemctl status "$SERVICE" --no-pager --lines=5 || true
else
  echo "==> ERROR: $SERVICE no arrancó. Últimos logs:" >&2
  sudo journalctl -u "$SERVICE" --no-pager --lines=40 >&2
  exit 1
fi
