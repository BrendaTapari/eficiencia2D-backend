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

# NOTA SOBRE SUDO: las reglas de /etc/sudoers.d matchean la línea de comando
# COMPLETA, flags incluidos. Por eso acá se invoca systemctl en su forma más
# simple (sin --quiet/--no-pager): cualquier flag extra no matchearía la regla y
# sudo pediría contraseña, cosa imposible desde GitHub Actions. Se usa `sudo -n`
# (no interactivo) para que, si falta el permiso, falle al instante con un error
# claro en vez de quedar colgado esperando una contraseña.

# 1. APAGAR ------------------------------------------------------------------
# Se detiene antes de tocar dependencias para no cambiarle el entorno a un
# proceso en ejecución. `|| true`: si aún no está instalado/activo, no aborta.
echo "==> Deteniendo el servicio..."
sudo -n systemctl stop "$SERVICE" || true

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
sudo -n systemctl daemon-reload
sudo -n systemctl start "$SERVICE"

# 4. VERIFICAR ---------------------------------------------------------------
# Margen para que uvicorn levante y conecte a la base antes de consultar estado.
sleep 5
# Sin --quiet: se compara la salida, así la regla de sudoers matchea exacto.
ESTADO="$(sudo -n systemctl is-active "$SERVICE" || true)"
if [[ "$ESTADO" == "active" ]]; then
  echo "==> OK: $SERVICE está corriendo"
else
  echo "==> ERROR: $SERVICE no arrancó (estado: $ESTADO). Últimos logs:" >&2
  sudo -n journalctl -u "$SERVICE" -n 40 --no-pager >&2 || true
  exit 1
fi
