#!/usr/bin/env bash
# Arranque del backend Eficiencia2D en Linux (Debian / Google Cloud VM).
# Equivalente Linux de run.ps1. Uso:
#   ./start-server.sh              # producción: 0.0.0.0:80
#   PORT=8081 RELOAD=true ./start-server.sh   # desarrollo local
set -euo pipefail
cd "$(dirname "$0")"

# Activar el entorno virtual si existe (creado con: python3 -m venv venv)
if [[ -f venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-80}"

# Un worker: el pipeline de geometría es CPU-intensivo y usa caché en disco por file_id.
# Para escalar, subir workers detrás de nginx (ver deploy/README_DEPLOY.md).
exec uvicorn main:app --host "$HOST" --port "$PORT" --workers "${WORKERS:-1}"
