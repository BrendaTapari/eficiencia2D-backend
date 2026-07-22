import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Cargar variables de entorno (.env) y configurar logging una sola vez,
# antes de importar/usar los módulos del pipeline.
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from database import init_db
    from utils.mailer import validate_mail_config

    try:
        init_db()
    except Exception:
        logger.exception("No se pudieron crear las tablas en PostgreSQL")
        raise

    mail_issues = validate_mail_config()
    if mail_issues:
        logger.warning("Correo no configurado correctamente: %s", "; ".join(mail_issues))
    else:
        logger.info("Configuración de correo OK")

    yield


from api.routes.auth import router as auth_router
from api.routes.coupons import router as coupons_router
from api.routes.projects import router as projects_router
from api.routes.settings import router as settings_router
from api.routes.uploads import router as uploads_router
from api.routes.users import router as users_router
from api.routes.planes import router as planes_router
from api.routes.suscripciones import router as suscripciones_router
from api.routes.mp_webhooks import router as mp_webhooks_router

app = FastAPI(
    title="Eficiencia2D Backend API",
    description="API para procesamiento de modelos arquitectónicos 3D a planos 2D",
    version="1.0.0",
    lifespan=lifespan,
)

# Configuración CORS por variables de entorno (se combinan, no se elige solo una):
#   ALLOWED_ORIGINS — lista separada por comas
#   FRONTEND_URL / FRONTEND_URL_VERCEL — un origen cada una
#   CORS_EXTRA_ORIGINS — orígenes adicionales (ej. http://localhost:3000)
# Por defecto se permiten localhost:3000 y 127.0.0.1:3000 para desarrollo local.
# Desactivar con DISABLE_LOCALHOST_CORS=true en producción estricta.
_LOCALHOST_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


def _resolve_cors_origins() -> list[str]:
    parts: list[str] = []
    for key in (
        "ALLOWED_ORIGINS",
        "FRONTEND_URL",
        "FRONTEND_URL_VERCEL",
        "CORS_EXTRA_ORIGINS",
    ):
        raw = os.environ.get(key, "")
        parts.extend(o.strip() for o in raw.split(",") if o.strip())

    if os.environ.get("DISABLE_LOCALHOST_CORS", "").lower() not in ("1", "true", "yes"):
        parts.extend(_LOCALHOST_ORIGINS)

    seen: set[str] = set()
    origins: list[str] = []
    for origin in parts:
        if origin not in seen:
            seen.add(origin)
            origins.append(origin)
    return origins


_cors_origins = _resolve_cors_origins()
if _cors_origins:
    allow_origins = _cors_origins
    allow_credentials = True
    logger.info("CORS orígenes permitidos: %s", ", ".join(allow_origins))
else:
    allow_origins = ["*"]
    allow_credentials = False
    logger.info("CORS: allow_origins=* (sin credenciales)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Comprime la respuesta (los buffers base64 de geometría comprimen bien) cuando
# el cliente envía Accept-Encoding: gzip — el navegador la descomprime solo.
# compresslevel=1: ~21MB en ~1s para un modelo de 50MB (nivel 9 tardaría ~5-8s
# para apenas ~4MB menos; no compensa en una respuesta de 100+MB).
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=1)

# Incluir las rutas
app.include_router(auth_router, prefix="/api", tags=["Auth"])
app.include_router(coupons_router, prefix="/api", tags=["Cupones"])
app.include_router(users_router, prefix="/api", tags=["Usuarios"])
app.include_router(projects_router, prefix="/api", tags=["Proyectos"])
app.include_router(settings_router, prefix="/api", tags=["Configuración"])
app.include_router(uploads_router, prefix="/api", tags=["Procesamiento"])
app.include_router(planes_router, prefix="/api", tags=["Planes"])
app.include_router(suscripciones_router, prefix="/api", tags=["Suscripciones"])
app.include_router(mp_webhooks_router, prefix="/api", tags=["Webhooks"])

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Eficiencia2D Backend"}

# Para correr localmente para pruebas sin la línea de comandos
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8081, reload=True)
