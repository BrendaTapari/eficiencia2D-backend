import os
import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Cargar variables de entorno (.env) y configurar logging una sola vez,
# antes de importar/usar los módulos del pipeline.
load_dotenv()
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)

from api.routes.uploads import router as uploads_router
from api.routes.planes import router as planes_router
from api.routes.suscripciones import router as suscripciones_router
from api.routes.mp_webhooks import router as mp_webhooks_router
from database.database import init_db

app = FastAPI(
    title="Eficiencia2D Backend API",
    description="API para procesamiento de modelos arquitectónicos 3D a planos 2D",
    version="1.0.0"
)

# Configuración CORS por variables de entorno.
#   ALLOWED_ORIGINS: lista separada por comas (tiene prioridad)
#   FRONTEND_URL: un único origen
# Si no hay orígenes configurados, se usa "*" SIN credenciales (allow_credentials
# y allow_origins=["*"] son incompatibles según la especificación CORS).
def _resolve_cors_origins() -> list[str]:
    raw = os.environ.get("ALLOWED_ORIGINS") or os.environ.get("FRONTEND_URL", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    vercel = os.environ.get("FRONTEND_URL_VERCEL", "")
    if vercel and vercel not in origins:
        origins.append(vercel)
    return origins

_cors_origins = _resolve_cors_origins()
if _cors_origins:
    allow_origins = _cors_origins
    allow_credentials = True
else:
    allow_origins = ["*"]
    allow_credentials = False

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
app.include_router(uploads_router, prefix="/api", tags=["Procesamiento"])
app.include_router(planes_router, prefix="/api", tags=["Planes"])
app.include_router(suscripciones_router, prefix="/api", tags=["Suscripciones"])
app.include_router(mp_webhooks_router, prefix="/api", tags=["Webhooks"])

@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Eficiencia2D Backend"}

# Para correr localmente para pruebas sin la línea de comandos
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
