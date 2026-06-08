import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from api.routes import uploads

load_dotenv()

app = FastAPI(title="3D a 2D")

FRONTEND_URL = os.getenv("FRONTEND_URL",)

# Configuración del Middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],            # Permite peticiones solo desde los orígenes de la lista
    allow_credentials=True,           # Permite el envío de cookies/tokens si fuera necesario
    allow_methods=["*"],              # Permite todos los métodos HTTP (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],              # Permite todos los encabezados HTTP
)

app.include_router(uploads.router, prefix="/api", tags=["Procesamiento de Modelos"])

@app.get("/")
def read_root():
    return {
        "mensaje": "Servidor láser inicializado.",
        "cors_configurado_para": FRONTEND_URL
    }