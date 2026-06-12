from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.uploads import router as uploads_router

app = FastAPI(
    title="Eficiencia2D Backend API",
    description="API para procesamiento de modelos arquitectónicos 3D a planos 2D",
    version="1.0.0"
)

# Configuración CORS para permitir peticiones desde el frontend (React/Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, reemplazar "*" por el dominio de tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir las rutas
app.include_router(uploads_router, prefix="/api", tags=["Procesamiento"])

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Eficiencia2D Backend"}

# Para correr localmente para pruebas sin la línea de comandos
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8081, reload=True)
