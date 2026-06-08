import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException

# Inicializamos el router para esta sección de la API
router = APIRouter()

# Definimos y aseguramos que la ruta temporal exista
UPLOAD_DIR = "temp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    """
    Recibe un archivo 3D, lo valida y lo guarda temporalmente en disco.
    """
    # 1. Validación básica de seguridad (solo aceptamos STL o OBJ por ahora)
    extensiones_permitidas = ('.stl', '.obj')
    if not file.filename.lower().endswith(extensiones_permitidas):
        raise HTTPException(
            status_code=400, 
            detail="Formato no soportado. Por favor, sube un archivo .stl o .obj"
        )

    # 2. Generar un ID único para el archivo (evita que dos maquetas con el mismo nombre colisionen)
    file_id = str(uuid.uuid4())
    file_extension = file.filename.split('.')[-1]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.{file_extension}")

    # 3. Guardado eficiente en disco (Chunking)
    try:
        with open(file_path, "wb") as buffer:
            # copyfileobj lee y escribe en pequeños fragmentos, ideal para archivos de 50MB+
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al guardar el archivo: {str(e)}")
    finally:
        # Siempre cerramos el archivo original en memoria
        file.file.close()

    # 4. Respuesta exitosa al frontend
    return {
        "message": "Archivo recibido y guardado con éxito.",
        "file_id": file_id,
        "original_filename": file.filename,
        "status": "listo_para_procesar"
    }