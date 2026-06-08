import os
import uuid
import shutil
import dataclasses
from typing import Dict, Optional
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Importaciones del motor de procesamiento
from core.services.obj_parser import parse_obj
from core.pipeline import parse_pipeline, generate_pipeline
from core.services.types import PipelineOptions

router = APIRouter()
UPLOAD_DIR = "temp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class GenerateRequest(BaseModel):
    file_id: str
    original_filename: str = "model.obj"
    scale_denom: float = 50.0
    paper: str = "A4"
    overrides: Optional[Dict[int, str]] = None
    wall_wall_decisions: Optional[Dict[int, int]] = None

def export_colored_obj(groups, faces) -> str:
    v_lines = []
    f_lines = []
    vertex_map = {}
    next_idx = 1
    
    category_faces = {"wall": [], "floor": [], "discard": []}
    for g in groups:
        cat = g.category
        if cat not in category_faces: 
            category_faces[cat] = []
        for fi in g.face_indices:
            category_faces[cat].append(faces[fi])
            
    for cat, cat_faces in category_faces.items():
        if not cat_faces: continue
        f_lines.append(f"g {cat}")
        for face in cat_faces:
            f_indices = []
            for v in face.vertices:
                v_key = (round(v.x, 4), round(v.y, 4), round(v.z, 4))
                if v_key not in vertex_map:
                    vertex_map[v_key] = next_idx
                    v_lines.append(f"v {v.x:.4f} {v.y:.4f} {v.z:.4f}")
                    next_idx += 1
                f_indices.append(str(vertex_map[v_key]))
            f_lines.append(f"f {' '.join(f_indices)}")
            
    return "\n".join(v_lines) + "\n" + "\n".join(f_lines)


@router.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    extensiones_permitidas = ('.stl', '.obj')
    if not file.filename.lower().endswith(extensiones_permitidas):
        raise HTTPException(status_code=400, detail="Formato no soportado.")

    file_id = str(uuid.uuid4())
    file_extension = file.filename.split('.')[-1]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.{file_extension}")
    
    try:
        content = await file.read()
        # Guardar en disco para cuando el frontend pida generar el PDF después del review
        with open(file_path, "wb") as f:
            f.write(content)
            
        text_content = content.decode('utf-8')
        
        parsed = parse_obj(text_content)
        result = parse_pipeline(file.filename, parsed["faces"], parsed["warnings"])
        preview_obj = export_colored_obj(result.groups, result.faces)
        
        wall_count = sum(1 for g in result.groups if g.category == "wall")
        floor_count = sum(1 for g in result.groups if g.category == "floor")
        discard_count = sum(1 for g in result.groups if g.category == "discard")

        return JSONResponse(content={
            "message": "Archivo procesado con éxito.",
            "file_id": file_id,
            "original_filename": file.filename,
            "summary": {
                "walls": wall_count,
                "floors": floor_count,
                "discards": discard_count,
                "total_groups": len(result.groups)
            },
            "topology": {
                "faces": [dataclasses.asdict(f) for f in result.faces],
                "groups": [dataclasses.asdict(g) for g in result.groups],
                "joints": [dataclasses.asdict(j) for j in result.joints],
                "adjustments": [dataclasses.asdict(a) for a in result.adjustments],
                "wall_wall_joints": [dataclasses.asdict(wj) for wj in result.wall_wall_joints], "raw_faces": [dataclasses.asdict(f) for f in result.raw_faces], "applied_axis": result.applied_axis, "pre_split_face_count": result.pre_split_face_count, "suggested_merges": result.suggested_merges
            },
            "preview_obj": preview_obj
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
    finally:
        await file.close()

@router.post("/generate")
async def generate_pdf(request: GenerateRequest):
    """
    Recibe la configuración final del usuario (overrides, uniones),
    re-procesa el OBJ original y genera el PDF con el Nesting aplanado.
    """
    file_path_obj = os.path.join(UPLOAD_DIR, f"{request.file_id}.obj")
    file_path_stl = os.path.join(UPLOAD_DIR, f"{request.file_id}.stl")
    
    file_path = file_path_obj if os.path.exists(file_path_obj) else file_path_stl
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Archivo original no encontrado en el servidor.")
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()
            
        parsed = parse_obj(text_content)
        
        # En el futuro, podríamos modificar parse_pipeline para que reciba wall_wall_decisions
        # Por ahora lo pasamos tal cual
        phase1 = parse_pipeline(request.original_filename, parsed["faces"], parsed["warnings"])
        
        opts = PipelineOptions(
            scale_denom=request.scale_denom,
            paper=request.paper
        )
        
        # Ejecutamos la fase 2 para generar los archivos PDF
        files = generate_pipeline(phase1, opts, overrides=request.overrides)
        
        # Como generate_pipeline actualmente devuelve una lista vacía en el template actual,
        # enviaremos una respuesta indicando éxito.
        # Más adelante acá se devolverá el archivo .pdf directamente con un FileResponse
        return JSONResponse(content={
            "message": "Proyecto generado correctamente.",
            "generated_files": [f.name for f in files]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en generación: {str(e)}")
