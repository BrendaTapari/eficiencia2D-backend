import os
import uuid
import gzip
import time
import pickle
import shutil
import logging
import dataclasses
from typing import Dict, Optional
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from core.profiler import PipelineTimer
from core.services.obj_parser import parse_obj
from core.pipeline import parse_pipeline, generate_pipeline, Phase1Result
from core.services.types import PipelineOptions

router = APIRouter()
logger = logging.getLogger("eficiencia2d.pipeline")

UPLOAD_DIR = "temp/uploads"
CACHE_DIR = "temp/cache"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Límite configurable (500 MB)
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024  # 1 MB por chunk


class GenerateRequest(BaseModel):
    file_id: str
    original_filename: str = "model.obj"
    scale_denom: float = 50.0
    paper: str = "A4"
    overrides: Optional[Dict[int, str]] = None
    wall_wall_decisions: Optional[Dict[int, int]] = None


# ---------------------------------------------------------------------------
# Caché de Phase1 en disco (pickle + gzip)
# ---------------------------------------------------------------------------

def _cache_path(file_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{file_id}_phase1.pkl.gz")


def save_phase1_cache(file_id: str, result: Phase1Result) -> None:
    """Serializa el Phase1Result comprimido a disco."""
    t0 = time.perf_counter()
    path = _cache_path(file_id)
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_kb = os.path.getsize(path) / 1024
    ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[cache] Phase1 guardado: {path} ({size_kb:.0f} KB) en {ms:.0f} ms")


def load_phase1_cache(file_id: str) -> Optional[Phase1Result]:
    """Carga el Phase1Result desde caché. Retorna None si no existe."""
    path = _cache_path(file_id)
    if not os.path.exists(path):
        return None
    t0 = time.perf_counter()
    try:
        with gzip.open(path, "rb") as f:
            result = pickle.load(f)
        ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[cache] Phase1 cargado desde caché en {ms:.0f} ms")
        return result
    except Exception as e:
        logger.warning(f"[cache] Error al cargar caché {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Preview OBJ (solo grupos, no faces individuales)
# ---------------------------------------------------------------------------

def export_colored_obj(groups, faces) -> str:
    """
    Genera un OBJ de preview con grupos de color.
    Limitado a los primeros MAX_PREVIEW_GROUPS grupos para no sobrecargar el JSON.
    """
    MAX_PREVIEW_GROUPS = 200  # límite de grupos en preview

    v_lines = []
    f_lines = []
    vertex_map = {}
    next_idx = 1

    category_faces = {"wall": [], "floor": [], "discard": []}
    groups_limited = groups[:MAX_PREVIEW_GROUPS]

    for g in groups_limited:
        cat = g.category
        if cat not in category_faces:
            category_faces[cat] = []
        for fi in g.face_indices:
            if 0 <= fi < len(faces):
                category_faces[cat].append(faces[fi])

    for cat, cat_faces in category_faces.items():
        if not cat_faces:
            continue
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


# ---------------------------------------------------------------------------
# Endpoint /upload — optimizado
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    timer = PipelineTimer("upload_endpoint")

    extensiones_permitidas = ('.stl', '.obj')
    if not file.filename.lower().endswith(extensiones_permitidas):
        raise HTTPException(status_code=400, detail="Formato no soportado. Use .obj o .stl")

    file_id = str(uuid.uuid4())
    file_extension = file.filename.split('.')[-1].lower()
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.{file_extension}")

    try:
        # --- Paso 1: Streaming upload a disco (sin cargar todo en RAM) ---
        with timer.step("stream_upload_to_disk"):
            total_bytes = 0
            with open(file_path, "wb") as out_file:
                while True:
                    chunk = await file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_FILE_SIZE_BYTES:
                        out_file.close()
                        os.remove(file_path)
                        raise HTTPException(
                            status_code=413,
                            detail=f"Archivo demasiado grande (máx {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB)."
                        )
                    out_file.write(chunk)

        file_size_mb = total_bytes / 1024 / 1024
        logger.info(f"[upload] Archivo recibido: {file.filename} ({file_size_mb:.2f} MB)")

        # --- Paso 2+3: Parseo OBJ en streaming (sin cargar el archivo entero en RAM) ---
        with timer.step("parse_obj", size_mb=round(file_size_mb, 2)):
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                parsed = parse_obj(f)

        face_count = len(parsed["faces"])
        logger.info(f"[upload] Caras parseadas: {face_count:,}")

        # --- Paso 4: Pipeline completo ---
        with timer.step("parse_pipeline", face_count=face_count):
            result = parse_pipeline(file.filename, parsed["faces"], parsed["warnings"])

        # --- Paso 5: Guardar caché de Phase1 en disco ---
        with timer.step("save_phase1_cache"):
            save_phase1_cache(file_id, result)

        # --- Paso 6: Generar preview OBJ (limitado) ---
        with timer.step("export_colored_obj_preview"):
            preview_obj = export_colored_obj(result.groups, result.faces)

        # --- Conteos ---
        wall_count = sum(1 for g in result.groups if g.category == "wall")
        floor_count = sum(1 for g in result.groups if g.category == "floor")
        discard_count = sum(1 for g in result.groups if g.category == "discard")

        timing_report = timer.report()

        # --- Respuesta: sin serializar faces individuales (ahorra GBs de JSON) ---
        # El frontend usa groups + joints para la UI, no las faces crudas
        return JSONResponse(content={
            "message": "Archivo procesado con éxito.",
            "file_id": file_id,
            "original_filename": file.filename,
            "file_size_mb": round(file_size_mb, 2),
            "summary": {
                "walls": wall_count,
                "floors": floor_count,
                "discards": discard_count,
                "total_groups": len(result.groups),
                "total_faces": len(result.faces),
                "total_joints": len(result.joints),
            },
            "topology": {
                "groups": [dataclasses.asdict(g) for g in result.groups],
                "joints": [dataclasses.asdict(j) for j in result.joints],
                "adjustments": [dataclasses.asdict(a) for a in result.adjustments],
                "wall_wall_joints": [dataclasses.asdict(wj) for wj in result.wall_wall_joints],
                "applied_axis": result.applied_axis,
                "pre_split_face_count": result.pre_split_face_count,
                "suggested_merges": result.suggested_merges,
            },
            "preview_obj": preview_obj,
            "timing": timing_report,  # Debug: reporte de timing
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[upload] Error procesando {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
    finally:
        await file.close()


# ---------------------------------------------------------------------------
# Endpoint /generate — usa caché de Phase1
# ---------------------------------------------------------------------------

@router.post("/generate")
async def generate_pdf_endpoint(request: GenerateRequest):
    """
    Genera el PDF final. Usa caché de Phase1 si está disponible
    para evitar re-procesar el OBJ completo.
    """
    timer = PipelineTimer("generate_endpoint")

    # --- Intentar cargar caché ---
    with timer.step("load_phase1_cache"):
        phase1 = load_phase1_cache(request.file_id)

    if phase1 is None:
        logger.info("[generate] Caché no encontrado, re-procesando OBJ...")
        # Fallback: re-procesar desde disco
        file_path_obj = os.path.join(UPLOAD_DIR, f"{request.file_id}.obj")
        file_path_stl = os.path.join(UPLOAD_DIR, f"{request.file_id}.stl")
        file_path = file_path_obj if os.path.exists(file_path_obj) else file_path_stl

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Archivo no encontrado. Por favor sube el archivo nuevamente.")

        try:
            with timer.step("parse_obj_fallback"):
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    parsed = parse_obj(f)

            with timer.step("parse_pipeline_fallback"):
                phase1 = parse_pipeline(request.original_filename, parsed["faces"], parsed["warnings"])

            with timer.step("save_phase1_cache_fallback"):
                save_phase1_cache(request.file_id, phase1)

        except Exception as e:
            logger.exception(f"[generate] Error en fallback: {e}")
            raise HTTPException(status_code=500, detail=f"Error en re-procesamiento: {str(e)}")
    else:
        logger.info("[generate] Usando caché de Phase1 ✓")

    try:
        opts = PipelineOptions(
            scale_denom=request.scale_denom,
            paper=request.paper
        )

        with timer.step("generate_pipeline"):
            files = generate_pipeline(phase1, opts, overrides=request.overrides)

        timing_report = timer.report()

        return JSONResponse(content={
            "message": "Proyecto generado correctamente.",
            "generated_files": [f.name for f in files],
            "timing": timing_report,
        })

    except Exception as e:
        logger.exception(f"[generate] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error en generación: {str(e)}")
