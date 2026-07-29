import os
import uuid
import gzip
import time
import array
import base64
import io
import zipfile
import pickle
import shutil
import logging
import dataclasses
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from api.deps import bearer_scheme, get_current_user
from database import Proyecto, get_db

from core.profiler import PipelineTimer
from core.services.obj_parser import parse_obj
from core.services.stl_parser import parse_stl
from core.pipeline import parse_pipeline, generate_pipeline, Phase1Result
from core.services.cutting_sheet import build_placements
from core.services.curvature import build_curvature_map
from core.services.assembly_check import compute_assembly_warnings
from core.services.assembly_fit import check_piece_collisions
from core.group_classifier import compute_assembly_steps
from core.services.assembly_adjuster import DEFAULT_MDF_THICKNESS_M
from core.services.types import PipelineOptions, SheetConfig

router = APIRouter()
logger = logging.getLogger("eficiencia2d.pipeline")

# Directorios de trabajo (uploads OBJ/STL y caché de Phase1, DXF/PDF temporales).
# Se anclan a una raíz ABSOLUTA para no depender del CWD (un servicio systemd puede
# arrancar desde cualquier directorio). Base configurable con DATA_DIR; por defecto la
# carpeta `temp/` del proyecto. os.path.join → separador correcto en Linux/Windows.
_PROJECT_DIR = Path(__file__).resolve().parents[2]
_DATA_DIR = os.environ.get("DATA_DIR") or os.path.join(_PROJECT_DIR, "temp")
UPLOAD_DIR = os.path.join(_DATA_DIR, "uploads")
CACHE_DIR = os.path.join(_DATA_DIR, "cache")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Límite configurable (500 MB)
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024  # 1 MB por chunk


# ---------------------------------------------------------------------------
# Codificación compacta de geometría (buffers binarios + base64)
#
# Serializar las caras como dicts (dataclasses.asdict) genera millones de
# objetos y revienta la RAM (MemoryError) en modelos grandes. En su lugar
# empaquetamos la geometría en buffers planos little-endian:
#   - coords:        Float32  (x,y,z por cada vértice de cada cara, en orden)
#   - normals:       Float32  (nx,ny,nz por cara)
#   - vertex_counts: UInt32   (cantidad de vértices por cara)
#   - idx_counts:    UInt32   (cantidad de vertex_indices por cara, 0 si no hay)
#   - indices:       UInt32   (vertex_indices concatenados)
# El front los decodifica a Face3D[] con un adaptador chico (Float32Array/…).
# inner_loops y panel_id se omiten: en este pipeline siempre son [] / None,
# y el front los reconstruye como vacíos.
# ---------------------------------------------------------------------------

def encode_faces_compact(faces: List) -> Dict:
    coords = array.array("f")
    normals = array.array("f")
    vertex_counts = array.array("I")
    idx_counts = array.array("I")
    indices = array.array("I")

    coords_append = coords.append
    for f in faces:
        verts = f.vertices
        vertex_counts.append(len(verts))
        for v in verts:
            coords_append(v.x)
            coords_append(v.y)
            coords_append(v.z)
        n = f.normal
        normals.append(n.x)
        normals.append(n.y)
        normals.append(n.z)
        vi = getattr(f, "vertex_indices", None)
        if vi and len(vi) == len(verts):
            idx_counts.append(len(vi))
            for k in vi:
                indices.append(k)
        else:
            idx_counts.append(0)

    def b64(a: array.array) -> str:
        return base64.b64encode(a.tobytes()).decode("ascii")

    return {
        "count": len(faces),
        "format": "packed-le-v1",
        "vertex_counts_b64": b64(vertex_counts),
        "coords_b64": b64(coords),
        "normals_b64": b64(normals),
        "idx_counts_b64": b64(idx_counts),
        "indices_b64": b64(indices),
    }


# ---------------------------------------------------------------------------
# Serialización compartida (topology / nesting) para /upload, /recompute,
# /nesting-preview. Mantiene una sola fuente de verdad del formato de respuesta.
# ---------------------------------------------------------------------------

def _serialize_group(g) -> Dict:
    """Serializa un grupo. Si es un muro sin espesor detectado (mallas de una sola
    cara), reporta el espesor MDF estándar (3 mm) como `thickness` y marca
    `thickness_assumed=True`. Así el front deja de mostrar "no se detectó espesor en
    ninguna de las dos" y habilita la UI para elegir quién corta a quién: al cortar en
    MDF ese grosor existe igual. El trim real ya usa el mismo fallback en
    compute_adjustments; acá sólo se expone el dato para la UI (no toca la geometría)."""
    d = dataclasses.asdict(g)
    if g.category == "wall" and (g.thickness is None or g.thickness <= 0.001):
        d["thickness"] = DEFAULT_MDF_THICKNESS_M
        d["thickness_assumed"] = True
    else:
        d["thickness_assumed"] = False
    return d


VALID_CATEGORIES = ("wall", "floor", "discard")


def apply_overrides_to_groups(groups, overrides: Optional[Dict[int, str]]):
    """Devuelve los grupos con la categoría EFECTIVA (la del usuario si la cambió).

    El instructivo se arma con placements/assembly_steps, que filtran por
    `category != "discard"`. Si se les pasan los grupos crudos, ignoran lo que el
    usuario descartó o rescató en el visor 3D: los descartados siguen apareciendo en
    el instructivo y los rescatados no aparecen. No muta los grupos originales.
    """
    if not overrides:
        return groups
    out = []
    for g in groups:
        cat = overrides.get(g.id)
        if cat in VALID_CATEGORIES and cat != g.category:
            out.append(dataclasses.replace(g, category=cat))
        else:
            out.append(g)
    return out


def serialize_topology(
    result: Phase1Result,
    panel_id_by_group: Optional[Dict] = None,
    overrides: Optional[Dict[int, str]] = None,
) -> Dict:
    # `groups` viaja con la categoría original (el front lleva su propio estado de
    # overrides para la lista de componentes); placements y assembly_steps, en cambio,
    # deben respetarlos, porque de ahí sale el instructivo de armado.
    groups_eff = apply_overrides_to_groups(result.groups, overrides)
    topo = {
        "groups": [_serialize_group(g) for g in result.groups],
        "joints": [dataclasses.asdict(j) for j in result.joints],
        "adjustments": [dataclasses.asdict(a) for a in result.adjustments],
        "wall_wall_joints": [dataclasses.asdict(wj) for wj in result.wall_wall_joints],
        "applied_axis": result.applied_axis,
        "stem": result.stem,
        "warnings": result.warnings,
        "pre_split_face_count": result.pre_split_face_count,
        "suggested_merges": result.suggested_merges,
        # Geometría empaquetada en buffers binarios (ver encode_faces_compact).
        "faces_packed": encode_faces_compact(result.faces),
        "raw_faces_packed": encode_faces_compact(result.raw_faces),
        # Marco de proyección 3D por pieza (instructivo de armado): group_id -> {origin,
        # u_axis, v_axis, normal, width_m, height_m, mirrored}. world = origin + u·u_axis
        # + v·v_axis. Caras en el mismo espacio que faces_packed (eje Y).
        "placements": build_placements(groups_eff, result.faces, "Y"),
        "assembly_steps": compute_assembly_steps(groups_eff),
        # Curvatura por grupo (contrato §2.1): el front marca componentes curvos y
        # sugiere método kerf (simple) / auxético (doble) + spacing inicial.
        "curvature": build_curvature_map(result.groups, result.faces),
    }
    if panel_id_by_group is not None:
        topo["panel_id_by_group"] = {str(k): v for k, v in panel_id_by_group.items()}
    return topo


def _serialize_nesting_panel(p) -> Dict:
    # Separar CORTE (edges) de GRABADO (mark_segments). Las aristas score=True
    # (mark_lines + cortes tipo "line") son grabado rojo y viajan aparte para que el
    # front las dibuje en rojo sin duplicarlas como corte (B2). En el mismo marco local
    # del panel → el front les aplica el mismo offset/rotación que a edges.
    edges = []
    mark_segments = []
    for e in p.edges:
        if getattr(e, "score", False):
            mark_segments.append({
                "a": {"x": e.a.x, "y": e.a.y},
                "b": {"x": e.b.x, "y": e.b.y},
            })
        else:
            # Se conserva el flag `joint` (encastre/intersección) para que el front pueda
            # mostrar la funcionalidad de intersecciones entre paredes. En la LÁMINA de
            # corte (DXF/PDF) no se dibuja (ver emit_panel_entities/pdf_writer); acá viaja
            # como dato para la UI de intersecciones/instructivo.
            edges.append({
                "a": {"x": e.a.x, "y": e.a.y},
                "b": {"x": e.b.x, "y": e.b.y},
                "hole": e.hole,
                "joint": getattr(e, "joint", False),
                "flex": getattr(e, "flex", False),
            })
    return {
        "id": p.id,
        "category": p.category,
        "width_m": p.width_m,
        "height_m": p.height_m,
        "edges": edges,
        "mark_segments": mark_segments,
        "is_mark": p.is_mark,
    }


def serialize_nesting(nesting) -> Dict:
    return {
        "sheets": [
            {
                "index": s.index,
                "panels": [
                    {
                        "panel": _serialize_nesting_panel(pp.panel),
                        "x": pp.x,
                        "y": pp.y,
                        "rotated": pp.rotated,
                        "effective_w": pp.effective_w,
                        "effective_h": pp.effective_h,
                    }
                    for pp in s.panels
                ],
                "utilization": s.utilization,
            }
            for s in nesting.sheets
        ],
        "config": {
            "width_m": nesting.config.width_m,
            "height_m": nesting.config.height_m,
            "gap_m": nesting.config.gap_m,
        },
        "scale_denom": nesting.scale_denom,
        "unplaced": [_serialize_nesting_panel(p) for p in nesting.unplaced],
    }


def serialize_plate_joints(plate_joints) -> List[Dict]:
    """Encastres 3D (Misión 1) en coordenadas del mundo, para superponer las ranuras en
    el instructivo. `cut_id` = grupo que recibe la ranura (la pieza que se revela); el
    front filtra por `cut_id == group.id` y dibuja el slot con (a, b, width)."""
    out: List[Dict] = []
    for pj in plate_joints or []:
        out.append({
            "cut_id": pj.cut_id,
            "cutter_id": pj.cutter_id,
            "a": {"x": pj.a.x, "y": pj.a.y, "z": pj.a.z},
            "b": {"x": pj.b.x, "y": pj.b.y, "z": pj.b.z},
            "width": pj.width,
        })
    return out


def ensure_model_on_disk(
    file_id: str,
    original_filename: str,
    *,
    ruta_r2: str | None = None,
) -> str:
    """Garantiza que el modelo esté en disco local; descarga desde R2 si hace falta."""
    extension = original_filename.rsplit(".", 1)[-1].lower()
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.{extension}")

    if os.path.exists(file_path):
        return file_path

    file_path_obj = os.path.join(UPLOAD_DIR, f"{file_id}.obj")
    file_path_stl = os.path.join(UPLOAD_DIR, f"{file_id}.stl")
    if os.path.exists(file_path_obj):
        return file_path_obj
    if os.path.exists(file_path_stl):
        return file_path_stl

    if not ruta_r2:
        raise HTTPException(
            status_code=404,
            detail="Archivo no encontrado. Por favor sube el archivo nuevamente.",
        )

    from database.storage import descargar_archivo

    logger.info("[load] Descargando modelo desde R2: %s", ruta_r2)
    data = descargar_archivo(ruta_r2)
    with open(file_path, "wb") as out_file:
        out_file.write(data)
    return file_path


def load_or_parse_phase1(
    file_id: str,
    original_filename: str,
    *,
    ruta_r2: str | None = None,
) -> Phase1Result:
    """Carga el Phase1 de caché o, si no está, re-parsea desde disco o R2."""
    phase1 = load_phase1_cache(file_id)
    if phase1 is not None:
        return phase1

    logger.info("[load] Caché no encontrado, re-procesando archivo...")
    file_path = ensure_model_on_disk(file_id, original_filename, ruta_r2=ruta_r2)
    if file_path.lower().endswith(".stl"):
        parsed = parse_stl(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            parsed = parse_obj(f)
    phase1 = parse_pipeline(original_filename, parsed["faces"], parsed["warnings"])
    save_phase1_cache(file_id, phase1)
    return phase1


def _original_filename_for_proyecto(nombre: str, formato: str) -> str:
    if nombre.lower().endswith(f".{formato}"):
        return nombre
    return f"{nombre}.{formato}"


def original_filename_for_proyecto(proyecto) -> str:
    """Nombre de archivo real del modelo guardado (para parseo y descarga desde R2)."""
    meta = proyecto.metadata_impresion or {}
    archivo = meta.get("archivo_original")
    if isinstance(archivo, str) and archivo.strip():
        return archivo.strip()

    if proyecto.url_archivo:
        name = Path(proyecto.url_archivo).name
        if name:
            return name

    return _original_filename_for_proyecto(proyecto.nombre, proyecto.formato)


def resolve_phase1_source(
    *,
    file_id: str | None,
    original_filename: str,
    proyecto_id: str | None,
    db: Session,
    user_id,
) -> tuple[str, str, str | None]:
    """Resuelve file_id, original_filename y ruta R2 desde file_id o proyecto_id."""
    if proyecto_id:
        try:
            proyecto_uuid = UUID(proyecto_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="proyecto_id inválido") from exc

        proyecto = db.get(Proyecto, proyecto_uuid)
        if proyecto is None or proyecto.usuario_id != user_id:
            raise HTTPException(status_code=404, detail="Proyecto no encontrado")

        resolved_file_id = str(proyecto.id)
        resolved_filename = original_filename_for_proyecto(proyecto)
        return resolved_file_id, resolved_filename, proyecto.url_archivo

    if not file_id:
        raise HTTPException(
            status_code=400,
            detail="Debes enviar file_id o proyecto_id",
        )

    return file_id, original_filename, None


def _load_phase1_for_request(
    request,
    db: Session,
    credentials: HTTPAuthorizationCredentials | None,
) -> Phase1Result:
    user_id = None
    if request.proyecto_id:
        user = get_current_user(credentials, db)
        user_id = user.id

    file_id, original_filename, ruta_r2 = resolve_phase1_source(
        file_id=request.file_id,
        original_filename=request.original_filename,
        proyecto_id=request.proyecto_id,
        db=db,
        user_id=user_id,
    )
    return load_or_parse_phase1(file_id, original_filename, ruta_r2=ruta_r2)


def _summarize_phase1(result: Phase1Result) -> dict:
    return {
        "walls": sum(1 for g in result.groups if g.category == "wall"),
        "floors": sum(1 for g in result.groups if g.category == "floor"),
        "discards": sum(1 for g in result.groups if g.category == "discard"),
        "total_groups": len(result.groups),
        "total_faces": len(result.faces),
        "total_joints": len(result.joints),
    }


def build_geometry_payload(
    file_id: str,
    original_filename: str,
    result: Phase1Result,
    *,
    file_size_mb: float | None = None,
    summary: dict | None = None,
    timing: dict | None = None,
) -> dict:
    """Respuesta de geometría compatible con /upload y abrir proyecto guardado."""
    from core.review_generate import compute_panel_id_by_group

    pid_by_group = compute_panel_id_by_group(result)
    payload = {
        "message": "Proyecto cargado correctamente.",
        "file_id": file_id,
        "proyecto_id": file_id,
        "original_filename": original_filename,
        "summary": summary or _summarize_phase1(result),
        "topology": serialize_topology(result, pid_by_group),
        "preview_obj": export_colored_obj(result.groups, result.faces),
    }
    if file_size_mb is not None:
        payload["file_size_mb"] = file_size_mb
    if timing is not None:
        payload["timing"] = timing
    return payload


def load_saved_proyecto_phase1(proyecto) -> tuple[Phase1Result, str]:
    """Descarga (si hace falta), parsea y devuelve Phase1 de un proyecto guardado."""
    file_id = str(proyecto.id)
    original_filename = original_filename_for_proyecto(proyecto)
    result = load_or_parse_phase1(
        file_id,
        original_filename,
        ruta_r2=proyecto.url_archivo,
    )
    return result, original_filename


class SheetConfigModel(BaseModel):
    width_m: float = 1.0
    height_m: float = 0.6
    gap_m: float = 0.003


class SplitModel(BaseModel):
    group_id: int
    mode: str = "components"  # "components" | "panels"


class GenerateRequest(BaseModel):
    file_id: str | None = None
    proyecto_id: str | None = None
    original_filename: str = "model.obj"
    scale_denom: float = 50.0
    paper: str = "A4"
    page_mode: str = "one_per_sheet"  # "one_per_sheet" | "single_page" (PDF de planchas)
    combine_sheets: bool = False  # paredes y pisos juntos en las mismas planchas
    min_area_m2: float = 1.0
    axis: Optional[str] = None  # "Y" | "Z" | None (usa el eje del upload)
    sheet_config: Optional[SheetConfigModel] = None
    overrides: Optional[Dict[int, str]] = None
    wall_wall_decisions: Optional[Dict[int, int]] = None
    merges: Optional[List[List[int]]] = None
    splits: Optional[List[SplitModel]] = None
    marks: Optional[List[int]] = None  # ids de componentes cuyas aberturas se graban
    user_cuts: Optional[List] = None  # reservado: corte manual futuro
    flex: Optional[List[dict]] = None  # patrón de flexión por grupo (kerf / auxético)
    mark_lines: Optional[List[dict]] = None  # polilíneas rojas a grabar (score) por grupo
    ribs: Optional[List[dict]] = None  # refuerzos: nervios/cartelas
    columns: Optional[List[dict]] = None  # refuerzos: columnas


class RecomputeRequest(BaseModel):
    file_id: str | None = None
    proyecto_id: str | None = None
    original_filename: str = "model.obj"
    axis: Optional[str] = None  # "Y" | "Z"
    min_area_m2: float = 1.0
    merges: Optional[List[List[int]]] = None
    splits: Optional[List[SplitModel]] = None
    # Recategorizaciones del usuario (id de grupo -> "wall"/"floor"/"discard"). Sin
    # esto, el instructivo seguía mostrando los componentes descartados en el visor.
    overrides: Optional[Dict[int, str]] = None


class NestingPreviewRequest(BaseModel):
    file_id: str | None = None
    proyecto_id: str | None = None
    original_filename: str = "model.obj"
    axis: Optional[str] = None
    min_area_m2: float = 1.0
    merges: Optional[List[List[int]]] = None
    splits: Optional[List[SplitModel]] = None
    overrides: Optional[Dict[int, str]] = None
    wall_wall_decisions: Optional[Dict[int, int]] = None
    marks: Optional[List[int]] = None
    user_cuts: Optional[List[dict]] = None
    flex: Optional[List[dict]] = None  # patrón de flexión por grupo (kerf / auxético)
    mark_lines: Optional[List[dict]] = None  # polilíneas rojas a grabar (score) por grupo
    ribs: Optional[List[dict]] = None  # refuerzos: nervios/cartelas
    columns: Optional[List[dict]] = None  # refuerzos: columnas
    sheet_config: Optional[SheetConfigModel] = None
    scale_denom: float = 50.0
    paper: str = "A4"
    page_mode: str = "one_per_sheet"  # cartón (deriva plancha del papel) | láser
    combine_sheets: bool = False  # paredes y pisos juntos en las mismas planchas


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
# Procesamiento compartido de modelos 3D
# ---------------------------------------------------------------------------

async def process_model_file(
    file: UploadFile,
    file_id: str,
    background_tasks: BackgroundTasks,
) -> dict:
    """Recibe un modelo, lo guarda en disco temporal y devuelve la respuesta de procesamiento."""
    timer = PipelineTimer("upload_endpoint")

    extensiones_permitidas = (".stl", ".obj")
    if not file.filename.lower().endswith(extensiones_permitidas):
        raise HTTPException(status_code=400, detail="Formato no soportado. Use .obj o .stl")

    file_extension = file.filename.split(".")[-1].lower()
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.{file_extension}")

    try:
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
                            detail=f"Archivo demasiado grande (máx {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB).",
                        )
                    out_file.write(chunk)

        file_size_mb = total_bytes / 1024 / 1024
        logger.info(f"[upload] Archivo recibido: {file.filename} ({file_size_mb:.2f} MB)")

        with timer.step("parse_model", size_mb=round(file_size_mb, 2), fmt=file_extension):
            if file_extension == "stl":
                parsed = parse_stl(file_path)
            else:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    parsed = parse_obj(f)

        face_count = len(parsed["faces"])
        logger.info(f"[upload] Caras parseadas: {face_count:,}")

        with timer.step("parse_pipeline", face_count=face_count):
            result = parse_pipeline(file.filename, parsed["faces"], parsed["warnings"])

        background_tasks.add_task(save_phase1_cache, file_id, result)

        with timer.step("export_colored_obj_preview"):
            preview_obj = export_colored_obj(result.groups, result.faces)

        wall_count = sum(1 for g in result.groups if g.category == "wall")
        floor_count = sum(1 for g in result.groups if g.category == "floor")
        discard_count = sum(1 for g in result.groups if g.category == "discard")

        with timer.step("panel_id_by_group"):
            from core.review_generate import compute_panel_id_by_group

            pid_by_group = compute_panel_id_by_group(result)

        timing_report = timer.report()

        return {
            "message": "Archivo procesado con éxito.",
            "file_id": file_id,
            "original_filename": file.filename,
            "file_size_mb": round(file_size_mb, 2),
            "file_size_bytes": total_bytes,
            "file_path": file_path,
            "file_extension": file_extension,
            "summary": {
                "walls": wall_count,
                "floors": floor_count,
                "discards": discard_count,
                "total_groups": len(result.groups),
                "total_faces": len(result.faces),
                "total_joints": len(result.joints),
            },
            "topology": serialize_topology(result, pid_by_group),
            "preview_obj": preview_obj,
            "timing": timing_report,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[upload] Error procesando {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}") from e
    finally:
        await file.close()


# ---------------------------------------------------------------------------
# Endpoint /upload — optimizado
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_model(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    file_id = str(uuid.uuid4())
    content = await process_model_file(file, file_id, background_tasks)
    content.pop("file_path", None)
    content.pop("file_size_bytes", None)
    content.pop("file_extension", None)
    return JSONResponse(content=content)


# ---------------------------------------------------------------------------
# Endpoint /generate — usa caché de Phase1
# ---------------------------------------------------------------------------

@router.post("/generate")
async def generate_pdf_endpoint(
    request: GenerateRequest,
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    """
    Genera el PDF final. Usa caché de Phase1 si está disponible
    para evitar re-procesar el OBJ completo.
    """
    timer = PipelineTimer("generate_endpoint")

    with timer.step("load_phase1"):
        base = _load_phase1_for_request(request, db, credentials)

    try:
        # Re-derivar topología determinística: eje → área mínima → splits.
        from core.pipeline import apply_review_edits
        with timer.step("apply_review_edits"):
            phase1 = apply_review_edits(
                base,
                axis=request.axis,
                min_real_area=request.min_area_m2,
                splits=[s.dict() for s in (request.splits or [])],
            )

        sheet = request.sheet_config
        sheet_cfg = (
            SheetConfig(
                width_m=sheet.width_m,
                height_m=sheet.height_m,
                gap_m=sheet.gap_m,
            )
            if sheet
            else SheetConfig(width_m=1.0, height_m=0.6, gap_m=0.003)
        )
        opts = PipelineOptions(
            scale_denom=request.scale_denom,
            paper=request.paper,
            include_cutting_sheet=True,
            sheet_config=sheet_cfg,
            min_area_m2=request.min_area_m2,
            page_mode=request.page_mode,
            combine_sheets=request.combine_sheets,
        )

        with timer.step("generate_pipeline"):
            files = generate_pipeline(
                phase1,
                opts,
                overrides=request.overrides,
                wall_wall_decisions=request.wall_wall_decisions,
                merges=request.merges,
                marks=request.marks,
                user_cuts=request.user_cuts,
                flex=request.flex,
                mark_lines=request.mark_lines,
                ribs=request.ribs,
                columns=request.columns,
            )

        if not files:
            raise HTTPException(
                status_code=422,
                detail="No se generaron archivos. Verificá clasificación y planchas.",
            )

        with timer.step("build_zip"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in files:
                    zf.writestr(f.name, f.blob)
            zip_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        timing_report = timer.report()

        return JSONResponse(content={
            "message": "Proyecto generado correctamente.",
            "generated_files": [f.name for f in files],
            "zip_base64": zip_b64,
            "zip_filename": f"{phase1.stem}_planos.zip",
            "timing": timing_report,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[generate] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error en generación: {str(e)}")


# ---------------------------------------------------------------------------
# Endpoint /recompute — re-deriva la topología tras una edición geométrica
# (eje, área mínima, fusiones, divisiones). Devuelve la misma estructura que
# /upload. El front reemplaza su phase1Result completo con esta respuesta.
# ---------------------------------------------------------------------------

@router.post("/recompute")
async def recompute_endpoint(
    request: RecomputeRequest,
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    timer = PipelineTimer("recompute_endpoint")

    with timer.step("load_phase1"):
        base = _load_phase1_for_request(request, db, credentials)

    try:
        from core.pipeline import apply_review_edits
        from core.review_generate import apply_merges, compute_panel_id_by_group

        with timer.step("apply_review_edits"):
            rebuilt = apply_review_edits(
                base,
                axis=request.axis,
                min_real_area=request.min_area_m2,
                splits=[s.dict() for s in (request.splits or [])],
            )

        with timer.step("apply_merges"):
            # Los overrides también entran acá: una fusión puede involucrar grupos
            # rescatados de "descartado" (ver apply_merges).
            merged = apply_merges(rebuilt, request.merges or [], request.overrides)

        with timer.step("panel_id_by_group"):
            pid_by_group = compute_panel_id_by_group(
                merged, overrides=request.overrides
            )

        timer.report()
        return JSONResponse(
            content={
                "topology": serialize_topology(
                    merged, pid_by_group, request.overrides
                )
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[recompute] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error en recálculo: {str(e)}")


# ---------------------------------------------------------------------------
# Endpoint /nesting-preview — layout de nesting para previsualizar antes de
# generar los archivos finales. Aplica la misma cadena determinística que
# /generate (eje → área → splits → merges → overrides/decisiones) pero en vez
# de exportar DXF/PDF devuelve la distribución de paneles en planchas.
# ---------------------------------------------------------------------------

@router.post("/nesting-preview")
async def nesting_preview_endpoint(
    request: NestingPreviewRequest,
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    timer = PipelineTimer("nesting_preview_endpoint")

    with timer.step("load_phase1"):
        base = _load_phase1_for_request(request, db, credentials)

    try:
        from core.pipeline import apply_review_edits
        from core.review_generate import compute_nesting

        with timer.step("apply_review_edits"):
            rebuilt = apply_review_edits(
                base,
                axis=request.axis,
                min_real_area=request.min_area_m2,
                splits=[s.dict() for s in (request.splits or [])],
            )

        sheet = request.sheet_config
        sheet_cfg = (
            SheetConfig(
                width_m=sheet.width_m,
                height_m=sheet.height_m,
                gap_m=sheet.gap_m,
            )
            if sheet
            else SheetConfig(width_m=1.0, height_m=0.6, gap_m=0.003)
        )
        opts = PipelineOptions(
            scale_denom=request.scale_denom,
            paper=request.paper,
            include_cutting_sheet=True,
            sheet_config=sheet_cfg,
            min_area_m2=request.min_area_m2,
            page_mode=request.page_mode,
            combine_sheets=request.combine_sheets,
        )

        with timer.step("compute_nesting"):
            work, wall_nesting, floor_nesting, cfg, panel_id_by_group, plate_joints, final_panels = compute_nesting(
                rebuilt,
                opts,
                overrides=request.overrides,
                wall_wall_decisions=request.wall_wall_decisions,
                merges=request.merges,
                marks=request.marks,
                user_cuts=request.user_cuts,
                flex=request.flex,
                mark_lines=request.mark_lines,
                ribs=request.ribs,
                columns=request.columns,
            )

        # Chequeo automático de ensamble: huecos/solapes/piezas sin apoyo (world coords),
        # para avisar ANTES de mandar a cortar. Vacío = ensamble verificado.
        with timer.step("assembly_check"):
            try:
                assembly_warnings = compute_assembly_warnings(
                    work.groups, work.faces, panel_id_by_group
                )
                # Además, "armar la maqueta" con las piezas TAL COMO se van a cortar:
                # detecta placas que se atraviesan, que es lo que hasta ahora sólo se
                # descubría pegando el MDF. Va en la misma lista de avisos.
                assembly_warnings = assembly_warnings + check_piece_collisions(
                    final_panels
                )
            except Exception:
                logger.exception("[nesting-preview] assembly_check falló")
                assembly_warnings = []

        timer.report()
        return JSONResponse(content={
            "wall_nesting": serialize_nesting(wall_nesting),
            "floor_nesting": serialize_nesting(floor_nesting),
            "config": {
                "width_m": cfg.width_m,
                "height_m": cfg.height_m,
                "gap_m": cfg.gap_m,
            },
            # Encastres 3D (world coords) para superponer ranuras en el instructivo.
            "plate_joints": serialize_plate_joints(plate_joints),
            # Chequeo de ensamble: [] = verificado; si no, gap/overlap/unsupported.
            "assembly_warnings": assembly_warnings,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[nesting-preview] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error en nesting-preview: {str(e)}")
