import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.deps import get_current_user
from api.schemas.rol import is_admin_user
from api.routes.uploads import (
    SheetConfigModel,
    SplitModel,
    build_geometry_payload,
    load_saved_proyecto_phase1,
    process_model_file,
)
from core.profiler import PipelineTimer
from database import Proyecto, Usuario, get_db
from database.storage import descargar_archivo, eliminar_archivo, obtener_url_archivo, subir_archivo

router = APIRouter()
logger = logging.getLogger(__name__)

STATE_FILENAME = "estado.json"
NOTE_TEXT_MAX_LENGTH = 2000


class ProyectoResponse(BaseModel):
    id: str
    nombre: str
    formato: str
    tamano_bytes: int
    url_archivo: str
    metadata_impresion: dict[str, Any] | None
    fecha_creacion: str
    fecha_ultima_edicion: str
    download_url: str | None = None


class ProyectoListResponse(BaseModel):
    proyectos: list[ProyectoResponse]
    total: int


class GroupNoteModel(BaseModel):
    """Nota de usuario asociada a un componente de topología (group_id ≥ 0)."""

    id: str = Field(min_length=1, max_length=128)
    group_id: int = Field(ge=0, description="ID de topología del backend; nunca negativo")
    text: str = Field(min_length=1, max_length=NOTE_TEXT_MAX_LENGTH)
    created_at: str | None = None
    updated_at: str | None = None

    @field_validator("text")
    @classmethod
    def text_must_be_non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text no puede estar vacío")
        if len(stripped) > NOTE_TEXT_MAX_LENGTH:
            raise ValueError(f"text no puede superar {NOTE_TEXT_MAX_LENGTH} caracteres")
        return stripped

    @field_validator("id")
    @classmethod
    def id_must_be_non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("id de nota no puede estar vacío")
        return stripped


class ProyectoPartialSaveRequest(BaseModel):
    """Campos editables del proyecto; solo se actualizan los enviados en el body."""
    nombre: str | None = Field(default=None, min_length=1, max_length=200)
    axis: str | None = None
    min_area_m2: float | None = None
    merges: list[list[int]] | None = None
    splits: list[SplitModel] | None = None
    overrides: dict[int, str] | None = None
    wall_wall_decisions: dict[int, int] | None = None
    marks: list[int] | None = None
    user_cuts: list[Any] | None = None
    mark_lines: list[Any] | None = None
    notes: list[GroupNoteModel] | None = None
    sheet_config: SheetConfigModel | None = None
    scale_denom: float | None = None
    paper: str | None = None
    page_mode: str | None = None


class ProyectoStateSaveResponse(BaseModel):
    proyecto_id: str
    nombre: str
    estado_r2: str
    estado_actualizado_at: str
    estado: dict[str, Any]
    message: str


class ProyectoRenameRequest(BaseModel):
    nombre: str = Field(min_length=1, max_length=200)

    @field_validator("nombre")
    @classmethod
    def normalize_nombre(cls, value: str) -> str:
        nombre = value.strip()
        if not nombre:
            raise ValueError("El nombre del proyecto no puede estar vacío")
        return nombre


class ProyectoRenameResponse(ProyectoResponse):
    """Nombre actualizado en BD; si había estado guardado, también en estado.json (R2)."""
    estado_sincronizado: bool = False
    estado_actualizado_at: str | None = None


def _load_estado_proyecto(proyecto: Proyecto) -> dict[str, Any]:
    meta = proyecto.metadata_impresion or {}
    estado_r2 = meta.get("estado_r2")
    if isinstance(estado_r2, str) and estado_r2.strip():
        try:
            raw = descargar_archivo(estado_r2.strip())
            loaded = json.loads(raw.decode("utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except (RuntimeError, json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("No se pudo leer estado guardado de R2 para proyecto %s", proyecto.id)

    estado = meta.get("estado")
    return dict(estado) if isinstance(estado, dict) else {}


def _proyecto_tiene_estado_guardado(proyecto: Proyecto, estado: dict[str, Any]) -> bool:
    """True si el proyecto ya tiene estado.json en R2 o un estado legacy en metadata."""
    meta = proyecto.metadata_impresion or {}
    estado_r2 = meta.get("estado_r2")
    if isinstance(estado_r2, str) and estado_r2.strip():
        return True
    return bool(estado)


def _touch_proyecto_edicion(proyecto: Proyecto) -> None:
    """Marca el proyecto como editado ahora (para ordenar listados)."""
    proyecto.fecha_ultima_edicion = datetime.now(timezone.utc)


def _commit_proyecto_nombre(db: Session, proyecto: Proyecto, nombre: str) -> None:
    proyecto.nombre = nombre
    _touch_proyecto_edicion(proyecto)
    try:
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Error al renombrar proyecto %s en BD", proyecto.id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo guardar el nombre del proyecto",
        ) from exc
    db.refresh(proyecto)


def _set_proyecto_nombre(
    db: Session,
    proyecto: Proyecto,
    nombre: str,
) -> tuple[bool, str | None]:
    """
    Actualiza el nombre visible del proyecto en PostgreSQL y, si existe estado
    guardado, también la clave `nombre` de estado.json en R2.

    No modifica `metadata_impresion.archivo_original` ni renombra objetos en R2
    (las rutas son {usuario_id}/{proyecto_id}/…).
    """
    estado = _load_estado_proyecto(proyecto)
    tiene_estado = _proyecto_tiene_estado_guardado(proyecto, estado)

    if tiene_estado:
        estado["nombre"] = nombre
        proyecto.nombre = nombre
        estado_actualizado_at = _persist_estado_proyecto(db, proyecto, estado)
        return True, estado_actualizado_at

    _commit_proyecto_nombre(db, proyecto, nombre)
    return False, None


def _rename_response(proyecto: Proyecto, *, estado_sincronizado: bool, estado_actualizado_at: str | None) -> ProyectoRenameResponse:
    base = _proyecto_to_response(proyecto)
    return ProyectoRenameResponse(
        **base.model_dump(),
        estado_sincronizado=estado_sincronizado,
        estado_actualizado_at=estado_actualizado_at,
    )


def _serialize_notes(notes: list[GroupNoteModel] | None) -> list[dict[str, Any]]:
    """Serializa notes para estado.json (array completo enviado por el front)."""
    if not notes:
        return []
    return [
        {
            "id": note.id,
            "group_id": note.group_id,
            "text": note.text,
            "created_at": note.created_at,
            "updated_at": note.updated_at,
        }
        for note in notes
    ]


def _merge_partial_estado(existing: dict[str, Any], patch: ProyectoPartialSaveRequest) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in patch.model_dump(exclude_unset=True, exclude={"nombre"}).items():
        if key == "notes":
            # Reemplazo completo del array (mismo patrón que marks / user_cuts).
            # No se valida existencia de group_id en topología: notas huérfanas se conservan.
            merged["notes"] = _serialize_notes(patch.notes)
        else:
            merged[key] = value
    return merged


def _persist_estado_proyecto(
    db: Session,
    proyecto: Proyecto,
    estado: dict[str, Any],
) -> str:
    estado_bytes = json.dumps(estado, ensure_ascii=False).encode("utf-8")
    try:
        ruta_r2 = subir_archivo(
            estado_bytes,
            proyecto.usuario_id,
            proyecto.id,
            STATE_FILENAME,
            content_type="application/json",
        )
    except RuntimeError as exc:
        logger.exception("Error al subir estado parcial del proyecto %s a R2", proyecto.id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo guardar el estado del proyecto en el almacenamiento en la nube",
        ) from exc

    actualizado_at = datetime.now(timezone.utc).isoformat()
    meta = dict(proyecto.metadata_impresion or {})
    meta["estado_r2"] = ruta_r2
    meta["estado_actualizado_at"] = actualizado_at
    proyecto.metadata_impresion = meta
    _touch_proyecto_edicion(proyecto)

    try:
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Error al persistir metadata de estado del proyecto %s", proyecto.id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo guardar el estado del proyecto en la base de datos",
        ) from exc

    db.refresh(proyecto)
    return actualizado_at


def _proyecto_to_response(proyecto: Proyecto, *, include_download_url: bool = False) -> ProyectoResponse:
    download_url = obtener_url_archivo(proyecto.url_archivo) if include_download_url else None
    fecha_edicion = proyecto.fecha_ultima_edicion or proyecto.fecha_creacion
    return ProyectoResponse(
        id=str(proyecto.id),
        nombre=proyecto.nombre,
        formato=proyecto.formato,
        tamano_bytes=proyecto.tamano_bytes,
        url_archivo=proyecto.url_archivo,
        metadata_impresion=proyecto.metadata_impresion,
        fecha_creacion=proyecto.fecha_creacion.isoformat(),
        fecha_ultima_edicion=fecha_edicion.isoformat(),
        download_url=download_url,
    )


def _get_user_proyecto(db: Session, user: Usuario, proyecto_id: UUID) -> Proyecto:
    """Devuelve el proyecto si el usuario es dueño o admin."""
    proyecto = db.get(Proyecto, proyecto_id)
    if proyecto is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Proyecto no encontrado")
    if proyecto.usuario_id != user.id and not is_admin_user(user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Proyecto no encontrado")
    return proyecto


def list_proyectos_for_user(db: Session, user: Usuario) -> ProyectoListResponse:
    proyectos = (
        db.query(Proyecto)
        .filter(Proyecto.usuario_id == user.id)
        .order_by(Proyecto.fecha_ultima_edicion.desc(), Proyecto.fecha_creacion.desc())
        .all()
    )
    return ProyectoListResponse(
        proyectos=[_proyecto_to_response(p) for p in proyectos],
        total=len(proyectos),
    )


async def create_proyecto_for_user(
    file: UploadFile,
    nombre: str | None,
    background_tasks: BackgroundTasks,
    user: Usuario,
    db: Session,
) -> JSONResponse:
    """Guarda un proyecto: procesa el modelo, sube a R2 y persiste metadatos en BD."""
    proyecto_id = uuid.uuid4()
    processed = await process_model_file(file, str(proyecto_id), background_tasks)

    original_filename = processed["original_filename"]
    file_extension = processed["file_extension"]
    file_path = processed["file_path"]
    total_bytes = processed["file_size_bytes"]
    display_name = (nombre or Path(original_filename).stem).strip()
    if not display_name:
        raise HTTPException(status_code=400, detail="El nombre del proyecto no puede estar vacío")

    try:
        with open(file_path, "rb") as archivo_local:
            ruta_r2 = subir_archivo(
                archivo_local,
                user.id,
                proyecto_id,
                original_filename,
            )
    except RuntimeError as exc:
        logger.exception("Error al subir proyecto a R2")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo guardar el archivo en el almacenamiento en la nube",
        ) from exc

    proyecto = Proyecto(
        id=proyecto_id,
        usuario_id=user.id,
        nombre=display_name,
        formato=file_extension,
        tamano_bytes=total_bytes,
        url_archivo=ruta_r2,
        metadata_impresion={
            "archivo_original": original_filename,
            "summary": processed["summary"],
        },
        fecha_ultima_edicion=datetime.now(timezone.utc),
    )
    db.add(proyecto)

    try:
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        try:
            eliminar_archivo(ruta_r2)
        except RuntimeError:
            logger.exception("No se pudo revertir el archivo en R2 tras fallo en BD")
        logger.exception("Error al guardar proyecto en la base de datos")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo guardar el proyecto en la base de datos",
        ) from exc

    db.refresh(proyecto)
    logger.info("Proyecto guardado: %s para usuario %s", proyecto.id, user.id)

    response_content = {
        **_proyecto_to_response(proyecto).model_dump(),
        "message": "Proyecto guardado correctamente.",
        "proyecto_id": str(proyecto.id),
        "file_id": str(proyecto.id),
        "original_filename": original_filename,
        "file_size_mb": processed["file_size_mb"],
        "summary": processed["summary"],
        "topology": processed["topology"],
        "preview_obj": processed["preview_obj"],
        "timing": processed["timing"],
    }
    return JSONResponse(content=response_content, status_code=status.HTTP_201_CREATED)


@router.post("/projects", status_code=status.HTTP_201_CREATED)
async def crear_proyecto(
    file: UploadFile = File(...),
    nombre: str | None = Form(default=None),
    background_tasks: BackgroundTasks = None,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Guarda un proyecto del usuario: procesa el modelo, sube el archivo pesado a R2 y persiste metadatos."""
    return await create_proyecto_for_user(file, nombre, background_tasks, current_user, db)


@router.get("/projects", response_model=ProyectoListResponse)
def listar_proyectos(
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return list_proyectos_for_user(db, current_user)


@router.get("/projects/{proyecto_id}", response_model=ProyectoResponse)
def obtener_proyecto(
    proyecto_id: UUID,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    proyecto = _get_user_proyecto(db, current_user, proyecto_id)
    return _proyecto_to_response(proyecto, include_download_url=True)


@router.patch("/projects/{proyecto_id}", response_model=ProyectoRenameResponse)
def renombrar_proyecto(
    proyecto_id: UUID,
    body: ProyectoRenameRequest,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Renombra el proyecto en la base de datos y sincroniza `nombre` en estado.json (R2)
    si el proyecto ya tiene estado guardado. No altera el archivo 3D ni `archivo_original`.
    """
    proyecto = _get_user_proyecto(db, current_user, proyecto_id)

    if proyecto.nombre == body.nombre:
        return _rename_response(proyecto, estado_sincronizado=False, estado_actualizado_at=None)

    estado_sincronizado, estado_actualizado_at = _set_proyecto_nombre(db, proyecto, body.nombre)
    logger.info(
        "Proyecto renombrado: %s → %r (usuario %s, estado_sync=%s)",
        proyecto.id,
        body.nombre,
        current_user.id,
        estado_sincronizado,
    )
    return _rename_response(
        proyecto,
        estado_sincronizado=estado_sincronizado,
        estado_actualizado_at=estado_actualizado_at,
    )


@router.post("/projects/{proyecto_id}/open")
def abrir_proyecto(
    proyecto_id: UUID,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Carga la geometría de un proyecto guardado: descarga el modelo desde R2,
    reprocesa el pipeline y devuelve topology + preview (igual que al subir).
    """
    proyecto = _get_user_proyecto(db, current_user, proyecto_id)
    timer = PipelineTimer("open_project_endpoint")

    try:
        with timer.step("load_saved_proyecto"):
            result, original_filename = load_saved_proyecto_phase1(proyecto)
    except HTTPException:
        raise
    except RuntimeError as exc:
        logger.exception("Error al descargar proyecto %s desde R2", proyecto.id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo descargar el archivo del proyecto desde el almacenamiento",
        ) from exc
    except Exception as exc:
        logger.exception("Error al abrir proyecto %s", proyecto.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo procesar el proyecto guardado",
        ) from exc

    meta = proyecto.metadata_impresion or {}
    summary = meta.get("summary")
    file_size_mb = round(proyecto.tamano_bytes / 1024 / 1024, 2)
    saved_state = _load_estado_proyecto(proyecto)

    response_content = {
        **_proyecto_to_response(proyecto).model_dump(),
        **build_geometry_payload(
            str(proyecto.id),
            original_filename,
            result,
            file_size_mb=file_size_mb,
            summary=summary,
            timing=timer.report(),
        ),
        "saved_state": saved_state or None,
        "estado_actualizado_at": meta.get("estado_actualizado_at"),
    }
    return JSONResponse(content=response_content)


@router.patch("/projects/{proyecto_id}/state", response_model=ProyectoStateSaveResponse)
def guardar_estado_parcial_proyecto(
    proyecto_id: UUID,
    body: ProyectoPartialSaveRequest,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Guarda de forma parcial las modificaciones del proyecto (clasificación, fusiones,
    planchas, etc.) y sube el estado consolidado a R2 como estado.json.
    """
    if not body.model_fields_set:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Debes enviar al menos un campo para guardar",
        )

    proyecto = _get_user_proyecto(db, current_user, proyecto_id)
    merged_estado = _merge_partial_estado(_load_estado_proyecto(proyecto), body)

    if "nombre" in body.model_fields_set and body.nombre is not None:
        nombre = body.nombre.strip()
        if not nombre:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El nombre del proyecto no puede estar vacío",
            )
        proyecto.nombre = nombre
        merged_estado["nombre"] = nombre

    estado_actualizado_at = _persist_estado_proyecto(db, proyecto, merged_estado)
    meta = proyecto.metadata_impresion or {}

    logger.info("Estado parcial guardado para proyecto %s (usuario %s)", proyecto.id, current_user.id)
    return ProyectoStateSaveResponse(
        proyecto_id=str(proyecto.id),
        nombre=proyecto.nombre,
        estado_r2=meta.get("estado_r2", ""),
        estado_actualizado_at=estado_actualizado_at,
        estado=merged_estado,
        message="Estado del proyecto guardado correctamente.",
    )


@router.get("/projects/{proyecto_id}/state", response_model=ProyectoStateSaveResponse)
def obtener_estado_proyecto(
    proyecto_id: UUID,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Devuelve el último estado guardado del proyecto desde R2."""
    proyecto = _get_user_proyecto(db, current_user, proyecto_id)
    meta = proyecto.metadata_impresion or {}
    estado = _load_estado_proyecto(proyecto)
    return ProyectoStateSaveResponse(
        proyecto_id=str(proyecto.id),
        nombre=proyecto.nombre,
        estado_r2=str(meta.get("estado_r2") or ""),
        estado_actualizado_at=str(meta.get("estado_actualizado_at") or ""),
        estado=estado,
        message="Estado del proyecto cargado correctamente.",
    )


@router.get("/projects/{proyecto_id}/download")
def descargar_proyecto(
    proyecto_id: UUID,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    proyecto = _get_user_proyecto(db, current_user, proyecto_id)
    try:
        download_url = obtener_url_archivo(proyecto.url_archivo)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo generar la URL de descarga",
        ) from exc

    return {
        "proyecto_id": str(proyecto.id),
        "nombre": proyecto.nombre,
        "formato": proyecto.formato,
        "download_url": download_url,
    }


@router.delete("/projects/{proyecto_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_proyecto(
    proyecto_id: UUID,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    proyecto = _get_user_proyecto(db, current_user, proyecto_id)
    ruta_r2 = proyecto.url_archivo
    meta = proyecto.metadata_impresion or {}
    estado_r2 = meta.get("estado_r2")

    db.delete(proyecto)
    try:
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Error al eliminar proyecto de la base de datos")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo eliminar el proyecto",
        ) from exc

    try:
        eliminar_archivo(ruta_r2)
    except RuntimeError:
        logger.exception("Proyecto eliminado en BD pero no en R2: %s", ruta_r2)

    if isinstance(estado_r2, str) and estado_r2.strip() and estado_r2 != ruta_r2:
        try:
            eliminar_archivo(estado_r2.strip())
        except RuntimeError:
            logger.exception("Estado del proyecto no eliminado en R2: %s", estado_r2)

    return None
