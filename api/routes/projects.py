import logging
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.deps import get_current_user
from api.routes.uploads import (
    build_geometry_payload,
    load_saved_proyecto_phase1,
    process_model_file,
)
from core.profiler import PipelineTimer
from database import Proyecto, Usuario, get_db
from database.storage import eliminar_archivo, obtener_url_archivo, subir_archivo

router = APIRouter()
logger = logging.getLogger(__name__)


class ProyectoResponse(BaseModel):
    id: str
    nombre: str
    formato: str
    tamano_bytes: int
    url_archivo: str
    metadata_impresion: dict[str, Any] | None
    fecha_creacion: str
    download_url: str | None = None


class ProyectoListResponse(BaseModel):
    proyectos: list[ProyectoResponse]
    total: int


def _proyecto_to_response(proyecto: Proyecto, *, include_download_url: bool = False) -> ProyectoResponse:
    download_url = obtener_url_archivo(proyecto.url_archivo) if include_download_url else None
    return ProyectoResponse(
        id=str(proyecto.id),
        nombre=proyecto.nombre,
        formato=proyecto.formato,
        tamano_bytes=proyecto.tamano_bytes,
        url_archivo=proyecto.url_archivo,
        metadata_impresion=proyecto.metadata_impresion,
        fecha_creacion=proyecto.fecha_creacion.isoformat(),
        download_url=download_url,
    )


def _get_user_proyecto(db: Session, user: Usuario, proyecto_id: UUID) -> Proyecto:
    proyecto = db.get(Proyecto, proyecto_id)
    if proyecto is None or proyecto.usuario_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Proyecto no encontrado")
    return proyecto


def list_proyectos_for_user(db: Session, user: Usuario) -> ProyectoListResponse:
    proyectos = (
        db.query(Proyecto)
        .filter(Proyecto.usuario_id == user.id)
        .order_by(Proyecto.fecha_creacion.desc())
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
    }
    return JSONResponse(content=response_content)


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

    return None
