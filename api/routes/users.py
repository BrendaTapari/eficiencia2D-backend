import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload

from api.deps import get_current_user
from api.routes.projects import (
    ProyectoListResponse,
    abrir_proyecto,
    create_proyecto_for_user,
    list_proyectos_for_user,
)
from api.schemas.rol import RolResponse, user_rol_fields
from core.security import hash_password, verify_password
from database import Proyecto, Usuario, get_db

router = APIRouter()
logger = logging.getLogger(__name__)


class UserProfileResponse(BaseModel):
    id: str
    email: str
    nombre: str | None
    estado: str
    rol_id: int
    rol: RolResponse
    fecha_creacion: str
    email_verified_at: str | None
    total_proyectos: int


class UpdateUserRequest(BaseModel):
    nombre: str = Field(min_length=1, max_length=120)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=6, max_length=128)


class ChangePasswordResponse(BaseModel):
    message: str


def _user_profile(db: Session, user: Usuario) -> UserProfileResponse:
    total_proyectos = db.query(Proyecto).filter(Proyecto.usuario_id == user.id).count()
    rol_id, rol_name = user_rol_fields(user)
    return UserProfileResponse(
        id=str(user.id),
        email=user.email,
        nombre=user.nombre,
        estado=user.estado,
        rol_id=rol_id,
        rol=RolResponse(id=rol_id, rol=rol_name),
        fecha_creacion=user.fecha_creacion.isoformat(),
        email_verified_at=(
            user.email_verified_at.isoformat() if user.email_verified_at else None
        ),
        total_proyectos=total_proyectos,
    )


def _load_user_with_rol(db: Session, user_id: UUID) -> Usuario:
    user = (
        db.query(Usuario)
        .options(joinedload(Usuario.rol))
        .filter(Usuario.id == user_id)
        .first()
    )
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado")
    return user


@router.get("/users/me", response_model=UserProfileResponse)
def get_my_profile(
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = _load_user_with_rol(db, current_user.id)
    return _user_profile(db, user)


@router.patch("/users/me", response_model=UserProfileResponse)
def update_my_profile(
    body: UpdateUserRequest,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    current_user.nombre = body.nombre.strip()
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Error al actualizar perfil de usuario %s", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo actualizar el perfil",
        ) from None
    db.refresh(current_user)
    user = _load_user_with_rol(db, current_user.id)
    return _user_profile(db, user)


@router.patch("/users/me/password", response_model=ChangePasswordResponse)
def change_my_password(
    body: ChangePasswordRequest,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not verify_password(body.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La contraseña actual es incorrecta",
        )

    if body.current_password == body.new_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La nueva contraseña debe ser distinta a la actual",
        )

    current_user.password_hash = hash_password(body.new_password)
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Error al cambiar contraseña de usuario %s", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo cambiar la contraseña",
        ) from None

    logger.info("Contraseña actualizada para usuario %s", current_user.id)
    return ChangePasswordResponse(message="Contraseña actualizada correctamente")


@router.get("/users/me/projects", response_model=ProyectoListResponse)
def list_my_projects(
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return list_proyectos_for_user(db, current_user)


@router.post("/users/me/projects/{proyecto_id}/open")
def open_my_project(
    proyecto_id: UUID,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return abrir_proyecto(proyecto_id, current_user, db)


@router.post("/users/me/projects", status_code=status.HTTP_201_CREATED)
async def create_my_project(
    file: UploadFile = File(...),
    nombre: str | None = Form(default=None),
    background_tasks: BackgroundTasks = None,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return await create_proyecto_for_user(file, nombre, background_tasks, current_user, db)
