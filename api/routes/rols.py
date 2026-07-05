import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload

from api.schemas.rol import ADMIN_ROL_ID, RolResponse, rol_to_response
from database import Rol, Usuario, get_db

router = APIRouter()
logger = logging.getLogger(__name__)


class RolUpdateRequest(BaseModel):
    rol_id: int = Field(..., ge=1)


@router.get("/rols", response_model=list[RolResponse])
def list_rols(db: Session = Depends(get_db)):
    return db.query(Rol).filter(Rol.id != ADMIN_ROL_ID).order_by(Rol.id).all()


def _get_user_with_rol(db: Session, user_id: UUID) -> Usuario:
    user = (
        db.query(Usuario)
        .options(joinedload(Usuario.rol))
        .filter(Usuario.id == user_id)
        .first()
    )
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado")
    return user


@router.get("/rols/user/{user_id}", response_model=RolResponse)
def get_rol_by_user_id(user_id: UUID, db: Session = Depends(get_db)):
    user = _get_user_with_rol(db, user_id)
    return rol_to_response(user.rol, rol_id=user.rol_id)


@router.patch("/rols/user/{user_id}", response_model=RolResponse)
def update_rol_by_user_id(
    user_id: UUID,
    body: RolUpdateRequest,
    db: Session = Depends(get_db),
):
    user = _get_user_with_rol(db, user_id)
    rol = db.get(Rol, body.rol_id)
    if rol is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rol no encontrado")
    if body.rol_id == ADMIN_ROL_ID:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se puede asignar el rol de administrador desde este endpoint",
        )

    user.rol_id = body.rol_id
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Error al actualizar rol del usuario %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo actualizar el rol del usuario",
        ) from None

    db.refresh(user)
    return rol_to_response(rol)
