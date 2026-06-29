import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.deps import get_current_user
from database import Rol, Usuario, get_db

router = APIRouter()
logger = logging.getLogger(__name__)

class RolResponse(BaseModel):
    id: int
    rol: str

class RolUpdateRequest(BaseModel):
    rol: str

ADMIN_ROL_ID = 2

@router.get("/rols", response_model=list[RolResponse])
def list_rols(db: Session = Depends(get_db)):
    return db.query(Rol).filter(Rol.id != ADMIN_ROL_ID).all()


@router.get("/rols/user/{user_id}", response_model=RolResponse)
def get_rol_by_user_id(user_id: UUID, db: Session = Depends(get_db)):
    user = db.get(Usuario, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado")
    return user.rol

@router.patch("/rols/user/{user_id}", response_model=RolResponse)
def update_rol_by_user_id(user_id: UUID, rol: RolUpdateRequest, db: Session = Depends(get_db)):
    user = db.get(Usuario, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado")
    user.rol = rol.rol
    db.commit()
    db.refresh(user)
    return user.rol