import logging
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.deps import get_current_user
from database import ConfiguracionUsuario, Usuario, get_db

router = APIRouter()
logger = logging.getLogger(__name__)


class SettingsResponse(BaseModel):
    tema_color: str
    idioma: str
    notificaciones_email: bool
    preferencias_interfaz: dict[str, Any] | None


class SettingsUpdateRequest(BaseModel):
    tema_color: str | None = Field(default=None, min_length=1, max_length=64)
    idioma: str | None = Field(default=None, min_length=2, max_length=5)
    notificaciones_email: bool | None = None
    preferencias_interfaz: dict[str, Any] | None = None


def _config_to_response(config: ConfiguracionUsuario) -> SettingsResponse:
    return SettingsResponse(
        tema_color=config.tema_color,
        idioma=config.idioma,
        notificaciones_email=config.notificaciones_email,
        preferencias_interfaz=config.preferencias_interfaz,
    )


def _get_or_create_config(db: Session, user: Usuario) -> ConfiguracionUsuario:
    config = (
        db.query(ConfiguracionUsuario)
        .filter(ConfiguracionUsuario.usuario_id == user.id)
        .first()
    )
    if config is None:
        config = ConfiguracionUsuario(usuario_id=user.id)
        db.add(config)
        db.commit()
        db.refresh(config)
        logger.info("Configuración creada para usuario %s", user.id)
    return config


@router.get("/settings/me", response_model=SettingsResponse)
def get_my_settings(
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    config = _get_or_create_config(db, current_user)
    return _config_to_response(config)


@router.patch("/settings/me", response_model=SettingsResponse)
def update_my_settings(
    body: SettingsUpdateRequest,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    config = _get_or_create_config(db, current_user)
    updates = body.model_dump(exclude_unset=True)

    for field, value in updates.items():
        setattr(config, field, value)

    db.commit()
    db.refresh(config)
    return _config_to_response(config)
