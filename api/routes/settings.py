import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.deps import get_current_user
from database import ConfiguracionUsuario, Usuario, get_db

router = APIRouter()
logger = logging.getLogger(__name__)

HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
MODELO_COLOR_CHANNELS = ("wall", "floor", "background")


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


def _validate_model_color(value: Any, channel: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"colores_modelo.{channel} debe ser un color hex (#RRGGBB) o null",
        )
    normalized = value.strip()
    if not normalized:
        return None
    if not HEX_COLOR_RE.fullmatch(normalized):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"colores_modelo.{channel} debe tener formato #RRGGBB",
        )
    return normalized.lower()


def _merge_colores_modelo(existing: Any, patch: Any) -> dict[str, str | None] | None:
    """
    Merge parcial de colores del modelo.
    `patch is None` elimina todos los overrides.
    Cada canal null en patch limpia ese canal; omitir conserva el valor previo.
    """
    if patch is None:
        return None

    if not isinstance(patch, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="colores_modelo debe ser un objeto o null",
        )

    unknown = set(patch.keys()) - set(MODELO_COLOR_CHANNELS)
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"colores_modelo contiene claves inválidas: {', '.join(sorted(unknown))}",
        )

    base: dict[str, str | None] = {}
    if isinstance(existing, dict):
        for channel in MODELO_COLOR_CHANNELS:
            if channel in existing:
                base[channel] = _validate_model_color(existing.get(channel), channel)

    for channel in MODELO_COLOR_CHANNELS:
        if channel not in patch:
            continue
        base[channel] = _validate_model_color(patch[channel], channel)

    if not base or all(base.get(channel) is None for channel in MODELO_COLOR_CHANNELS):
        return None

    return base


def _merge_preferencias_interfaz(
    existing: dict[str, Any] | None,
    patch: dict[str, Any],
) -> dict[str, Any]:
    """Merge superficial de preferencias_interfaz; colores_modelo merge por canal."""
    merged = dict(existing or {})

    for key, value in patch.items():
        if key == "colores_modelo":
            merged_colors = _merge_colores_modelo(merged.get("colores_modelo"), value)
            if merged_colors is None:
                merged.pop("colores_modelo", None)
            else:
                merged["colores_modelo"] = merged_colors
        else:
            merged[key] = value

    return merged


def _normalize_config(config: ConfiguracionUsuario) -> bool:
    """Rellena NULLs de filas legacy y devuelve si hubo cambios."""
    changed = False
    if config.tema_color is None:
        config.tema_color = "oscuro"
        changed = True
    if config.idioma is None:
        config.idioma = "es"
        changed = True
    if config.notificaciones_email is None:
        config.notificaciones_email = True
        changed = True
    return changed


def _config_to_response(config: ConfiguracionUsuario) -> SettingsResponse:
    return SettingsResponse(
        tema_color=config.tema_color or "oscuro",
        idioma=config.idioma or "es",
        notificaciones_email=(
            True if config.notificaciones_email is None else config.notificaciones_email
        ),
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
        try:
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            logger.exception("Error de base de datos al crear configuración")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No se pudo conectar con la base de datos. Intentá de nuevo en unos segundos.",
            ) from None
        db.refresh(config)
        logger.info("Configuración creada para usuario %s", user.id)
    elif _normalize_config(config):
        try:
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            logger.exception("Error de base de datos al normalizar configuración")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No se pudo conectar con la base de datos. Intentá de nuevo en unos segundos.",
            ) from None
        db.refresh(config)
        logger.info("Configuración legacy normalizada para usuario %s", user.id)
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

    if "preferencias_interfaz" in updates:
        patch_prefs = updates.pop("preferencias_interfaz")
        if patch_prefs is None:
            config.preferencias_interfaz = None
        elif isinstance(patch_prefs, dict):
            config.preferencias_interfaz = _merge_preferencias_interfaz(
                config.preferencias_interfaz,
                patch_prefs,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="preferencias_interfaz debe ser un objeto o null",
            )

    for field, value in updates.items():
        setattr(config, field, value)

    _normalize_config(config)
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Error de base de datos al actualizar configuración")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo conectar con la base de datos. Intentá de nuevo en unos segundos.",
        ) from None
    db.refresh(config)
    return _config_to_response(config)
