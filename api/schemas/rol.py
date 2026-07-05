from pydantic import BaseModel

from database import Rol, Usuario

ADMIN_ROL_ID = 2
DEFAULT_ROL_ID = 1


class RolResponse(BaseModel):
    id: int
    rol: str


def rol_to_response(rol: Rol | None, *, rol_id: int | None = None) -> RolResponse:
    if rol is not None:
        return RolResponse(id=rol.id, rol=rol.rol)
    resolved_id = rol_id if rol_id is not None else DEFAULT_ROL_ID
    label = "admin" if resolved_id == ADMIN_ROL_ID else "estudiante"
    return RolResponse(id=resolved_id, rol=label)


def user_rol_fields(user: Usuario) -> tuple[int, str]:
    rol_id = user.rol_id if user.rol_id is not None else DEFAULT_ROL_ID
    if user.rol is not None:
        return rol_id, user.rol.rol
    return rol_id, "admin" if rol_id == ADMIN_ROL_ID else "estudiante"


def is_admin_user(user: Usuario) -> bool:
    return user.rol_id == ADMIN_ROL_ID
