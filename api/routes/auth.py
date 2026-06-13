import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.deps import get_current_user
from core.security import create_access_token, hash_password, verify_password
from database import ConfiguracionUsuario, Usuario, get_db

router = APIRouter()
logger = logging.getLogger(__name__)


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    nombre: str | None = Field(default=None, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseModel):
    id: str
    email: str
    nombre: str | None
    estado: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


def _user_to_response(user: Usuario) -> UserResponse:
    return UserResponse(
        id=str(user.id),
        email=user.email,
        nombre=user.nombre,
        estado=user.estado,
    )


def _build_auth_response(user: Usuario) -> AuthResponse:
    token = create_access_token(str(user.id))
    return AuthResponse(access_token=token, user=_user_to_response(user))


@router.post("/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(Usuario).filter(Usuario.email == body.email.lower()).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ya existe una cuenta con ese email",
        )

    user = Usuario(
        email=body.email.lower(),
        password_hash=hash_password(body.password),
        nombre=body.nombre.strip() if body.nombre else None,
    )
    config = ConfiguracionUsuario(usuario=user)

    db.add(user)
    db.add(config)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        logger.exception("Error de integridad al registrar usuario")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No se pudo crear la cuenta",
        ) from None

    db.refresh(user)
    return _build_auth_response(user)


@router.post("/auth/login", response_model=AuthResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(Usuario).filter(Usuario.email == body.email.lower()).first()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email o contraseña incorrectos",
        )

    if user.estado != "activo":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cuenta inactiva",
        )

    return _build_auth_response(user)


@router.get("/auth/me", response_model=UserResponse)
def me(current_user: Usuario = Depends(get_current_user)):
    return _user_to_response(current_user)
