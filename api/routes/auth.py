import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.deps import get_current_user
from core.security import create_access_token, hash_password, verify_password
from database import ConfiguracionUsuario, Usuario, get_db
from utils.mailer import send_verification_email

router = APIRouter()
logger = logging.getLogger(__name__)

ESTADO_ACTIVO = "activo"
ESTADO_PENDIENTE = "pendiente_verificacion"


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    nombre: str | None = Field(default=None, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class VerifyEmailRequest(BaseModel):
    token: str = Field(min_length=36, max_length=64)


class UserResponse(BaseModel):
    id: str
    email: str
    nombre: str | None
    estado: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class RegisterResponse(BaseModel):
    message: str
    email: str
    verification_email_sent: bool
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


def _create_verification_token() -> str:
    return str(uuid.uuid4())


async def _send_verification_email_task(
    email: str,
    token: str,
    nombre: str | None,
) -> None:
    try:
        await send_verification_email(recipient=email, token=token, nombre=nombre)
    except Exception:
        logger.exception(
            "No se pudo enviar el correo de verificación en segundo plano a %s",
            email,
        )


@router.post("/auth/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(
    body: RegisterRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    email = body.email.lower()
    existing = db.query(Usuario).filter(Usuario.email == email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ya existe una cuenta con ese email",
        )

    verification_token = _create_verification_token()
    user = Usuario(
        email=email,
        password_hash=hash_password(body.password),
        nombre=body.nombre.strip() if body.nombre else None,
        estado=ESTADO_PENDIENTE,
        email_verification_token=verification_token,
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

    background_tasks.add_task(
        _send_verification_email_task,
        user.email,
        verification_token,
        user.nombre,
    )

    return RegisterResponse(
        message="Cuenta creada. Revisá tu correo para verificar tu cuenta.",
        email=user.email,
        verification_email_sent=True,
        user=_user_to_response(user),
    )


@router.post("/auth/verify-email", response_model=AuthResponse)
def verify_email(body: VerifyEmailRequest, db: Session = Depends(get_db)):
    user = (
        db.query(Usuario)
        .filter(Usuario.email_verification_token == body.token.strip())
        .first()
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token de verificación inválido o expirado",
        )

    user.estado = ESTADO_ACTIVO
    user.email_verification_token = None
    user.email_verified_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)

    logger.info("Cuenta verificada: %s", user.email)
    return _build_auth_response(user)


@router.post("/auth/login", response_model=AuthResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(Usuario).filter(Usuario.email == body.email.lower()).first()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email o contraseña incorrectos",
        )

    if user.estado == ESTADO_PENDIENTE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Debes verificar tu correo electrónico antes de iniciar sesión",
        )

    if user.estado != ESTADO_ACTIVO:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cuenta inactiva",
        )

    return _build_auth_response(user)


@router.get("/auth/me", response_model=UserResponse)
def me(current_user: Usuario = Depends(get_current_user)):
    return _user_to_response(current_user)
