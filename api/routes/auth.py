import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from api.deps import get_current_user
from core.security import create_access_token, hash_password, verify_password
from database import ConfiguracionUsuario, Usuario, get_db
from utils.mailer import (
    build_verification_url,
    get_mail_config_status,
    is_mail_configured,
    send_verification_email,
    validate_mail_config,
)

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
    verification_email_scheduled: bool
    user: UserResponse


class TestEmailRequest(BaseModel):
    email: EmailStr
    nombre: str | None = Field(default="Usuario de prueba", max_length=120)


class TestEmailResponse(BaseModel):
    ok: bool
    message: str
    verification_url: str


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
    logger.info("Iniciando envío de correo de verificación a %s", email)
    try:
        await send_verification_email(recipient=email, token=token, nombre=nombre)
    except Exception:
        logger.exception(
            "No se pudo enviar el correo de verificación en segundo plano a %s",
            email,
        )


@router.post("/auth/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(
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
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Error de base de datos al registrar usuario")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo conectar con la base de datos. Intentá de nuevo en unos segundos.",
        ) from None

    db.refresh(user)

    mail_ok = is_mail_configured()
    if mail_ok:
        background_tasks.add_task(
            _send_verification_email_task,
            user.email,
            verification_token,
            user.nombre,
        )
        logger.info("Correo de verificación programado para %s", user.email)
    else:
        logger.error(
            "Correo NO programado para %s — configuración incompleta: %s",
            user.email,
            "; ".join(validate_mail_config()),
        )

    return RegisterResponse(
        message=(
            "Cuenta creada. Revisá tu correo para verificar tu cuenta."
            if mail_ok
            else "Cuenta creada, pero el servidor no pudo programar el correo de verificación. "
            "Contactá al administrador o probá POST /api/auth/test-email."
        ),
        email=user.email,
        verification_email_scheduled=mail_ok,
        user=_user_to_response(user),
    )


@router.get("/auth/mail-status")
def mail_status():
    """
    Muestra qué configuración SMTP lee el proceso (sin exponer la contraseña).
    Compará password_length con 16 y env_file con la ruta del servidor.
    """
    return get_mail_config_status()


@router.post("/auth/test-email", response_model=TestEmailResponse)
async def test_email(body: TestEmailRequest):
    """
    Envía un correo de verificación de prueba de forma síncrona.
    Usá este endpoint en Swagger para ver el error exacto si el SMTP falla.
    """
    issues = validate_mail_config()
    if issues:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": "Configuración de correo incompleta", "issues": issues},
        )

    token = _create_verification_token()

    try:
        await send_verification_email(
            recipient=body.email.lower(),
            token=token,
            nombre=body.nombre,
        )
    except Exception as exc:
        logger.exception("Fallo test-email a %s", body.email)
        detail = str(exc)
        if "535" in detail or "BadCredentials" in detail or "not accepted" in detail.lower():
            detail = (
                "Gmail rechazó usuario/contraseña (535). "
                "Generá una nueva App Password en Google (16 caracteres), "
                "actualizá MAIL_PASSWORD en el .env DEL SERVIDOR y reiniciá el servicio. "
                "Verificá GET /api/auth/mail-status (password_length debe ser 16). "
                f"Detalle SMTP: {exc}"
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
        ) from exc

    return TestEmailResponse(
        ok=True,
        message=f"Correo de prueba enviado a {body.email}",
        verification_url=build_verification_url(token),
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
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Error de base de datos al verificar correo")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo conectar con la base de datos. Intentá de nuevo en unos segundos.",
        ) from None
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
