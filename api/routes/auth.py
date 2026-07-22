import logging
import os
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from api.deps import get_current_user
from api.datetime_utils import dt_to_iso
from api.schemas.rol import DEFAULT_ROL_ID, RolResponse, user_rol_fields
from core.security import create_access_token, hash_password, verify_password
from database import ConfiguracionUsuario, Usuario, get_db
from database.database import get_db_config_status
from utils.mailer import (
    build_verification_url,
    get_mail_config_status,
    is_mail_configured,
    send_password_reset_email,
    send_verification_email,
    validate_mail_config,
)

router = APIRouter()
logger = logging.getLogger(__name__)

ESTADO_ACTIVO = "activo"
ESTADO_PENDIENTE = "pendiente_verificacion"
PASSWORD_RESET_EXPIRE_MINUTES = int(os.environ.get("PASSWORD_RESET_EXPIRE_MINUTES", "60"))


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    nombre: str | None = Field(default=None, max_length=120)
    acepto_terminos: bool = False


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
    rol_id: int
    rol: RolResponse
    acepto_terminos_at: str | None = None


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


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ForgotPasswordResponse(BaseModel):
    message: str


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=36, max_length=64)
    new_password: str = Field(min_length=6, max_length=128)


class ResetPasswordResponse(BaseModel):
    message: str


def _user_to_response(user: Usuario) -> UserResponse:
    rol_id, rol_name = user_rol_fields(user)
    return UserResponse(
        id=str(user.id),
        email=user.email,
        nombre=user.nombre,
        estado=user.estado,
        rol_id=rol_id,
        rol=RolResponse(id=rol_id, rol=rol_name),
        acepto_terminos_at=dt_to_iso(user.acepto_terminos_at),
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


async def _send_password_reset_email_task(
    email: str,
    token: str,
    nombre: str | None,
) -> None:
    logger.info("Iniciando envío de correo de recuperación a %s", email)
    try:
        await send_password_reset_email(recipient=email, token=token, nombre=nombre)
    except Exception:
        logger.exception(
            "No se pudo enviar el correo de recuperación en segundo plano a %s",
            email,
        )


@router.post("/auth/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    email = body.email.lower()
    if not body.acepto_terminos:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Debés aceptar los términos y condiciones para registrarte",
        )

    existing = db.query(Usuario).filter(Usuario.email == email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ya existe una cuenta con ese email",
        )

    verification_token = _create_verification_token()
    now = datetime.now(timezone.utc)
    user = Usuario(
        email=email,
        password_hash=hash_password(body.password),
        nombre=body.nombre.strip() if body.nombre else None,
        estado=ESTADO_PENDIENTE,
        email_verification_token=verification_token,
        rol_id=DEFAULT_ROL_ID,
        acepto_terminos_at=now,
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


@router.get("/auth/db-status")
def db_status():
    """
    Muestra a qué base de datos está conectado el servidor (sin credenciales).
    Verificá que database_host sea de Supabase y no localhost.
    """
    return get_db_config_status()


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


@router.post("/auth/forgot-password", response_model=ForgotPasswordResponse)
async def forgot_password(
    body: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Solicita un enlace de recuperación por correo.
    Siempre responde igual aunque el email no exista (por seguridad).
    """
    email = body.email.lower()
    user = db.query(Usuario).filter(Usuario.email == email).first()

    if user is not None and is_mail_configured():
        token = _create_verification_token()
        user.password_reset_token = token
        user.password_reset_expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=PASSWORD_RESET_EXPIRE_MINUTES
        )
        try:
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            logger.exception("Error al guardar token de recuperación para %s", email)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No se pudo procesar la solicitud. Intentá de nuevo en unos segundos.",
            ) from None

        background_tasks.add_task(
            _send_password_reset_email_task,
            user.email,
            token,
            user.nombre,
        )
        logger.info("Correo de recuperación programado para %s", user.email)
    elif user is not None:
        logger.error(
            "Recuperación NO programada para %s — correo no configurado: %s",
            email,
            "; ".join(validate_mail_config()),
        )

    return ForgotPasswordResponse(
        message=(
            "Si el correo está registrado, recibirás un enlace para restablecer tu contraseña."
        ),
    )


@router.post("/auth/reset-password", response_model=ResetPasswordResponse)
def reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Restablece la contraseña usando el token recibido por correo."""
    token = body.token.strip()
    user = (
        db.query(Usuario)
        .filter(Usuario.password_reset_token == token)
        .first()
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El enlace de recuperación es inválido o ya fue utilizado",
        )

    expires_at = user.password_reset_expires_at
    if expires_at is None or expires_at < datetime.now(timezone.utc):
        user.password_reset_token = None
        user.password_reset_expires_at = None
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El enlace de recuperación expiró. Solicitá uno nuevo.",
        )

    user.password_hash = hash_password(body.new_password)
    user.password_reset_token = None
    user.password_reset_expires_at = None

    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Error al restablecer contraseña")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo restablecer la contraseña. Intentá de nuevo en unos segundos.",
        ) from None

    logger.info("Contraseña restablecida para %s", user.email)
    return ResetPasswordResponse(message="Contraseña actualizada correctamente")


@router.get("/auth/me", response_model=UserResponse)
def me(current_user: Usuario = Depends(get_current_user)):
    return _user_to_response(current_user)
