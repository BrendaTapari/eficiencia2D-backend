import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")
TEMPLATE_DIR = PROJECT_DIR / "templates" / "email"


def _env(key: str, default: str = "") -> str:
    """Lee una variable de entorno y elimina comillas envolventes."""
    return os.environ.get(key, default).strip().strip('"').strip("'")


def _mail_password() -> str:
    """App Password de Gmail: quitar espacios/saltos de línea invisibles."""
    return "".join(_env("MAIL_PASSWORD").split())


def get_frontend_base_url() -> str:
    """
    URL base del frontend para links en correos.
    Prioridad: FRONTEND_URL_VERCEL → FRONTEND_URL.
    """
    raw = _env("FRONTEND_URL_VERCEL") or _env("FRONTEND_URL")
    return raw.rstrip("/")


def build_verification_url(token: str) -> str:
    return f"{get_frontend_base_url()}/verificar-correo?token={token}"


def build_password_reset_url(token: str) -> str:
    return f"{get_frontend_base_url()}/restablecer-contrasena?token={token}"


def _mail_port() -> int:
    return int(_env("MAIL_PORT", "587"))


def _connection_config() -> ConnectionConfig:
    port = _mail_port()
    use_ssl = port == 465

    return ConnectionConfig(
        MAIL_USERNAME=_env("MAIL_USERNAME"),
        MAIL_PASSWORD=_mail_password(),
        MAIL_FROM=_env("MAIL_FROM"),
        MAIL_FROM_NAME=_env("MAIL_FROM_NAME", "Eficiencia 2D"),
        MAIL_PORT=port,
        MAIL_SERVER=_env("MAIL_SERVER", "smtp.gmail.com"),
        MAIL_STARTTLS=not use_ssl,
        MAIL_SSL_TLS=use_ssl,
        USE_CREDENTIALS=True,
        VALIDATE_CERTS=True,
        TEMPLATE_FOLDER=TEMPLATE_DIR,
        SUPPRESS_SEND=0,
    )


def get_fastmail() -> FastMail:
    """Instancia nueva en cada envío para no cachear credenciales viejas."""
    return FastMail(_connection_config())


def get_mail_config_status() -> dict[str, object]:
    """Diagnóstico seguro: no expone la contraseña, solo metadatos."""
    pwd = _mail_password()
    username = _env("MAIL_USERNAME")
    mail_from = _env("MAIL_FROM")
    port = _mail_port()
    env_path = PROJECT_DIR / ".env"
    issues = validate_mail_config()

    if username and mail_from and username.lower() != mail_from.lower():
        issues.append("Para Gmail, MAIL_FROM debe ser igual a MAIL_USERNAME")
    if pwd and len(pwd) != 16:
        issues.append(
            f"MAIL_PASSWORD tiene {len(pwd)} caracteres; una App Password de Google tiene 16"
        )

    return {
        "env_file": str(env_path),
        "env_file_exists": env_path.is_file(),
        "mail_username": username,
        "mail_from": mail_from,
        "mail_server": _env("MAIL_SERVER", "smtp.gmail.com"),
        "mail_port": port,
        "mail_ssl_tls": port == 465,
        "mail_starttls": port != 465,
        "password_length": len(pwd),
        "password_is_16_chars": len(pwd) == 16,
        "template_dir": str(TEMPLATE_DIR),
        "frontend_base_url": get_frontend_base_url(),
        "issues": issues,
        "ok": len(issues) == 0,
    }


def validate_mail_config() -> list[str]:
    """Devuelve una lista de problemas de configuración (vacía = OK)."""
    issues: list[str] = []
    if not _env("MAIL_USERNAME"):
        issues.append("MAIL_USERNAME no está definido")
    if not _mail_password():
        issues.append("MAIL_PASSWORD no está definido")
    if not _env("MAIL_FROM"):
        issues.append("MAIL_FROM no está definido")
    if not _env("MAIL_SERVER"):
        issues.append("MAIL_SERVER no está definido")
    if not TEMPLATE_DIR.is_dir():
        issues.append(f"No existe la carpeta de plantillas: {TEMPLATE_DIR}")
    elif not (TEMPLATE_DIR / "verify_account.html").is_file():
        issues.append("Falta templates/email/verify_account.html")
    if not get_frontend_base_url():
        issues.append("FRONTEND_URL_VERCEL o FRONTEND_URL no está definido")
    return issues


def is_mail_configured() -> bool:
    return len(validate_mail_config()) == 0


def _app_name() -> str:
    return _env("MAIL_FROM_NAME", "Eficiencia 2D")


async def send_verification_email(
    recipient: str,
    token: str,
    nombre: str | None = None,
) -> None:
    """Envía el correo de verificación de cuenta usando la plantilla HTML."""
    verification_url = build_verification_url(token)
    display_name = (nombre or "").strip() or recipient.split("@")[0]

    message = MessageSchema(
        subject=f"Verifica tu cuenta — {_app_name()}",
        recipients=[recipient],
        template_body={
            "app_name": _app_name(),
            "nombre": display_name,
            "verification_url": verification_url,
            "token": token,
        },
        subtype=MessageType.html,
    )

    try:
        await get_fastmail().send_message(message, template_name="verify_account.html")
        logger.info("Correo de verificación enviado a %s", recipient)
    except Exception:
        logger.exception("Error al enviar correo de verificación a %s", recipient)
        raise


async def send_password_reset_email(
    recipient: str,
    token: str,
    nombre: str | None = None,
) -> None:
    """Envía el correo de recuperación de contraseña (plantilla reutilizable)."""
    reset_url = build_password_reset_url(token)
    display_name = (nombre or "").strip() or recipient.split("@")[0]

    message = MessageSchema(
        subject=f"Restablecer contraseña — {_app_name()}",
        recipients=[recipient],
        template_body={
            "app_name": _app_name(),
            "nombre": display_name,
            "reset_url": reset_url,
            "token": token,
        },
        subtype=MessageType.html,
    )

    try:
        await get_fastmail().send_message(message, template_name="password_reset.html")
        logger.info("Correo de recuperación enviado a %s", recipient)
    except Exception:
        logger.exception("Error al enviar correo de recuperación a %s", recipient)
        raise
