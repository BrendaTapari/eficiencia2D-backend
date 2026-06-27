import logging
import os
import socket
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, Integer, String, Numeric, BigInteger, ForeignKey, DateTime, create_engine, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)

# Ruta fija al .env del proyecto (systemd no siempre arranca desde ahí).
PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

Base = declarative_base()


def _prefer_ipv4_database_url(url: str) -> str:
    """
    En VPS sin ruteo IPv6, Supabase puede resolverse a IPv6 y fallar con 'No route to host'.
    Reemplaza el hostname por su dirección IPv4 cuando sea posible.
    """
    if "supabase.co" not in url:
        return url

    dialect_prefix = ""
    parse_url = url
    if url.startswith("postgresql+psycopg2://"):
        dialect_prefix = "postgresql+psycopg2://"
        parse_url = "postgresql://" + url[len(dialect_prefix) :]
    elif url.startswith("postgresql://"):
        dialect_prefix = "postgresql://"

    parsed = urlparse(parse_url)
    host = parsed.hostname
    if not host or host.replace(".", "").isdigit():
        return url

    try:
        infos = socket.getaddrinfo(host, parsed.port or 5432, socket.AF_INET, socket.SOCK_STREAM)
        ipv4 = infos[0][4][0]
    except OSError:
        logger.warning("No se pudo resolver IPv4 para %s; se usa el hostname original", host)
        return url

    port = parsed.port or 5432
    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"

    rebuilt = urlunparse(parsed._replace(netloc=f"{userinfo}{ipv4}:{port}"))
    if dialect_prefix == "postgresql+psycopg2://":
        rebuilt = rebuilt.replace("postgresql://", dialect_prefix, 1)

    if ipv4 != host:
        logger.info("Conexión Supabase forzada a IPv4: %s -> %s", host, ipv4)
    return rebuilt


def _get_database_url() -> str:
    """Lee DATABASE_URL del .env y la adapta para SQLAlchemy + psycopg2."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL no está definida. Configurala en el archivo .env del proyecto."
        )

    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    elif not url.startswith("postgresql+psycopg2://"):
        raise RuntimeError(
            "DATABASE_URL debe usar el esquema postgresql:// o postgresql+psycopg2://"
        )

    # Supabase exige SSL en conexiones remotas.
    if "supabase.co" in url and "sslmode=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}sslmode=require"

    url = _prefer_ipv4_database_url(url)
    return url


def get_db_config_status() -> dict:
    """Diagnóstico de conexión a BD (sin exponer credenciales)."""
    raw_url = os.environ.get("DATABASE_URL", "")
    env_path = PROJECT_DIR / ".env"
    issues: list[str] = []

    if not raw_url:
        issues.append("DATABASE_URL no está definida")

    host = ""
    database = ""
    uses_ssl = False
    is_supabase = False
    is_local = False

    if raw_url:
        try:
            without_scheme = raw_url.split("://", 1)[-1]
            host_part = without_scheme.split("@")[-1]
            host = host_part.split("/")[0].split("?")[0]
            database = host_part.split("/")[1].split("?")[0] if "/" in host_part else ""
            uses_ssl = "sslmode=require" in raw_url or "supabase.co" in raw_url
            is_supabase = "supabase.co" in raw_url
            is_local = host.startswith("localhost") or host.startswith("127.0.0.1")
            if is_local:
                issues.append("DATABASE_URL apunta a PostgreSQL local, no a Supabase")
        except Exception:
            issues.append("DATABASE_URL tiene un formato inválido")

    user_count = None
    sample_emails: list[str] = []
    if not issues:
        try:
            with engine.connect() as conn:
                user_count = conn.execute(text("SELECT COUNT(*) FROM usuarios")).scalar()
                rows = conn.execute(
                    text("SELECT email FROM usuarios ORDER BY fecha_creacion LIMIT 5")
                ).fetchall()
                sample_emails = [row[0] for row in rows]
        except Exception as exc:
            issues.append(f"No se pudo conectar a la base de datos: {exc}")

    return {
        "env_file": str(env_path),
        "env_file_exists": env_path.is_file(),
        "database_host": host,
        "database_name": database,
        "is_supabase": is_supabase,
        "is_localhost": is_local,
        "uses_ssl": uses_ssl,
        "usuarios_count": user_count,
        "sample_emails": sample_emails,
        "issues": issues,
        "ok": len(issues) == 0,
    }


DATABASE_URL = _get_database_url()

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Crea las tablas en PostgreSQL si aún no existen."""
    tables = list(Base.metadata.tables.keys())
    logger.info("Conectando a PostgreSQL para crear tablas: %s", tables)
    Base.metadata.create_all(bind=engine)
    logger.info("Tablas verificadas/creadas correctamente")
    _migrate_configuraciones_usuario_schema()
    _migrate_usuario_email_verification_schema()
    _migrate_usuario_password_reset_schema()
    _migrate_usuario_rol_schema()
    _migrate_cupones_schema()
    _backfill_configuraciones_usuario()


def _migrate_configuraciones_usuario_schema() -> None:
    """Alinea columnas legacy (p. ej. notificaciones_email como timestamp) al esquema actual."""
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'configuraciones_usuario'
                  AND column_name = 'notificaciones_email'
                """
            )
        ).fetchone()
        if row is None:
            return

        if row[0] == "boolean":
            return

        logger.info(
            "Migrando configuraciones_usuario.notificaciones_email: %s -> boolean",
            row[0],
        )
        conn.execute(
            text(
                """
                ALTER TABLE configuraciones_usuario
                ALTER COLUMN notificaciones_email DROP DEFAULT
                """
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE configuraciones_usuario
                ALTER COLUMN notificaciones_email TYPE BOOLEAN
                USING TRUE
                """
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE configuraciones_usuario
                ALTER COLUMN notificaciones_email SET DEFAULT TRUE
                """
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE configuraciones_usuario
                ALTER COLUMN notificaciones_email SET NOT NULL
                """
            )
        )
        logger.info("Columna notificaciones_email migrada a boolean")


def _migrate_usuario_email_verification_schema() -> None:
    """Agrega columnas de verificación de correo en usuarios si aún no existen."""
    statements = (
        """
        ALTER TABLE usuarios
        ADD COLUMN IF NOT EXISTS email_verification_token VARCHAR(64)
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ix_usuarios_email_verification_token
        ON usuarios (email_verification_token)
        WHERE email_verification_token IS NOT NULL
        """,
        """
        ALTER TABLE usuarios
        ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMPTZ
        """,
    )
    with engine.begin() as conn:
        for sql in statements:
            conn.execute(text(sql))
    logger.info("Esquema de verificación de correo en usuarios verificado")


def _migrate_usuario_password_reset_schema() -> None:
    """Agrega columnas de recuperación de contraseña en usuarios si aún no existen."""
    statements = (
        """
        ALTER TABLE usuarios
        ADD COLUMN IF NOT EXISTS password_reset_token VARCHAR(64)
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ix_usuarios_password_reset_token
        ON usuarios (password_reset_token)
        WHERE password_reset_token IS NOT NULL
        """,
        """
        ALTER TABLE usuarios
        ADD COLUMN IF NOT EXISTS password_reset_expires_at TIMESTAMPTZ
        """,
    )
    with engine.begin() as conn:
        for sql in statements:
            conn.execute(text(sql))
    logger.info("Esquema de recuperación de contraseña en usuarios verificado")


def _migrate_usuario_rol_schema() -> None:
    """Agrega columna rol en usuarios si aún no existe."""
    statements = (
        """
        ALTER TABLE usuarios
        ADD COLUMN IF NOT EXISTS rol VARCHAR NOT NULL DEFAULT 'estudiante'
        """,
        """
        UPDATE usuarios SET rol = 'estudiante' WHERE rol IS NULL
        """,
    )
    with engine.begin() as conn:
        for sql in statements:
            conn.execute(text(sql))
    logger.info("Esquema de rol en usuarios verificado")


def _migrate_cupones_schema() -> None:
    """Crea tablas de cupones y registro de usos si aún no existen."""
    statements = (
        """
        CREATE TABLE IF NOT EXISTS cupones (
            id UUID PRIMARY KEY,
            codigo VARCHAR NOT NULL UNIQUE,
            descripcion VARCHAR,
            limite_usos INTEGER NOT NULL DEFAULT 1,
            limite_usos_por_usuario INTEGER NOT NULL DEFAULT 1,
            plan_id INTEGER,
            descuento_porcentaje NUMERIC(5, 2),
            descuento_monto NUMERIC(10, 2),
            fecha_inicio TIMESTAMPTZ,
            fecha_expiracion TIMESTAMPTZ,
            activo BOOLEAN NOT NULL DEFAULT TRUE,
            limitaciones JSONB,
            fecha_creacion TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT fk_cupones_plan FOREIGN KEY (plan_id) REFERENCES planes (id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS usos_cupon (
            id UUID PRIMARY KEY,
            cupon_id UUID NOT NULL,
            usuario_id UUID NOT NULL,
            fecha_uso TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT fk_usos_cupon_cupon FOREIGN KEY (cupon_id) REFERENCES cupones (id),
            CONSTRAINT fk_usos_cupon_usuario FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_usos_cupon_cupon_usuario
        ON usos_cupon (cupon_id, usuario_id)
        """,
    )
    with engine.begin() as conn:
        for sql in statements:
            conn.execute(text(sql))
    logger.info("Esquema de cupones verificado")


def _backfill_configuraciones_usuario() -> None:
    """Corrige filas legacy con NULLs en configuraciones_usuario."""
    statements = (
        "UPDATE configuraciones_usuario SET tema_color = 'oscuro' WHERE tema_color IS NULL",
        "UPDATE configuraciones_usuario SET idioma = 'es' WHERE idioma IS NULL",
        "UPDATE configuraciones_usuario SET notificaciones_email = TRUE WHERE notificaciones_email IS NULL",
    )
    with engine.begin() as conn:
        for sql in statements:
            result = conn.execute(text(sql))
            if result.rowcount:
                logger.info("Backfill configuraciones_usuario: %s (%s filas)", sql, result.rowcount)

class Usuario(Base):
    __tablename__ = 'usuarios'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    nombre = Column(String, nullable=True)
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    estado = Column(String, nullable=False, default='activo')
    email_verification_token = Column(String(64), nullable=True, unique=True)
    email_verified_at = Column(DateTime(timezone=True), nullable=True)
    password_reset_token = Column(String(64), nullable=True, unique=True)
    password_reset_expires_at = Column(DateTime(timezone=True), nullable=True)
    rol = Column(String, nullable=False, default='estudiante', server_default='estudiante')

    # Relaciones
    suscripcion = relationship("Suscripcion", back_populates="usuario", uselist=False) # Relación 1 a 1
    proyectos = relationship("Proyecto", back_populates="usuario")
    pagos = relationship("Pago", back_populates="usuario")
    usos_cupon = relationship("UsoCupon", back_populates="usuario")
    configuracion_usuario = relationship(
        "ConfiguracionUsuario", back_populates="usuario", uselist=False
    )


class Plan(Base):
    __tablename__ = 'planes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(String, nullable=False)
    precio = Column(Numeric(10, 2), nullable=False)
    limite_almacenamiento_mb = Column(Integer, nullable=True)
    limite_proyectos = Column(Integer, nullable=True)

    # Relaciones
    suscripciones = relationship("Suscripcion", back_populates="plan")
    cupones = relationship("Cupon", back_populates="plan")


class Suscripcion(Base):
    __tablename__ = 'suscripciones'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    usuario_id = Column(UUID(as_uuid=True), ForeignKey('usuarios.id', ondelete='RESTRICT'), unique=True, nullable=False)
    plan_id = Column(Integer, ForeignKey('planes.id', ondelete='RESTRICT'), nullable=False)
    estado = Column(String, nullable=False) # 'active', 'canceled', etc.
    fecha_inicio = Column(DateTime(timezone=True), nullable=False)
    fecha_fin = Column(DateTime(timezone=True), nullable=False)
    proveedor_pago_id = Column(String, nullable=True)

    # Relaciones
    usuario = relationship("Usuario", back_populates="suscripcion")
    plan = relationship("Plan", back_populates="suscripciones")
    pagos = relationship("Pago", back_populates="suscripcion")


class Proyecto(Base):
    __tablename__ = 'proyectos'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    usuario_id = Column(UUID(as_uuid=True), ForeignKey('usuarios.id', ondelete='RESTRICT'), nullable=False)
    nombre = Column(String, nullable=False)
    formato = Column(String, nullable=False) # 'stl', 'obj'
    tamano_bytes = Column(BigInteger, nullable=False)
    url_archivo = Column(String, nullable=False)
    metadata_impresion = Column(JSONB, nullable=True)
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relaciones
    usuario = relationship("Usuario", back_populates="proyectos")


class Pago(Base):
    __tablename__ = 'pagos'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    usuario_id = Column(UUID(as_uuid=True), ForeignKey('usuarios.id', ondelete='RESTRICT'), nullable=False)
    suscripcion_id = Column(UUID(as_uuid=True), ForeignKey('suscripciones.id', ondelete='RESTRICT'), nullable=False)
    monto = Column(Numeric(10, 2), nullable=False)
    moneda = Column(String(3), nullable=False)
    estado = Column(String, nullable=False) # 'exitoso', 'fallido', etc.
    pasarela_pago = Column(String, nullable=True)
    transaccion_externa_id = Column(String, unique=True, nullable=False)
    fecha_pago = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relaciones
    usuario = relationship("Usuario", back_populates="pagos")
    suscripcion = relationship("Suscripcion", back_populates="pagos")


class ConfiguracionUsuario(Base):
    __tablename__ = 'configuraciones_usuario'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Relación 1 a 1 con usuarios. Si se borra el usuario, se borra su configuración (CASCADE)
    usuario_id = Column(UUID(as_uuid=True), ForeignKey('usuarios.id', ondelete='CASCADE'), unique=True, nullable=False)
    
    # Preferencias específicas de la interfaz
    tema_color = Column(String, nullable=False, default='oscuro')
    idioma = Column(String(5), nullable=False, default='es') # 'es', 'en', etc.
    notificaciones_email = Column(Boolean, nullable=False, default=True)
    
    # Flexibilidad para el futuro (ej: atajos de teclado personalizados, filtros por defecto)
    preferencias_interfaz = Column(JSONB, nullable=True)

    # Relación inversa hacia Usuario
    usuario = relationship("Usuario", back_populates="configuracion_usuario")


class Cupon(Base):
    __tablename__ = 'cupones'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    codigo = Column(String, unique=True, nullable=False)
    descripcion = Column(String, nullable=True)
    limite_usos = Column(Integer, nullable=False, default=1)
    limite_usos_por_usuario = Column(Integer, nullable=False, default=1)
    plan_id = Column(Integer, ForeignKey('planes.id', ondelete='RESTRICT'), nullable=True)
    descuento_porcentaje = Column(Numeric(5, 2), nullable=True)
    descuento_monto = Column(Numeric(10, 2), nullable=True)
    fecha_inicio = Column(DateTime(timezone=True), nullable=True)
    fecha_expiracion = Column(DateTime(timezone=True), nullable=True)
    activo = Column(Boolean, nullable=False, default=True)
    limitaciones = Column(JSONB, nullable=True)
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    plan = relationship("Plan", back_populates="cupones")
    usos = relationship("UsoCupon", back_populates="cupon")


class UsoCupon(Base):
    __tablename__ = 'usos_cupon'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cupon_id = Column(UUID(as_uuid=True), ForeignKey('cupones.id', ondelete='RESTRICT'), nullable=False)
    usuario_id = Column(UUID(as_uuid=True), ForeignKey('usuarios.id', ondelete='RESTRICT'), nullable=False)
    fecha_uso = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    cupon = relationship("Cupon", back_populates="usos")
    usuario = relationship("Usuario", back_populates="usos_cupon")