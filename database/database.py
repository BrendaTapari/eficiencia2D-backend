import logging
import os
import uuid
from pathlib import Path

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

# En el servidor con Docker: host=localhost y puerto=5433 (mapeo en docker-compose).
# Si la API corre dentro de la misma red Docker: host=postgres_db y puerto=5432.
# La contraseña con "ñ" debe ir URL-encoded (%C3%B1) dentro de DATABASE_URL.
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://eficiencia_db:%C3%B1StefaBren_bd@localhost:5433/eficiencia_db",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
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

    # Relaciones
    suscripcion = relationship("Suscripcion", back_populates="usuario", uselist=False) # Relación 1 a 1
    proyectos = relationship("Proyecto", back_populates="usuario")
    pagos = relationship("Pago", back_populates="usuario")
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