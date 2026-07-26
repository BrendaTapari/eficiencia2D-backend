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
    Opcional: si el host de Supabase tiene A-record IPv4, fija `hostaddr` para
    preferir IPv4 (útil en redes dual-stack raras).

    NO corta el arranque si no hay IPv4: muchos VPS llegan bien por IPv6 a
    `db.*.supabase.co`. Forzar error acá planchaba el backend sin tocar el .env.
    """
    if "supabase.co" not in url or "hostaddr=" in url:
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
        ipv4 = socket.gethostbyname(host)
    except OSError:
        # Sin A-record (típico en db.* solo-IPv6): dejar la URL intacta.
        logger.info(
            "Sin IPv4 para %s; se usa la URL tal cual (IPv6/DNS de libpq)",
            host,
        )
        return url

    query = parsed.query
    rebuilt_query = f"{query}&hostaddr={ipv4}" if query else f"hostaddr={ipv4}"
    rebuilt = urlunparse(parsed._replace(query=rebuilt_query))
    if dialect_prefix == "postgresql+psycopg2://":
        rebuilt = rebuilt.replace("postgresql://", dialect_prefix, 1)

    logger.info("Conexión Supabase con hostaddr IPv4: %s -> %s", host, ipv4)
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
    _migrate_usuario_terminos_schema()
    _migrate_usuario_first_time_schema()
    _migrate_cupones_schema()
    _migrate_planes_contract_schema()
    _migrate_precios_plan_schema()
    _migrate_suscripciones_contract_schema()
    _backfill_configuraciones_usuario()
    _seed_planes()


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
    """Migra roles legacy (columna VARCHAR) a tabla rol + usuarios.rol_id."""
    bootstrap = (
        """
        CREATE TABLE IF NOT EXISTS rol (
            id SERIAL PRIMARY KEY,
            rol VARCHAR NOT NULL UNIQUE
        )
        """,
        """
        INSERT INTO rol (id, rol) VALUES (1, 'estudiante')
        ON CONFLICT (id) DO NOTHING
        """,
        """
        INSERT INTO rol (id, rol) VALUES (2, 'admin')
        ON CONFLICT (id) DO NOTHING
        """,
        """
        ALTER TABLE usuarios
        ADD COLUMN IF NOT EXISTS rol_id INTEGER
        """,
    )
    with engine.begin() as conn:
        for sql in bootstrap:
            conn.execute(text(sql))

        legacy_rol = conn.execute(
            text(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'usuarios'
                  AND column_name = 'rol'
                """
            )
        ).fetchone()

        if legacy_rol:
            conn.execute(
                text(
                    """
                    UPDATE usuarios u
                    SET rol_id = r.id
                    FROM rol r
                    WHERE u.rol_id IS NULL
                      AND LOWER(u.rol::text) = LOWER(r.rol)
                    """
                )
            )
            conn.execute(
                text(
                    """
                    UPDATE usuarios
                    SET rol_id = 2
                    WHERE rol_id IS NULL AND LOWER(rol::text) = 'admin'
                    """
                )
            )
            conn.execute(text("ALTER TABLE usuarios DROP COLUMN rol"))

        finalize = (
            """
            UPDATE usuarios SET rol_id = 1 WHERE rol_id IS NULL
            """,
            """
            ALTER TABLE usuarios ALTER COLUMN rol_id SET DEFAULT 1
            """,
            """
            ALTER TABLE usuarios ALTER COLUMN rol_id SET NOT NULL
            """,
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'fk_usuarios_rol'
                ) THEN
                    ALTER TABLE usuarios
                    ADD CONSTRAINT fk_usuarios_rol
                    FOREIGN KEY (rol_id) REFERENCES rol (id);
                END IF;
            END $$
            """,
        )
        for sql in finalize:
            conn.execute(text(sql))

    logger.info("Esquema de roles (tabla rol + usuarios.rol_id) verificado")


def _migrate_usuario_terminos_schema() -> None:
    """Agrega acepto_terminos_at a usuarios si falta."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                ALTER TABLE usuarios
                ADD COLUMN IF NOT EXISTS acepto_terminos_at TIMESTAMPTZ
                """
            )
        )
    logger.info("Esquema de términos (usuarios.acepto_terminos_at) verificado")


def _migrate_usuario_first_time_schema() -> None:
    """Agrega first_time a usuarios si falta (no-op si ya existe en Supabase)."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                ALTER TABLE usuarios
                ADD COLUMN IF NOT EXISTS first_time BOOLEAN NOT NULL DEFAULT TRUE
                """
            )
        )
    logger.info("Esquema de first_time (usuarios.first_time) verificado")


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


def _migrate_planes_contract_schema() -> None:
    """Columnas de catálogo en planes (sin precios embebidos)."""
    statements = (
        "ALTER TABLE planes ADD COLUMN IF NOT EXISTS slug VARCHAR UNIQUE",
        "ALTER TABLE planes ADD COLUMN IF NOT EXISTS descripcion VARCHAR DEFAULT ''",
        "ALTER TABLE planes ADD COLUMN IF NOT EXISTS features JSONB DEFAULT '[]'::jsonb",
        "ALTER TABLE planes ADD COLUMN IF NOT EXISTS destacado BOOLEAN DEFAULT FALSE",
        "ALTER TABLE planes ADD COLUMN IF NOT EXISTS activo BOOLEAN DEFAULT TRUE",
        "ALTER TABLE planes ADD COLUMN IF NOT EXISTS orden INTEGER DEFAULT 0",
    )
    with engine.begin() as conn:
        for sql in statements:
            conn.execute(text(sql))
    logger.info("Esquema de planes (catálogo) verificado")


def _migrate_precios_plan_schema() -> None:
    """Tabla precio_plan (FK planes_id) y migración desde legacy / precios_plan."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS precio_plan (
                    id SERIAL PRIMARY KEY,
                    precio NUMERIC(10, 2) NOT NULL,
                    moneda VARCHAR DEFAULT 'ARS',
                    periodo VARCHAR DEFAULT 'mes',
                    planes_id INTEGER NOT NULL,
                    CONSTRAINT fk_precio_plan_planes
                        FOREIGN KEY (planes_id) REFERENCES planes (id)
                )
                """
            )
        )

        # Si quedó una tabla mal nombrada (precios_plan) con datos, copiarlos.
        has_legacy_table = conn.execute(
            text(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'precios_plan'
                """
            )
        ).first()
        if has_legacy_table:
            conn.execute(
                text(
                    """
                    INSERT INTO precio_plan (precio, moneda, periodo, planes_id)
                    SELECT pp.precio, pp.moneda, pp.periodo, pp.planes_id
                    FROM precios_plan pp
                    WHERE NOT EXISTS (
                        SELECT 1 FROM precio_plan p
                        WHERE p.planes_id = pp.planes_id
                          AND p.moneda IS NOT DISTINCT FROM pp.moneda
                          AND p.periodo IS NOT DISTINCT FROM pp.periodo
                    )
                    """
                )
            )

        legacy_cols = {
            row[0]
            for row in conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'planes'
                      AND column_name IN ('precio', 'precio_mensual', 'moneda', 'periodo')
                    """
                )
            ).fetchall()
        }

        if legacy_cols:
            if "precio_mensual" in legacy_cols and "precio" in legacy_cols:
                precio_expr = "COALESCE(p.precio_mensual, p.precio, 0)"
            elif "precio_mensual" in legacy_cols:
                precio_expr = "COALESCE(p.precio_mensual, 0)"
            elif "precio" in legacy_cols:
                precio_expr = "COALESCE(p.precio, 0)"
            else:
                precio_expr = "0"

            moneda_expr = "COALESCE(p.moneda, 'ARS')" if "moneda" in legacy_cols else "'ARS'"
            periodo_expr = "COALESCE(p.periodo, 'mes')" if "periodo" in legacy_cols else "'mes'"

            conn.execute(
                text(
                    f"""
                    INSERT INTO precio_plan (precio, moneda, periodo, planes_id)
                    SELECT {precio_expr}, {moneda_expr}, {periodo_expr}, p.id
                    FROM planes p
                    WHERE NOT EXISTS (
                        SELECT 1 FROM precio_plan pp WHERE pp.planes_id = p.id
                    )
                    """
                )
            )

    logger.info("Esquema de precio_plan verificado")


def _migrate_suscripciones_contract_schema() -> None:
    """Agrega columnas del contrato de suscripciones si aún no existen."""
    statements = (
        "ALTER TABLE suscripciones ADD COLUMN IF NOT EXISTS proveedor VARCHAR",
        "ALTER TABLE suscripciones ADD COLUMN IF NOT EXISTS cancela_al_fin BOOLEAN DEFAULT FALSE",
    )
    with engine.begin() as conn:
        for sql in statements:
            conn.execute(text(sql))
    logger.info("Esquema de suscripciones (contrato) verificado")


def _seed_planes() -> None:
    """Inserta planes y precios iniciales si no hay slugs."""
    with SessionLocal() as session:
        existing = session.execute(
            text("SELECT COUNT(*) FROM planes WHERE slug IS NOT NULL")
        ).scalar()
        if existing and existing > 0:
            _backfill_precios_faltantes(session)
            return

        seeds = [
            {
                "nombre": "Gratis", "slug": "gratis", "precio": 0,
                "moneda": "ARS", "periodo": "mes",
                "descripcion": "Para probar la herramienta.",
                "features": '["1 proyecto activo", "Visor 3D y revisión", "Exporta con marca de agua"]',
                "destacado": False, "activo": True, "orden": 1,
                "limite_almacenamiento_mb": 100, "limite_proyectos": 1,
            },
            {
                "nombre": "Pro", "slug": "pro", "precio": 8000,
                "moneda": "ARS", "periodo": "mes",
                "descripcion": "Para uso profesional.",
                "features": '["Proyectos ilimitados", "Exporta sin marca de agua", "Instructivo de armado", "Soporte prioritario"]',
                "destacado": True, "activo": True, "orden": 2,
                "limite_almacenamiento_mb": None, "limite_proyectos": None,
            },
            {
                "nombre": "Estudio", "slug": "estudio", "precio": 20000,
                "moneda": "ARS", "periodo": "mes",
                "descripcion": "Para equipos y estudios.",
                "features": '["Todo lo de Pro", "Múltiples usuarios", "Prioridad de cómputo", "Facturación por equipo"]',
                "destacado": False, "activo": True, "orden": 3,
                "limite_almacenamiento_mb": None, "limite_proyectos": None,
            },
        ]
        for s in seeds:
            plan_row = session.execute(
                text(
                    """
                    INSERT INTO planes (nombre, slug, descripcion, features, destacado, activo, orden,
                                        limite_almacenamiento_mb, limite_proyectos)
                    VALUES (:nombre, :slug, :descripcion, CAST(:features AS jsonb), :destacado, :activo,
                            :orden, :limite_almacenamiento_mb, :limite_proyectos)
                    RETURNING id
                    """
                ),
                {k: v for k, v in s.items() if k not in ("precio", "moneda", "periodo")},
            ).fetchone()
            session.execute(
                text(
                    """
                    INSERT INTO precio_plan (precio, moneda, periodo, planes_id)
                    VALUES (:precio, :moneda, :periodo, :planes_id)
                    """
                ),
                {
                    "precio": s["precio"],
                    "moneda": s["moneda"],
                    "periodo": s["periodo"],
                    "planes_id": plan_row[0],
                },
            )
        session.commit()
        logger.info("Seed: %d planes insertados", len(seeds))


def _backfill_precios_faltantes(session) -> None:
    """Si hay planes sin fila en precio_plan, inserta el precio default por slug."""
    defaults = {
        "gratis": 0,
        "pro": 8000,
        "estudio": 20000,
    }
    inserted = 0
    for slug, precio in defaults.items():
        row = session.execute(
            text(
                """
                SELECT p.id
                FROM planes p
                WHERE p.slug = :slug
                  AND NOT EXISTS (
                      SELECT 1 FROM precio_plan pp WHERE pp.planes_id = p.id
                  )
                """
            ),
            {"slug": slug},
        ).fetchone()
        if not row:
            continue
        session.execute(
            text(
                """
                INSERT INTO precio_plan (precio, moneda, periodo, planes_id)
                VALUES (:precio, 'ARS', 'mes', :planes_id)
                """
            ),
            {"precio": precio, "planes_id": row[0]},
        )
        inserted += 1
    if inserted:
        session.commit()
        logger.info("Backfill: %d precios insertados en precio_plan", inserted)


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
    acepto_terminos_at = Column(DateTime(timezone=True), nullable=True)
    first_time = Column(Boolean, nullable=False, default=True, server_default='true')
    rol_id = Column(
        Integer,
        ForeignKey('rol.id', ondelete='RESTRICT'),
        nullable=False,
        default=1,
        server_default='1',
    )

    # Relaciones
    suscripcion = relationship("Suscripcion", back_populates="usuario", uselist=False) # Relación 1 a 1
    proyectos = relationship("Proyecto", back_populates="usuario")
    pagos = relationship("Pago", back_populates="usuario")
    usos_cupon = relationship("UsoCupon", back_populates="usuario")
    rol = relationship("Rol", back_populates="usuarios")
    configuracion_usuario = relationship(
        "ConfiguracionUsuario", back_populates="usuario", uselist=False
    )


class Plan(Base):
    __tablename__ = 'planes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(String, nullable=False)
    limite_almacenamiento_mb = Column(Integer, nullable=True)
    limite_proyectos = Column(Integer, nullable=True)
    slug = Column(String, unique=True, nullable=True)  
    descripcion = Column(String, default="")
    features = Column(JSONB, default=list)
    destacado = Column(Boolean, default=False)
    activo = Column(Boolean, default=True)
    orden = Column(Integer, default=0)

    # Relaciones
    suscripciones = relationship("Suscripcion", back_populates="plan")
    cupones = relationship("Cupon", back_populates="plan")
    precios = relationship(
        "PrecioPlan",
        back_populates="plan",
        cascade="all, delete-orphan",
        foreign_keys="PrecioPlan.planes_id",
    )


class PrecioPlan(Base):
    __tablename__ = "precio_plan"

    id = Column(Integer, primary_key=True, autoincrement=True)
    precio = Column(Numeric(10, 2), nullable=False)
    moneda = Column(String, default="ARS")
    periodo = Column(String, default="mes")
    planes_id = Column(Integer, ForeignKey("planes.id", ondelete="RESTRICT"), nullable=False)

    plan = relationship("Plan", back_populates="precios", foreign_keys=[planes_id])


class Suscripcion(Base):
    __tablename__ = 'suscripciones'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    usuario_id = Column(UUID(as_uuid=True), ForeignKey('usuarios.id', ondelete='RESTRICT'), unique=True, nullable=False)
    plan_id = Column(Integer, ForeignKey('planes.id', ondelete='RESTRICT'), nullable=False)
    estado = Column(String, nullable=False) # 'activa', 'pendiente', 'cancelada'
    fecha_inicio = Column(DateTime(timezone=True), nullable=False)
    fecha_fin = Column(DateTime(timezone=True), nullable=False)
    proveedor_pago_id = Column(String, nullable=True)
    proveedor = Column(String, nullable=True)
    cancela_al_fin = Column(Boolean, default=False)

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


class Rol(Base):
    __tablename__ = 'rol'

    id = Column(Integer, primary_key=True, autoincrement=True)
    rol = Column(String, nullable=False, unique=True)

    usuarios = relationship("Usuario", back_populates="rol")
