import os
import uuid

from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, Numeric, BigInteger, ForeignKey, DateTime, create_engine
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

load_dotenv()

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

class Usuario(Base):
    __tablename__ = 'usuarios'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    nombre = Column(String, nullable=True)
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    estado = Column(String, nullable=False, default='activo')

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
    tema_color = Column(String, nullable=False, default='oscuro') # 'claro', 'oscuro', 'sistema'
    idioma = Column(String(5), nullable=False, default='es') # 'es', 'en', etc.
    notificaciones_email = Column(DateTime, nullable=True) # Para registrar preferencias de avisos
    
    # Flexibilidad para el futuro (ej: atajos de teclado personalizados, filtros por defecto)
    preferencias_interfaz = Column(JSONB, nullable=True)

    # Relación inversa hacia Usuario
    usuario = relationship("Usuario", back_populates="configuracion_usuario")