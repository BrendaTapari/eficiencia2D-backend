from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String,
)
from sqlalchemy.dialects.postgresql import JSON as PG_JSON
from sqlalchemy.types import JSON
from sqlalchemy.orm import relationship

from database.database import Base


class Plan(Base):
    __tablename__ = "planes"

    id = Column(String, primary_key=True)
    slug = Column(String, unique=True, nullable=False)
    nombre = Column(String, nullable=False)
    precio_mensual = Column(Float, default=0)
    moneda = Column(String, default="ARS")
    periodo = Column(String, default="mes")
    descripcion = Column(String, default="")
    features = Column(JSON, default=list)
    destacado = Column(Boolean, default=False)
    activo = Column(Boolean, default=True)
    orden = Column(Integer, default=0)


class Suscripcion(Base):
    __tablename__ = "suscripciones"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, unique=True, index=True)
    plan_id = Column(String, ForeignKey("planes.id"), nullable=True)
    estado = Column(String, default="ninguna")
    proveedor = Column(String, nullable=True)
    proveedor_ref = Column(String, nullable=True)
    periodo_inicio = Column(DateTime, nullable=True)
    periodo_fin = Column(DateTime, nullable=True)
    cancela_al_fin = Column(Boolean, default=False)

    plan = relationship("Plan")
