import os
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

log = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./eficiencia2d.db")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    **({"connect_args": {"check_same_thread": False}} if DATABASE_URL.startswith("sqlite") else {}),
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def seed_planes(session):
    from database.models import Plan

    if session.query(Plan).first() is not None:
        return

    seeds = [
        Plan(
            id="gratis", slug="gratis", nombre="Gratis",
            precio_mensual=0, moneda="ARS", periodo="mes",
            descripcion="Para probar la herramienta.",
            features=["1 proyecto activo", "Visor 3D y revisión", "Exporta con marca de agua"],
            destacado=False, activo=True, orden=1,
        ),
        Plan(
            id="pro", slug="pro", nombre="Pro",
            precio_mensual=8000, moneda="ARS", periodo="mes",
            descripcion="Para uso profesional.",
            features=["Proyectos ilimitados", "Exporta sin marca de agua",
                       "Instructivo de armado", "Soporte prioritario"],
            destacado=True, activo=True, orden=2,
        ),
        Plan(
            id="estudio", slug="estudio", nombre="Estudio",
            precio_mensual=20000, moneda="ARS", periodo="mes",
            descripcion="Para equipos y estudios.",
            features=["Todo lo de Pro", "Múltiples usuarios",
                       "Prioridad de cómputo", "Facturación por equipo"],
            destacado=False, activo=True, orden=3,
        ),
    ]
    session.add_all(seeds)
    session.commit()
    log.info("Seed: %d planes insertados", len(seeds))


def init_db():
    from database.models import Plan, Suscripcion  # noqa: F401 — register models
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        seed_planes(session)
    log.info("Base de datos inicializada")
