from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session, joinedload

from api.services.plan_pricing import plan_tarifa, serialize_precio_plan
from database import Plan, get_db

router = APIRouter()


def serialize_plan(p: Plan, db: Session) -> dict:
    precio, moneda, periodo = plan_tarifa(db, p)
    precios = [serialize_precio_plan(row) for row in (p.precios or [])]
    return {
        "id": p.id,
        "slug": p.slug,
        "nombre": p.nombre,
        "precio_mensual": precio,
        "moneda": moneda,
        "periodo": periodo,
        "precios": precios,
        "descripcion": p.descripcion or "",
        "features": p.features or [],
        "destacado": bool(p.destacado),
        "orden": p.orden or 0,
    }


@router.get("/planes")
def list_planes(db: Session = Depends(get_db)):
    planes = (
        db.query(Plan)
        .options(joinedload(Plan.precios))
        .filter(Plan.activo.is_(True))
        .order_by(Plan.orden)
        .all()
    )
    return [serialize_plan(p, db) for p in planes]
