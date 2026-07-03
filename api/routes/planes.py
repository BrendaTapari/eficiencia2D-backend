from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import Plan, get_db

router = APIRouter()


def serialize_plan(p: Plan) -> dict:
    return {
        "id": p.id,
        "slug": p.slug,
        "nombre": p.nombre,
        "precio_mensual": float(p.precio_mensual or p.precio or 0),
        "moneda": p.moneda or "ARS",
        "periodo": p.periodo or "mes",
        "descripcion": p.descripcion or "",
        "features": p.features or [],
        "destacado": bool(p.destacado),
        "orden": p.orden or 0,
    }


@router.get("/planes")
def list_planes(db: Session = Depends(get_db)):
    planes = (
        db.query(Plan)
        .filter(Plan.activo.is_(True))
        .order_by(Plan.orden)
        .all()
    )
    return [serialize_plan(p) for p in planes]
