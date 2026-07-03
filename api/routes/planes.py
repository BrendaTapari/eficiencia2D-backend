from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.deps import get_db
from database.models import Plan

router = APIRouter()


def serialize_plan(p: Plan) -> dict:
    return {
        "id": p.id,
        "slug": p.slug,
        "nombre": p.nombre,
        "precio_mensual": p.precio_mensual,
        "moneda": p.moneda,
        "periodo": p.periodo,
        "descripcion": p.descripcion,
        "features": p.features or [],
        "destacado": p.destacado,
        "orden": p.orden,
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
