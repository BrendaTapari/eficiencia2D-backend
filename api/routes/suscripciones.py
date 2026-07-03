import os
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.deps import get_current_user
from database import Plan, Suscripcion, Usuario, get_db

log = logging.getLogger(__name__)
router = APIRouter()

MP_ACCESS_TOKEN = os.environ.get("MP_ACCESS_TOKEN", "")
FRONTEND_URL = os.environ.get("FRONTEND_URL_VERCEL") or os.environ.get("FRONTEND_URL", "http://localhost:3000")


class SuscripcionRequest(BaseModel):
    plan_id: int


def serialize_suscripcion(s: Suscripcion) -> dict:
    return {
        "plan_id": s.plan_id,
        "estado": s.estado,
        "periodo_fin": s.fecha_fin.isoformat() if s.fecha_fin else None,
        "cancela_al_fin": bool(s.cancela_al_fin),
    }


def _empty_suscripcion() -> dict:
    return {"plan_id": None, "estado": "ninguna", "periodo_fin": None, "cancela_al_fin": False}


def _create_mp_preapproval(plan: Plan, user_id: str) -> dict:
    import mercadopago

    if not MP_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="MP_ACCESS_TOKEN no configurado")

    sdk = mercadopago.SDK(MP_ACCESS_TOKEN)
    preapproval_data = {
        "reason": f"Suscripción {plan.nombre}",
        "auto_recurring": {
            "frequency": 1,
            "frequency_type": "months",
            "transaction_amount": float(plan.precio_mensual or plan.precio),
            "currency_id": plan.moneda or "ARS",
        },
        "back_url": f"{FRONTEND_URL}/payment-callback?sub=1",
        "external_reference": user_id,
    }
    result = sdk.preapproval().create(preapproval_data)
    response = result.get("response", {})

    if result.get("status") not in (200, 201):
        log.error("MercadoPago preapproval error: %s", result)
        raise HTTPException(status_code=502, detail="Error al crear suscripción en MercadoPago")

    return {
        "id": response.get("id"),
        "init_point": response.get("init_point"),
    }


@router.get("/users/me/suscripcion")
def get_suscripcion(
    user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sub = db.query(Suscripcion).filter(Suscripcion.usuario_id == user.id).first()
    if not sub:
        return _empty_suscripcion()
    return serialize_suscripcion(sub)


@router.post("/users/me/suscripcion")
def create_or_change_suscripcion(
    body: SuscripcionRequest,
    user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    plan = db.query(Plan).filter(Plan.id == body.plan_id, Plan.activo.is_(True)).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan no encontrado o inactivo")

    sub = db.query(Suscripcion).filter(Suscripcion.usuario_id == user.id).first()

    if sub and sub.plan_id == body.plan_id and sub.estado == "activa":
        raise HTTPException(status_code=409, detail="Ya suscripto a este plan")

    precio = float(plan.precio_mensual or plan.precio or 0)

    if precio == 0:
        now = datetime.now(timezone.utc)
        if sub:
            sub.plan_id = plan.id
            sub.estado = "activa"
            sub.fecha_inicio = now
            sub.fecha_fin = now
            sub.cancela_al_fin = False
            sub.proveedor = None
            sub.proveedor_pago_id = None
        else:
            sub = Suscripcion(
                usuario_id=user.id,
                plan_id=plan.id,
                estado="activa",
                fecha_inicio=now,
                fecha_fin=now,
            )
            db.add(sub)
        db.commit()
        db.refresh(sub)
        return serialize_suscripcion(sub)

    mp = _create_mp_preapproval(plan, str(user.id))
    now = datetime.now(timezone.utc)
    if sub:
        sub.plan_id = plan.id
        sub.estado = "pendiente"
        sub.proveedor = "mercadopago"
        sub.proveedor_pago_id = mp["id"]
        sub.cancela_al_fin = False
        sub.fecha_inicio = now
        sub.fecha_fin = now
    else:
        sub = Suscripcion(
            usuario_id=user.id,
            plan_id=plan.id,
            estado="pendiente",
            proveedor="mercadopago",
            proveedor_pago_id=mp["id"],
            fecha_inicio=now,
            fecha_fin=now,
        )
        db.add(sub)
    db.commit()
    return {"checkout_url": mp["init_point"]}


@router.delete("/users/me/suscripcion")
def cancel_suscripcion(
    user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sub = db.query(Suscripcion).filter(Suscripcion.usuario_id == user.id).first()
    if not sub or sub.estado not in ("activa", "pendiente"):
        raise HTTPException(status_code=404, detail="No hay suscripción activa para cancelar")

    sub.cancela_al_fin = True
    db.commit()
    db.refresh(sub)
    return serialize_suscripcion(sub)
