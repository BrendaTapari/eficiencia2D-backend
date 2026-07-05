import os
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from api.deps import get_current_user
from api.services.cupon_service import (
    assert_cupon_vigente,
    calcular_precio_con_descuento,
    cupon_tipo,
    get_cupon_por_codigo,
    registrar_uso_cupon,
)
from api.services.plan_pricing import plan_tarifa
from database import Plan, Suscripcion, Usuario, get_db

log = logging.getLogger(__name__)
router = APIRouter()

MP_ACCESS_TOKEN = os.environ.get("MP_ACCESS_TOKEN", "")
FRONTEND_URL = os.environ.get("FRONTEND_URL_VERCEL") or os.environ.get("FRONTEND_URL", "http://localhost:3000")


class SuscripcionRequest(BaseModel):
    plan_id: int
    cupon_codigo: str | None = Field(default=None, max_length=64)

    @field_validator("cupon_codigo")
    @classmethod
    def normalize_cupon(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        return normalized or None


def serialize_suscripcion(s: Suscripcion) -> dict:
    return {
        "plan_id": s.plan_id,
        "estado": s.estado,
        "periodo_fin": s.fecha_fin.isoformat() if s.fecha_fin else None,
        "cancela_al_fin": bool(s.cancela_al_fin),
    }


def _empty_suscripcion() -> dict:
    return {"plan_id": None, "estado": "ninguna", "periodo_fin": None, "cancela_al_fin": False}


def _create_mp_preapproval(plan: Plan, user_id: str, *, monto: float, moneda: str) -> dict:
    import mercadopago

    if not MP_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="MP_ACCESS_TOKEN no configurado")

    sdk = mercadopago.SDK(MP_ACCESS_TOKEN)
    preapproval_data = {
        "reason": f"Suscripción {plan.nombre}",
        "auto_recurring": {
            "frequency": 1,
            "frequency_type": "months",
            "transaction_amount": monto,
            "currency_id": moneda,
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


def _activar_suscripcion(
    db: Session,
    user: Usuario,
    plan: Plan,
    sub: Suscripcion | None,
) -> Suscripcion:
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
    return sub


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
    cupon = None
    if body.cupon_codigo:
        cupon = get_cupon_por_codigo(db, body.cupon_codigo)
        assert_cupon_vigente(cupon, db, usuario=user)

    if cupon is not None and cupon_tipo(cupon) == "plan":
        plan = db.query(Plan).filter(Plan.id == cupon.plan_id, Plan.activo.is_(True)).first()
        if plan is None:
            raise HTTPException(status_code=404, detail="Plan del cupón no encontrado o inactivo")
        if body.plan_id != plan.id:
            raise HTTPException(
                status_code=400,
                detail=f"Este cupón aplica al plan {plan.nombre}",
            )
        sub = db.query(Suscripcion).filter(Suscripcion.usuario_id == user.id).first()
        if sub and sub.plan_id == plan.id and sub.estado == "activa":
            raise HTTPException(status_code=409, detail="Ya tenés activo el plan de este cupón")

        sub = _activar_suscripcion(db, user, plan, sub)
        registrar_uso_cupon(db, cupon, user)
        db.commit()
        db.refresh(sub)
        return serialize_suscripcion(sub)

    plan = db.query(Plan).filter(Plan.id == body.plan_id, Plan.activo.is_(True)).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan no encontrado o inactivo")

    sub = db.query(Suscripcion).filter(Suscripcion.usuario_id == user.id).first()

    if sub and sub.plan_id == body.plan_id and sub.estado == "activa":
        raise HTTPException(status_code=409, detail="Ya suscripto a este plan")

    precio, moneda, _periodo = plan_tarifa(db, plan)

    if cupon is not None:
        if cupon_tipo(cupon) != "descuento":
            raise HTTPException(status_code=400, detail="Cupón incompatible con este plan")
        if precio <= 0:
            raise HTTPException(
                status_code=400,
                detail="Los cupones de descuento aplican a planes de pago",
            )
        precio = calcular_precio_con_descuento(precio, cupon)

    if precio == 0:
        sub = _activar_suscripcion(db, user, plan, sub)
        if cupon is not None:
            registrar_uso_cupon(db, cupon, user)
            db.commit()
            db.refresh(sub)
        return serialize_suscripcion(sub)

    mp = _create_mp_preapproval(plan, str(user.id), monto=precio, moneda=moneda)
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

    if cupon is not None:
        registrar_uso_cupon(db, cupon, user)

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
