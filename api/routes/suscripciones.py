import os
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
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
MP_BYPASS_KEY = os.environ.get("MP_BYPASS_KEY", "")
FRONTEND_URL = os.environ.get("FRONTEND_URL_VERCEL") or os.environ.get("FRONTEND_URL", "http://localhost:3000")
MP_TIMEOUT_SEC = 20


class SuscripcionRequest(BaseModel):
    plan_id: int
    cupon_codigo: str | None = Field(default=None, max_length=64)
    """Pago único ya aprobado en MP (Wallet Brick / preference). Activa sin preapproval."""
    mp_payment_id: str | None = Field(default=None, max_length=128)
    """Código de bypass de desarrollo (mismo que MP_BYPASS_KEY)."""
    bypass_key: str | None = Field(default=None, max_length=128)

    @field_validator("cupon_codigo")
    @classmethod
    def normalize_cupon(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        return normalized or None

    @field_validator("mp_payment_id", "bypass_key")
    @classmethod
    def strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
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


def _mp_sdk():
    import mercadopago

    if not MP_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="MP_ACCESS_TOKEN no configurado")
    return mercadopago.SDK(MP_ACCESS_TOKEN)


def _create_mp_preapproval(plan: Plan, user_id: str, *, monto: float, moneda: str) -> dict:
    sdk = _mp_sdk()
    preapproval_data = {
        "reason": f"Suscripción {plan.nombre}",
        "auto_recurring": {
            "frequency": 1,
            "frequency_type": "months",
            "transaction_amount": float(monto),
            "currency_id": (moneda or "ARS").upper(),
        },
        "back_url": f"{FRONTEND_URL.rstrip('/')}/payment-callback?sub=1",
        "external_reference": str(user_id),
    }

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(sdk.preapproval().create, preapproval_data).result(
                timeout=MP_TIMEOUT_SEC
            )
    except FuturesTimeout as exc:
        log.error("Timeout creando preapproval MP para plan %s", plan.id)
        raise HTTPException(
            status_code=504,
            detail="Mercado Pago no respondió a tiempo. Probá de nuevo en unos minutos.",
        ) from exc
    except Exception as exc:
        log.exception("Error de red creando preapproval MP")
        raise HTTPException(
            status_code=502,
            detail=f"No se pudo contactar a Mercado Pago: {exc}",
        ) from exc

    response = result.get("response", {}) if isinstance(result, dict) else {}

    if result.get("status") not in (200, 201):
        log.error("MercadoPago preapproval error: %s", result)
        detail = response.get("message") or response.get("error") or "Error al crear suscripción en MercadoPago"
        raise HTTPException(status_code=502, detail=str(detail))

    init_point = response.get("init_point") or response.get("sandbox_init_point")
    if not init_point:
        log.error("Preapproval sin init_point: %s", response)
        raise HTTPException(
            status_code=502,
            detail="Mercado Pago no devolvió URL de checkout",
        )

    return {
        "id": response.get("id"),
        "init_point": init_point,
    }


def _verify_mp_payment_approved(payment_id: str) -> None:
    sdk = _mp_sdk()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(sdk.payment().get, payment_id).result(timeout=MP_TIMEOUT_SEC)
    except FuturesTimeout as exc:
        raise HTTPException(
            status_code=504,
            detail="Mercado Pago no respondió al verificar el pago.",
        ) from exc
    except Exception as exc:
        log.exception("Error verificando pago MP %s", payment_id)
        raise HTTPException(
            status_code=502,
            detail=f"No se pudo verificar el pago: {exc}",
        ) from exc

    if result.get("status") != 200:
        log.error("MP payment.get error: %s", result)
        raise HTTPException(status_code=502, detail="No se pudo consultar el pago en Mercado Pago")

    payment = result.get("response", {})
    status = payment.get("status")
    if status != "approved":
        raise HTTPException(
            status_code=402,
            detail=f"El pago no está aprobado (estado: {status or 'desconocido'})",
        )


def _bypass_ok(key: str | None) -> bool:
    if not key or not MP_BYPASS_KEY:
        return False
    return key.strip() == MP_BYPASS_KEY.strip()


def _activar_suscripcion(
    db: Session,
    user: Usuario,
    plan: Plan,
    sub: Suscripcion | None,
    *,
    proveedor: str | None = None,
    proveedor_pago_id: str | None = None,
) -> Suscripcion:
    now = datetime.now(timezone.utc)
    if sub:
        sub.plan_id = plan.id
        sub.estado = "activa"
        sub.fecha_inicio = now
        sub.fecha_fin = now
        sub.cancela_al_fin = False
        sub.proveedor = proveedor
        sub.proveedor_pago_id = proveedor_pago_id
    else:
        sub = Suscripcion(
            usuario_id=user.id,
            plan_id=plan.id,
            estado="activa",
            fecha_inicio=now,
            fecha_fin=now,
            proveedor=proveedor,
            proveedor_pago_id=proveedor_pago_id,
        )
        db.add(sub)
    db.commit()
    db.refresh(sub)
    return sub


def _cancel_mp_preapproval(preapproval_id: str | None) -> None:
    if not preapproval_id or not MP_ACCESS_TOKEN:
        return
    try:
        sdk = _mp_sdk()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(
                sdk.preapproval().update,
                preapproval_id,
                {"status": "cancelled"},
            ).result(timeout=MP_TIMEOUT_SEC)
    except Exception:
        log.exception("No se pudo cancelar preapproval MP %s", preapproval_id)


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

    # Pago ya cobrado en el front (Wallet Brick) → activar.
    if body.mp_payment_id:
        if precio <= 0:
            raise HTTPException(status_code=400, detail="Este plan no requiere pago")
        _verify_mp_payment_approved(body.mp_payment_id)
        sub = _activar_suscripcion(
            db,
            user,
            plan,
            sub,
            proveedor="mercadopago",
            proveedor_pago_id=body.mp_payment_id,
        )
        if cupon is not None:
            registrar_uso_cupon(db, cupon, user)
            db.commit()
            db.refresh(sub)
        return serialize_suscripcion(sub)

    # Bypass de desarrollo.
    if body.bypass_key:
        if not _bypass_ok(body.bypass_key):
            raise HTTPException(status_code=403, detail="Código de bypass inválido")
        sub = _activar_suscripcion(
            db,
            user,
            plan,
            sub,
            proveedor="bypass",
            proveedor_pago_id="bypass",
        )
        if cupon is not None:
            registrar_uso_cupon(db, cupon, user)
            db.commit()
            db.refresh(sub)
        return serialize_suscripcion(sub)

    if precio == 0:
        sub = _activar_suscripcion(db, user, plan, sub)
        if cupon is not None:
            registrar_uso_cupon(db, cupon, user)
            db.commit()
            db.refresh(sub)
        return serialize_suscripcion(sub)

    # Fallback legacy: preapproval recurrente (puede fallar si MP no responde desde el VPS).
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

    if sub.proveedor == "mercadopago" and sub.proveedor_pago_id:
        _cancel_mp_preapproval(str(sub.proveedor_pago_id))

    now = datetime.now(timezone.utc)
    sub.cancela_al_fin = True
    if not sub.fecha_fin or sub.fecha_fin <= now:
        sub.estado = "cancelada"

    db.commit()
    db.refresh(sub)
    return serialize_suscripcion(sub)
