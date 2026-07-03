import os
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from database import Suscripcion, get_db

log = logging.getLogger(__name__)
router = APIRouter()

MP_ACCESS_TOKEN = os.environ.get("MP_ACCESS_TOKEN", "")


@router.post("/webhooks/mp")
async def mp_webhook(request: Request, db: Session = Depends(get_db)):
    try:
        body = await request.json()
    except Exception:
        return {"status": "ignored"}

    topic = body.get("type") or body.get("topic", "")
    if "preapproval" not in topic:
        return {"status": "ignored"}

    resource_id = None
    if body.get("data", {}).get("id"):
        resource_id = str(body["data"]["id"])
    elif body.get("id"):
        resource_id = str(body["id"])

    if not resource_id:
        log.warning("Webhook MP sin id de recurso: %s", body)
        return {"status": "ignored"}

    import mercadopago

    if not MP_ACCESS_TOKEN:
        log.error("MP_ACCESS_TOKEN no configurado; ignorando webhook")
        return {"status": "error"}

    sdk = mercadopago.SDK(MP_ACCESS_TOKEN)
    result = sdk.preapproval().get(resource_id)
    if result.get("status") != 200:
        log.error("No se pudo consultar preapproval %s: %s", resource_id, result)
        return {"status": "error"}

    preapproval = result.get("response", {})
    mp_status = preapproval.get("status", "")
    external_ref = preapproval.get("external_reference", "")

    sub = db.query(Suscripcion).filter(Suscripcion.proveedor_pago_id == resource_id).first()
    if not sub and external_ref:
        from database import Usuario
        user = db.query(Usuario).filter(Usuario.id == external_ref).first()
        if user:
            sub = db.query(Suscripcion).filter(Suscripcion.usuario_id == user.id).first()

    if not sub:
        log.warning("Webhook MP: no se encontró suscripción para preapproval %s (ref=%s)",
                     resource_id, external_ref)
        return {"status": "not_found"}

    if mp_status in ("authorized", "active"):
        sub.estado = "activa"
        sub.fecha_inicio = datetime.now(timezone.utc)
        date_str = preapproval.get("next_payment_date")
        if date_str:
            try:
                sub.fecha_fin = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        log.info("Suscripción activada para user %s (preapproval %s)", sub.usuario_id, resource_id)
    elif mp_status in ("cancelled", "paused"):
        sub.estado = "cancelada"
        log.info("Suscripción cancelada para user %s (preapproval %s)", sub.usuario_id, resource_id)
    elif mp_status == "pending":
        sub.estado = "pendiente"
    else:
        log.info("Webhook MP status no manejado: %s", mp_status)

    db.commit()
    return {"status": "ok"}
import os
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from database import Suscripcion, get_db

log = logging.getLogger(__name__)
router = APIRouter()

MP_ACCESS_TOKEN = os.environ.get("MP_ACCESS_TOKEN", "")


@router.post("/webhooks/mp")
async def mp_webhook(request: Request, db: Session = Depends(get_db)):
    try:
        body = await request.json()
    except Exception:
        return {"status": "ignored"}

    topic = body.get("type") or body.get("topic", "")
    if "preapproval" not in topic:
        return {"status": "ignored"}

    resource_id = None
    if body.get("data", {}).get("id"):
        resource_id = str(body["data"]["id"])
    elif body.get("id"):
        resource_id = str(body["id"])

    if not resource_id:
        log.warning("Webhook MP sin id de recurso: %s", body)
        return {"status": "ignored"}

    import mercadopago

    if not MP_ACCESS_TOKEN:
        log.error("MP_ACCESS_TOKEN no configurado; ignorando webhook")
        return {"status": "error"}

    sdk = mercadopago.SDK(MP_ACCESS_TOKEN)
    result = sdk.preapproval().get(resource_id)
    if result.get("status") != 200:
        log.error("No se pudo consultar preapproval %s: %s", resource_id, result)
        return {"status": "error"}

    preapproval = result.get("response", {})
    mp_status = preapproval.get("status", "")
    external_ref = preapproval.get("external_reference", "")

    sub = db.query(Suscripcion).filter(Suscripcion.proveedor_pago_id == resource_id).first()
    if not sub and external_ref:
        from database import Usuario
        user = db.query(Usuario).filter(Usuario.id == external_ref).first()
        if user:
            sub = db.query(Suscripcion).filter(Suscripcion.usuario_id == user.id).first()

    if not sub:
        log.warning("Webhook MP: no se encontró suscripción para preapproval %s (ref=%s)",
                     resource_id, external_ref)
        return {"status": "not_found"}

    if mp_status in ("authorized", "active"):
        sub.estado = "activa"
        sub.fecha_inicio = datetime.now(timezone.utc)
        date_str = preapproval.get("next_payment_date")
        if date_str:
            try:
                sub.fecha_fin = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        log.info("Suscripción activada para user %s (preapproval %s)", sub.usuario_id, resource_id)
    elif mp_status in ("cancelled", "paused"):
        sub.estado = "cancelada"
        log.info("Suscripción cancelada para user %s (preapproval %s)", sub.usuario_id, resource_id)
    elif mp_status == "pending":
        sub.estado = "pendiente"
    else:
        log.info("Webhook MP status no manejado: %s", mp_status)

    db.commit()
    return {"status": "ok"}
