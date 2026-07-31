import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from api.services.plan_pricing import plan_tarifa
from database import Cupon, Plan, UsoCupon, Usuario

CouponTipo = Literal["descuento", "plan"]


def cupon_tipo(cupon: Cupon) -> CouponTipo:
    if cupon.plan_id is not None:
        return "plan"
    return "descuento"


def tiene_descuento(
    descuento_porcentaje: Decimal | None,
    descuento_monto: Decimal | None,
) -> bool:
    if descuento_porcentaje is not None and descuento_porcentaje > 0:
        return True
    if descuento_monto is not None and descuento_monto > 0:
        return True
    return False


def validar_beneficio_cupon(
    *,
    plan_id: int | None,
    descuento_porcentaje: Decimal | None,
    descuento_monto: Decimal | None,
) -> None:
    """Exige plan XOR descuento (porcentaje o monto fijo, no ambos)."""
    has_plan = plan_id is not None
    has_pct = descuento_porcentaje is not None and descuento_porcentaje > 0
    has_amt = descuento_monto is not None and descuento_monto > 0
    has_discount = has_pct or has_amt

    if has_plan and has_discount:
        raise ValueError("Indicó un plan o un descuento, no ambos")
    if not has_plan and not has_discount:
        raise ValueError("Indicó plan_id o un descuento (porcentaje o monto)")
    if has_pct and has_amt:
        raise ValueError("Indicó descuento_porcentaje o descuento_monto, no ambos")


def resolve_fecha_inicio(fecha_inicio: datetime | None) -> datetime:
    if fecha_inicio is None:
        return datetime.now(timezone.utc)
    if fecha_inicio.tzinfo is None:
        return fecha_inicio.replace(tzinfo=timezone.utc)
    return fecha_inicio


def count_usos_totales(db: Session, cupon_id: uuid.UUID) -> int:
    return db.query(UsoCupon).filter(UsoCupon.cupon_id == cupon_id).count()


def count_usos_totales_batch(db: Session, cupon_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
    if not cupon_ids:
        return {}
    rows = (
        db.query(UsoCupon.cupon_id, func.count())
        .filter(UsoCupon.cupon_id.in_(cupon_ids))
        .group_by(UsoCupon.cupon_id)
        .all()
    )
    return {cupon_id: count for cupon_id, count in rows}


def count_usos_usuario(db: Session, cupon_id: uuid.UUID, usuario_id: uuid.UUID) -> int:
    return (
        db.query(UsoCupon)
        .filter(UsoCupon.cupon_id == cupon_id, UsoCupon.usuario_id == usuario_id)
        .count()
    )


def assert_cupon_vigente(
    cupon: Cupon,
    db: Session,
    *,
    usuario: Usuario | None = None,
    now: datetime | None = None,
) -> None:
    moment = now or datetime.now(timezone.utc)

    if not cupon.activo:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El cupón no está activo",
        )

    if cupon.fecha_inicio and moment < cupon.fecha_inicio:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El cupón aún no está vigente",
        )

    if cupon.fecha_expiracion and moment > cupon.fecha_expiracion:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El cupón expiró",
        )

    usos_totales = count_usos_totales(db, cupon.id)
    if usos_totales >= cupon.limite_usos:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El cupón alcanzó el límite de usos",
        )

    if usuario is not None:
        usos_usuario = count_usos_usuario(db, cupon.id, usuario.id)
        if usos_usuario >= cupon.limite_usos_por_usuario:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ya usaste este cupón el máximo de veces permitido",
            )


def get_cupon_por_codigo(db: Session, codigo: str) -> Cupon:
    normalized = codigo.strip().upper()
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ingresó un código de cupón",
        )
    cupon = (
        db.query(Cupon)
        .filter(Cupon.codigo == normalized, Cupon.activo.is_(True))
        .first()
    )
    if cupon is None:
        cupon = db.query(Cupon).filter(Cupon.codigo == normalized).first()
    if cupon is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cupón no encontrado",
        )
    return cupon


def assert_codigo_cupon_disponible(
    db: Session,
    codigo: str,
    *,
    excluir_cupon_id: uuid.UUID | None = None,
) -> None:
    """Impide crear/activar un cupón si ya hay otro activo con el mismo código."""
    normalized = codigo.strip().upper()
    query = db.query(Cupon).filter(
        Cupon.codigo == normalized,
        Cupon.activo.is_(True),
    )
    if excluir_cupon_id is not None:
        query = query.filter(Cupon.id != excluir_cupon_id)
    if query.first() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ya existe un cupón activo con ese código",
        )


def calcular_precio_con_descuento(precio: float, cupon: Cupon) -> float:
    if cupon.descuento_porcentaje is not None and cupon.descuento_porcentaje > 0:
        factor = float(cupon.descuento_porcentaje) / 100.0
        return max(round(precio * (1.0 - factor), 2), 0.0)
    if cupon.descuento_monto is not None and cupon.descuento_monto > 0:
        return max(round(precio - float(cupon.descuento_monto), 2), 0.0)
    return precio


def registrar_uso_cupon(db: Session, cupon: Cupon, usuario: Usuario) -> None:
    db.add(
        UsoCupon(
            id=uuid.uuid4(),
            cupon_id=cupon.id,
            usuario_id=usuario.id,
        )
    )


def serialize_plan_brief(plan: Plan | None, db: Session | None = None) -> dict | None:
    if plan is None:
        return None
    if db is not None:
        precio, moneda, _periodo = plan_tarifa(db, plan)
    elif plan.precios:
        row = plan.precios[0]
        precio, moneda = float(row.precio), row.moneda or "ARS"
    else:
        precio, moneda = 0.0, "ARS"
    return {
        "id": plan.id,
        "slug": plan.slug,
        "nombre": plan.nombre,
        "precio_mensual": precio,
        "moneda": moneda,
    }


def mensaje_cupon_valido(cupon: Cupon, plan: Plan | None) -> str:
    if cupon_tipo(cupon) == "plan" and plan is not None:
        return f"Cupón válido: acceso al plan {plan.nombre}"
    if cupon.descuento_porcentaje is not None:
        return f"Cupón válido: {float(cupon.descuento_porcentaje):g}% de descuento"
    if cupon.descuento_monto is not None:
        return f"Cupón válido: ${float(cupon.descuento_monto):g} de descuento"
    return "Cupón válido"
