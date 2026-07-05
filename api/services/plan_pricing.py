from sqlalchemy.orm import Session

from database import PrecioPlan, Plan

DEFAULT_MONEDA = "ARS"
DEFAULT_PERIODO = "mes"


def get_precio_plan(
    db: Session,
    plan_id: int,
    *,
    moneda: str = DEFAULT_MONEDA,
    periodo: str = DEFAULT_PERIODO,
) -> PrecioPlan | None:
    """Precio preferido por moneda/período; si no hay, el primero del plan."""
    row = (
        db.query(PrecioPlan)
        .filter(
            PrecioPlan.planes_id == plan_id,
            PrecioPlan.moneda == moneda,
            PrecioPlan.periodo == periodo,
        )
        .first()
    )
    if row is not None:
        return row
    return (
        db.query(PrecioPlan)
        .filter(PrecioPlan.planes_id == plan_id)
        .order_by(PrecioPlan.id)
        .first()
    )


def get_precio_plan_for_plan(
    db: Session,
    plan: Plan,
    *,
    moneda: str = DEFAULT_MONEDA,
    periodo: str = DEFAULT_PERIODO,
) -> PrecioPlan | None:
    if plan.precios:
        for precio in plan.precios:
            if precio.moneda == moneda and precio.periodo == periodo:
                return precio
        return plan.precios[0]
    return get_precio_plan(db, plan.id, moneda=moneda, periodo=periodo)


def plan_tarifa(
    db: Session,
    plan: Plan,
    *,
    moneda: str = DEFAULT_MONEDA,
    periodo: str = DEFAULT_PERIODO,
) -> tuple[float, str, str]:
    """Devuelve (precio, moneda, periodo) del plan."""
    row = get_precio_plan_for_plan(db, plan, moneda=moneda, periodo=periodo)
    if row is None:
        return 0.0, moneda, periodo
    return float(row.precio), row.moneda or moneda, row.periodo or periodo


def serialize_precio_plan(row: PrecioPlan) -> dict:
    return {
        "id": row.id,
        "precio": float(row.precio),
        "moneda": row.moneda or DEFAULT_MONEDA,
        "periodo": row.periodo or DEFAULT_PERIODO,
    }
