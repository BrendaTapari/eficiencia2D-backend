import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, joinedload

from api.deps import get_admin_user, get_optional_current_user
from api.services.plan_pricing import plan_tarifa
from api.services.cupon_service import (
    assert_cupon_vigente,
    calcular_precio_con_descuento,
    count_usos_totales,
    cupon_tipo,
    get_cupon_por_codigo,
    mensaje_cupon_valido,
    resolve_fecha_inicio,
    serialize_plan_brief,
    validar_beneficio_cupon,
)
from database import Cupon, Plan, Usuario, get_db

router = APIRouter()
logger = logging.getLogger(__name__)


class CreateCouponRequest(BaseModel):
    codigo: str = Field(min_length=2, max_length=64)
    descripcion: str = Field(min_length=1, max_length=500)
    limite_usos: int = Field(..., ge=1)
    limite_usos_por_usuario: int = Field(default=1, ge=1)
    plan_id: int | None = None
    descuento_porcentaje: Decimal | None = Field(default=None, ge=0, le=100)
    descuento_monto: Decimal | None = Field(default=None, ge=0)
    fecha_inicio: datetime | None = None
    fecha_expiracion: datetime | None = None
    activo: bool = True
    limitaciones: dict[str, Any] | None = None

    @field_validator("codigo")
    @classmethod
    def normalize_codigo(cls, value: str) -> str:
        codigo = value.strip().upper()
        if not codigo:
            raise ValueError("El código del cupón no puede estar vacío")
        return codigo

    @model_validator(mode="after")
    def validate_coupon(self) -> "CreateCouponRequest":
        validar_beneficio_cupon(
            plan_id=self.plan_id,
            descuento_porcentaje=self.descuento_porcentaje,
            descuento_monto=self.descuento_monto,
        )
        if (
            self.fecha_inicio is not None
            and self.fecha_expiracion is not None
            and self.fecha_expiracion <= self.fecha_inicio
        ):
            raise ValueError("fecha_expiracion debe ser posterior a fecha_inicio")
        return self


class ValidarCuponRequest(BaseModel):
    codigo: str = Field(min_length=2, max_length=64)
    plan_id: int | None = None

    @field_validator("codigo")
    @classmethod
    def normalize_codigo(cls, value: str) -> str:
        return value.strip().upper()


class PlanBriefResponse(BaseModel):
    id: int
    slug: str | None
    nombre: str
    precio_mensual: float
    moneda: str


class CouponResponse(BaseModel):
    id: str
    codigo: str
    descripcion: str | None
    tipo: Literal["descuento", "plan"]
    limite_usos: int
    limite_usos_por_usuario: int
    plan_id: int | None
    plan: PlanBriefResponse | None = None
    descuento_porcentaje: float | None
    descuento_monto: float | None
    fecha_inicio: str | None
    fecha_expiracion: str | None
    activo: bool
    limitaciones: dict[str, Any] | None
    fecha_creacion: str
    usos_totales: int
    usos_actuales: int


class ValidarCuponResponse(BaseModel):
    cupon: CouponResponse
    tipo: Literal["descuento", "plan"]
    message: str
    precio_original: float | None = None
    precio_final: float | None = None


class CouponListResponse(BaseModel):
    cupones: list[CouponResponse]
    total: int


class UpdateCouponRequest(BaseModel):
    activo: bool


def _decimal_to_float(value: Decimal | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if numeric > 0 else None


def _plan_brief(plan: Plan | None, db: Session) -> PlanBriefResponse | None:
    data = serialize_plan_brief(plan, db)
    if data is None:
        return None
    return PlanBriefResponse(**data)


def _coupon_to_response(
    cupon: Cupon,
    *,
    usos_totales: int,
    plan: Plan | None = None,
    db: Session | None = None,
) -> CouponResponse:
    tipo = cupon_tipo(cupon)
    linked_plan = plan
    if linked_plan is None and cupon.plan_id is not None:
        linked_plan = cupon.plan

    return CouponResponse(
        id=str(cupon.id),
        codigo=cupon.codigo,
        descripcion=cupon.descripcion,
        tipo=tipo,
        limite_usos=cupon.limite_usos,
        limite_usos_por_usuario=cupon.limite_usos_por_usuario,
        plan_id=cupon.plan_id,
        plan=_plan_brief(linked_plan, db) if db is not None else None,
        descuento_porcentaje=_decimal_to_float(cupon.descuento_porcentaje),
        descuento_monto=_decimal_to_float(cupon.descuento_monto),
        fecha_inicio=cupon.fecha_inicio.isoformat() if cupon.fecha_inicio else None,
        fecha_expiracion=cupon.fecha_expiracion.isoformat() if cupon.fecha_expiracion else None,
        activo=cupon.activo,
        limitaciones=cupon.limitaciones,
        fecha_creacion=cupon.fecha_creacion.isoformat(),
        usos_totales=usos_totales,
        usos_actuales=usos_totales,
    )


def _validate_plan(db: Session, plan_id: int | None) -> Plan | None:
    if plan_id is None:
        return None
    plan = db.get(Plan, plan_id)
    if plan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Plan no encontrado",
        )
    return plan


@router.post("/cupones", response_model=CouponResponse, status_code=status.HTTP_201_CREATED)
def crear_cupon(
    body: CreateCouponRequest,
    admin: Usuario = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Crea un cupón de plan o descuento. Solo admin."""
    plan = _validate_plan(db, body.plan_id)

    existing = db.query(Cupon).filter(Cupon.codigo == body.codigo).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ya existe un cupón con ese código",
        )

    fecha_inicio = resolve_fecha_inicio(body.fecha_inicio)
    is_plan_cupon = body.plan_id is not None

    cupon = Cupon(
        id=uuid.uuid4(),
        codigo=body.codigo,
        descripcion=body.descripcion.strip(),
        limite_usos=body.limite_usos,
        limite_usos_por_usuario=body.limite_usos_por_usuario,
        plan_id=body.plan_id if is_plan_cupon else None,
        descuento_porcentaje=None if is_plan_cupon else body.descuento_porcentaje,
        descuento_monto=None if is_plan_cupon else body.descuento_monto,
        fecha_inicio=fecha_inicio,
        fecha_expiracion=body.fecha_expiracion,
        activo=body.activo,
        limitaciones=body.limitaciones,
    )

    db.add(cupon)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No se pudo crear el cupón (código duplicado o plan inválido)",
        ) from None
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Error al crear cupón")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo guardar el cupón",
        ) from None

    db.refresh(cupon)
    logger.info("Cupón creado: %s (%s) por admin %s", cupon.codigo, cupon_tipo(cupon), admin.id)
    return _coupon_to_response(cupon, usos_totales=0, plan=plan, db=db)


@router.post("/cupones/validar", response_model=ValidarCuponResponse)
def validar_cupon(
    body: ValidarCuponRequest,
    db: Session = Depends(get_db),
    current_user: Usuario | None = Depends(get_optional_current_user),
):
    """Valida un cupón por código (opcionalmente contra un plan seleccionado)."""
    cupon = get_cupon_por_codigo(db, body.codigo)
    assert_cupon_vigente(cupon, db, usuario=current_user)

    tipo = cupon_tipo(cupon)
    plan_obj: Plan | None = None
    precio_original: float | None = None
    precio_final: float | None = None

    if tipo == "plan":
        plan_obj = _validate_plan(db, cupon.plan_id)
        if body.plan_id is not None and body.plan_id != cupon.plan_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Este cupón aplica a otro plan",
            )
    else:
        if body.plan_id is not None:
            plan_obj = (
                db.query(Plan)
                .options(joinedload(Plan.precios))
                .filter(Plan.id == body.plan_id, Plan.activo.is_(True))
                .first()
            )
            if plan_obj is None:
                raise HTTPException(status_code=404, detail="Plan no encontrado o inactivo")
            precio_original, _moneda, _periodo = plan_tarifa(db, plan_obj)
            if precio_original <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Los cupones de descuento aplican a planes de pago",
                )
            precio_final = calcular_precio_con_descuento(precio_original, cupon)

    response_cupon = _coupon_to_response(
        cupon,
        usos_totales=count_usos_totales(db, cupon.id),
        plan=plan_obj or (cupon.plan if cupon.plan_id else None),
        db=db,
    )

    return ValidarCuponResponse(
        cupon=response_cupon,
        tipo=tipo,
        message=mensaje_cupon_valido(cupon, plan_obj or cupon.plan),
        precio_original=precio_original,
        precio_final=precio_final,
    )


@router.get("/cupones", response_model=CouponListResponse)
def listar_cupones(
    admin: Usuario = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Lista todos los cupones. Solo admin."""
    cupones = db.query(Cupon).order_by(Cupon.fecha_creacion.desc()).all()
    return CouponListResponse(
        cupones=[
            _coupon_to_response(c, usos_totales=count_usos_totales(db, c.id), db=db)
            for c in cupones
        ],
        total=len(cupones),
    )


@router.get("/cupones/{cupon_id}", response_model=CouponResponse)
def obtener_cupon(
    cupon_id: UUID,
    admin: Usuario = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Obtiene un cupón por ID. Solo admin."""
    cupon = db.get(Cupon, cupon_id)
    if cupon is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cupón no encontrado")
    return _coupon_to_response(cupon, usos_totales=count_usos_totales(db, cupon.id), db=db)


@router.patch("/cupones/{cupon_id}", response_model=CouponResponse)
def actualizar_cupon(
    cupon_id: UUID,
    body: UpdateCouponRequest,
    admin: Usuario = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Actualiza un cupón (p. ej. desactivarlo). Solo admin."""
    cupon = db.get(Cupon, cupon_id)
    if cupon is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cupón no encontrado")

    if cupon.activo == body.activo:
        return _coupon_to_response(
            cupon,
            usos_totales=count_usos_totales(db, cupon.id),
            db=db,
        )

    cupon.activo = body.activo

    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Error al actualizar cupón %s", cupon_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo actualizar el cupón",
        ) from None

    db.refresh(cupon)
    action = "activado" if body.activo else "desactivado"
    logger.info("Cupón %s (%s) por admin %s", action, cupon.codigo, admin.id)
    return _coupon_to_response(cupon, usos_totales=count_usos_totales(db, cupon.id), db=db)
