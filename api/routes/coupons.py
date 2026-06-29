import logging
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from api.deps import get_admin_user
from database import Cupon, Plan, UsoCupon, Usuario, get_db

router = APIRouter()
logger = logging.getLogger(__name__)


class CreateCouponRequest(BaseModel):
    codigo: str = Field(min_length=2, max_length=64)
    descripcion: str | None = Field(default=None, max_length=500)
    limite_usos: int = Field(default=1, ge=1)
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
    def validate_dates(self) -> "CreateCouponRequest":
        if (
            self.fecha_inicio is not None
            and self.fecha_expiracion is not None
            and self.fecha_expiracion <= self.fecha_inicio
        ):
            raise ValueError("fecha_expiracion debe ser posterior a fecha_inicio")
        return self


class CouponResponse(BaseModel):
    id: str
    codigo: str
    descripcion: str | None
    limite_usos: int
    limite_usos_por_usuario: int
    plan_id: int | None
    descuento_porcentaje: float | None
    descuento_monto: float | None
    fecha_inicio: str | None
    fecha_expiracion: str | None
    activo: bool
    limitaciones: dict[str, Any] | None
    fecha_creacion: str
    usos_totales: int


class CouponListResponse(BaseModel):
    cupones: list[CouponResponse]
    total: int


def _decimal_to_float(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


def _coupon_to_response(cupon: Cupon, *, usos_totales: int) -> CouponResponse:
    return CouponResponse(
        id=str(cupon.id),
        codigo=cupon.codigo,
        descripcion=cupon.descripcion,
        limite_usos=cupon.limite_usos,
        limite_usos_por_usuario=cupon.limite_usos_por_usuario,
        plan_id=cupon.plan_id,
        descuento_porcentaje=_decimal_to_float(cupon.descuento_porcentaje),
        descuento_monto=_decimal_to_float(cupon.descuento_monto),
        fecha_inicio=cupon.fecha_inicio.isoformat() if cupon.fecha_inicio else None,
        fecha_expiracion=cupon.fecha_expiracion.isoformat() if cupon.fecha_expiracion else None,
        activo=cupon.activo,
        limitaciones=cupon.limitaciones,
        fecha_creacion=cupon.fecha_creacion.isoformat(),
        usos_totales=usos_totales,
    )


def _count_usos(db: Session, cupon_id: UUID) -> int:
    return db.query(UsoCupon).filter(UsoCupon.cupon_id == cupon_id).count()


def _validate_plan(db: Session, plan_id: int | None) -> None:
    if plan_id is None:
        return
    if db.get(Plan, plan_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Plan no encontrado",
        )


@router.post("/cupones", response_model=CouponResponse, status_code=status.HTTP_201_CREATED)
def crear_cupon(
    body: CreateCouponRequest,
    admin: Usuario = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Crea un cupón. Solo usuarios con rol admin."""
    _validate_plan(db, body.plan_id)

    existing = db.query(Cupon).filter(Cupon.codigo == body.codigo).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ya existe un cupón con ese código",
        )

    cupon = Cupon(
        id=uuid.uuid4(),
        codigo=body.codigo,
        descripcion=body.descripcion,
        limite_usos=body.limite_usos,
        limite_usos_por_usuario=body.limite_usos_por_usuario,
        plan_id=body.plan_id,
        descuento_porcentaje=body.descuento_porcentaje,
        descuento_monto=body.descuento_monto,
        fecha_inicio=body.fecha_inicio,
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
    logger.info("Cupón creado: %s por admin %s", cupon.codigo, admin.id)
    return _coupon_to_response(cupon, usos_totales=0)


@router.get("/cupones", response_model=CouponListResponse)
def listar_cupones(
    admin: Usuario = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Lista todos los cupones. Solo usuarios con rol admin."""
    cupones = db.query(Cupon).order_by(Cupon.fecha_creacion.desc()).all()
    return CouponListResponse(
        cupones=[
            _coupon_to_response(c, usos_totales=_count_usos(db, c.id)) for c in cupones
        ],
        total=len(cupones),
    )


@router.get("/cupones/{cupon_id}", response_model=CouponResponse)
def obtener_cupon(
    cupon_id: UUID,
    admin: Usuario = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Obtiene un cupón por ID. Solo usuarios con rol admin."""
    cupon = db.get(Cupon, cupon_id)
    if cupon is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cupón no encontrado")
    return _coupon_to_response(cupon, usos_totales=_count_usos(db, cupon.id))
