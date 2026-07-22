"""Helpers para serializar datetimes que a veces llegan como str desde la BD."""

from __future__ import annotations

from datetime import date, datetime


def dt_to_iso(value: datetime | date | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return str(value)
