"""Tests de merge parcial de preferencias_interfaz."""

import pytest
from fastapi import HTTPException

from api.routes.settings import _merge_colores_modelo, _merge_preferencias_interfaz


def test_merge_colores_modelo_partial_channel():
    result = _merge_colores_modelo(
        {"wall": "#111111", "floor": "#222222"},
        {"wall": "#4ade80"},
    )
    assert result == {"wall": "#4ade80", "floor": "#222222"}


def test_merge_colores_modelo_null_resets_all():
    assert _merge_colores_modelo({"wall": "#111111"}, None) is None


def test_merge_colores_modelo_all_channels_null():
    assert _merge_colores_modelo({"wall": "#111111"}, {"wall": None, "floor": None, "background": None}) is None


def test_merge_colores_modelo_validates_hex():
    with pytest.raises(HTTPException) as exc:
        _merge_colores_modelo({}, {"wall": "red"})
    assert exc.value.status_code == 400


def test_merge_preferencias_interfaz_preserves_other_keys():
    merged = _merge_preferencias_interfaz(
        {"navegacion_camara": "blender", "colores_modelo": {"wall": "#111111"}},
        {"colores_modelo": {"floor": "#94a3b8"}},
    )
    assert merged["navegacion_camara"] == "blender"
    assert merged["colores_modelo"] == {"wall": "#111111", "floor": "#94a3b8"}


def test_merge_preferencias_interfaz_new_navegacion():
    merged = _merge_preferencias_interfaz(
        {"colores_modelo": {"wall": "#111111"}},
        {"navegacion_camara": "cad"},
    )
    assert merged["navegacion_camara"] == "cad"
    assert merged["colores_modelo"] == {"wall": "#111111"}
