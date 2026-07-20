"""Tests de merge parcial del estado del proyecto (estado.json)."""

from unittest.mock import MagicMock, patch

from api.routes.projects import (
    ProyectoPartialSaveRequest,
    _merge_partial_estado,
    _proyecto_tiene_estado_guardado,
    _set_proyecto_nombre,
)


def test_merge_mark_lines_preserves_other_fields():
    existing = {"marks": [1], "user_cuts": [{"group_id": 2}]}
    patch = ProyectoPartialSaveRequest(
        mark_lines=[
            {
                "id": "mkl-abc123",
                "group_id": 12,
                "points": [[0.1, 0.2], [0.5, 0.8], [0.9, 0.3]],
            }
        ]
    )
    merged = _merge_partial_estado(existing, patch)
    assert merged["marks"] == [1]
    assert merged["user_cuts"] == [{"group_id": 2}]
    assert merged["mark_lines"][0]["id"] == "mkl-abc123"
    assert merged["mark_lines"][0]["group_id"] == 12


def test_merge_mark_lines_replaces_array():
    existing = {"mark_lines": [{"id": "old", "group_id": 1, "points": [[0, 0], [1, 1]]}]}
    patch = ProyectoPartialSaveRequest(mark_lines=[])
    merged = _merge_partial_estado(existing, patch)
    assert merged["mark_lines"] == []


def test_merge_nombre_via_state_patch():
    existing = {"marks": [1]}
    patch = ProyectoPartialSaveRequest(nombre="Nuevo nombre")
    merged = _merge_partial_estado(existing, patch)
    assert "nombre" not in merged  # nombre se aplica fuera del merge genérico


def test_proyecto_tiene_estado_guardado_por_r2():
    proyecto = MagicMock()
    proyecto.metadata_impresion = {"estado_r2": "user/proj/estado.json"}
    assert _proyecto_tiene_estado_guardado(proyecto, {}) is True


def test_proyecto_tiene_estado_guardado_por_contenido():
    proyecto = MagicMock()
    proyecto.metadata_impresion = {}
    assert _proyecto_tiene_estado_guardado(proyecto, {"marks": [1]}) is True


@patch("api.routes.projects._persist_estado_proyecto")
@patch("api.routes.projects._load_estado_proyecto")
def test_set_proyecto_nombre_sincroniza_estado(mock_load, mock_persist):
    proyecto = MagicMock()
    proyecto.id = "pid"
    proyecto.metadata_impresion = {"estado_r2": "u/p/estado.json"}
    mock_load.return_value = {"marks": [2], "nombre": "Viejo"}
    mock_persist.return_value = "2026-07-16T00:00:00+00:00"
    db = MagicMock()

    sync, updated_at = _set_proyecto_nombre(db, proyecto, "Nuevo")

    assert sync is True
    assert updated_at == "2026-07-16T00:00:00+00:00"
    assert proyecto.nombre == "Nuevo"
    mock_persist.assert_called_once()
    estado_guardado = mock_persist.call_args[0][2]
    assert estado_guardado["nombre"] == "Nuevo"
    assert estado_guardado["marks"] == [2]


@patch("api.routes.projects._commit_proyecto_nombre")
@patch("api.routes.projects._load_estado_proyecto")
def test_set_proyecto_nombre_solo_bd_sin_estado(mock_load, mock_commit):
    proyecto = MagicMock()
    proyecto.metadata_impresion = {"archivo_original": "modelo.obj"}
    mock_load.return_value = {}
    db = MagicMock()

    sync, updated_at = _set_proyecto_nombre(db, proyecto, "Solo BD")

    assert sync is False
    assert updated_at is None
    mock_commit.assert_called_once_with(db, proyecto, "Solo BD")
