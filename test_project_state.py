"""Tests de merge parcial del estado del proyecto (estado.json)."""

from api.routes.projects import ProyectoPartialSaveRequest, _merge_partial_estado


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
