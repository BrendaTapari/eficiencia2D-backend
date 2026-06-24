"""Tests unitarios para la secuencia de ensamble."""
from core.services.assembly_logic import (
    AssemblyPiece,
    BoundingBox3D,
    generate_assembly_sequence,
)
from core.services.types import Vec3


def _box(
    pid: str,
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    normal: Vec3,
) -> AssemblyPiece:
    return AssemblyPiece(
        id=pid,
        normal=normal,
        bbox=BoundingBox3D(x0, y0, z0, x1, y1, z1),
        category="wall" if abs(normal.y) < 0.5 else "floor",
    )


def test_base_walls_roof_sequence():
    base = _box("B-01", 0, 0, 0, 4, 0.15, 3, Vec3(0, 1, 0))
    wall_a = _box("M-01", 0, 0.15, 0, 0.12, 2.5, 3, Vec3(1, 0, 0))
    wall_b = _box("M-02", 3.88, 0.15, 0, 4, 2.5, 3, Vec3(-1, 0, 0))
    roof = _box("T-01", 0, 2.5, 0, 4, 2.62, 3, Vec3(0, 1, 0))

    steps = generate_assembly_sequence([base, wall_a, wall_b, roof], epsilon=0.02)

    assert len(steps) >= 3
    assert steps[0]["part_ids"] == ["B-01"]
    assert "B-01" not in steps[1]["part_ids"]
    assert set(steps[1]["part_ids"]) | set(steps[2]["part_ids"]) >= {"M-01", "M-02"}
    assert steps[-1]["part_ids"] == ["T-01"]
    assert "camera_focus" in steps[0]
    assert "x" in steps[0]["camera_focus"]


def test_empty_pieces():
    assert generate_assembly_sequence([]) == []
