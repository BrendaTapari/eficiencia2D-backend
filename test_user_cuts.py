"""Tests for user-defined panel cuts (mirrors frontend user-cuts.ts)."""

from core.services.cutting_sheet import Edge2D
from core.services.types import Vec2
from core.services.user_cuts import (
    UserCut,
    _edges_to_polygons,
    _rings_to_polygon,
    apply_user_cuts_to_panel,
)


def _rect_panel(w: float, h: float) -> tuple[float, float, list[Edge2D]]:
    edges = [
        Edge2D(a=Vec2(0, 0), b=Vec2(w, 0)),
        Edge2D(a=Vec2(w, 0), b=Vec2(w, h)),
        Edge2D(a=Vec2(w, h), b=Vec2(0, h)),
        Edge2D(a=Vec2(0, h), b=Vec2(0, 0)),
    ]
    return w, h, edges


def _piece_area(piece) -> float:
    rings = _edges_to_polygons(piece.edges)
    poly = _rings_to_polygon(rings)
    return poly.area if poly else piece.width_m * piece.height_m


def test_rect_cut_splits_panel():
    w, h, edges = _rect_panel(4.0, 3.0)
    cut = UserCut(
        id="c1",
        group_id=1,
        kind="rect",
        u0=1.0,
        v0=1.0,
        u1=2.0,
        v1=2.0,
    )
    pieces = apply_user_cuts_to_panel(w, h, edges, [cut])
    assert len(pieces) == 1
    assert _piece_area(pieces[0]) < w * h - 0.5


def test_line_cut_keeps_positive_half():
    w, h, edges = _rect_panel(4.0, 3.0)
    cut = UserCut(
        id="c2",
        group_id=1,
        kind="line",
        u0=2.0,
        v0=0.0,
        u1=2.0,
        v1=3.0,
        keep_positive=True,
    )
    pieces = apply_user_cuts_to_panel(w, h, edges, [cut])
    assert len(pieces) == 1
    assert abs(_piece_area(pieces[0]) - w * h / 2) < 0.05
    assert abs(pieces[0].width_m - w / 2) < 0.05


def test_circle_cut_removes_material():
    w, h, edges = _rect_panel(5.0, 5.0)
    cut = UserCut(
        id="c3",
        group_id=1,
        kind="circle",
        u0=2.5,
        v0=2.5,
        u1=3.5,
        v1=2.5,
    )
    pieces = apply_user_cuts_to_panel(w, h, edges, [cut])
    assert len(pieces) == 1
    assert _piece_area(pieces[0]) < w * h - 0.5


if __name__ == "__main__":
    test_rect_cut_splits_panel()
    test_line_cut_keeps_positive_half()
    test_circle_cut_removes_material()
    print("OK")
