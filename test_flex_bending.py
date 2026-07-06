"""Tests del patrón de flexión (kerf / auxético) y del desarrollo de superficies curvas."""

import math

from core.services.cutting_sheet import Edge2D, emit_panel_entities
from core.services.types import Face3D, Vec2, Vec3
from core.services.flex_bending import apply_flex_to_panel, build_flex_by_group, parse_flex
from core.services.curvature import detect_curvature, unroll


def _rect_panel(w, h):
    return w, h, [
        Edge2D(a=Vec2(0, 0), b=Vec2(w, 0)),
        Edge2D(a=Vec2(w, 0), b=Vec2(w, h)),
        Edge2D(a=Vec2(w, h), b=Vec2(0, h)),
        Edge2D(a=Vec2(0, h), b=Vec2(0, 0)),
    ]


def _face(a, b, c):
    n = (
        (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
        (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
    )
    l = math.sqrt(sum(x * x for x in n)) or 1.0
    return Face3D(
        vertices=[Vec3(*a), Vec3(*b), Vec3(*c)],
        normal=Vec3(n[0] / l, n[1] / l, n[2] / l),
        inner_loops=[],
    )


# --- flex: geometría del patrón --------------------------------------------


def test_parse_flex_validates_and_clamps():
    specs = parse_flex(
        [
            {"group_id": "x", "method": "kerf", "spacing_m": 0.02},   # gid inválido
            {"method": "nope", "spacing_m": 0.02},                    # método inválido
            {"group_id": 5, "method": "kerf", "spacing_m": 0.02},     # ok
            {"group_id": 6, "method": "kerf", "spacing_m": 0.0001},   # spacing clampeado
        ]
    )
    assert [s.group_id for s in specs] == [5, 6]
    assert specs[1].spacing_m >= 0.003  # clamp mínimo


def test_kerf_smaller_spacing_more_slots():
    w, h, edges = _rect_panel(0.6, 0.4)
    big = parse_flex([{"group_id": 1, "method": "kerf", "spacing_m": 0.03}])[0]
    small = parse_flex([{"group_id": 1, "method": "kerf", "spacing_m": 0.008}])[0]
    n_big = len(apply_flex_to_panel(w, h, edges, big))
    n_small = len(apply_flex_to_panel(w, h, edges, small))
    assert n_small > n_big > 0


def test_kerf_removes_rectangular_holes():
    """v2: el kerf REMUEVE ranuras rectangulares (huecos cerrados), no líneas."""
    from shapely.geometry import Polygon as _P

    from core.services.flex_bending import FlexSpec, _kerf_slots

    spec = FlexSpec(group_id=1, method="kerf", spacing_m=0.03, ligament_m=0.012)
    slots = _kerf_slots(0.6, 0.4, spec)
    assert slots, "kerf debe generar huecos"
    assert all(isinstance(s, _P) and s.area > 0 for s in slots), "deben ser rectángulos cerrados"


def test_kerf_wider_spacing_removes_more_material():
    """v2: al aumentar spacing la ranura se ensancha (slot_w = spacing − ligament)."""
    from core.services.flex_bending import FlexSpec, _kerf_slots

    lig = 0.012
    s3 = _kerf_slots(0.6, 0.4, FlexSpec(1, "kerf", 0.03, lig))
    s5 = _kerf_slots(0.6, 0.4, FlexSpec(1, "kerf", 0.05, lig))
    width = lambda s: s.bounds[2] - s.bounds[0]
    assert width(s5[0]) > width(s3[0]), "más spacing ⇒ ranura más ancha"


def test_auxetic_rotating_removes_cells():
    """v2: los auxéticos también REMUEVEN celdas (huecos), no líneas."""
    from shapely.geometry import Polygon as _P

    from core.services.flex_bending import FlexSpec, _auxetic_rotating

    holes = _auxetic_rotating(0.4, 0.4, FlexSpec(1, "auxetic_rotating", 0.03, 0.008))
    assert holes and all(isinstance(h, _P) and h.area > 0 for h in holes)


def test_each_method_produces_clipped_flex_edges():
    w, h, edges = _rect_panel(0.6, 0.4)
    for method in ("kerf", "auxetic_rotating", "auxetic_reentrant", "auxetic_chiral"):
        spec = parse_flex([{"group_id": 1, "method": method, "spacing_m": 0.02}])[0]
        fe = apply_flex_to_panel(w, h, edges, spec)
        assert fe, method
        assert all(e.flex for e in fe), method
        # Recortado al panel (con margen): todo dentro de [0,w]x[0,h].
        for e in fe:
            for p in (e.a, e.b):
                assert -1e-6 <= p.x <= w + 1e-6 and -1e-6 <= p.y <= h + 1e-6, method


# --- curvatura: detección y desarrollo -------------------------------------


def _flat_grid():
    fs = []
    for i in range(8):
        for j in range(6):
            x0, x1, y0, y1 = i * 0.1, (i + 1) * 0.1, j * 0.1, (j + 1) * 0.1
            fs.append(_face((x0, y0, 0), (x1, y0, 0), (x1, y1, 0)))
            fs.append(_face((x0, y0, 0), (x1, y1, 0), (x0, y1, 0)))
    return fs


def _cylinder(R=0.5, sweep=math.pi, N=24, H=6):
    fs = []
    for i in range(N):
        for j in range(H):
            t0, t1 = sweep * i / N, sweep * (i + 1) / N
            y0, y1 = j * 0.15, (j + 1) * 0.15
            p00 = (R * math.cos(t0), y0, R * math.sin(t0))
            p10 = (R * math.cos(t1), y0, R * math.sin(t1))
            p01 = (R * math.cos(t0), y1, R * math.sin(t0))
            p11 = (R * math.cos(t1), y1, R * math.sin(t1))
            fs.append(_face(p00, p10, p11))
            fs.append(_face(p00, p11, p01))
    return fs


def _sphere_cap(R=0.5, M=16):
    fs = []
    for i in range(M):
        for j in range(M // 2):
            u0, u1 = math.pi * i / M, math.pi * (i + 1) / M
            v0 = (math.pi / 2) * (j / (M / 2)) * 0.7
            v1 = (math.pi / 2) * ((j + 1) / (M / 2)) * 0.7

            def sp(u, v):
                return (R * math.sin(v) * math.cos(u), R * math.cos(v), R * math.sin(v) * math.sin(u))

            fs.append(_face(sp(u0, v0), sp(u1, v0), sp(u1, v1)))
            fs.append(_face(sp(u0, v0), sp(u1, v1), sp(u0, v1)))
    return fs


def test_flat_is_not_curved():
    info = detect_curvature(_flat_grid())
    assert not info.curved and info.kind == "flat"


def test_cylinder_is_single_curvature():
    info = detect_curvature(_cylinder())
    assert info.curved and info.kind == "single"
    assert abs(info.bend_radius_m - 0.5) < 0.05


def test_sphere_is_double_curvature():
    info = detect_curvature(_sphere_cap())
    assert info.curved and info.kind == "double"


def test_cylinder_unroll_preserves_arc_length():
    R = 0.5
    faces = _cylinder(R=R, sweep=math.pi)
    up = unroll(faces)
    assert up is not None and up.kind == "single"
    # ancho desarrollado ≈ media circunferencia R·π
    assert abs(up.width_m - R * math.pi) < 0.05
    assert abs(up.height_m - 0.9) < 0.05


# --- integración DXF: el patrón cae en la capa FLEX_CUT --------------------


def test_dxf_emits_flex_layer():
    lines = []
    edges = [
        Edge2D(a=Vec2(0, 0), b=Vec2(0.1, 0)),               # contorno -> CUT_EXTERIOR
        Edge2D(a=Vec2(0.02, 0.02), b=Vec2(0.02, 0.08), flex=True),  # patrón -> FLEX_CUT
    ]
    emit_panel_entities(lines, edges, 0.1, 0.1, "A1", 0.0, 0.0, include_text=False)
    assert "FLEX_CUT" in lines
    assert "CUT_EXTERIOR" in lines


def test_dxf_no_flex_layer_without_flex_edges():
    lines = []
    edges = [Edge2D(a=Vec2(0, 0), b=Vec2(0.1, 0))]
    emit_panel_entities(lines, edges, 0.1, 0.1, "A1", 0.0, 0.0, include_text=False)
    assert "FLEX_CUT" not in lines


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
