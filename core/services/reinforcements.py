"""
Refuerzos estructurales: NERVIOS (cartelas) — Fase 1.

Un nervio es una cartela (triángulo rectángulo) que se apoya sobre la arista de
intersección de dos placas perpendiculares (p. ej. pared-piso), en la posición `pos_t`
(0..1) a lo largo de esa arista. Se genera:
  - la PIEZA física plana (triángulo + 2 pestañas) para cortar y nestear (afecta precio),
  - las MUESCAS (ranuras) en las dos placas receptoras por donde entran las pestañas.

Todo en el mismo espacio 3D Y-up que `faces`/`placements`. La geometría exacta la define
el backend; el front sólo manda id/group_a/group_b/size_m/pos_t.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from core.group_classifier import GeometryGroup
from core.services.cutting_sheet import Edge2D
from core.services.plate_intersect import group_plane
from core.services.types import Face3D, Vec2, Vec3, cross, dot, normalize, sub

MIN_SIZE_M = 0.03
MAX_SIZE_M = 1.0
DEFAULT_THICKNESS_M = 0.003


@dataclass
class Rib:
    id: str
    group_a: int
    group_b: int
    size_m: float
    pos_t: float


@dataclass
class RibGeometry:
    rib_id: str
    group_a: int
    group_b: int
    width_m: float
    height_m: float
    edges: List[Edge2D]                      # contorno 2D de la cartela (triángulo + pestañas)
    notch_a: Optional[Tuple[Vec3, Vec3, float]]  # muesca en group_a (P0, P1, ancho)
    notch_b: Optional[Tuple[Vec3, Vec3, float]]  # muesca en group_b


def parse_ribs(raw: Optional[List[dict]]) -> List[Rib]:
    out: List[Rib] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        ga, gb = item.get("group_a"), item.get("group_b")
        if not isinstance(ga, int) or not isinstance(gb, int) or ga == gb:
            continue
        try:
            size = float(item.get("size_m"))
        except (TypeError, ValueError):
            continue
        size = min(max(size, MIN_SIZE_M), MAX_SIZE_M)
        try:
            pos_t = float(item.get("pos_t", 0.5))
        except (TypeError, ValueError):
            pos_t = 0.5
        pos_t = min(max(pos_t, 0.0), 1.0)
        out.append(Rib(id=str(item.get("id") or ""), group_a=ga, group_b=gb,
                       size_m=size, pos_t=pos_t))
    return out


def _v(a) -> Vec3:
    return Vec3(float(a[0]), float(a[1]), float(a[2]))


def _group_verts(g: GeometryGroup, faces: List[Face3D]) -> List[Vec3]:
    return [v for fi in g.face_indices if 0 <= fi < len(faces) for v in faces[fi].vertices]


def _intersection_edge(
    ga: GeometryGroup, gb: GeometryGroup, faces: List[Face3D]
) -> Optional[Tuple[Vec3, Vec3, Vec3]]:
    """Arista (segmento) donde se cruzan los planos de ga y gb, recortada al solape de
    ambas placas a lo largo de la dirección de la arista. Devuelve (P0, P1, dir)."""
    na, da = group_plane(ga, faces)
    nb, db = group_plane(gb, faces)
    e = cross(na, nb)
    if math.sqrt(e.x ** 2 + e.y ** 2 + e.z ** 2) < 1e-6:
        return None
    e = normalize(e)
    M = np.array([[na.x, na.y, na.z], [nb.x, nb.y, nb.z], [e.x, e.y, e.z]])
    rhs = np.array([da, db, 0.0])
    try:
        p0 = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None
    base = Vec3(float(p0[0]), float(p0[1]), float(p0[2]))

    # rango de t = e·(v - base) donde AMBAS placas existen (solape de proyecciones)
    def trange(g):
        ts = [dot(e, sub(v, base)) for v in _group_verts(g, faces)]
        return (min(ts), max(ts)) if ts else None
    ra, rb = trange(ga), trange(gb)
    if not ra or not rb:
        return None
    t0 = max(ra[0], rb[0])
    t1 = min(ra[1], rb[1])
    if t1 - t0 < 1e-4:
        return None
    P0 = Vec3(base.x + e.x * t0, base.y + e.y * t0, base.z + e.z * t0)
    P1 = Vec3(base.x + e.x * t1, base.y + e.y * t1, base.z + e.z * t1)
    return P0, P1, e


def _dir_into(g: GeometryGroup, faces: List[Face3D], e: Vec3, P: Vec3) -> Vec3:
    """Dirección en la superficie de g, perpendicular a la arista e, hacia el interior."""
    n, _ = group_plane(g, faces)
    u = normalize(cross(n, e))
    # orientar hacia el centroide del grupo
    c = g.centroid
    if dot(u, sub(c, P)) < 0:
        u = Vec3(-u.x, -u.y, -u.z)
    return u


def _rib_panel_edges(size: float, thickness: float) -> Tuple[List[Edge2D], float, float]:
    """Contorno 2D de la cartela: triángulo rectángulo (catetos `size`) con una pestaña
    en cada cateto (rectángulo que sobresale `thickness*·` para encastrar en la placa)."""
    s = size
    td = max(thickness, DEFAULT_THICKNESS_M) * 2.0  # profundidad de pestaña (visible)
    t0, t1 = 0.30 * s, 0.62 * s
    pts = [
        (0.0, 0.0),
        (t0, 0.0), (t0, -td), (t1, -td), (t1, 0.0),   # pestaña sobre el cateto horizontal
        (s, 0.0),
        (0.0, s),                                      # hipotenusa
        (0.0, t1), (-td, t1), (-td, t0), (0.0, t0),    # pestaña sobre el cateto vertical
    ]
    edges = [Edge2D(a=Vec2(*pts[i]), b=Vec2(*pts[(i + 1) % len(pts)]))
             for i in range(len(pts))]
    # normalizar a origen 0-based
    minx = min(p[0] for p in pts); miny = min(p[1] for p in pts)
    edges = [Edge2D(a=Vec2(e.a.x - minx, e.a.y - miny),
                    b=Vec2(e.b.x - minx, e.b.y - miny)) for e in edges]
    maxx = max(p[0] for p in pts) - minx
    maxy = max(p[1] for p in pts) - miny
    return edges, maxx, maxy


def build_rib_geometry(
    rib: Rib, ga: GeometryGroup, gb: GeometryGroup, faces: List[Face3D]
) -> Optional[RibGeometry]:
    edge = _intersection_edge(ga, gb, faces)
    if edge is None:
        return None
    P0, P1, e = edge
    P = Vec3(P0.x + (P1.x - P0.x) * rib.pos_t,
             P0.y + (P1.y - P0.y) * rib.pos_t,
             P0.z + (P1.z - P0.z) * rib.pos_t)

    thickness = ga.thickness or gb.thickness or DEFAULT_THICKNESS_M
    edges, w, h = _rib_panel_edges(rib.size_m, thickness)

    # Muescas: por donde el nervio entra en cada placa (segmento en su superficie, desde
    # la arista hacia el interior). Ancho = grosor del nervio.
    u_a = _dir_into(ga, faces, e, P)
    u_b = _dir_into(gb, faces, e, P)
    notch_len = rib.size_m * 0.6
    na = (P, Vec3(P.x + u_a.x * notch_len, P.y + u_a.y * notch_len, P.z + u_a.z * notch_len),
          thickness)
    nb = (P, Vec3(P.x + u_b.x * notch_len, P.y + u_b.y * notch_len, P.z + u_b.z * notch_len),
          thickness)
    return RibGeometry(rib_id=rib.id, group_a=rib.group_a, group_b=rib.group_b,
                       width_m=w, height_m=h, edges=edges, notch_a=na, notch_b=nb)


def build_ribs(
    ribs: List[Rib], groups: List[GeometryGroup], faces: List[Face3D]
) -> List[RibGeometry]:
    by_id = {g.id: g for g in groups}
    out: List[RibGeometry] = []
    for rib in ribs:
        ga, gb = by_id.get(rib.group_a), by_id.get(rib.group_b)
        if ga is None or gb is None:
            continue
        geo = build_rib_geometry(rib, ga, gb, faces)
        if geo is not None:
            out.append(geo)
    return out
