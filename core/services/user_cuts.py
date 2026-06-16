"""
User-defined subtractive cuts on review panels (mirrors src/core/user-cuts.ts).

Coordinates are metres in the panel-local frame after project_faces_to_2d
normalisation (same as the frontend after mirrorEdgesHorizontal).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

from core.services.cutting_sheet import Edge2D
from core.services.types import Vec2

CutShapeKind = Literal["rect", "square", "circle", "line"]

SNAP = 10_000
MIN_PIECE_AREA = 0.0001
MIN_SHAPE_SIZE = 0.02


@dataclass
class UserCut:
    id: str
    group_id: int
    kind: CutShapeKind
    u0: float
    v0: float
    u1: float
    v1: float
    keep_positive: bool = True


@dataclass
class PanelPiece:
    width_m: float
    height_m: float
    edges: List[Edge2D]


def parse_user_cuts(raw: Optional[List[dict]]) -> List[UserCut]:
    if not raw:
        return []
    cuts: List[UserCut] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        group_id = item.get("group_id")
        kind = item.get("kind")
        if not isinstance(group_id, int) or kind not in ("rect", "square", "circle", "line"):
            continue
        keep = item.get("keep_positive")
        cuts.append(
            UserCut(
                id=str(item.get("id") or ""),
                group_id=group_id,
                kind=kind,  # type: ignore[arg-type]
                u0=float(item["u0"]),
                v0=float(item["v0"]),
                u1=float(item["u1"]),
                v1=float(item["v1"]),
                keep_positive=True if keep is None else bool(keep),
            )
        )
    return cuts


def build_cuts_by_group(
    cuts: List[UserCut],
    merge_target: Optional[Dict[int, int]] = None,
) -> Dict[int, List[UserCut]]:
    """Group cuts by group_id, remapping merged-away ids to the survivor."""
    merge_target = merge_target or {}
    by_group: Dict[int, List[UserCut]] = {}
    for cut in cuts:
        gid = merge_target.get(cut.group_id, cut.group_id)
        by_group.setdefault(gid, []).append(cut)
    return by_group


def build_merge_target_map(
    groups: list,
    merges: Optional[List[List[int]]],
) -> Dict[int, int]:
    """Map each merged group id → survivor id (same rule as apply_merges)."""
    if not merges:
        return {}
    group_by_id = {g.id: g for g in groups}
    target: Dict[int, int] = {}
    for merge_set in merges:
        members = [
            group_by_id[gid]
            for gid in merge_set
            if gid in group_by_id and group_by_id[gid].category != "discard"
        ]
        if len(members) < 2:
            continue
        survivor = max(members, key=lambda g: g.total_area)
        for member in members:
            target[member.id] = survivor.id
    return target


def _snap(v: float) -> float:
    return round(v * SNAP) / SNAP


def _snap_key(x: float, y: float) -> str:
    return f"{_snap(x)},{_snap(y)}"


def _ring_area(ring: List[Tuple[float, float]]) -> float:
    area = 0.0
    for i in range(len(ring) - 1):
        area += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
    return area / 2.0


def _edges_to_polygons(edges: List[Edge2D]) -> List[List[Tuple[float, float]]]:
    adj: Dict[str, List[Vec2]] = {}

    def add_edge(a: Vec2, b: Vec2) -> None:
        ka, kb = _snap_key(a.x, a.y), _snap_key(b.x, b.y)
        adj.setdefault(ka, []).append(b)
        adj.setdefault(kb, []).append(a)

    for edge in edges:
        add_edge(edge.a, edge.b)

    used: set[str] = set()
    rings: List[List[Tuple[float, float]]] = []

    for start_key, neighbors in adj.items():
        if not neighbors:
            continue
        edge_key = f"{start_key}|{_snap_key(neighbors[0].x, neighbors[0].y)}"
        if edge_key in used:
            continue

        ring: List[Tuple[float, float]] = []
        cur = start_key
        prev: Optional[str] = None
        guard = 0

        while guard < len(adj) * 4:
            guard += 1
            cx, cy = map(float, cur.split(","))
            ring.append((cx, cy))
            nbs = adj[cur]
            nxt: Optional[str] = None
            for nb in nbs:
                nk = _snap_key(nb.x, nb.y)
                if nk == prev:
                    continue
                ek = cur + "|" + nk if cur < nk else nk + "|" + cur
                if ek in used:
                    continue
                nxt = nk
                used.add(ek)
                break
            if not nxt or nxt == start_key:
                break
            prev = cur
            cur = nxt

        if len(ring) >= 4:
            rings.append(ring)

    return rings


def _polygons_to_edges(polys: List[List[Tuple[float, float]]]) -> List[Edge2D]:
    edges: List[Edge2D] = []
    for ri, ring in enumerate(polys):
        is_hole = ri >= 1
        for i in range(len(ring) - 1):
            edges.append(
                Edge2D(
                    a=Vec2(ring[i][0], ring[i][1]),
                    b=Vec2(ring[i + 1][0], ring[i + 1][1]),
                    hole=is_hole,
                )
            )
    return edges


def _normalize_edges(edges: List[Edge2D]) -> Optional[PanelPiece]:
    if len(edges) < 3:
        return None
    min_u = min(min(e.a.x, e.b.x) for e in edges)
    max_u = max(max(e.a.x, e.b.x) for e in edges)
    min_v = min(min(e.a.y, e.b.y) for e in edges)
    max_v = max(max(e.a.y, e.b.y) for e in edges)
    w, h = max_u - min_u, max_v - min_v
    if w < 0.01 or h < 0.01:
        return None
    return PanelPiece(
        width_m=w,
        height_m=h,
        edges=[
            Edge2D(
                a=Vec2(e.a.x - min_u, e.a.y - min_v),
                b=Vec2(e.b.x - min_u, e.b.y - min_v),
                hole=e.hole,
            )
            for e in edges
        ],
    )


def _cut_shape_polygon(
    cut: UserCut,
    width_m: float,
    height_m: float,
) -> Optional[List[Tuple[float, float]]]:
    kind = cut.kind
    u0, v0, u1, v1 = cut.u0, cut.v0, cut.u1, cut.v1

    if kind == "line":
        return None

    if kind == "circle":
        radius = math.hypot(u1 - u0, v1 - v0)
        if radius < MIN_SHAPE_SIZE:
            return None
        ring: List[Tuple[float, float]] = []
        steps = 32
        for i in range(steps + 1):
            t = (i / steps) * math.pi * 2
            ring.append((u0 + math.cos(t) * radius, v0 + math.sin(t) * radius))
        return ring

    min_u, max_u = min(u0, u1), max(u0, u1)
    min_v, max_v = min(v0, v1), max(v0, v1)

    if kind == "square":
        side = max(max_u - min_u, max_v - min_v)
        max_u = min_u + side
        max_v = min_v + side

    if max_u - min_u < MIN_SHAPE_SIZE or max_v - min_v < MIN_SHAPE_SIZE:
        return None

    min_u = max(0.0, min_u)
    min_v = max(0.0, min_v)
    max_u = min(width_m, max_u)
    max_v = min(height_m, max_v)

    return [
        (min_u, min_v),
        (max_u, min_v),
        (max_u, max_v),
        (min_u, max_v),
        (min_u, min_v),
    ]


def _clip_panel_by_line(
    edges: List[Edge2D],
    u0: float,
    v0: float,
    u1: float,
    v1: float,
    keep_positive: bool,
) -> Optional[List[Edge2D]]:
    dx, dy = u1 - u0, v1 - v0

    def side(p: Vec2) -> bool:
        cross = dx * (p.y - v0) - dy * (p.x - u0)
        return cross >= -1e-9 if keep_positive else cross <= 1e-9

    out: List[Edge2D] = []
    crossings: List[Vec2] = []

    for edge in edges:
        a_in, b_in = side(edge.a), side(edge.b)
        if a_in and b_in:
            out.append(edge)
        elif not a_in and not b_in:
            continue
        else:
            denom = (edge.b.x - edge.a.x) * dy - (edge.b.y - edge.a.y) * dx
            if abs(denom) < 1e-12:
                continue
            t = ((u0 - edge.a.x) * dy - (v0 - edge.a.y) * dx) / denom
            cut_pt = Vec2(
                edge.a.x + t * (edge.b.x - edge.a.x),
                edge.a.y + t * (edge.b.y - edge.a.y),
            )
            crossings.append(cut_pt)
            if a_in:
                out.append(Edge2D(a=edge.a, b=cut_pt, hole=edge.hole))
            else:
                out.append(Edge2D(a=cut_pt, b=edge.b, hole=edge.hole))

    if len(crossings) >= 2:
        crossings.sort(key=lambda p: (p.x, p.y))
        for i in range(0, len(crossings) - 1, 2):
            out.append(Edge2D(a=crossings[i], b=crossings[i + 1]))

    return out if len(out) >= 3 else None


def _rings_to_polygon(rings: List[List[Tuple[float, float]]]) -> Optional[Polygon]:
    if not rings:
        return None
    ordered = sorted(rings, key=lambda r: abs(_ring_area(r)), reverse=True)
    outer = [(x, y) for x, y in ordered[0]]
    holes = [[(x, y) for x, y in ring] for ring in ordered[1:]]
    try:
        poly = Polygon(outer, holes=holes or None)
    except Exception:
        try:
            poly = Polygon(outer)
        except Exception:
            return None
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty:
        return None
    return poly


def _polygon_to_rings(poly: Polygon) -> List[List[Tuple[float, float]]]:
    rings: List[List[Tuple[float, float]]] = []
    if poly.geom_type == "Polygon":
        polys = [poly]
    elif poly.geom_type == "MultiPolygon":
        polys = list(poly.geoms)
    else:
        return rings

    for p in polys:
        if p.is_empty:
            continue
        ext = list(p.exterior.coords)
        if len(ext) >= 4:
            rings.append([(x, y) for x, y in ext])
        for interior in p.interiors:
            hole = list(interior.coords)
            if len(hole) >= 4:
                rings.append([(x, y) for x, y in hole])
    return rings


def _apply_shape_cut(
    width_m: float,
    height_m: float,
    edges: List[Edge2D],
    cut: UserCut,
) -> List[PanelPiece]:
    cut_ring = _cut_shape_polygon(cut, width_m, height_m)
    if not cut_ring:
        return [PanelPiece(width_m, height_m, edges)]

    panel_rings = _edges_to_polygons(edges)
    if not panel_rings:
        return [PanelPiece(width_m, height_m, edges)]

    panel_poly = _rings_to_polygon(panel_rings)
    if panel_poly is None:
        return [PanelPiece(width_m, height_m, edges)]

    try:
        cut_poly = Polygon(cut_ring)
        if not cut_poly.is_valid:
            cut_poly = make_valid(cut_poly)
        result = panel_poly.difference(cut_poly)
    except Exception:
        return [PanelPiece(width_m, height_m, edges)]

    if result.is_empty:
        return []

    if isinstance(result, MultiPolygon):
        geoms = list(result.geoms)
    else:
        geoms = [result]

    pieces: List[PanelPiece] = []
    for geom in geoms:
        if geom.is_empty:
            continue
        rings = _polygon_to_rings(geom)
        if not rings:
            continue
        sorted_rings = sorted(rings, key=lambda r: abs(_ring_area(r)), reverse=True)
        piece_edges = _polygons_to_edges(sorted_rings)
        norm = _normalize_edges(piece_edges)
        if norm and norm.width_m * norm.height_m >= MIN_PIECE_AREA:
            pieces.append(norm)
    return pieces


def _apply_single_cut(
    width_m: float,
    height_m: float,
    edges: List[Edge2D],
    cut: UserCut,
) -> List[PanelPiece]:
    if cut.kind == "line":
        clipped = _clip_panel_by_line(
            edges, cut.u0, cut.v0, cut.u1, cut.v1, cut.keep_positive
        )
        if not clipped:
            return []
        norm = _normalize_edges(clipped)
        return [norm] if norm else []

    return _apply_shape_cut(width_m, height_m, edges, cut)


def apply_user_cuts_to_panel(
    width_m: float,
    height_m: float,
    edges: List[Edge2D],
    cuts: List[UserCut],
) -> List[PanelPiece]:
    """Apply all cuts for one panel; may split into several pieces."""
    pieces: List[PanelPiece] = [PanelPiece(width_m, height_m, list(edges))]

    for cut in cuts:
        nxt: List[PanelPiece] = []
        for piece in pieces:
            nxt.extend(
                _apply_single_cut(piece.width_m, piece.height_m, piece.edges, cut)
            )
        pieces = [p for p in nxt if len(p.edges) >= 3]
        if not pieces:
            break

    return pieces if pieces else []

