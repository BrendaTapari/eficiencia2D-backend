import math
from typing import List, Dict, Set, Optional, Tuple, Literal
from dataclasses import dataclass, field
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# Importamos nuestros tipos y servicios
from core.services.types import (
    Face3D,
    Vec2,
    Vec3,
    cross,
    dot,
    get_vertex_indices,
    normalize,
    sub,
    vlength,
)
from core.services.floor_plan_extractor import detect_floor_levels
from core.services.wall_pairing import are_thin_twins, TwinCandidate
from core.services.sheet_nester import NestingResult, rotate_edges

# ============================================================================
# Cutting Sheet — Plancha de Corte
#
# Descompone el modelo 3D en componentes estructurales individuales usando
# agrupamiento por coplanaridad geométrica.
# ============================================================================

GAP_M = 0.003  # 3mm de brecha para corte láser
SHEET_SPACING_M = 0.10  # Brecha visual entre planchas en el DXF
NORMAL_CLUSTER_DOT = 0.85  # Caras con dot > esto son de la "misma dirección"
NEAR_PARALLEL_EPS = 0.01  # Tolerancia cercana a cero para producto cruzado
THIN_TWIN_THRESHOLD = 0.40  # Fusionar grupos coplanares gemelos más cercanos que esto

UpAxis = Literal["Y", "Z"]
PanelCategory = Literal["wall", "floor"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_up(v: Vec3, up: UpAxis) -> float:
    return v.y if up == "Y" else v.z


def get_up_vec(up: UpAxis) -> Vec3:
    return Vec3(0, 1, 0) if up == "Y" else Vec3(0, 0, 1)


def face_area(face: Face3D) -> float:
    verts = face.vertices
    if len(verts) < 3:
        return 0.0
    sx, sy, sz = 0.0, 0.0, 0.0
    for i in range(1, len(verts) - 1):
        e1 = sub(verts[i], verts[0])
        e2 = sub(verts[i + 1], verts[0])
        c = cross(e1, e2)
        sx += c.x
        sy += c.y
        sz += c.z
    return 0.5 * math.sqrt(sx**2 + sy**2 + sz**2)


def snap(v: float) -> float:
    return round(v * 100) / 100.0


def vert_key(x: float, y: float) -> str:
    return f"{snap(x)},{snap(y)}"


def edge_key(ax: float, ay: float, bx: float, by: float) -> str:
    a = vert_key(ax, ay)
    b = vert_key(bx, by)
    return f"{a}|{b}" if a < b else f"{b}|{a}"


def snap3(v: float) -> float:
    return round(v * 100) / 100.0


def r_str(n: float) -> str:
    # Evitar notación científica y limitar a 4 decimales
    return f"{n:.4f}".rstrip("0").rstrip(".") if "." in f"{n:.4f}" else f"{n:.4f}"


MIN_HOLE_AREA = 0.0025


def ring_2d_area(ring: List[Tuple[float, float]]) -> float:
    a = 0.0
    for i in range(len(ring) - 1):
        a += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
    return a / 2.0


def sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


# ---------------------------------------------------------------------------
# Tipos de Paneles
# ---------------------------------------------------------------------------


@dataclass
class Edge2D:
    a: Vec2
    b: Vec2


@dataclass
class Panel:
    id: str
    group_name: str
    category: PanelCategory
    floor_index: int
    width_m: float
    height_m: float
    edges: List[Edge2D]
    source_group_id: int


@dataclass
class PlacedPanel:
    panel: Panel
    x: float
    y: float


@dataclass
class CoplanarGroup:
    normal: Vec3
    d: float
    faces: List[Face3D]
    total_area: float
    category: PanelCategory


@dataclass
class RawEdge:
    ax: float
    ay: float
    bx: float
    by: float
    via: Optional[int] = None
    vib: Optional[int] = None


# ---------------------------------------------------------------------------
# Agrupamiento por Coplanaridad
# ---------------------------------------------------------------------------


def cluster_by_coplanarity(faces: List[Face3D], up: UpAxis) -> List[CoplanarGroup]:
    groups: List[CoplanarGroup] = []
    D_TOLERANCE = 0.15

    for face in faces:
        n = face.normal
        if vlength(n) < 0.01:
            continue
        area = face_area(face)
        if area < 1e-6:
            continue

        d = dot(n, face.vertices[0])
        up_comp = abs(get_up(n, up))
        category: PanelCategory = "floor" if up_comp > 0.75 else "wall"

        placed = False
        for group in groups:
            if (
                group.category == category
                and abs(dot(n, group.normal)) > NORMAL_CLUSTER_DOT
                and abs(d - group.d) < D_TOLERANCE
            ):
                group.faces.append(face)
                group.total_area += area
                placed = True
                break

        if not placed:
            groups.append(
                CoplanarGroup(
                    normal=normalize(n),
                    d=d,
                    faces=[face],
                    total_area=area,
                    category=category,
                )
            )

    return groups


def split_connected_components(faces: List[Face3D]) -> List[List[Face3D]]:
    if len(faces) <= 1:
        return [faces]

    vert_to_faces: Dict[str, List[int]] = {}
    for fi, face in enumerate(faces):
        indices = get_vertex_indices(face)
        if indices:
            for vi in indices:
                key = str(vi)
                if key not in vert_to_faces:
                    vert_to_faces[key] = []
                vert_to_faces[key].append(fi)
        else:
            for v in face.vertices:
                key = f"{snap3(v.x)},{snap3(v.y)},{snap3(v.z)}"
                if key not in vert_to_faces:
                    vert_to_faces[key] = []
                vert_to_faces[key].append(fi)

    parent = list(range(len(faces)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for face_indices in vert_to_faces.values():
        for i in range(1, len(face_indices)):
            union(face_indices[0], face_indices[i])

    comp_map: Dict[int, List[Face3D]] = {}
    for fi, face in enumerate(faces):
        root = find(fi)
        if root not in comp_map:
            comp_map[root] = []
        comp_map[root].append(face)

    return list(comp_map.values())


# ---------------------------------------------------------------------------
# Rastreo de Contornos (Half-Edge Dart Traversal)
# ---------------------------------------------------------------------------


def trace_contours(boundary_edges: List[RawEdge]) -> List[RawEdge]:
    if len(boundary_edges) <= 2:
        return boundary_edges

    def vert_id(e: RawEdge, side: str) -> str:
        if side == "a":
            return f"i{e.via}" if e.via is not None else vert_key(e.ax, e.ay)
        return f"i{e.vib}" if e.vib is not None else vert_key(e.bx, e.by)

    adj: Dict[str, List[int]] = {}
    for i, e in enumerate(boundary_edges):
        for side in ("a", "b"):
            vk = vert_id(e, side)
            if vk not in adj:
                adj[vk] = []
            adj[vk].append(i)

    removed: Set[int] = set()
    changed = True
    while changed:
        changed = False
        for vk, indices in list(adj.items()):
            live = [i for i in indices if i not in removed]
            if len(live) == 1:
                removed.add(live[0])
                changed = True
            adj[vk] = [] if len(live) <= 1 else live

    vert_coord: Dict[str, Vec2] = {}
    for e in boundary_edges:
        ak, bk = vert_id(e, "a"), vert_id(e, "b")
        if ak not in vert_coord:
            vert_coord[ak] = Vec2(e.ax, e.ay)
        if bk not in vert_coord:
            vert_coord[bk] = Vec2(e.bx, e.by)

    live_edges = [i for i in range(len(boundary_edges)) if i not in removed]
    if not live_edges:
        return []

    def dart_from(dart: int) -> str:
        ei = dart >> 1
        return vert_id(boundary_edges[ei], "a" if (dart & 1) == 0 else "b")

    def dart_to(dart: int) -> str:
        ei = dart >> 1
        return vert_id(boundary_edges[ei], "b" if (dart & 1) == 0 else "a")

    def dart_angle(dart: int) -> float:
        frm, to = vert_coord[dart_from(dart)], vert_coord[dart_to(dart)]
        return math.atan2(to.y - frm.y, to.x - frm.x)

    outgoing: Dict[str, List[int]] = {}
    for ei in live_edges:
        for dir_bit in (0, 1):
            dart = ei * 2 + dir_bit
            frm = dart_from(dart)
            if frm not in outgoing:
                outgoing[frm] = []
            outgoing[frm].append(dart)

    for arr in outgoing.values():
        arr.sort(key=dart_angle)

    def next_dart(dart: int) -> int:
        w = dart_to(dart)
        arr = outgoing[w]
        twin = (dart >> 1) * 2 + 1 if (dart & 1) == 0 else (dart >> 1) * 2
        idx = arr.index(twin)
        prev = (idx - 1 + len(arr)) % len(arr)
        return arr[prev]

    dart_face: Dict[int, int] = {}
    face_sign: List[int] = []
    face_id = 0

    for ei in live_edges:
        for dir_bit in (0, 1):
            start = ei * 2 + dir_bit
            if start in dart_face:
                continue

            loop_darts: List[int] = []
            d = start
            guard = 0
            limit = len(live_edges) * 2 + 4
            while guard < limit:
                if d in dart_face:
                    break
                dart_face[d] = face_id
                loop_darts.append(d)
                d = next_dart(d)
                if d == start:
                    break
                guard += 1

            area2 = 0.0
            for dd in loop_darts:
                frm, to = vert_coord[dart_from(dd)], vert_coord[dart_to(dd)]
                area2 += frm.x * to.y - to.x * frm.y
            face_sign.append(sign(area2))
            face_id += 1

    kept: Set[int] = set()
    for ei in live_edges:
        f0 = dart_face.get(ei * 2)
        f1 = dart_face.get(ei * 2 + 1)
        if f0 is None or f1 is None:
            continue
        s0, s1 = face_sign[f0], face_sign[f1]
        if s0 != 0 and s1 != 0 and s0 != s1:
            kept.add(ei)

    return [e for i, e in enumerate(boundary_edges) if i in kept]


# ---------------------------------------------------------------------------
# Boolean Union & Proyecciones (Reemplazo de polyclip-ts por Shapely)
# ---------------------------------------------------------------------------

UNION_SNAP = 1e4


def snap_union(v: float) -> float:
    return round(v * UNION_SNAP) / UNION_SNAP


def union_outline(
    faces: List[Face3D], u_axis: Vec3, v_axis: Vec3
) -> Optional[List[RawEdge]]:
    polys: List[Polygon] = []
    for face in faces:
        ring = [
            (snap_union(dot(v, u_axis)), snap_union(dot(v, v_axis)))
            for v in face.vertices
        ]
        if len(ring) < 3:
            continue
        if abs(ring_2d_area(ring)) < 1e-7:
            continue

        # Shapely requiere anillos cerrados, pero asume cierre si se lo damos en orden
        polys.append(Polygon(ring))

    if not polys:
        return None

    try:
        merged = unary_union(polys)
    except Exception:
        return None

    out: List[RawEdge] = []

    def process_poly(poly: Polygon):
        # Boundary Exterior
        ext = list(poly.exterior.coords)
        for i in range(len(ext) - 1):
            out.append(
                RawEdge(ax=ext[i][0], ay=ext[i][1], bx=ext[i + 1][0], by=ext[i + 1][1])
            )

        # Huecos Internos (Windows/Doors)
        for interior in poly.interiors:
            int_coords = list(interior.coords)
            if abs(ring_2d_area(int_coords)) < MIN_HOLE_AREA:
                continue
            for i in range(len(int_coords) - 1):
                out.append(
                    RawEdge(
                        ax=int_coords[i][0],
                        ay=int_coords[i][1],
                        bx=int_coords[i + 1][0],
                        by=int_coords[i + 1][1],
                    )
                )

    if isinstance(merged, MultiPolygon):
        for p in merged.geoms:
            process_poly(p)
    elif isinstance(merged, Polygon):
        process_poly(merged)
    else:
        return None

    return out if out else None


def legacy_boundary(faces: List[Face3D], u_axis: Vec3, v_axis: Vec3) -> List[RawEdge]:
    edge_face_count: Dict[str, int] = {}
    edge_coords: Dict[str, RawEdge] = {}

    for face in faces:
        pts = [Vec2(dot(v, u_axis), dot(v, v_axis)) for v in face.vertices]
        vi = get_vertex_indices(face)
        n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            key = (
                f"{vi[i]}|{vi[j]}"
                if (vi and vi[i] < vi[j])
                else (
                    f"{vi[j]}|{vi[i]}"
                    if vi
                    else edge_key(pts[i].x, pts[i].y, pts[j].x, pts[j].y)
                )
            )

            edge_face_count[key] = edge_face_count.get(key, 0) + 1
            if key not in edge_coords:
                edge_coords[key] = RawEdge(
                    ax=pts[i].x,
                    ay=pts[i].y,
                    bx=pts[j].x,
                    by=pts[j].y,
                    via=vi[i] if vi else None,
                    vib=vi[j] if vi else None,
                )

    boundary_edges = [
        edge_coords[key] for key, count in edge_face_count.items() if count == 1
    ]
    if not boundary_edges:
        return []
    return trace_contours(boundary_edges)


@dataclass
class ProjectedFace:
    width_m: float
    height_m: float
    edges: List[Edge2D]
    v_up: float
    u_axis: Vec3
    v_axis: Vec3
    origin_u: float
    origin_v: float


def project_faces_to_2d(
    faces: List[Face3D], group_normal: Vec3, up: UpAxis
) -> Optional[ProjectedFace]:
    if not faces:
        return None

    world_up = get_up_vec(up)
    u_axis = normalize(cross(world_up, group_normal))

    if vlength(u_axis) < NEAR_PARALLEL_EPS:
        u_axis = Vec3(1, 0, 0)
        v_axis = normalize(cross(group_normal, u_axis))
        if vlength(v_axis) < NEAR_PARALLEL_EPS:
            v_axis = Vec3(0, 0, 1) if up == "Y" else Vec3(0, 1, 0)
    else:
        v_axis = normalize(cross(group_normal, u_axis))

    contoured = union_outline(faces, u_axis, v_axis) or legacy_boundary(
        faces, u_axis, v_axis
    )
    if not contoured:
        return None

    min_u, max_u = float("inf"), float("-inf")
    min_v, max_v = float("inf"), float("-inf")

    for e in contoured:
        min_u = min(min_u, e.ax, e.bx)
        max_u = max(max_u, e.ax, e.bx)
        min_v = min(min_v, e.ay, e.by)
        max_v = max(max_v, e.ay, e.by)

    w, h = max_u - min_u, max_v - min_v
    if w < 0.01 or h < 0.01:
        return None

    edges = [
        Edge2D(a=Vec2(e.ax - min_u, e.ay - min_v), b=Vec2(e.bx - min_u, e.by - min_v))
        for e in contoured
    ]

    return ProjectedFace(
        width_m=w,
        height_m=h,
        edges=edges,
        v_up=dot(v_axis, world_up),
        u_axis=u_axis,
        v_axis=v_axis,
        origin_u=min_u,
        origin_v=min_v,
    )


# ---------------------------------------------------------------------------
# Recortes y Simetría (Assembly Compensation)
# ---------------------------------------------------------------------------


def clip_panel_at_v(
    edges: List[Edge2D], cut: float, keep_above: bool
) -> Optional[Dict]:
    def in_side(y: float) -> bool:
        return y >= cut - 1e-9 if keep_above else y <= cut + 1e-9

    out: List[Edge2D] = []
    crossings: List[float] = []

    for e in edges:
        a_in, b_in = in_side(e.a.y), in_side(e.b.y)
        if a_in and b_in:
            out.append(e)
        elif not a_in and not b_in:
            continue
        else:
            t = (cut - e.a.y) / (e.b.y - e.a.y)
            ix = e.a.x + t * (e.b.x - e.a.x)
            cut_pt = Vec2(ix, cut)
            keep = e.a if a_in else e.b
            out.append(Edge2D(a=keep, b=cut_pt) if a_in else Edge2D(a=cut_pt, b=keep))
            crossings.append(ix)

    crossings.sort()
    for i in range(0, len(crossings) - 1, 2):
        out.append(Edge2D(a=Vec2(crossings[i], cut), b=Vec2(crossings[i + 1], cut)))

    if len(out) < 3:
        return None

    min_u, min_v = float("inf"), float("inf")
    max_u, max_v = float("-inf"), float("-inf")
    for e in out:
        min_u = min(min_u, e.a.x, e.b.x)
        max_u = max(max_u, e.a.x, e.b.x)
        min_v = min(min_v, e.a.y, e.b.y)
        max_v = max(max_v, e.a.y, e.b.y)

    w, h = max_u - min_u, max_v - min_v
    if w < 0.01 or h < 0.01:
        return None

    normalized = [
        Edge2D(
            a=Vec2(e.a.x - min_u, e.a.y - min_v), b=Vec2(e.b.x - min_u, e.b.y - min_v)
        )
        for e in out
    ]
    return {"width_m": w, "height_m": h, "edges": normalized}


def clip_panel_at_u(
    edges: List[Edge2D], cut: float, keep_right: bool
) -> Optional[Dict]:
    def in_side(x: float) -> bool:
        return x >= cut - 1e-9 if keep_right else x <= cut + 1e-9

    out: List[Edge2D] = []
    crossings: List[float] = []

    for e in edges:
        a_in, b_in = in_side(e.a.x), in_side(e.b.x)
        if a_in and b_in:
            out.append(e)
        elif not a_in and not b_in:
            continue
        else:
            t = (cut - e.a.x) / (e.b.x - e.a.x)
            iy = e.a.y + t * (e.b.y - e.a.y)
            cut_pt = Vec2(cut, iy)
            keep = e.a if a_in else e.b
            out.append(Edge2D(a=keep, b=cut_pt) if a_in else Edge2D(a=cut_pt, b=keep))
            crossings.append(iy)

    crossings.sort()
    for i in range(0, len(crossings) - 1, 2):
        out.append(Edge2D(a=Vec2(cut, crossings[i]), b=Vec2(cut, crossings[i + 1])))

    if len(out) < 3:
        return None

    min_u, min_v = float("inf"), float("inf")
    max_u, max_v = float("-inf"), float("-inf")
    for e in out:
        min_u = min(min_u, e.a.x, e.b.x)
        max_u = max(max_u, e.a.x, e.b.x)
        min_v = min(min_v, e.a.y, e.b.y)
        max_v = max(max_v, e.a.y, e.b.y)

    w, h = max_u - min_u, max_v - min_v
    if w < 0.01 or h < 0.01:
        return None

    normalized = [
        Edge2D(
            a=Vec2(e.a.x - min_u, e.a.y - min_v), b=Vec2(e.b.x - min_u, e.b.y - min_v)
        )
        for e in out
    ]
    return {"width_m": w, "height_m": h, "edges": normalized}


def mirror_edges_horizontal(edges: List[Edge2D], width_m: float) -> List[Edge2D]:
    return [
        Edge2D(a=Vec2(width_m - e.a.x, e.a.y), b=Vec2(width_m - e.b.x, e.b.y))
        for e in edges
    ]


# ---------------------------------------------------------------------------
# Filtrado Simple y Twin Merging
# ---------------------------------------------------------------------------


def compute_group_geom(group: CoplanarGroup) -> Dict:
    sx, sy, sz, count = 0.0, 0.0, 0.0, 0
    min_x, min_y, min_z = float("inf"), float("inf"), float("inf")
    max_x, max_y, max_z = float("-inf"), float("-inf"), float("-inf")

    for face in group.faces:
        for v in face.vertices:
            sx += v.x
            sy += v.y
            sz += v.z
            count += 1
            if v.x < min_x:
                min_x = v.x
            if v.y < min_y:
                min_y = v.y
            if v.z < min_z:
                min_z = v.z
            if v.x > max_x:
                max_x = v.x
            if v.y > max_y:
                max_y = v.y
            if v.z > max_z:
                max_z = v.z

    return {
        "centroid": Vec3(sx / count, sy / count, sz / count),
        "extent": max(max_x - min_x, max_y - min_y, max_z - min_z),
    }


def merge_thin_twin_groups(groups: List[CoplanarGroup]) -> List[CoplanarGroup]:
    geom = [compute_group_geom(g) for g in groups]
    drop: Set[int] = set()

    for i in range(len(groups)):
        if i in drop:
            continue
        for j in range(i + 1, len(groups)):
            if j in drop:
                continue
            if groups[i].category != groups[j].category:
                continue

            a = TwinCandidate(
                normal=groups[i].normal,
                d=groups[i].d,
                centroid=geom[i]["centroid"],
                extent=geom[i]["extent"],
            )
            b = TwinCandidate(
                normal=groups[j].normal,
                d=groups[j].d,
                centroid=geom[j]["centroid"],
                extent=geom[j]["extent"],
            )

            if not are_thin_twins(a, b, THIN_TWIN_THRESHOLD):
                continue

            if groups[i].total_area >= groups[j].total_area:
                drop.add(j)
            else:
                drop.add(i)
                break

    return [g for i, g in enumerate(groups) if i not in drop]


def filter_groups_for_simple_mode(groups: List[CoplanarGroup]) -> List[CoplanarGroup]:
    walls = [g for g in groups if g.category == "wall"]
    floors = [g for g in groups if g.category == "floor"]

    used: Set[int] = set()
    kept: List[CoplanarGroup] = []

    for i in range(len(walls)):
        if i in used:
            continue
        used.add(i)
        best = walls[i]

        for j in range(i + 1, len(walls)):
            if j in used:
                continue
            if dot(walls[i].normal, walls[j].normal) > -0.85:
                continue
            if abs(walls[i].d + walls[j].d) > 0.5:
                continue

            used.add(j)
            if walls[j].total_area > best.total_area:
                best = walls[j]

        kept.append(best)

    max_area = max([g.total_area for g in kept] + [0])
    filtered = [g for g in kept if g.total_area >= max_area * 0.10]

    return filtered + floors


# ---------------------------------------------------------------------------
# API Pública - Decomposición y DXF
# ---------------------------------------------------------------------------


def decompose_into_panels(
    faces: List[Face3D], up: UpAxis, simple_mode: bool, min_area_m2: float = 0.01
) -> List[Panel]:
    coplanar_groups = cluster_by_coplanarity(faces, up)
    coplanar_groups = merge_thin_twin_groups(coplanar_groups)
    if simple_mode:
        coplanar_groups = filter_groups_for_simple_mode(coplanar_groups)

    levels = detect_floor_levels(faces, up)
    raw_panels = []

    for group in coplanar_groups:
        components = split_connected_components(group.faces)
        for comp_faces in components:
            res = project_faces_to_2d(comp_faces, group.normal, up)
            if not res or (res.width_m * res.height_m < min_area_m2):
                continue

            all_elevs = [get_up(v, up) for f in comp_faces for v in f.vertices]
            mid = (min(all_elevs) + max(all_elevs)) / 2.0

            floor_idx = 0
            for i in range(len(levels) - 1, -1, -1):
                if mid >= levels[i] - 0.5:
                    floor_idx = i
                    break

            raw_panels.append(
                {
                    "category": group.category,
                    "floor_index": floor_idx,
                    "width_m": res.width_m,
                    "height_m": res.height_m,
                    "edges": res.edges,
                }
            )

    walls = sorted(
        [p for p in raw_panels if p["category"] == "wall"],
        key=lambda x: (x["floor_index"], -(x["width_m"] * x["height_m"])),
    )
    floors = sorted(
        [p for p in raw_panels if p["category"] == "floor"],
        key=lambda x: (x["floor_index"], -(x["width_m"] * x["height_m"])),
    )

    panels: List[Panel] = []

    for i, rp in enumerate(walls, 1):
        panels.append(
            Panel(
                id=f"A{i}",
                group_name=f"wall_{i}",
                category="wall",
                floor_index=rp["floor_index"],
                width_m=rp["width_m"],
                height_m=rp["height_m"],
                edges=rp["edges"],
                source_group_id=-1,
            )
        )

    for i, rp in enumerate(floors, 1):
        panels.append(
            Panel(
                id=f"B{i}",
                group_name=f"floor_{i}",
                category="floor",
                floor_index=rp["floor_index"],
                width_m=rp["width_m"],
                height_m=rp["height_m"],
                edges=rp["edges"],
                source_group_id=-1,
            )
        )

    return panels


def layout_panels(panels: List[Panel]) -> List[PlacedPanel]:
    if not panels:
        return []
    sorted_panels = sorted(panels, key=lambda p: p.height_m, reverse=True)
    max_row_w = max([p.width_m for p in sorted_panels] + [2.0]) * 4

    placed: List[PlacedPanel] = []
    row_x, row_y, row_max_h = 0.0, 0.0, 0.0

    for panel in sorted_panels:
        pw, ph = panel.width_m, panel.height_m
        if row_x > 0 and row_x + pw > max_row_w:
            row_y += row_max_h + GAP_M
            row_x, row_max_h = 0.0, 0.0

        placed.append(PlacedPanel(panel=panel, x=row_x, y=row_y))
        row_x += pw + GAP_M
        row_max_h = max(row_max_h, ph)

    return placed


CS_LAYERS = [
    {"name": "CUT_EXTERIOR", "aci": "7"},
    {"name": "ENGRAVE_VECTOR", "aci": "5"},
    {"name": "ENGRAVE_RASTER", "aci": "8"},
    {"name": "CUT_INTERIOR", "aci": "3"},
]


def emit_dxf_header(lines: List[str], layer_count: int):
    lines.extend(
        [
            "0",
            "SECTION",
            "2",
            "HEADER",
            "9",
            "$ACADVER",
            "1",
            "AC1009",
            "9",
            "$INSUNITS",
            "70",
            "6",
            "0",
            "ENDSEC",
            "0",
            "SECTION",
            "2",
            "TABLES",
            "0",
            "TABLE",
            "2",
            "LTYPE",
            "70",
            "1",
            "0",
            "LTYPE",
            "2",
            "CONTINUOUS",
            "70",
            "0",
            "3",
            "Solid line",
            "72",
            "65",
            "73",
            "0",
            "40",
            "0.0",
            "0",
            "ENDTAB",
            "0",
            "TABLE",
            "2",
            "LAYER",
            "70",
            str(layer_count),
        ]
    )
    for l in CS_LAYERS:
        lines.extend(
            ["0", "LAYER", "2", l["name"], "70", "0", "62", l["aci"], "6", "CONTINUOUS"]
        )
    lines.extend(["0", "ENDTAB", "0", "ENDSEC"])


def fit_text_height(text: str, max_w: float, max_h: float, target_h: float) -> float:
    if not text:
        return 0.0
    by_width = (max_w * 0.88) / (len(text) * 0.62)
    return min(target_h, by_width, max_h)


def emit_panel_entities(
    lines: List[str],
    edges: List[Edge2D],
    pw: float,
    ph: float,
    panel_id: str,
    ox: float,
    oy: float,
    scale_denom: float = 1.0,
    include_text: bool = True,
):
    for edge in edges:
        lines.extend(
            [
                "0",
                "LINE",
                "8",
                "CUT_EXTERIOR",
                "62",
                "7",
                "10",
                r_str(ox + edge.a.x),
                "20",
                r_str(oy + edge.a.y),
                "11",
                r_str(ox + edge.b.x),
                "21",
                r_str(oy + edge.b.y),
            ]
        )

    if not include_text:
        return

    real_w, real_h = pw * scale_denom, ph * scale_denom
    dim_text = f"{real_w:.2f} x {real_h:.2f} m"
    label_h = fit_text_height(panel_id, pw, ph * 0.45, 0.008)
    dim_h = fit_text_height(dim_text, pw, ph * 0.30, 0.005)

    if label_h >= 0.002:
        lx, ly = r_str(ox + pw / 2), r_str(oy + ph - label_h * 1.5)
        lines.extend(
            [
                "0",
                "TEXT",
                "8",
                "ENGRAVE_VECTOR",
                "62",
                "5",
                "10",
                lx,
                "20",
                ly,
                "40",
                r_str(label_h),
                "1",
                panel_id,
                "72",
                "1",
                "11",
                lx,
                "21",
                ly,
            ]
        )

    if dim_h >= 0.002 and label_h + dim_h * 3 < ph:
        dx, dy = r_str(ox + pw / 2), r_str(oy + dim_h * 0.6)
        lines.extend(
            [
                "0",
                "TEXT",
                "8",
                "ENGRAVE_RASTER",
                "62",
                "8",
                "10",
                dx,
                "20",
                dy,
                "40",
                r_str(dim_h),
                "1",
                dim_text,
                "72",
                "1",
                "11",
                dx,
                "21",
                dy,
            ]
        )


def panels_to_dxf(placed: List[PlacedPanel]) -> str:
    lines: List[str] = []
    emit_dxf_header(lines, len(CS_LAYERS))
    lines.extend(["0", "SECTION", "2", "ENTITIES"])
    for p in placed:
        emit_panel_entities(
            lines,
            p.panel.edges,
            p.panel.width_m,
            p.panel.height_m,
            p.panel.id,
            p.x,
            p.y,
        )
    lines.extend(["0", "ENDSEC", "0", "EOF"])
    return "\n".join(lines) + "\n"


def generate_cutting_sheets(
    faces: List[Face3D], up_axis: UpAxis, scale_denom: float, mode: str = "simple"
) -> List[Dict[str, str]]:
    panels = decompose_into_panels(faces, up_axis, mode == "simple")
    if not panels:
        return []

    results = []
    wall_panels = [p for p in panels if p.category == "wall"]
    floor_panels = [p for p in panels if p.category == "floor"]

    if wall_panels:
        results.append(
            {
                "name": "Descomposicion_Paredes.dxf",
                "content": panels_to_dxf(layout_panels(wall_panels)),
            }
        )
    if floor_panels:
        results.append(
            {
                "name": "Descomposicion_Pisos.dxf",
                "content": panels_to_dxf(layout_panels(floor_panels)),
            }
        )

    return results


def nested_sheets_to_dxf(nesting: NestingResult, include_text: bool = True) -> str:
    sheets, config = nesting.sheets, nesting.config
    if not sheets:
        return ""

    lines: List[str] = []
    emit_dxf_header(lines, len(CS_LAYERS))
    lines.extend(["0", "SECTION", "2", "ENTITIES"])

    cols = min(len(sheets), 3)

    for si, sheet in enumerate(sheets):
        col, row = si % cols, si // cols
        sx = col * (config.width_m + SHEET_SPACING_M)
        sy = -(row * (config.height_m + SHEET_SPACING_M))

        x0, y0, x1, y1 = sx, sy, sx + config.width_m, sy + config.height_m
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for ci in range(4):
            ax, ay = corners[ci]
            bx, by = corners[(ci + 1) % 4]
            lines.extend(
                [
                    "0",
                    "LINE",
                    "8",
                    "ENGRAVE_RASTER",
                    "62",
                    "7",
                    "10",
                    r_str(ax),
                    "20",
                    r_str(ay),
                    "11",
                    r_str(bx),
                    "21",
                    r_str(by),
                ]
            )

        if include_text:
            cx, cy = r_str(sx + config.width_m / 2), r_str(sy + config.height_m + 0.02)
            lines.extend(
                [
                    "0",
                    "TEXT",
                    "8",
                    "ENGRAVE_RASTER",
                    "62",
                    "8",
                    "10",
                    cx,
                    "20",
                    cy,
                    "40",
                    r_str(0.03),
                    "1",
                    f"Plancha {si + 1}",
                    "72",
                    "1",
                    "11",
                    cx,
                    "21",
                    cy,
                ]
            )

        for placed in sheet.panels:
            edges = (
                rotate_edges(placed.panel.edges, placed.panel.width_m)
                if placed.rotated
                else placed.panel.edges
            )
            emit_panel_entities(
                lines,
                edges,
                placed.effective_w,
                placed.effective_h,
                placed.panel.id,
                sx + placed.x,
                sy + placed.y,
                nesting.scale_denom,
                include_text,
            )

    lines.extend(["0", "ENDSEC", "0", "EOF"])
    return "\n".join(lines) + "\n"
