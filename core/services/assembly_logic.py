"""
Secuencia de ensamble 3D — heurística bottom-up para maquetas arquitectónicas.

El pipeline interno normaliza modelos Z-up a Y-up; por defecto ``up_axis="Y"``.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from core.group_classifier import GeometryGroup
from core.pipeline import Phase1Result
from core.services.assembly_guide import _panel_centroid_3d
from core.services.cutting_sheet import Panel
from core.services.types import Face3D, Vec3, cross, dot, normalize, vlength

UpAxis = Literal["Y", "Z"]

HORIZONTAL_DOT = 0.85
VERTICAL_DOT = 0.50
DEFAULT_EPSILON = 0.01
# Grosor visual de placa para el instructivo 3D (corte láser / cartón).
DEFAULT_VIEWER_THICKNESS_M = 0.012
MAX_VIEWER_THICKNESS_M = 0.05
VIEWER_SCHEMA = "oriented_box_v1"


@dataclass(frozen=True, slots=True)
class BoundingBox3D:
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
            (self.min_z + self.max_z) * 0.5,
        )

    @property
    def size(self) -> Tuple[float, float, float]:
        return (
            max(self.max_x - self.min_x, 1e-4),
            max(self.max_y - self.min_y, 1e-4),
            max(self.max_z - self.min_z, 1e-4),
        )

    def min_up(self, up: UpAxis) -> float:
        return self.min_y if up == "Y" else self.min_z

    def max_up(self, up: UpAxis) -> float:
        return self.max_y if up == "Y" else self.max_z

    def expanded(self, epsilon: float) -> "BoundingBox3D":
        return BoundingBox3D(
            self.min_x - epsilon,
            self.min_y - epsilon,
            self.min_z - epsilon,
            self.max_x + epsilon,
            self.max_y + epsilon,
            self.max_z + epsilon,
        )


@dataclass(slots=True)
class AssemblyPiece:
    """Pieza de entrada para ``generate_assembly_sequence``."""

    id: str
    normal: Vec3
    bbox: BoundingBox3D
    vertices: List[Vec3] = field(default_factory=list)
    category: Optional[str] = None
    group_id: Optional[int] = None
    center_3d: Optional[Vec3] = None
    panel_width_m: Optional[float] = None
    panel_height_m: Optional[float] = None

    @property
    def centroid(self) -> Tuple[float, float, float]:
        if self.center_3d is not None:
            return (self.center_3d.x, self.center_3d.y, self.center_3d.z)
        if self.vertices:
            n = len(self.vertices)
            sx = sum(v.x for v in self.vertices)
            sy = sum(v.y for v in self.vertices)
            sz = sum(v.z for v in self.vertices)
            return (sx / n, sy / n, sz / n)
        return self.bbox.center


def _up_vec(up: UpAxis) -> Vec3:
    return Vec3(0.0, 1.0, 0.0) if up == "Y" else Vec3(0.0, 0.0, 1.0)


def _compute_bbox(vertices: Sequence[Vec3]) -> BoundingBox3D:
    xs = [v.x for v in vertices]
    ys = [v.y for v in vertices]
    zs = [v.z for v in vertices]
    return BoundingBox3D(min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def _aabb_intersect(a: BoundingBox3D, b: BoundingBox3D) -> bool:
    return (
        a.min_x <= b.max_x
        and a.max_x >= b.min_x
        and a.min_y <= b.max_y
        and a.max_y >= b.min_y
        and a.min_z <= b.max_z
        and a.max_z >= b.min_z
    )


def _is_horizontal(normal: Vec3, up: UpAxis) -> bool:
    return abs(dot(normalize(normal), _up_vec(up))) >= HORIZONTAL_DOT


def _is_vertical(normal: Vec3, up: UpAxis) -> bool:
    return abs(dot(normalize(normal), _up_vec(up))) <= VERTICAL_DOT


def _orientation_bucket(normal: Vec3, up: UpAxis) -> int:
    """Agrupa muros por dirección horizontal (8 sectores de 45°)."""
    n = normalize(normal)
    if up == "Y":
        hx, hz = n.x, n.z
    else:
        hx, hz = n.x, n.y
    if math.hypot(hx, hz) < 1e-6:
        return 0
    angle = math.atan2(hz, hx)
    return int(round(angle / (math.pi / 4))) % 8


def _build_adjacency(
    pieces: Sequence[AssemblyPiece], epsilon: float
) -> Dict[str, List[str]]:
    by_id = {p.id: p for p in pieces}
    adj: Dict[str, List[str]] = {p.id: [] for p in pieces}
    ids = [p.id for p in pieces]
    expanded = {p.id: p.bbox.expanded(epsilon) for p in pieces}

    for i, id_a in enumerate(ids):
        box_a = expanded[id_a]
        for id_b in ids[i + 1 :]:
            if _aabb_intersect(box_a, expanded[id_b]):
                adj[id_a].append(id_b)
                adj[id_b].append(id_a)
    return adj


def _identify_base_ids(
    pieces: Sequence[AssemblyPiece], up: UpAxis
) -> List[str]:
    horizontals = [p for p in pieces if _is_horizontal(p.normal, up)]
    if not horizontals:
        if not pieces:
            return []
        lowest = min(p.bbox.min_up(up) for p in pieces)
        tol = 0.02
        return [
            p.id
            for p in pieces
            if p.bbox.min_up(up) <= lowest + tol and _is_horizontal(p.normal, up)
        ] or [min(pieces, key=lambda p: p.bbox.min_up(up)).id]

    lowest = min(p.bbox.min_up(up) for p in horizontals)
    tol = 0.02
    return [p.id for p in horizontals if p.bbox.min_up(up) <= lowest + tol]


def _assign_bfs_levels(
    pieces: Sequence[AssemblyPiece],
    base_ids: List[str],
    adj: Dict[str, List[str]],
    up: UpAxis,
) -> Dict[str, int]:
    by_id = {p.id: p for p in pieces}
    levels: Dict[str, int] = {}

    for bid in base_ids:
        levels[bid] = 0

    queue: deque[str] = deque(base_ids)
    while queue:
        pid = queue.popleft()
        cur = levels[pid]
        piece = by_id[pid]

        for nid in adj[pid]:
            if nid in levels:
                continue

            neighbor = by_id[nid]
            if _is_horizontal(neighbor.normal, up):
                if neighbor.bbox.min_up(up) <= piece.bbox.min_up(up) + 0.02:
                    levels[nid] = 0
                else:
                    levels[nid] = 3
            elif _is_vertical(neighbor.normal, up):
                levels[nid] = 1 if cur == 0 else 2
            else:
                levels[nid] = min(cur + 1, 3)

            queue.append(nid)

    for p in pieces:
        if p.id not in levels:
            if _is_horizontal(p.normal, up):
                levels[p.id] = 3
            elif _is_vertical(p.normal, up):
                levels[p.id] = 2
            else:
                levels[p.id] = 2

    return levels


def _group_ids_for_step(
    piece_ids: List[str],
    by_id: Dict[str, AssemblyPiece],
    level: int,
    up: UpAxis,
) -> List[List[str]]:
    """Sub-agrupa piezas del mismo nivel (p. ej. muros por orientación)."""
    if level == 0 or len(piece_ids) <= 4:
        return [sorted(piece_ids)]

    if level in (1, 2):
        buckets: Dict[int, List[str]] = {}
        for pid in piece_ids:
            bucket = _orientation_bucket(by_id[pid].normal, up)
            buckets.setdefault(bucket, []).append(pid)
        groups = [sorted(v) for v in buckets.values()]
        groups.sort(key=lambda g: by_id[g[0]].bbox.min_up(up))
        return groups

    return [sorted(piece_ids)]


def _step_title(level: int, group_index: int, group_count: int) -> str:
    if level == 0:
        return "Colocar Base"
    if level == 1:
        if group_count > 1:
            return f"Levantar Muros Principales ({group_index + 1}/{group_count})"
        return "Levantar Muros Principales"
    if level == 2:
        if group_count > 1:
            return f"Completar Muros Secundarios ({group_index + 1}/{group_count})"
        return "Completar Muros Secundarios"
    if group_count > 1:
        return f"Instalar Techos y Cubiertas ({group_index + 1}/{group_count})"
    return "Instalar Techos y Cubiertas"


def _step_description(level: int, part_ids: List[str]) -> str:
    ids = ", ".join(part_ids)
    if level == 0:
        return (
            f"Ubica {'la pieza base' if len(part_ids) == 1 else 'las piezas base'} "
            f"({ids}) en la superficie de trabajo."
        )
    if level == 1:
        return (
            f"Ensambla los muros ({ids}) que conectan con la base. "
            "Alineá las ranuras de encastre antes de fijar."
        )
    if level == 2:
        return (
            f"Completa los muros ({ids}) apoyados sobre los muros principales."
        )
    return (
        f"Coloca {'la cubierta' if len(part_ids) == 1 else 'las cubiertas'} ({ids}) "
        "sobre los muros ya ensamblados."
    )


def _camera_focus(piece_ids: List[str], by_id: Dict[str, AssemblyPiece]) -> Dict[str, float]:
    if not piece_ids:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    cx = cy = cz = 0.0
    for pid in piece_ids:
        x, y, z = by_id[pid].centroid
        cx += x
        cy += y
        cz += z
    n = len(piece_ids)
    return {"x": round(cx / n, 4), "y": round(cy / n, 4), "z": round(cz / n, 4)}


def generate_assembly_sequence(
    pieces: Sequence[AssemblyPiece],
    *,
    epsilon: float = DEFAULT_EPSILON,
    up_axis: UpAxis = "Y",
) -> List[Dict]:
    """
    Calcula la secuencia de ensamble paso a paso.

    Heurística:
      1. Base = piezas horizontales más bajas.
      2. Grafo de adyacencia por AABB + tolerancia.
      3. BFS: nivel 0 base → 1 muros sobre base → 2 muros sobre muros → 3 techos.
    """
    if not pieces:
        return []

    by_id = {p.id: p for p in pieces}
    adj = _build_adjacency(pieces, epsilon)
    base_ids = _identify_base_ids(pieces, up_axis)
    levels = _assign_bfs_levels(pieces, base_ids, adj, up_axis)

    level_buckets: Dict[int, List[str]] = {}
    for pid, lvl in levels.items():
        level_buckets.setdefault(lvl, []).append(pid)

    steps: List[Dict] = []
    step_index = 0

    for level in sorted(level_buckets.keys()):
        ids_at_level = level_buckets[level]
        subgroups = _group_ids_for_step(ids_at_level, by_id, level, up_axis)

        for gi, group in enumerate(subgroups):
            steps.append(
                {
                    "step_index": step_index,
                    "title": _step_title(level, gi, len(subgroups)),
                    "description": _step_description(level, group),
                    "part_ids": group,
                    "camera_focus": _camera_focus(group, by_id),
                }
            )
            step_index += 1

    return steps


def _collect_group_vertices(
    group: GeometryGroup, faces: Sequence[Face3D]
) -> List[Vec3]:
    verts: List[Vec3] = []
    for fi in group.face_indices:
        if 0 <= fi < len(faces):
            verts.extend(faces[fi].vertices)
    return verts


def pieces_from_groups(
    groups: Sequence[GeometryGroup],
    faces: Sequence[Face3D],
    panel_id_by_group: Optional[Dict[int, str]] = None,
    *,
    exclude_discard: bool = True,
) -> List[AssemblyPiece]:
    """Construye piezas de ensamble a partir de grupos del pipeline Phase1."""
    panel_id_by_group = panel_id_by_group or {}
    out: List[AssemblyPiece] = []

    for group in groups:
        if exclude_discard and group.category == "discard":
            continue

        vertices = _collect_group_vertices(group, faces)
        if not vertices:
            continue

        piece_id = panel_id_by_group.get(group.id) or group.label or f"G-{group.id}"
        out.append(
            AssemblyPiece(
                id=piece_id,
                normal=group.representative_normal,
                bbox=_compute_bbox(vertices),
                vertices=vertices,
                category=group.category,
                group_id=group.id,
            )
        )

    return out


def pieces_from_phase1(
    phase1: Phase1Result,
    panel_id_by_group: Optional[Dict[int, str]] = None,
) -> List[AssemblyPiece]:
    return pieces_from_groups(
        phase1.groups,
        phase1.faces,
        panel_id_by_group,
    )


def _panels_per_group(panels: Sequence[Panel]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for panel in panels:
        gid = panel.source_group_id
        counts[gid] = counts.get(gid, 0) + 1
    return counts


def _thickness_along_normal(vertices: Sequence[Vec3], normal: Vec3) -> float:
    n = normalize(normal)
    projs = [dot(v, n) for v in vertices]
    return max(projs) - min(projs) if projs else 0.05


def _approx_panel_bbox(
    panel: Panel,
    center: Vec3,
    thickness: float,
    normal: Vec3,
) -> BoundingBox3D:
    """AABB aproximado alrededor del centro del panel (cortes manuales)."""
    n = normalize(normal)
    up = Vec3(0.0, 1.0, 0.0)
    if abs(dot(n, up)) > 0.95:
        u = Vec3(1.0, 0.0, 0.0)
    else:
        u = normalize(cross(up, n))
    v = normalize(cross(n, u))
    hw = panel.width_m * 0.5
    hh = panel.height_m * 0.5
    ht = max(thickness, 0.01) * 0.5
    corners: List[Vec3] = []
    for su in (-1.0, 1.0):
        for sv in (-1.0, 1.0):
            for sn in (-1.0, 1.0):
                corners.append(
                    Vec3(
                        center.x + u.x * hw * su + v.x * hh * sv + n.x * ht * sn,
                        center.y + u.y * hw * su + v.y * hh * sv + n.y * ht * sn,
                        center.z + u.z * hw * su + v.z * hh * sv + n.z * ht * sn,
                    )
                )
    return _compute_bbox(corners)


def pieces_from_panels(
    wall_panels: Sequence[Panel],
    floor_panels: Sequence[Panel],
    groups: Sequence[GeometryGroup],
    faces: Sequence[Face3D],
) -> List[AssemblyPiece]:
    """Una pieza por panel descompuesto (A1, B2, …) con posición 3D real."""
    group_by_id = {g.id: g for g in groups}
    all_panels = list(wall_panels) + list(floor_panels)
    per_group = _panels_per_group(all_panels)
    face_list = list(faces)
    out: List[AssemblyPiece] = []

    for panel in all_panels:
        group = group_by_id.get(panel.source_group_id)
        if group is None:
            continue

        vertices = _collect_group_vertices(group, face_list)
        if not vertices:
            continue

        center = _panel_centroid_3d(panel, group, face_list)
        thickness = _thickness_along_normal(vertices, group.representative_normal)
        multi = per_group.get(panel.source_group_id, 1) > 1

        if multi:
            bbox = _approx_panel_bbox(
                panel, center, thickness, group.representative_normal
            )
        else:
            bbox = _compute_bbox(vertices)

        out.append(
            AssemblyPiece(
                id=panel.id,
                normal=group.representative_normal,
                bbox=bbox,
                vertices=vertices,
                category=panel.category,
                group_id=group.id,
                center_3d=center,
                panel_width_m=panel.width_m,
                panel_height_m=panel.height_m,
            )
        )

    return out


def _piece_color(category: Optional[str]) -> str:
    if category == "floor":
        return "#64748b"
    if category == "wall":
        return "#475569"
    return "#334155"


def _euler_from_normal(normal: Vec3) -> List[float]:
    """Euler XYZ en radianes (Three.js) para orientar la pieza según su normal."""
    n = normalize(normal)
    if vlength(n) < 1e-9:
        return [0.0, 0.0, 0.0]
    yaw = math.atan2(n.x, n.z)
    pitch = -math.asin(max(-1.0, min(1.0, n.y)))
    return [round(pitch, 6), round(yaw, 6), 0.0]


def _panel_thickness_m(
    piece: AssemblyPiece, group: Optional[GeometryGroup]
) -> float:
    """
    Grosor de placa para el visor.

    Usa el grosor detectado del muro (gemelas) si es razonable; nunca el AABB
    completo del grupo (incluye ambas pieles y genera bloques gruesos).
    """
    if group is not None and group.thickness is not None:
        t = float(group.thickness)
        if 0.005 <= t <= MAX_VIEWER_THICKNESS_M:
            return t
    return DEFAULT_VIEWER_THICKNESS_M


def _oriented_size(
    piece: AssemblyPiece, group: Optional[GeometryGroup] = None
) -> Tuple[float, float, float]:
    """[ancho_plano, alto_plano, grosor] en metros — placa fina orientada por ``normal``."""
    if piece.panel_width_m and piece.panel_height_m:
        thickness = _panel_thickness_m(piece, group)
        return (piece.panel_width_m, piece.panel_height_m, thickness)
    sx, sy, sz = piece.bbox.size
    n = normalize(piece.normal)
    if abs(n.y) >= HORIZONTAL_DOT:
        return (sx, max(sz, DEFAULT_VIEWER_THICKNESS_M), sy)
    if abs(n.x) >= HORIZONTAL_DOT:
        return (sy, sz, max(sx, DEFAULT_VIEWER_THICKNESS_M))
    return (sx, sy, max(sz, DEFAULT_VIEWER_THICKNESS_M))


def build_viewer_pieces(
    pieces: Sequence[AssemblyPiece],
    groups: Sequence[GeometryGroup],
    faces: Sequence[Face3D],
) -> List[Dict]:
    """
    Placa fina orientada por panel (2D de corte + grosor).

    No exporta malla del grupo 3D: incluye ambas pieles del muro y se ve mal
    en el instructivo interactivo.
    """
    del faces  # reservado; el visor usa cajas orientadas, no malla cruda.
    group_by_id = {g.id: g for g in groups}
    viewer: List[Dict] = []

    for p in pieces:
        cx, cy, cz = p.centroid
        group = group_by_id.get(p.group_id) if p.group_id is not None else None
        sx, sy, sz = _oriented_size(p, group)
        viewer.append(
            {
                "id": p.id,
                "kind": "oriented_box",
                "position": [round(cx, 4), round(cy, 4), round(cz, 4)],
                "size": [round(sx, 4), round(sy, 4), round(sz, 4)],
                "normal": [
                    round(p.normal.x, 5),
                    round(p.normal.y, 5),
                    round(p.normal.z, 5),
                ],
                "rotation": _euler_from_normal(p.normal),
                "color": _piece_color(p.category),
                "category": p.category or "unknown",
            }
        )

    return viewer


def build_assembly_guide_payload(
    phase1: Phase1Result,
    panel_id_by_group: Optional[Dict[int, str]] = None,
    *,
    wall_panels: Optional[Sequence[Panel]] = None,
    floor_panels: Optional[Sequence[Panel]] = None,
    epsilon: float = DEFAULT_EPSILON,
) -> Dict:
    """
    Payload completo para el visor React: pasos + piezas 3D.

    Incluye claves en inglés (API) y español (compatibilidad con el frontend).
    """
    # Caras en Phase1Result están en espacio Y-up canónico (Z-up se rota al parsear).
    if wall_panels is not None and floor_panels is not None:
        pieces = pieces_from_panels(
            wall_panels, floor_panels, phase1.groups, phase1.faces
        )
    else:
        pieces = pieces_from_phase1(phase1, panel_id_by_group)

    steps = generate_assembly_sequence(pieces, epsilon=epsilon, up_axis="Y")
    viewer_pieces = build_viewer_pieces(pieces, phase1.groups, phase1.faces)

    pasos = [
        {
            "titulo": s["title"],
            "descripcion": s["description"],
            "piezaIds": s["part_ids"],
            "camera_focus": s.get("camera_focus"),
        }
        for s in steps
    ]

    return {
        "steps": steps,
        "pieces": viewer_pieces,
        "pasos": pasos,
        "piezas": viewer_pieces,
        "meta": {
            "piece_count": len(pieces),
            "step_count": len(steps),
            "applied_axis": phase1.applied_axis,
            "epsilon": epsilon,
            "viewer_schema": VIEWER_SCHEMA,
        },
    }
