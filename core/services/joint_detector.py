import math
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field

# Importamos nuestros tipos (asegurate de tenerlos disponibles en core/types.py)
from core.services.types import (
    Face3D,
    Vec3,
    dot,
    get_vertex_indices,
    normalize,
    sub,
    vlength,
)
from core.group_classifier import GeometryGroup

# ============================================================================
# Joint Detector
#
# Identifica aristas compartidas entre grupos de geometría en el espacio 3D.
# Una arista compartida significa que dos componentes se unen físicamente en
# ese límite — una "unión" (joint) donde se deben tomar decisiones de ensamblaje.
# ============================================================================


@dataclass
class Joint:
    group_a: int
    group_b: int
    total_length: float
    dihedral_angle: float
    edge_mid: Vec3
    edge_dir: Vec3
    # Fracción de la longitud de la arista compartida que es aproximadamente horizontal (|dir.y|/|dir| <= 0.3)
    horizontal_frac: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def snap3(v: float) -> float:
    return round(v * 100) / 100.0


def edge_key_3d(
    ax: float, ay: float, az: float, bx: float, by: float, bz: float
) -> str:
    sax, say, saz = snap3(ax), snap3(ay), snap3(az)
    sbx, sby, sbz = snap3(bx), snap3(by), snap3(bz)

    # Lexicographical comparison logic
    if sax < sbx or (sax == sbx and (say < sby or (say == sby and saz < sbz))):
        return f"{sax},{say},{saz}|{sbx},{sby},{sbz}"
    return f"{sbx},{sby},{sbz}|{sax},{say},{saz}"


def edge_length(a: Vec3, b: Vec3) -> float:
    dx, dy, dz = b.x - a.x, b.y - a.y, b.z - a.z
    return math.sqrt(dx**2 + dy**2 + dz**2)


def pair_key(a: int, b: int) -> str:
    return f"{a}|{b}" if a < b else f"{b}|{a}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class EdgeData:
    a: Vec3
    b: Vec3
    len: float


@dataclass
class PairData:
    groups: Tuple[int, int]
    total_length: float = 0.0
    edges: List[EdgeData] = field(default_factory=list)


def detect_joints(faces: List[Face3D], groups: List[GeometryGroup]) -> List[Joint]:
    """
    Detecta uniones (aristas 3D compartidas) entre grupos de geometría.
    Devuelve un Joint por par de grupos que comparten al menos una arista.
    """
    # edge_key -> set of group IDs
    edge_to_groups: Dict[str, Set[int]] = {}
    edge_lengths: Dict[str, float] = {}
    edge_verts: Dict[str, Tuple[Vec3, Vec3]] = {}

    for group in groups:
        if group.category == "discard":
            continue

        for fi in group.face_indices:
            if fi < 0 or fi >= len(faces):
                continue

            face = faces[fi]
            verts = face.vertices
            vi = get_vertex_indices(face)
            n_verts = len(verts)

            for i in range(n_verts):
                j = (i + 1) % n_verts
                if vi:
                    key = f"{vi[i]}|{vi[j]}" if vi[i] < vi[j] else f"{vi[j]}|{vi[i]}"
                else:
                    key = edge_key_3d(
                        verts[i].x,
                        verts[i].y,
                        verts[i].z,
                        verts[j].x,
                        verts[j].y,
                        verts[j].z,
                    )

                if key in edge_to_groups:
                    edge_to_groups[key].add(group.id)
                else:
                    edge_to_groups[key] = {group.id}
                    edge_lengths[key] = edge_length(verts[i], verts[j])
                    edge_verts[key] = (verts[i], verts[j])

    # Recolectar longitudes de uniones + geometría de aristas por par de grupos
    pair_data_map: Dict[str, PairData] = {}

    for key, group_ids in edge_to_groups.items():
        if len(group_ids) < 2:
            continue

        ids = list(group_ids)
        length = edge_lengths.get(key, 0.0)
        verts = edge_verts.get(key)

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pk = pair_key(ids[i], ids[j])
                pd = pair_data_map.get(pk)
                if not pd:
                    pd = PairData(groups=(ids[i], ids[j]))
                    pair_data_map[pk] = pd

                pd.total_length += length
                if verts:
                    pd.edges.append(EdgeData(a=verts[0], b=verts[1], len=length))

    # Construir lookup normal representativa por grupo
    group_normals: Dict[int, Vec3] = {g.id: g.representative_normal for g in groups}

    HORIZ_DIR_TOL = 0.3
    joints: List[Joint] = []

    for pd in pair_data_map.values():
        if pd.total_length < 0.01:
            continue

        g_a, g_b = pd.groups
        n_a = group_normals.get(g_a)
        n_b = group_normals.get(g_b)

        dihedral_angle = 90.0
        if n_a and n_b:
            abs_dot = abs(dot(n_a, n_b))
            dihedral_angle = (math.acos(min(1.0, abs_dot)) * 180.0) / math.pi

        # Punto medio ponderado y dirección dominante de las aristas compartidas
        mx, my, mz = 0.0, 0.0, 0.0
        horiz_len = 0.0
        longest_edge: Optional[EdgeData] = pd.edges[0] if pd.edges else None

        for e in pd.edges:
            w = e.len
            mx += (e.a.x + e.b.x) * 0.5 * w
            my += (e.a.y + e.b.y) * 0.5 * w
            mz += (e.a.z + e.b.z) * 0.5 * w

            direction = sub(e.b, e.a)
            dir_len = vlength(direction)
            if dir_len > 1e-9:
                abs_y_frac = abs(direction.y) / dir_len
                if abs_y_frac <= HORIZ_DIR_TOL:
                    horiz_len += e.len

            if not longest_edge or e.len > longest_edge.len:
                longest_edge = e

        if pd.total_length > 0:
            edge_mid = Vec3(
                mx / pd.total_length, my / pd.total_length, mz / pd.total_length
            )
            horizontal_frac = horiz_len / pd.total_length
        else:
            edge_mid = Vec3(0.0, 0.0, 0.0)
            horizontal_frac = 0.0

        if longest_edge:
            edge_dir = normalize(sub(longest_edge.b, longest_edge.a))
        else:
            edge_dir = Vec3(1.0, 0.0, 0.0)

        joints.append(
            Joint(
                group_a=g_a,
                group_b=g_b,
                total_length=pd.total_length,
                dihedral_angle=dihedral_angle,
                edge_mid=edge_mid,
                edge_dir=edge_dir,
                horizontal_frac=horizontal_frac,
            )
        )

    joints.sort(key=lambda j: j.total_length, reverse=True)
    return joints
