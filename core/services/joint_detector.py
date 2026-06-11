import math
import time
import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field

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

logger = logging.getLogger("eficiencia2d.pipeline")

# ============================================================================
# Joint Detector — Optimizado
#
# Cambio clave: edge keys como tuplas de ints en vez de f-strings.
# Para 1M de aristas esto ahorra millones de allocations de strings.
# ============================================================================


@dataclass
class Joint:
    group_a: int
    group_b: int
    total_length: float
    dihedral_angle: float
    edge_mid: Vec3
    edge_dir: Vec3
    horizontal_frac: float


# ---------------------------------------------------------------------------
# Helpers — edge keys como tuplas numéricas (mucho más rápido que strings)
# ---------------------------------------------------------------------------

SNAP_FACTOR = 100  # resolución 1cm


def _snap_int(v: float) -> int:
    return round(v * SNAP_FACTOR)


def edge_key_numeric(
    ax: float, ay: float, az: float, bx: float, by: float, bz: float
) -> Tuple[int, int, int, int, int, int]:
    """
    Clave de arista como tupla de 6 ints ordenada lexicográficamente.
    Mucho más rápido que construir f-strings con redondeo de floats.
    """
    sax, say, saz = _snap_int(ax), _snap_int(ay), _snap_int(az)
    sbx, sby, sbz = _snap_int(bx), _snap_int(by), _snap_int(bz)
    if (sax, say, saz) <= (sbx, sby, sbz):
        return (sax, say, saz, sbx, sby, sbz)
    return (sbx, sby, sbz, sax, say, saz)


def edge_length(a: Vec3, b: Vec3) -> float:
    dx, dy, dz = b.x - a.x, b.y - a.y, b.z - a.z
    return math.sqrt(dx**2 + dy**2 + dz**2)


def pair_key(a: int, b: int) -> Tuple[int, int]:
    """Clave de par como tupla (no string) — más rápido."""
    return (a, b) if a < b else (b, a)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EdgeRecord:
    """Datos de una arista única, consolidados en un solo objeto (antes 3 dicts)."""
    groups: Set[int]
    length: float
    va: Vec3
    vb: Vec3


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
    
    Optimizaciones:
    - Edge keys como tuplas de ints (sin f-strings)
    - Pair keys como tuplas de ints
    - Una sola pasada sobre edge_to_groups
    """
    t0 = time.perf_counter()

    # edge_tuple → EdgeRecord (grupos + longitud + vértices, en un solo dict)
    edges: Dict[Tuple, EdgeRecord] = {}

    total_edges = 0

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
                total_edges += 1

                if vi:
                    # Usar índices numéricos directamente (mejor camino)
                    a_idx, b_idx = vi[i], vi[j]
                    key = (min(a_idx, b_idx), max(a_idx, b_idx), -1, -1, -1, -1)
                else:
                    key = edge_key_numeric(
                        verts[i].x, verts[i].y, verts[i].z,
                        verts[j].x, verts[j].y, verts[j].z,
                    )

                rec = edges.get(key)
                if rec is not None:
                    rec.groups.add(group.id)
                else:
                    edges[key] = EdgeRecord(
                        groups={group.id},
                        length=edge_length(verts[i], verts[j]),
                        va=verts[i],
                        vb=verts[j],
                    )

    t_edges = time.perf_counter()
    logger.debug(
        f"  detect_joints: {total_edges:,} aristas procesadas, "
        f"{len(edges):,} únicas — {(t_edges-t0)*1000:.1f} ms"
    )

    # Recolectar pares
    pair_data_map: Dict[Tuple[int, int], PairData] = {}

    shared_count = 0
    for rec in edges.values():
        if len(rec.groups) < 2:
            continue
        shared_count += 1

        ids = list(rec.groups)
        length = rec.length

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pk = pair_key(ids[i], ids[j])
                pd = pair_data_map.get(pk)
                if not pd:
                    pd = PairData(groups=(ids[i], ids[j]))
                    pair_data_map[pk] = pd
                pd.total_length += length
                pd.edges.append(EdgeData(a=rec.va, b=rec.vb, len=length))

    t_pairs = time.perf_counter()
    logger.debug(
        f"  detect_joints: {shared_count:,} aristas compartidas, "
        f"{len(pair_data_map)} pares — {(t_pairs-t_edges)*1000:.1f} ms"
    )

    # Construir joints
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

    total_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"  detect_joints: {len(joints)} joints detectados en {total_ms:.1f} ms"
    )
    return joints
