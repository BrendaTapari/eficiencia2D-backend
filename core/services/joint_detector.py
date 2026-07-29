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

    # Complemento GEOMÉTRICO: la detección de arriba es topológica (exige que dos
    # grupos compartan una arista). En una unión en T —el borde de un muro apoyado
    # sobre la CARA de otro— no hay arista compartida, así que esas uniones se perdían
    # por completo: no se sugería recorte ni aparecían en la lista para elegir quién
    # corta a quién, y al cortar las piezas chocaban. Ver detect_contact_joints.
    joints.extend(detect_contact_joints(faces, groups, joints))

    joints.sort(key=lambda j: j.total_length, reverse=True)

    total_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"  detect_joints: {len(joints)} joints detectados en {total_ms:.1f} ms"
    )
    return joints


# ---------------------------------------------------------------------------
# Detección geométrica de contactos muro-muro (complemento del método topológico)
# ---------------------------------------------------------------------------

# Separación máxima entre superficies para considerar que dos muros se tocan. Cubre la
# holgura de MODELADO (huecos de fracciones de mm, muros que terminan a ras), no huecos
# de diseño: a 5 cm los muros están de verdad separados y recortarlos abriría una luz
# visible en la maqueta en vez de cerrarla.
CONTACT_TOL_M = 0.02
# Altura mínima del contacto para que la unión sea relevante al cortar.
MIN_VERTICAL_OVERLAP_M = 0.30
# Ángulo mínimo entre dos muros para tratarlos como unión real. Por debajo son casi
# coplanares: se solapan, no se cruzan, y no hay nada que encastrar.
MIN_JOINT_ANGLE_DEG = 20.0


def detect_contact_joints(
    faces: List[Face3D],
    groups: List[GeometryGroup],
    existing: List[Joint],
) -> List[Joint]:
    """Uniones muro-muro que se TOCAN pero no comparten aristas en la malla.

    El detector topológico sólo ve uniones donde las mallas están soldadas. Un caso
    muy común no lo está: el muro B termina contra la CARA de A (unión en T), o el
    exportador dejó vértices duplicados en la esquina. Geométricamente hay contacto y
    al cortar las piezas se superponen, pero el sistema no lo veía.

    Se mide sobre la GEOMETRÍA REAL (distancia punto-contra-cara), no sobre un plano
    medio: los muros no planos o de doble piel hacían fallar cualquier aproximación
    por plano. Dos muros forman unión si sus superficies se acercan a menos de
    CONTACT_TOL_M, no son casi coplanares y el contacto abarca altura suficiente.
    Devuelve sólo los pares que no fueron detectados ya.
    """
    import numpy as np

    known = {pair_key(j.group_a, j.group_b) for j in existing}
    walls = [
        g for g in groups
        if g.category == "wall" and math.hypot(
            g.representative_normal.x, g.representative_normal.z
        ) >= 0.5
    ]
    if len(walls) < 2:
        return []

    tris: Dict[int, "np.ndarray"] = {}
    verts: Dict[int, "np.ndarray"] = {}
    bbox: Dict[int, Tuple[float, float, float, float, float, float]] = {}
    for g in walls:
        tt: List = []
        vv: List = []
        for fi in g.face_indices:
            if fi < 0 or fi >= len(faces):
                continue
            v = faces[fi].vertices
            if len(v) < 3:
                continue
            vv.extend((p.x, p.y, p.z) for p in v)
            for k in range(1, len(v) - 1):
                tt.append(
                    ((v[0].x, v[0].y, v[0].z),
                     (v[k].x, v[k].y, v[k].z),
                     (v[k + 1].x, v[k + 1].y, v[k + 1].z))
                )
        if not tt or not vv:
            continue
        A = np.asarray(vv, dtype=float)
        tris[g.id] = np.asarray(tt, dtype=float)
        verts[g.id] = A
        bbox[g.id] = (
            float(A[:, 0].min()), float(A[:, 0].max()),
            float(A[:, 1].min()), float(A[:, 1].max()),
            float(A[:, 2].min()), float(A[:, 2].max()),
        )

    tol = CONTACT_TOL_M

    def touching_points(P: "np.ndarray", T: "np.ndarray") -> List[Tuple[float, float, float]]:
        """Puntos de P que están a <= tol de la superficie T (el contacto en sí)."""
        A, B, C = T[:, 0], T[:, 1], T[:, 2]
        AB = B - A
        AC = C - A
        n = np.cross(AB, AC)
        nn = (n * n).sum(1)
        nn[nn < 1e-18] = 1e-18
        out: List[Tuple[float, float, float]] = []
        for p in P:
            AP = p - A
            # Coordenadas baricéntricas de la proyección, recortadas al triángulo.
            u = np.clip((np.cross(AP, AC) * n).sum(1) / nn, 0.0, 1.0)
            v = np.clip((np.cross(AB, AP) * n).sum(1) / nn, 0.0, 1.0)
            s = u + v
            f = s > 1.0
            if f.any():
                u[f] /= s[f]
                v[f] /= s[f]
            Q = A + u[:, None] * AB + v[:, None] * AC
            if float(((p - Q) ** 2).sum(1).min()) <= tol * tol:
                out.append((float(p[0]), float(p[1]), float(p[2])))
        return out

    ids = sorted(tris)
    out: List[Joint] = []
    group_by_id: Dict[int, GeometryGroup] = {g.id: g for g in walls}

    for i in range(len(ids)):
        a = ids[i]
        ga = group_by_id[a]
        ax0, ax1, ay0, ay1, az0, az1 = bbox[a]
        for j in range(i + 1, len(ids)):
            b = ids[j]
            if pair_key(a, b) in known:
                continue
            gb = group_by_id[b]

            # Casi coplanares: se solapan, no se cruzan. Nada que encastrar.
            cos_ang = abs(dot(ga.representative_normal, gb.representative_normal))
            dihedral = math.degrees(math.acos(min(1.0, cos_ang)))
            if dihedral < MIN_JOINT_ANGLE_DEG:
                continue

            bx0, bx1, by0, by1, bz0, bz1 = bbox[b]
            # Descarte barato: cajas envolventes que ni se rozan.
            if (ax0 - tol > bx1 or bx0 - tol > ax1
                    or ay0 - tol > by1 or by0 - tol > ay1
                    or az0 - tol > bz1 or bz0 - tol > az1):
                continue

            # Contacto real: puntos de un muro que tocan la superficie del otro.
            contact = touching_points(verts[b], tris[a])
            contact += touching_points(verts[a], tris[b])
            if not contact:
                continue
            ys = [p[1] for p in contact]
            y_lo, y_hi = min(ys), max(ys)
            if y_hi - y_lo < MIN_VERTICAL_OVERLAP_M:
                continue

            # edge_mid DEBE ser el punto de contacto: decompose lo proyecta sobre el eje
            # del panel para decidir QUÉ EXTREMO del muro recortar. Con el centroide del
            # muro el recorte caía en el extremo equivocado y la unión seguía solapada.
            cx = sum(p[0] for p in contact) / len(contact)
            cz = sum(p[2] for p in contact) / len(contact)

            out.append(
                Joint(
                    group_a=a,
                    group_b=b,
                    total_length=y_hi - y_lo,
                    dihedral_angle=dihedral,
                    edge_mid=Vec3(cx, (y_lo + y_hi) * 0.5, cz),
                    # La arista de contacto de dos muros verticales es vertical.
                    edge_dir=Vec3(0.0, 1.0, 0.0),
                    horizontal_frac=0.0,
                )
            )

    if out:
        logger.info(
            f"  detect_joints: +{len(out)} uniones por contacto geométrico "
            f"(mallas no soldadas)"
        )
    return out
