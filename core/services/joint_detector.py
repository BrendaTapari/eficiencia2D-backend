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


def build_group_adjacency(joints: List[Joint]) -> Dict[int, List[int]]:
    """Grafo de vecindad entre grupos: `group_id -> [ids vecinos]`, ordenado.

    Vive en `Phase1Result` y es la ÚNICA respuesta a "quién toca a quién". Hasta ahora la
    adyacencia se reconstruía en cada consumidor y con criterios distintos: acá por
    aristas/contacto real, y en `assembly_logic._build_adjacency` por solape de AABB
    expandido, que es una aproximación mucho más gruesa (dos muros paralelos y separados
    solapan AABB sin tocarse). Que el instructivo y los encastres opinen distinto sobre la
    misma maqueta era una fuente de confusión.

    Se construye a partir de las juntas ya detectadas, que son la fuente autoritativa, y
    las listas salen ordenadas: el resultado no depende del orden de iteración.
    """
    adj: Dict[int, set] = {}
    for j in joints:
        if j.group_a == j.group_b:
            continue
        adj.setdefault(j.group_a, set()).add(j.group_b)
        adj.setdefault(j.group_b, set()).add(j.group_a)
    return {gid: sorted(vecinos) for gid, vecinos in sorted(adj.items())}


# ---------------------------------------------------------------------------
# Detección geométrica de contactos muro-muro (complemento del método topológico)
# ---------------------------------------------------------------------------

# Separación máxima entre superficies para considerar que dos muros se tocan. Cubre la
# holgura de MODELADO (huecos de fracciones de mm, muros que terminan a ras), no huecos
# de diseño: a 5 cm los muros están de verdad separados y recortarlos abriría una luz
# visible en la maqueta en vez de cerrarla.
CONTACT_TOL_M = 0.02
# Cada cuánto se prueba un punto a lo largo de las aristas (ver `detect_contact_joints`).
ARISTA_PASO_M = 0.1
# Altura mínima del contacto para que la unión sea relevante al cortar.
MIN_VERTICAL_OVERLAP_M = 0.30
# Ángulo mínimo entre dos muros para tratarlos como unión real. Por debajo son casi
# coplanares: se solapan, no se cruzan, y no hay nada que encastrar.
MIN_JOINT_ANGLE_DEG = 20.0


# Memoria de contactos ya calculados entre pares de muros; ver `detect_contact_joints`.
_CONTACTOS: Dict[tuple, list] = {}
_MAX_CONTACTOS = 400_000


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

    # El contacto entre dos muros depende sólo de la geometría de los dos. Cada fusión
    # (del usuario o automática) vuelve a detectar las juntas de TODO el modelo aunque
    # haya cambiado un solo grupo: en un modelo de 300 piezas eran 2.5 s por pasada y
    # hasta tres pasadas por pedido. Se recuerda el resultado por par.
    #
    # Cada grupo se identifica por las COORDENADAS de sus caras, no por su id ni por
    # índices: un id o un índice se repiten entre dos modelos distintos (o entre dos
    # versiones del mismo archivo), y confundirlos devolvería las juntas de otro edificio.
    firma = {gid: hash(verts[gid].tobytes()) for gid in verts}

    def touching_points(P: "np.ndarray", T: "np.ndarray") -> List[Tuple[float, float, float]]:
        """Puntos de P que están a <= tol de la superficie T (el contacto en sí)."""
        A, B, C = T[:, 0], T[:, 1], T[:, 2]
        AB = B - A
        AC = C - A
        n = np.cross(AB, AC)
        nn = (n * n).sum(1)
        nn[nn < 1e-18] = 1e-18
        out: List[Tuple[float, float, float]] = []
        # Sólo los puntos que caen en la caja de la otra superficie: el resto no puede
        # tocarla, y probarlos uno por uno contra todos sus triángulos era la mayor parte
        # del costo.
        lo_ = T.reshape(-1, 3).min(0) - tol
        hi_ = T.reshape(-1, 3).max(0) + tol
        P = P[((P >= lo_) & (P <= hi_)).all(1)]
        if not len(P):
            return out
        # En bloques de puntos contra todos los triángulos a la vez (antes, punto por
        # punto en Python: un grupo grande recién fusionado tardaba segundos).
        bloque = max(1, 500_000 // max(1, len(A)))
        for k0 in range(0, len(P), bloque):
            Pk = P[k0:k0 + bloque]
            AP = Pk[:, None, :] - A[None, :, :]
            u = np.clip((np.cross(AP, AC[None]) * n[None]).sum(2) / nn[None], 0.0, 1.0)
            v = np.clip((np.cross(AB[None], AP) * n[None]).sum(2) / nn[None], 0.0, 1.0)
            s_ = u + v
            f = s_ > 1.0
            u = np.where(f, u / np.where(f, s_, 1.0), u)
            v = np.where(f, v / np.where(f, s_, 1.0), v)
            Q = A[None] + u[..., None] * AB[None] + v[..., None] * AC[None]
            d2 = ((Pk[:, None, :] - Q) ** 2).sum(2).min(1)
            for p in Pk[d2 <= tol * tol]:
                out.append((float(p[0]), float(p[1]), float(p[2])))
        return out

    # Los puntos que se prueban contra la cara del otro muro: los vértices Y puntos a lo
    # largo de las aristas. Con sólo vértices se perdían los contactos de una arista que
    # apoya entera sobre una cara cuyos extremos caen fuera de ella: un muro que llega a
    # una fachada partida en franjas horizontales (zócalo, franja, dintel) toca a cada
    # franja con un tramo de su arista vertical, pero sus dos vértices quedan arriba y
    # abajo de todas. Medido en demo.obj: esa esquina no tenía ninguna unión.
    cache_muestras: Dict[int, "np.ndarray"] = {}

    def muestras(gid: int) -> "np.ndarray":
        if gid in cache_muestras:
            return cache_muestras[gid]
        T = tris[gid]
        pts = [verts[gid]]
        for i0, i1 in ((0, 1), (1, 2), (2, 0)):
            A_, B_ = T[:, i0], T[:, i1]
            largo = np.sqrt(((B_ - A_) ** 2).sum(1))
            k = np.maximum(1, np.ceil(largo / ARISTA_PASO_M)).astype(int)
            for n_ in np.unique(k):
                if n_ <= 1:
                    continue
                sel = k == n_
                t = (np.arange(1, n_) / n_)[None, :, None]
                pts.append((A_[sel][:, None, :] + t * (B_[sel] - A_[sel])[:, None, :]).reshape(-1, 3))
        P_ = np.unique(np.round(np.concatenate(pts), 4), axis=0)
        cache_muestras[gid] = P_
        return P_

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
            clave = (firma[a], firma[b])
            contact = _CONTACTOS.get(clave)
            if contact is None:
                contact = touching_points(muestras(b), tris[a])
                contact += touching_points(muestras(a), tris[b])
                if len(_CONTACTOS) >= _MAX_CONTACTOS:
                    _CONTACTOS.clear()
                _CONTACTOS[clave] = contact
            if not contact:
                continue
            ys = [p[1] for p in contact]
            y_lo, y_hi = min(ys), max(ys)
            # Una franja más baja que el mínimo (un zócalo de 38 cm) se toca en toda su
            # altura y es una unión: el mínimo es relativo a la más baja de las dos.
            alto_min = min(ay1 - ay0, by1 - by0)
            if y_hi - y_lo < min(MIN_VERTICAL_OVERLAP_M, 0.5 * alto_min):
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
