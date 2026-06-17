"""
Topología geométrica compartida (Refactors 1 y 3 de la auditoría de Berg).

- weld_vertices: soldado canónico de vértices (snap-rounding ÚNICO) -> asigna a cada
  cara `vertex_indices` enteros canónicos, de modo que las etapas posteriores comparen
  por IGUALDAD ENTERA (no por epsilon disperso). Suelda vértices coincidentes/jitter
  que vienen duplicados del export (causa de sobre-corte por "costuras").
- connected_components: una sola implementación de componentes conexas por vértices
  compartidos (union-find), reusada por la clasificación (antes estaba inline en
  group_classifier.split_connected).

Los booleanos 2D (uniones/recortes) siguen en shapely/GEOS: es más robusto que un
DCEL hecho a mano, así que NO se reemplaza esa parte.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from shapely import STRtree
from shapely.geometry import Polygon
from shapely.ops import unary_union

from core.services.types import Face3D, IndexedFace3D, Vec3, cross, dot, normalize, vlength


# Tolerancia de soldado: 0.1 mm. Suelda coincidentes/jitter de float sin fundir
# vértices realmente distintos (que en arquitectura están a >= cm).
WELD_TOL = 1e-4


def weld_vertices(faces: Sequence[Face3D], tol: float = WELD_TOL) -> int:
    """
    Asigna `vertex_indices` canónicos (in-place) a cada IndexedFace3D según una grilla
    de paso `tol`. Dos vértices que caen en la misma celda comparten id. Devuelve la
    cantidad de vértices canónicos. Las caras sin el campo (Face3D plano) se ignoran.
    """
    inv = 1.0 / tol
    canon: Dict[Tuple[int, int, int], int] = {}
    next_id = 0
    for f in faces:
        if not isinstance(f, IndexedFace3D):
            continue
        vi: List[int] = []
        for v in f.vertices:
            key = (round(v.x * inv), round(v.y * inv), round(v.z * inv))
            cid = canon.get(key)
            if cid is None:
                cid = next_id
                canon[key] = cid
                next_id += 1
            vi.append(cid)
        f.vertex_indices = vi
    return next_id


def _coord_key(v, snap: float = 0.01) -> Tuple[float, float, float]:
    inv = 1.0 / snap
    return (round(v.x * inv) / inv, round(v.y * inv) / inv, round(v.z * inv) / inv)


def connected_components(
    items: List,
    face_of: Callable[[object], Face3D],
    coord_snap: float = 0.01,
) -> List[List]:
    """
    Agrupa `items` en componentes conexas por vértices compartidos. Usa `vertex_indices`
    (igualdad entera exacta) si están disponibles; si no, cae a una clave de coordenada
    redondeada (`coord_snap`). `face_of(item)` devuelve la Face3D del item.

    Reemplaza la lógica que estaba inline en group_classifier.split_connected.
    """
    n = len(items)
    if n <= 1:
        return [items]

    vert_to_idx: Dict[object, List[int]] = {}
    for i, it in enumerate(items):
        face = face_of(it)
        vi = (
            face.vertex_indices
            if isinstance(face, IndexedFace3D)
            and len(face.vertex_indices) == len(face.vertices)
            else None
        )
        if vi:
            for k in vi:
                vert_to_idx.setdefault(("i", k), []).append(i)
        else:
            for v in face.vertices:
                vert_to_idx.setdefault(_coord_key(v, coord_snap), []).append(i)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for bucket in vert_to_idx.values():
        for j in range(1, len(bucket)):
            ra, rb = find(bucket[0]), find(bucket[j])
            if ra != rb:
                parent[ra] = rb

    comp_map: Dict[int, List] = {}
    for i, it in enumerate(items):
        comp_map.setdefault(find(i), []).append(it)
    return list(comp_map.values())


# ---------------------------------------------------------------------------
# Adyacencia coplanar por CONTACTO real (Misión 2 — de Berg cap. 2/10)
#
# Reemplaza la validación de "plano infinito + bbox con gap" por un test
# topológico: dos componentes coplanares se unen SÓLO si sus polígonos,
# proyectados al plano del muro, se tocan/solapan (distancia <= eps ≈ 0).
# Corrige falsos positivos (hueco real -> no tocan -> no unen) y falsos
# negativos (se tocan sin compartir vértice -> distancia 0 -> unen).
# Broad-phase con STRtree (R-tree) -> O(n log n + k), no O(n²).
# ---------------------------------------------------------------------------


def _inplane_basis(n: Vec3) -> Tuple[Vec3, Vec3]:
    """Base ortonormal (u, v) DENTRO del plano de normal n."""
    wu = Vec3(0.0, 1.0, 0.0)
    dd = dot(wu, n)
    vip = Vec3(wu.x - dd * n.x, wu.y - dd * n.y, wu.z - dd * n.z)
    if vlength(vip) < 1e-9:  # cara ~horizontal (n ~ Y): usar X como "arriba" auxiliar
        wu = Vec3(1.0, 0.0, 0.0)
        dd = dot(wu, n)
        vip = Vec3(wu.x - dd * n.x, wu.y - dd * n.y, wu.z - dd * n.z)
    v = normalize(vip)
    u = normalize(cross(n, v))
    return u, v


def merge_coplanar_by_contact(
    components: List[List],
    normal: Vec3,
    faces: List[Face3D],
    face_of: Callable[[object], Face3D],
    eps: float = 1e-3,
) -> List[List]:
    """
    Une componentes coplanares cuyos polígonos proyectados al plano se tocan (<= eps).
    `face_of(item)` da la Face3D de cada item del componente.
    """
    n_comp = len(components)
    if n_comp <= 1:
        return components

    u, v = _inplane_basis(normalize(normal))

    def project(vert) -> Tuple[float, float]:
        return (
            vert.x * u.x + vert.y * u.y + vert.z * u.z,
            vert.x * v.x + vert.y * v.y + vert.z * v.z,
        )

    polys: List[Optional[Polygon]] = []
    for comp in components:
        rings = []
        for it in comp:
            f = face_of(it)
            if len(f.vertices) < 3:
                continue
            p = Polygon([project(vert) for vert in f.vertices])
            if not p.is_valid:
                p = p.buffer(0)
            if (not p.is_empty) and p.area > 1e-9:
                rings.append(p)
        if not rings:
            polys.append(None)
            continue
        merged = unary_union(rings)
        if not merged.is_valid:
            merged = merged.buffer(0)
        polys.append(merged if not merged.is_empty else None)

    parent = list(range(n_comp))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    valid = [i for i in range(n_comp) if polys[i] is not None]
    if len(valid) > 1:
        geoms = [polys[i] for i in valid]
        tree = STRtree(geoms)
        for i in valid:
            for m in tree.query(polys[i], predicate="dwithin", distance=eps):
                j = valid[int(m)]
                if j > i:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj

    out: Dict[int, List] = {}
    for i in range(n_comp):
        out.setdefault(find(i), []).extend(components[i])
    return list(out.values())
