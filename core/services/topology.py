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

from core.services.types import Face3D, IndexedFace3D


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
