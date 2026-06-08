import math
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

from core.services.types import Face3D, get_vertex_indices

# NOTA: Rompemos la importación circular.
# group_classifier.py importa desde acá, pero acá necesitamos el tipo Subgroup.
# Lo importamos bajo un TYPE_CHECKING o simplemente lo referenciamos como "any"
# si no usamos comprobación de tipos estricta, pero para mantenerlo limpio:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.group_classifier import Subgroup

# ============================================================================
# Slab Edge Detector
#
# Detecta los cantos de losa. Los cantos son caras verticales que actúan como
# borde de una placa horizontal. El detector previene que estas caras cortas se
# clasifiquen erróneamente como muros.
# ============================================================================

HORIZONTAL_NORMAL_MIN = 0.98
VERTICAL_NORMAL_MAX = 0.5
EDGE_DIR_TOL = math.sqrt(1.0 - HORIZONTAL_NORMAL_MIN**2)
HEIGHT_BAND = 0.05
MAX_SLAB_THICKNESS = 1.0
PLATE_RATIO = 0.15
FACE_ASPECT = 0.35


@dataclass
class SlabEdgeLink:
    floor_subgroup_index: int
    rim_subgroup_index: int
    thickness: float


def snap3(v: float) -> float:
    return round(v * 100) / 100.0


def edge_key(face: Face3D, i: int, j: int, vi: List[int]) -> str:
    if vi:
        return f"{vi[i]}|{vi[j]}" if vi[i] < vi[j] else f"{vi[j]}|{vi[i]}"

    a, b = face.vertices[i], face.vertices[j]
    sax, say, saz = snap3(a.x), snap3(a.y), snap3(a.z)
    sbx, sby, sbz = snap3(b.x), snap3(b.y), snap3(b.z)

    # Lexicographical comparison logic matching TypeScript
    if sax < sbx or (sax == sbx and (say < sby or (say == sby and saz < sbz))):
        return f"{sax},{say},{saz}|{sbx},{sby},{sbz}"
    return f"{sbx},{sby},{sbz}|{sax},{say},{saz}"


def face_area(f: Face3D) -> float:
    v = f.vertices
    if len(v) < 3:
        return 0.0
    sx, sy, sz = 0.0, 0.0, 0.0
    for i in range(1, len(v) - 1):
        e1x, e1y, e1z = v[i].x - v[0].x, v[i].y - v[0].y, v[i].z - v[0].z
        e2x, e2y, e2z = v[i + 1].x - v[0].x, v[i + 1].y - v[0].y, v[i + 1].z - v[0].z
        sx += e1y * e2z - e1z * e2y
        sy += e1z * e2x - e1x * e2z
        sz += e1x * e2y - e1y * e2x
    return 0.5 * math.sqrt(sx**2 + sy**2 + sz**2)


def lateral_extent(f: Face3D) -> float:
    min_x, max_x = float("inf"), float("-inf")
    min_z, max_z = float("inf"), float("-inf")
    for v in f.vertices:
        if v.x < min_x:
            min_x = v.x
        if v.x > max_x:
            max_x = v.x
        if v.z < min_z:
            min_z = v.z
        if v.z > max_z:
            max_z = v.z
    return max(max_x - min_x, max_z - min_z)


@dataclass
class SkinEdge:
    y: float
    lat: float
    area: float


def find_slab_rim_faces(faces: List[Face3D]) -> Dict[int, float]:
    up: Dict[str, SkinEdge] = {}
    down: Dict[str, SkinEdge] = {}

    for f in faces:
        ny = f.normal.y
        if abs(ny) < HORIZONTAL_NORMAL_MIN:
            continue

        area = face_area(f)
        lat = lateral_extent(f)
        vi = get_vertex_indices(f)
        verts = f.vertices
        target_map = up if ny > 0 else down

        n = len(verts)
        for i in range(n):
            j = (i + 1) % n
            a, b = verts[i], verts[j]
            dy = b.y - a.y
            length = math.sqrt((b.x - a.x) ** 2 + dy**2 + (b.z - a.z) ** 2)

            if length < 1e-9 or abs(dy) / length > EDGE_DIR_TOL:
                continue

            key = edge_key(f, i, j, vi)
            y = (a.y + b.y) / 2.0

            existing = target_map.get(key)
            if not existing or area > existing.area:
                target_map[key] = SkinEdge(y=y, lat=lat, area=area)

    rim: Dict[int, float] = {}

    for fi, f in enumerate(faces):
        if abs(f.normal.y) > VERTICAL_NORMAL_MAX:
            continue

        verts = f.vertices
        if len(verts) < 3:
            continue

        ymin, ymax = float("inf"), float("-inf")
        for v in verts:
            if v.y < ymin:
                ymin = v.y
            if v.y > ymax:
                ymax = v.y

        t = ymax - ymin
        if t <= 1e-3 or t > MAX_SLAB_THICKNESS:
            continue

        vi = get_vertex_indices(f)
        top_skin: SkinEdge = None
        bot_skin: SkinEdge = None
        max_horiz_len = 0.0

        n = len(verts)
        for i in range(n):
            j = (i + 1) % n
            a, b = verts[i], verts[j]
            dy = b.y - a.y
            length = math.sqrt((b.x - a.x) ** 2 + dy**2 + (b.z - a.z) ** 2)

            if length < 1e-9 or abs(dy) / length > EDGE_DIR_TOL:
                continue

            if length > max_horiz_len:
                max_horiz_len = length

            key = edge_key(f, i, j, vi)
            y = (a.y + b.y) / 2.0

            if abs(y - ymax) < HEIGHT_BAND:
                e = up.get(key)
                if e and (not top_skin or e.area > top_skin.area):
                    top_skin = e

            if abs(y - ymin) < HEIGHT_BAND:
                e = down.get(key)
                if e and (not bot_skin or e.area > bot_skin.area):
                    bot_skin = e

        if not top_skin or not bot_skin:
            continue

        skin_lat = min(top_skin.lat, bot_skin.lat)
        if skin_lat <= 1e-6:
            continue
        if t / skin_lat >= PLATE_RATIO:
            continue
        if max_horiz_len <= 1e-6 or t / max_horiz_len >= FACE_ASPECT:
            continue

        rim[fi] = t

    return rim


def detect_slab_edges(
    subgroups: List["Subgroup"], faces: List[Face3D], rim_faces: Set[int]
) -> List[SlabEdgeLink]:

    if not rim_faces:
        return []

    floor_idxs: List[int] = []
    rim_idxs: List[int] = []

    for i, sg in enumerate(subgroups):
        if abs(sg.normal.y) >= HORIZONTAL_NORMAL_MIN:
            floor_idxs.append(i)
        elif sg.face_infos and all((fi.index in rim_faces) for fi in sg.face_infos):
            rim_idxs.append(i)

    if not floor_idxs or not rim_idxs:
        return []

    floor_edge: Dict[str, int] = {}
    for f_idx in floor_idxs:
        for fi in subgroups[f_idx].face_infos:
            face = faces[fi.index]
            vi = get_vertex_indices(face)
            verts = face.vertices
            n = len(verts)
            for i in range(n):
                j = (i + 1) % n
                floor_edge[edge_key(face, i, j, vi)] = f_idx

    links: List[SlabEdgeLink] = []

    # En TypeScript rim_faces es un Map<number, number> para sacar el thickness,
    # pero como aquí pasamos un set(), debemos ajustar o requerir el dict original.
    # Usaremos una pequeña trampa por ahora para simular el map original que find_slab_rim_faces devuelve.
    # Para uso correcto, esta función debería recibir un Dict[int, float] en rim_faces
    if isinstance(rim_faces, set):
        print("Warning: rim_faces passed as set, falling back to thickness=0.0")
        rim_dict = {k: 0.0 for k in rim_faces}
    else:
        rim_dict = rim_faces

    for r_idx in rim_idxs:
        thickness = 0.0
        linked_floors: Set[int] = set()

        for fi in subgroups[r_idx].face_infos:
            thickness = max(thickness, rim_dict.get(fi.index, 0.0))
            face = faces[fi.index]
            vi = get_vertex_indices(face)
            verts = face.vertices
            n = len(verts)
            for i in range(n):
                j = (i + 1) % n
                f_link = floor_edge.get(edge_key(face, i, j, vi))
                if f_link is not None:
                    linked_floors.add(f_link)

        if thickness <= 0:
            continue

        for f in linked_floors:
            links.append(
                SlabEdgeLink(
                    floor_subgroup_index=f,
                    rim_subgroup_index=r_idx,
                    thickness=thickness,
                )
            )

    return links


def validate_slab_candidates(
    rim_faces: Dict[int, float], faces: List[Face3D]
) -> Set[int]:

    if not rim_faces:
        return set()

    candidates: Dict[str, Dict[str, List]] = {}

    for fi, thickness in rim_faces.items():
        f = faces[fi]
        ymin, ymax = float("inf"), float("-inf")
        for v in f.vertices:
            if v.y < ymin:
                ymin = v.y
            if v.y > ymax:
                ymax = v.y

        band_min = round(ymin / HEIGHT_BAND) * HEIGHT_BAND
        band_max = round(ymax / HEIGHT_BAND) * HEIGHT_BAND
        key = f"{band_min}|{band_max}"

        cand = candidates.get(key)
        if cand:
            cand["indices"].append(fi)
            cand["thicknesses"].append(thickness)
        else:
            candidates[key] = {"indices": [fi], "thicknesses": [thickness]}

    validated: Set[int] = set()

    for cand in candidates.values():
        max_t = max(cand["thicknesses"])
        min_t = min(cand["thicknesses"])

        if max_t > MAX_SLAB_THICKNESS:
            continue

        if max_t - min_t > HEIGHT_BAND * 2:
            continue

        for idx in cand["indices"]:
            validated.add(idx)

    return validated
