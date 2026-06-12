import math
import re
import time
import logging
from typing import List, Dict, Set, Optional, Literal, Tuple
from dataclasses import dataclass

from core.services.types import (
    Face3D,
    Vec3,
    cross,
    dot,
    get_vertex_indices,
    normalize,
    sub,
    vlength,
)
from core.services.wall_pairing import are_thin_twins

logger = logging.getLogger("eficiencia2d.pipeline")

# ============================================================================
# Group Classifier — Optimizado
# ============================================================================

FaceCategory = Literal["floor", "wall", "discard"]

HORIZONTAL_THRESHOLD = 0.98
VERTICAL_THRESHOLD = 0.5
MIN_AREA = 1e-6
HEIGHT_BAND = 0.05
THIN_WALL_THRESHOLD = 0.40
# Las dos pieles de un muro gemelo deben tener áreas comparables: la menor >= 20%
# de la mayor. Evita encadenar una fachada con las barras finas de una baranda.
TWIN_AREA_RATIO = 0.20
DEFAULT_MIN_REAL_AREA = 1.0
WALL_PEEL_MIN_HEIGHT = 1.0


@dataclass
class GeometryGroup:
    id: int
    label: str
    category: FaceCategory
    face_indices: List[int]
    total_area: float
    centroid: Vec3
    orientation: str
    representative_normal: Vec3
    thickness: Optional[float] = None
    min_y: Optional[float] = None
    max_y: Optional[float] = None
    original_category: Optional[FaceCategory] = None


# ---------------------------------------------------------------------------
# Clasificación por cara — optimizada: un solo loop
# ---------------------------------------------------------------------------


def face_area(f: Face3D) -> float:
    verts = f.vertices
    n = len(verts)
    if n < 3:
        return 0.0
    # Fast path: triangulo (caso mas comun en OBJ)
    if n == 3:
        v0, v1, v2 = verts[0], verts[1], verts[2]
        e1x, e1y, e1z = v1.x - v0.x, v1.y - v0.y, v1.z - v0.z
        e2x, e2y, e2z = v2.x - v0.x, v2.y - v0.y, v2.z - v0.z
        cx = e1y * e2z - e1z * e2y
        cy = e1z * e2x - e1x * e2z
        cz = e1x * e2y - e1y * e2x
        return 0.5 * math.sqrt(cx*cx + cy*cy + cz*cz)
    # Fast path: quad
    if n == 4:
        v0, v1, v2, v3 = verts[0], verts[1], verts[2], verts[3]
        # Triangulo 1: v0, v1, v2
        e1x, e1y, e1z = v1.x - v0.x, v1.y - v0.y, v1.z - v0.z
        e2x, e2y, e2z = v2.x - v0.x, v2.y - v0.y, v2.z - v0.z
        sx = e1y * e2z - e1z * e2y
        sy = e1z * e2x - e1x * e2z
        sz = e1x * e2y - e1y * e2x
        # Triangulo 2: v0, v2, v3
        e1x, e1y, e1z = v2.x - v0.x, v2.y - v0.y, v2.z - v0.z
        e2x, e2y, e2z = v3.x - v0.x, v3.y - v0.y, v3.z - v0.z
        sx += e1y * e2z - e1z * e2y
        sy += e1z * e2x - e1x * e2z
        sz += e1x * e2y - e1y * e2x
        return 0.5 * math.sqrt(sx*sx + sy*sy + sz*sz)
    # General: fan triangulation
    sx, sy, sz = 0.0, 0.0, 0.0
    v0 = verts[0]
    for i in range(1, n - 1):
        vi, vi1 = verts[i], verts[i + 1]
        e1x, e1y, e1z = vi.x - v0.x, vi.y - v0.y, vi.z - v0.z
        e2x, e2y, e2z = vi1.x - v0.x, vi1.y - v0.y, vi1.z - v0.z
        sx += e1y * e2z - e1z * e2y
        sy += e1z * e2x - e1x * e2z
        sz += e1x * e2y - e1y * e2x
    return 0.5 * math.sqrt(sx*sx + sy*sy + sz*sz)


def face_centroid(f: Face3D) -> Vec3:
    verts = f.vertices
    if not verts:
        return Vec3(0.0, 0.0, 0.0)
    n = len(verts)
    sx, sy, sz = 0.0, 0.0, 0.0
    for v in verts:
        sx += v.x
        sy += v.y
        sz += v.z
    return Vec3(sx / n, sy / n, sz / n)


@dataclass
class FaceInfo:
    index: int
    area: float
    centroid: Vec3
    orientation: Literal["horizontal", "vertical", "inclined"]
    category: FaceCategory


def classify_all_faces(faces: List[Face3D]) -> List[FaceInfo]:
    """
    Optimizaciones aplicadas:
    1. Un solo loop para clasificar + calcular bounds (era 2 loops)
    2. face_area() inline para triangulos/quads (evita Vec3 allocations)
    3. Agrupamiento de horizontales con dict de bins O(1)
    """
    t0 = time.perf_counter()
    infos: List[FaceInfo] = []

    min_x, max_x = float("inf"), float("-inf")
    min_z, max_z = float("inf"), float("-inf")

    for i, face in enumerate(faces):
        area = face_area(face)
        if area < MIN_AREA:
            continue

        abs_y = abs(face.normal.y)
        if abs_y >= HORIZONTAL_THRESHOLD:
            orientation = "horizontal"
        elif abs_y <= VERTICAL_THRESHOLD:
            orientation = "vertical"
        else:
            orientation = "inclined"

        centroid = face_centroid(face)
        infos.append(
            FaceInfo(
                index=i,
                area=area,
                centroid=centroid,
                orientation=orientation,
                category="discard",
            )
        )

        # Calcular bounds en el mismo loop (evita segundo loop)
        for v in face.vertices:
            if v.x < min_x:
                min_x = v.x
            if v.x > max_x:
                max_x = v.x
            if v.z < min_z:
                min_z = v.z
            if v.z > max_z:
                max_z = v.z

    range_x = max_x - min_x if min_x != float("inf") else 0
    range_z = max_z - min_z if min_z != float("inf") else 0

    if range_x < 1.0 or range_z < 1.0:
        for fi in infos:
            if fi.orientation == "horizontal":
                fi.category = "floor"
            elif fi.orientation == "vertical":
                fi.category = "wall"
        ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"  classify_all_faces (simple): {len(infos)} infos en {ms:.1f} ms")
        return infos

    # Agrupar horizontales por nivel con bin-rounding O(1)
    level_groups: Dict[float, List[FaceInfo]] = {}
    BAND_ROUND = HEIGHT_BAND  # 0.05m

    for fi in infos:
        if fi.orientation != "horizontal":
            continue
        h = fi.centroid.y
        bin_key = round(h / BAND_ROUND) * BAND_ROUND

        placed = False
        for candidate in (bin_key, bin_key - BAND_ROUND, bin_key + BAND_ROUND):
            candidate = round(candidate, 6)
            if candidate in level_groups:
                ref = next(iter(level_groups[candidate])).centroid.y
                if abs(ref - h) < BAND_ROUND:
                    level_groups[candidate].append(fi)
                    placed = True
                    break

        if not placed:
            level_groups[bin_key] = [fi]

    band_area = {
        key: sum(fi.area for fi in group) for key, group in level_groups.items()
    }
    max_band_area = max(band_area.values()) if band_area else 0.0
    LEVEL_THRESHOLD = max(1.0, 0.03 * max_band_area)

    real_level_keys = {
        key for key, area in band_area.items() if area >= LEVEL_THRESHOLD
    }

    for key, group in level_groups.items():
        is_real = (key in real_level_keys) or (len(real_level_keys) == 0)
        for fi in group:
            fi.category = "floor" if is_real else "discard"

    for fi in infos:
        if fi.orientation == "vertical":
            fi.category = "wall"
        elif fi.orientation == "inclined":
            fi.category = "floor"

    ms = (time.perf_counter() - t0) * 1000
    logger.debug(
        f"  classify_all_faces: {len(faces)} caras totales, {len(infos)} validas, "
        f"{len(level_groups)} niveles, {ms:.1f} ms"
    )
    return infos



# ---------------------------------------------------------------------------
# Clustering Coplanar — Optimizado O(N) con bin-indexing
# ---------------------------------------------------------------------------

NORMAL_CLUSTER_DOT = 0.985
D_TOLERANCE = 0.15


def snap3(v: float) -> float:
    return round(v * 100) / 100.0


@dataclass
class CoplanarCluster:
    normal: Vec3
    d: float
    face_infos: List[FaceInfo]


def _normal_bin_key(n: Vec3) -> Tuple[int, int, int]:
    """
    Convierte una normal a una clave de bin discreta.
    Resolución: ~5.7° (NORMAL_CLUSTER_DOT=0.985 → arccos(0.985)≈10°, bin de 0.1)
    """
    BIN = 10  # escala: 1 unidad = 0.1 en normal
    return (round(n.x * BIN), round(n.y * BIN), round(n.z * BIN))


def _d_bin_key(d: float) -> int:
    """Convierte un valor d de plano a clave de bin (resolución: D_TOLERANCE/2)."""
    return round(d / (D_TOLERANCE / 2))


def cluster_coplanar(
    infos: List[FaceInfo], faces: List[Face3D]
) -> List[CoplanarCluster]:
    """
    Versión optimizada con bin-indexing.
    
    Original: O(N * C) donde C = número de clusters creciente → O(N²) en peor caso.
    Optimizado: O(N) amortizado usando dict de bins para lookup de candidatos.
    
    La clave del bin es (normal_bin, d_bin). Para cada cara, buscamos en los bins
    adyacentes (±1 en cada dimensión de d) para cubrir casos en borde de tolerancia.
    """
    clusters: List[CoplanarCluster] = []
    # bin_key → índice en clusters[]
    bin_to_cluster: Dict[Tuple, int] = {}

    for fi in infos:
        face = faces[fi.index]
        n = face.normal
        if vlength(n) < 0.01:
            continue
        n_norm = normalize(n)
        d = dot(n_norm, face.vertices[0])

        nb = _normal_bin_key(n_norm)
        db = _d_bin_key(d)

        # Buscar en bins adyacentes de d (±1)
        found_idx = -1
        for d_candidate in (db - 1, db, db + 1):
            key = (nb, d_candidate)
            idx = bin_to_cluster.get(key, -1)
            if idx != -1:
                cl = clusters[idx]
                # Verificar tolerancia exacta
                if (
                    dot(n_norm, cl.normal) > NORMAL_CLUSTER_DOT
                    and abs(d - cl.d) < D_TOLERANCE
                ):
                    found_idx = idx
                    break

        if found_idx != -1:
            clusters[found_idx].face_infos.append(fi)
        else:
            new_idx = len(clusters)
            clusters.append(CoplanarCluster(normal=n_norm, d=d, face_infos=[fi]))
            # Registrar en el bin principal
            key = (nb, db)
            bin_to_cluster[key] = new_idx

    return clusters


# ---------------------------------------------------------------------------
# Componentes Conectados (Union-Find) — sin cambios, ya es eficiente
# ---------------------------------------------------------------------------


def split_connected(
    face_infos: List[FaceInfo], faces: List[Face3D]
) -> List[List[FaceInfo]]:
    if len(face_infos) <= 1:
        return [face_infos]

    vert_to_idx: Dict[Tuple, List[int]] = {}

    for i, fi in enumerate(face_infos):
        face = faces[fi.index]
        indices = get_vertex_indices(face)
        if indices:
            for vi in indices:
                key = (vi,)
                bucket = vert_to_idx.get(key)
                if bucket is None:
                    vert_to_idx[key] = [i]
                else:
                    bucket.append(i)
        else:
            for v in face.vertices:
                key = (snap3(v.x), snap3(v.y), snap3(v.z))
                bucket = vert_to_idx.get(key)
                if bucket is None:
                    vert_to_idx[key] = [i]
                else:
                    bucket.append(i)

    parent = list(range(len(face_infos)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for indices in vert_to_idx.values():
        for i in range(1, len(indices)):
            union(indices[0], indices[i])

    comp_map: Dict[int, List[FaceInfo]] = {}
    for i, fi in enumerate(face_infos):
        root = find(i)
        if root not in comp_map:
            comp_map[root] = []
        comp_map[root].append(fi)

    return list(comp_map.values())


# ---------------------------------------------------------------------------
# Etiquetas y Subgrupos
# ---------------------------------------------------------------------------


def orientation_label(normal: Vec3, orientation: str) -> str:
    if orientation == "horizontal":
        return "Horizontal"
    angle = math.atan2(normal.x, normal.z)
    deg = ((math.degrees(angle)) % 360 + 360) % 360
    if deg < 45 or deg >= 315:
        return "Vertical - Norte"
    if deg < 135:
        return "Vertical - Este"
    if deg < 225:
        return "Vertical - Sur"
    return "Vertical - Oeste"


CATEGORY_LABELS = {
    "floor": "Piso",
    "wall": "Pared",
    "discard": "Descartado",
}


@dataclass
class Subgroup:
    category: FaceCategory
    face_infos: List[FaceInfo]
    normal: Vec3
    d: float
    centroid: Vec3
    extent: float
    total_area: float


def build_subgroup(
    category: FaceCategory,
    cluster: CoplanarCluster,
    comp: List[FaceInfo],
    faces: List[Face3D],
) -> Optional[Subgroup]:
    total_area = sum(fi.area for fi in comp)
    if total_area < 0.01:
        return None

    cx = sum(fi.centroid.x * fi.area for fi in comp) / total_area
    cy = sum(fi.centroid.y * fi.area for fi in comp) / total_area
    cz = sum(fi.centroid.z * fi.area for fi in comp) / total_area

    min_x, min_y, min_z = float("inf"), float("inf"), float("inf")
    max_x, max_y, max_z = float("-inf"), float("-inf"), float("-inf")

    for fi in comp:
        for v in faces[fi.index].vertices:
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

    extent = max(max_x - min_x, max_y - min_y, max_z - min_z)

    return Subgroup(
        category=category,
        face_infos=comp,
        normal=cluster.normal,
        d=cluster.d,
        centroid=Vec3(cx, cy, cz),
        extent=extent,
        total_area=total_area,
    )


# ---------------------------------------------------------------------------
# API Pública
# ---------------------------------------------------------------------------


def classify_into_groups(
    faces: List[Face3D],
    min_real_area: float = DEFAULT_MIN_REAL_AREA,
    out_warnings: Optional[List[str]] = None,
) -> List[GeometryGroup]:
    t0 = time.perf_counter()

    if not faces:
        return []

    # 1. Clasificar caras individuales
    t_step = time.perf_counter()
    all_infos = classify_all_faces(faces)
    logger.debug(f"  → classify_all_faces: {(time.perf_counter()-t_step)*1000:.1f} ms")

    rim_faces: Set[int] = set()

    by_category: Dict[FaceCategory, List[FaceInfo]] = {}
    for fi in all_infos:
        if fi.category not in by_category:
            by_category[fi.category] = []
        by_category[fi.category].append(fi)

    logger.debug(
        f"  → por categoría: "
        + ", ".join(f"{k}={len(v)}" for k, v in by_category.items())
    )

    # 2. Clustering coplanar
    t_step = time.perf_counter()
    subgroups: List[Subgroup] = []
    for category, infos in by_category.items():
        clusters = cluster_coplanar(infos, faces)
        logger.debug(
            f"  → cluster_coplanar({category}): {len(infos)} faces → {len(clusters)} clusters"
        )
        for cluster in clusters:
            components = split_connected(cluster.face_infos, faces)
            for comp in components:
                sg = build_subgroup(category, cluster, comp, faces)
                if sg:
                    subgroups.append(sg)
    logger.debug(
        f"  → clustering total: {(time.perf_counter()-t_step)*1000:.1f} ms, {len(subgroups)} subgrupos"
    )

    # 3. Union-Find para gemelos (thin twins) — OPTIMIZADO con bin-indexing
    #
    # Original: O(N²) — 4565² = 10M comparaciones
    # Optimizado: Agrupar subgrupos por normal bin, comparar solo los candidatos
    # cuya normal podría ser "opuesta" (dot < -0.985, es decir normal_bin ≈ -normal_bin_i)
    # Resultado: 10M → ~pocos miles de comparaciones
    t_step = time.perf_counter()
    parent = list(range(len(subgroups)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    twin_contrib: List[Dict] = []
    slab_contrib: List[Dict] = []

    is_rim_subgroup = [
        len(sg.face_infos) > 0 and all(fi.index in rim_faces for fi in sg.face_infos)
        for sg in subgroups
    ]

    # Construir índice: normal_bin -> [subgroup_indices]
    # Resolución de bin: 0.1 en cada componente (~5.7° de tolerancia)
    TWIN_BIN = 10
    normal_bin_index: Dict[Tuple, List[int]] = {}
    active_indices: List[int] = []

    for i, sg in enumerate(subgroups):
        if sg.category == "discard" or is_rim_subgroup[i]:
            continue
        active_indices.append(i)
        nb = (
            round(sg.normal.x * TWIN_BIN),
            round(sg.normal.y * TWIN_BIN),
            round(sg.normal.z * TWIN_BIN),
        )
        if nb not in normal_bin_index:
            normal_bin_index[nb] = []
        normal_bin_index[nb].append(i)

    comparisons = 0
    for i in active_indices:
        sg_i = subgroups[i]
        ni = sg_i.normal

        # El "opuesto" de normal (nx, ny, nz) es (-nx, -ny, -nz)
        # Buscar en bins adyacentes al opuesto (±1 por componente en cada eje)
        opp_bx = round(-ni.x * TWIN_BIN)
        opp_by = round(-ni.y * TWIN_BIN)
        opp_bz = round(-ni.z * TWIN_BIN)

        candidates: List[int] = []
        # Buscar en los 27 bins vecinos al opuesto (rango ±1 en cada dimensión)
        for dbx in (-1, 0, 1):
            for dby in (-1, 0, 1):
                for dbz in (-1, 0, 1):
                    key = (opp_bx + dbx, opp_by + dby, opp_bz + dbz)
                    for j in normal_bin_index.get(key, []):
                        if j > i:  # solo pares (i, j) con i < j para no duplicar
                            candidates.append(j)

        for j in candidates:
            sg_j = subgroups[j]
            # Guarda de similitud de área: las dos pieles del MISMO muro tienen
            # áreas comparables. Si son muy dispares (p. ej. fachada de 75 m² vs una
            # barra de baranda de 0.28 m²) NO son gemelas; sin esto, el union-find
            # encadena fachada→barra→barra→pared-de-balcón y fusiona todo el balcón.
            lo = min(sg_i.total_area, sg_j.total_area)
            hi = max(sg_i.total_area, sg_j.total_area)
            if hi <= 0 or lo / hi < TWIN_AREA_RATIO:
                continue
            comparisons += 1
            thickness = are_thin_twins(sg_i, sg_j, THIN_WALL_THRESHOLD)
            if thickness is not None:
                union(i, j)
                twin_contrib.append({"idx": i, "thickness": thickness})

    logger.debug(
        f"  -> thin twins: {(time.perf_counter()-t_step)*1000:.1f} ms, "
        f"{len(twin_contrib)} pares, {comparisons} comparaciones (vs {len(active_indices)**2//2} O(N2))"
    )

    # 4. Losas contiguas
    WIDE_SLAB_MAX_GAP = 0.5
    FOOTPRINT_TOL = 0.05

    sg_bounds = []
    for sg in subgroups:
        min_x, max_x, min_y, max_y, min_z, max_z = (
            float("inf"), float("-inf"),
            float("inf"), float("-inf"),
            float("inf"), float("-inf"),
        )
        for fi in sg.face_infos:
            for v in faces[fi.index].vertices:
                if v.x < min_x: min_x = v.x
                if v.x > max_x: max_x = v.x
                if v.y < min_y: min_y = v.y
                if v.y > max_y: max_y = v.y
                if v.z < min_z: min_z = v.z
                if v.z > max_z: max_z = v.z
        sg_bounds.append({"minX": min_x, "maxX": max_x, "minY": min_y, "maxY": max_y, "minZ": min_z, "maxZ": max_z})

    def is_floor_horiz(i: int) -> bool:
        return (
            subgroups[i].category == "floor"
            and abs(subgroups[i].normal.y) >= HORIZONTAL_THRESHOLD
        )

    # Spatial hashing sobre el footprint X/Z para evitar el O(N²) de comparar
    # todas las losas contra todas. Cada losa se indexa en TODAS las celdas que
    # toca su AABB de footprint expandido por FOOTPRINT_TOL; dos losas que se
    # solapan (overlap >= -FOOTPRINT_TOL en X y Z) comparten al menos una celda,
    # así que ningún par candidato del O(N²) se pierde. Las pruebas de overlap/gap
    # y la unión se aplican exactamente igual que antes.
    SLAB_CELL = 2.0  # tamaño de celda de la grilla (m)

    floor_horiz = [i for i in range(len(subgroups)) if is_floor_horiz(i)]

    def _cell_range(i: int) -> Tuple[int, int, int, int]:
        b = sg_bounds[i]
        cx0 = math.floor((b["minX"] - FOOTPRINT_TOL) / SLAB_CELL)
        cx1 = math.floor((b["maxX"] + FOOTPRINT_TOL) / SLAB_CELL)
        cz0 = math.floor((b["minZ"] - FOOTPRINT_TOL) / SLAB_CELL)
        cz1 = math.floor((b["maxZ"] + FOOTPRINT_TOL) / SLAB_CELL)
        return cx0, cx1, cz0, cz1

    slab_grid: Dict[Tuple[int, int], List[int]] = {}
    for i in floor_horiz:
        cx0, cx1, cz0, cz1 = _cell_range(i)
        for cx in range(cx0, cx1 + 1):
            for cz in range(cz0, cz1 + 1):
                slab_grid.setdefault((cx, cz), []).append(i)

    for i in floor_horiz:
        ib = sg_bounds[i]
        cx0, cx1, cz0, cz1 = _cell_range(i)
        candidates: Set[int] = set()
        for cx in range(cx0, cx1 + 1):
            for cz in range(cz0, cz1 + 1):
                for j in slab_grid.get((cx, cz), ()):
                    if j > i:
                        candidates.add(j)
        for j in candidates:
            if find(i) == find(j):
                continue
            jb = sg_bounds[j]
            overlap_x = min(ib["maxX"], jb["maxX"]) - max(ib["minX"], jb["minX"])
            overlap_z = min(ib["maxZ"], jb["maxZ"]) - max(ib["minZ"], jb["minZ"])
            if overlap_x < -FOOTPRINT_TOL or overlap_z < -FOOTPRINT_TOL:
                continue
            gap = max(ib["minY"], jb["minY"]) - min(ib["maxY"], jb["maxY"])
            if gap >= WIDE_SLAB_MAX_GAP:
                continue
            union(i, j)

    # 5. Construir grupos finales
    slab_thickness: Dict[int, float] = {}
    for contrib in slab_contrib:
        root = find(contrib["idx"])
        if root not in slab_thickness or contrib["thickness"] > slab_thickness[root]:
            slab_thickness[root] = contrib["thickness"]

    twin_thickness: Dict[int, float] = {}
    for contrib in twin_contrib:
        root = find(contrib["idx"])
        if root not in twin_thickness or contrib["thickness"] < twin_thickness[root]:
            twin_thickness[root] = contrib["thickness"]

    union_map: Dict[int, List[Subgroup]] = {}
    for i in range(len(subgroups)):
        root = find(i)
        if root not in union_map:
            union_map[root] = []
        union_map[root].append(subgroups[i])

    groups: List[GeometryGroup] = []
    next_id = 1
    counters: Dict[str, int] = {}

    for root_idx, merged in union_map.items():
        all_face_indices: List[int] = []
        total_area = 0.0
        cx, cy, cz = 0.0, 0.0, 0.0
        group_min_y, group_max_y = float("inf"), float("-inf")

        biggest = merged[0]
        biggest_orient = merged[0].face_infos[0].orientation

        for sg in merged:
            for fi in sg.face_infos:
                all_face_indices.append(fi.index)
                for v in faces[fi.index].vertices:
                    if v.y < group_min_y:
                        group_min_y = v.y
                    if v.y > group_max_y:
                        group_max_y = v.y
            total_area += sg.total_area
            cx += sg.centroid.x * sg.total_area
            cy += sg.centroid.y * sg.total_area
            cz += sg.centroid.z * sg.total_area
            if sg.total_area > biggest.total_area:
                biggest = sg
                biggest_orient = sg.face_infos[0].orientation

        cx /= total_area
        cy /= total_area
        cz /= total_area

        dominant = biggest.category
        orient = orientation_label(biggest.normal, biggest_orient)
        cat_label = CATEGORY_LABELS[dominant]
        key = f"{cat_label}_{orient}"
        counters[key] = counters.get(key, 0) + 1

        detected_thickness = slab_thickness.get(root_idx, twin_thickness.get(root_idx))

        groups.append(
            GeometryGroup(
                id=next_id,
                label=f"{cat_label} {orient} #{counters[key]}",
                category=dominant,
                face_indices=all_face_indices,
                total_area=total_area,
                centroid=Vec3(cx, cy, cz),
                orientation=orient,
                representative_normal=biggest.normal,
                thickness=detected_thickness,
                min_y=None if group_min_y == float("inf") else group_min_y,
                max_y=None if group_max_y == float("-inf") else group_max_y,
                original_category=dominant,
            )
        )
        next_id += 1

    for g in groups:
        if g.total_area < min_real_area and g.category != "discard":
            g.category = "discard"
            g.label = f"{CATEGORY_LABELS['discard']} {g.orientation} #{g.id}"

    ORDER = {"floor": 0, "wall": 1, "discard": 2}
    groups.sort(key=lambda g: (ORDER[g.category], -g.total_area))

    total_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"  classify_into_groups: {len(faces)} caras → {len(groups)} grupos en {total_ms:.1f} ms"
    )
    return groups


def polish_groups(groups: List[GeometryGroup], min_real_area: float) -> None:
    for g in groups:
        if g.category == "discard":
            continue
        is_horizontal = g.orientation == "Horizontal"
        if is_horizontal and g.category == "wall":
            g.category = "floor"
            g.original_category = "floor"
            g.label = re.sub(r"^Pared", CATEGORY_LABELS["floor"], g.label)
        elif not is_horizontal and g.category == "floor":
            g.category = "wall"
            g.original_category = "wall"
            g.label = re.sub(r"^Piso", CATEGORY_LABELS["wall"], g.label)

    for g in groups:
        if g.total_area < min_real_area and g.category != "discard":
            g.category = "discard"
            g.label = re.sub(r"^(Piso|Pared)", CATEGORY_LABELS["discard"], g.label)

    ORDER = {"floor": 0, "wall": 1, "discard": 2}
    groups.sort(key=lambda g: (ORDER[g.category], -g.total_area))


def peel_buried_walls(
    faces: List[Face3D], groups: List[GeometryGroup]
) -> List[GeometryGroup]:
    next_id = max((g.id for g in groups), default=0) + 1

    wall_counters: Dict[str, int] = {}
    for g in groups:
        m = re.match(r"^Pared (.+) #(\d+)$", g.label)
        if m:
            key, num = m.group(1), int(m.group(2))
            wall_counters[key] = max(wall_counters.get(key, 0), num)

    def info_for(idx: int) -> FaceInfo:
        f = faces[idx]
        abs_y = abs(f.normal.y)
        if abs_y >= HORIZONTAL_THRESHOLD:
            orientation = "horizontal"
        elif abs_y <= VERTICAL_THRESHOLD:
            orientation = "vertical"
        else:
            orientation = "inclined"
        return FaceInfo(
            index=idx,
            area=face_area(f),
            centroid=face_centroid(f),
            orientation=orientation,
            category="floor",
        )

    out: List[GeometryGroup] = []

    for g in groups:
        if g.category != "floor":
            out.append(g)
            continue

        infos = [info_for(idx) for idx in g.face_indices]
        vertical_infos = [fi for fi in infos if fi.orientation == "vertical"]

        if not vertical_infos:
            out.append(g)
            continue

        clusters = split_connected(vertical_infos, faces)
        peel_clusters: List[List[FaceInfo]] = []
        keep_vertical: List[FaceInfo] = []

        for cluster in clusters:
            min_y, max_y = float("inf"), float("-inf")
            for fi in cluster:
                for v in faces[fi.index].vertices:
                    if v.y < min_y:
                        min_y = v.y
                    if v.y > max_y:
                        max_y = v.y
            if max_y - min_y > WALL_PEEL_MIN_HEIGHT:
                peel_clusters.append(cluster)
            else:
                keep_vertical.extend(cluster)

        if not peel_clusters:
            out.append(g)
            continue

        rest_infos = [fi for fi in infos if fi.orientation != "vertical"]
        new_walls: List[GeometryGroup] = []
        reabsorbed: List[FaceInfo] = []

        for cluster in peel_clusters:
            biggest = max(cluster, key=lambda fi: fi.area)
            normal = normalize(faces[biggest.index].normal)
            pseudo_cluster = CoplanarCluster(normal=normal, d=0, face_infos=cluster)

            sg = build_subgroup("wall", pseudo_cluster, cluster, faces)
            if not sg:
                reabsorbed.extend(cluster)
                continue

            min_y, max_y = float("inf"), float("-inf")
            for fi in cluster:
                for v in faces[fi.index].vertices:
                    if v.y < min_y:
                        min_y = v.y
                    if v.y > max_y:
                        max_y = v.y

            orient = orientation_label(normal, "vertical")
            wall_counters[orient] = wall_counters.get(orient, 0) + 1

            new_walls.append(
                GeometryGroup(
                    id=next_id,
                    label=f"Pared {orient} #{wall_counters[orient]}",
                    category="wall",
                    face_indices=[fi.index for fi in cluster],
                    total_area=sg.total_area,
                    centroid=sg.centroid,
                    orientation=orient,
                    representative_normal=normal,
                    thickness=None,
                    min_y=min_y,
                    max_y=max_y,
                    original_category="wall",
                )
            )
            next_id += 1

        remaining = rest_infos + keep_vertical + reabsorbed
        if remaining:
            total_area, cx, cy, cz = 0.0, 0.0, 0.0, 0.0
            min_y, max_y = float("inf"), float("-inf")

            for fi in remaining:
                total_area += fi.area
                cx += fi.centroid.x * fi.area
                cy += fi.centroid.y * fi.area
                cz += fi.centroid.z * fi.area
                for v in faces[fi.index].vertices:
                    if v.y < min_y:
                        min_y = v.y
                    if v.y > max_y:
                        max_y = v.y

            if total_area > 0:
                cx /= total_area
                cy /= total_area
                cz /= total_area

            out.append(
                GeometryGroup(
                    id=g.id,
                    label=g.label,
                    category=g.category,
                    face_indices=[fi.index for fi in remaining],
                    total_area=total_area,
                    centroid=Vec3(cx, cy, cz),
                    orientation=g.orientation,
                    representative_normal=g.representative_normal,
                    thickness=g.thickness,
                    original_category=g.original_category,
                    min_y=min_y,
                    max_y=max_y,
                )
            )

        out.extend(new_walls)

    ORDER = {"floor": 0, "wall": 1, "discard": 2}
    out.sort(key=lambda g: (ORDER[g.category], -g.total_area))
    return out
