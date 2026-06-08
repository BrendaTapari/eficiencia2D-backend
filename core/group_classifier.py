import math
import re
from typing import List, Dict, Set, Optional, Literal, Tuple
from dataclasses import dataclass

# Importamos nuestros tipos y helpers
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

# NOTA: Este import asume que el próximo archivo que traduciremos será el slab-edge-detector
# from core.services.slab_edge_detector import detect_slab_edges, find_slab_rim_faces, validate_slab_candidates

# ============================================================================
# Group Classifier
# ============================================================================

FaceCategory = Literal["floor", "wall", "discard"]


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
# Clasificación por cara
# ---------------------------------------------------------------------------

HORIZONTAL_THRESHOLD = 0.98
VERTICAL_THRESHOLD = 0.5
MIN_AREA = 1e-6
HEIGHT_BAND = 0.05
THIN_WALL_THRESHOLD = 0.40
DEFAULT_MIN_REAL_AREA = 1.0
WALL_PEEL_MIN_HEIGHT = 1.0


def face_area(f: Face3D) -> float:
    verts = f.vertices
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


def face_centroid(f: Face3D) -> Vec3:
    verts = f.vertices
    if not verts:
        return Vec3(0.0, 0.0, 0.0)
    sx = sum(v.x for v in verts)
    sy = sum(v.y for v in verts)
    sz = sum(v.z for v in verts)
    n = len(verts)
    return Vec3(sx / n, sy / n, sz / n)


@dataclass
class FaceInfo:
    index: int
    area: float
    centroid: Vec3
    orientation: Literal["horizontal", "vertical", "inclined"]
    category: FaceCategory


def classify_all_faces(faces: List[Face3D]) -> List[FaceInfo]:
    infos: List[FaceInfo] = []

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

        infos.append(
            FaceInfo(
                index=i,
                area=area,
                centroid=face_centroid(face),
                orientation=orientation,
                category="discard",
            )
        )

    min_x, max_x = float("inf"), float("-inf")
    min_z, max_z = float("inf"), float("-inf")
    for fi in infos:
        for v in faces[fi.index].vertices:
            if v.x < min_x:
                min_x = v.x
            if v.x > max_x:
                max_x = v.x
            if v.z < min_z:
                min_z = v.z
            if v.z > max_z:
                max_z = v.z

    range_x = max_x - min_x
    range_z = max_z - min_z

    if range_x < 1.0 or range_z < 1.0:
        for fi in infos:
            if fi.orientation == "horizontal":
                fi.category = "floor"
            elif fi.orientation == "vertical":
                fi.category = "wall"
            else:
                fi.category = "discard"
        return infos

    horizontals = [fi for fi in infos if fi.orientation == "horizontal"]
    level_groups: Dict[float, List[FaceInfo]] = {}

    for fi in horizontals:
        h = fi.centroid.y
        found_key = None
        for key in level_groups:
            if abs(key - h) < HEIGHT_BAND:
                found_key = key
                break
        key = found_key if found_key is not None else h
        if key not in level_groups:
            level_groups[key] = []
        level_groups[key].append(fi)

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

    return infos


# ---------------------------------------------------------------------------
# Clustering Coplanar
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


def cluster_coplanar(
    infos: List[FaceInfo], faces: List[Face3D]
) -> List[CoplanarCluster]:
    clusters: List[CoplanarCluster] = []

    for fi in infos:
        face = faces[fi.index]
        n = face.normal
        if vlength(n) < 0.01:
            continue
        d = dot(n, face.vertices[0])

        placed = False
        for cl in clusters:
            if dot(n, cl.normal) > NORMAL_CLUSTER_DOT and abs(d - cl.d) < D_TOLERANCE:
                cl.face_infos.append(fi)
                placed = True
                break

        if not placed:
            clusters.append(CoplanarCluster(normal=normalize(n), d=d, face_infos=[fi]))

    return clusters


# ---------------------------------------------------------------------------
# Componentes Conectados (Union-Find)
# ---------------------------------------------------------------------------


def split_connected(
    face_infos: List[FaceInfo], faces: List[Face3D]
) -> List[List[FaceInfo]]:
    if len(face_infos) <= 1:
        return [face_infos]

    vert_to_idx: Dict[str, List[int]] = {}

    for i, fi in enumerate(face_infos):
        face = faces[fi.index]
        indices = get_vertex_indices(face)
        if indices:
            for vi in indices:
                key = str(vi)
                if key not in vert_to_idx:
                    vert_to_idx[key] = []
                vert_to_idx[key].append(i)
        else:
            for v in face.vertices:
                key = f"{snap3(v.x)},{snap3(v.y)},{snap3(v.z)}"
                if key not in vert_to_idx:
                    vert_to_idx[key] = []
                vert_to_idx[key].append(i)

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
    if not faces:
        return []

    all_infos = classify_all_faces(faces)

    # TODO: Cuando traduzcamos slab_edge_detector.py, descomentar esto:
    # raw_rim_faces = find_slab_rim_faces(faces)
    # rim_faces = validate_slab_candidates(raw_rim_faces, faces)
    # if len(rim_faces) == 0 and len(raw_rim_faces) > 0 and out_warnings is not None:
    #     out_warnings.append("Detección de losa desactivada: resultados no pasaron validación")
    # if rim_faces:
    #     for fi in all_infos:
    #         if fi.index in rim_faces: fi.category = "floor"
    rim_faces: Set[int] = set()  # Mock temporal hasta traer la librería

    by_category: Dict[FaceCategory, List[FaceInfo]] = {}
    for fi in all_infos:
        if fi.category not in by_category:
            by_category[fi.category] = []
        by_category[fi.category].append(fi)

    subgroups: List[Subgroup] = []
    for category, infos in by_category.items():
        clusters = cluster_coplanar(infos, faces)
        for cluster in clusters:
            components = split_connected(cluster.face_infos, faces)
            for comp in components:
                sg = build_subgroup(category, cluster, comp, faces)
                if sg:
                    subgroups.append(sg)

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

    twin_contrib: List[Dict[str, float]] = []
    slab_contrib: List[Dict[str, float]] = []

    is_rim_subgroup = [
        len(sg.face_infos) > 0 and all(fi.index in rim_faces for fi in sg.face_infos)
        for sg in subgroups
    ]

    for i in range(len(subgroups)):
        if subgroups[i].category == "discard" or is_rim_subgroup[i]:
            continue
        for j in range(i + 1, len(subgroups)):
            if subgroups[j].category == "discard" or is_rim_subgroup[j]:
                continue

            # Pasamos los parámetros posicionales que are_thin_twins espera: (TwinCandidate, TwinCandidate, float)
            # Como Subgroup tiene los mismos atributos que TwinCandidate, el duck-typing de Python lo permite.
            thickness = are_thin_twins(subgroups[i], subgroups[j], THIN_WALL_THRESHOLD)
            if thickness is not None:
                union(i, j)
                twin_contrib.append({"idx": i, "thickness": thickness})

    # TODO: Cuando traduzcamos slab_edge_detector.py, descomentar esto:
    # for link in detect_slab_edges(subgroups, faces, rim_faces):
    #     union(link.floor_subgroup_index, link.rim_subgroup_index)
    #     slab_contrib.append({"idx": link.floor_subgroup_index, "thickness": link.thickness})

    WIDE_SLAB_MAX_GAP = 0.5
    FOOTPRINT_TOL = 0.05

    sg_bounds = []
    for sg in subgroups:
        min_x, max_x, min_y, max_y, min_z, max_z = (
            float("inf"),
            float("-inf"),
            float("inf"),
            float("-inf"),
            float("inf"),
            float("-inf"),
        )
        for fi in sg.face_infos:
            for v in faces[fi.index].vertices:
                if v.x < min_x:
                    min_x = v.x
                if v.x > max_x:
                    max_x = v.x
                if v.y < min_y:
                    min_y = v.y
                if v.y > max_y:
                    max_y = v.y
                if v.z < min_z:
                    min_z = v.z
                if v.z > max_z:
                    max_z = v.z
        sg_bounds.append(
            {
                "minX": min_x,
                "maxX": max_x,
                "minY": min_y,
                "maxY": max_y,
                "minZ": min_z,
                "maxZ": max_z,
            }
        )

    def is_floor_horiz(i: int) -> bool:
        return (
            subgroups[i].category == "floor"
            and abs(subgroups[i].normal.y) >= HORIZONTAL_THRESHOLD
        )

    for i in range(len(subgroups)):
        if not is_floor_horiz(i):
            continue
        ib = sg_bounds[i]
        for j in range(i + 1, len(subgroups)):
            if not is_floor_horiz(j):
                continue
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
