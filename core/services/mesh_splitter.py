import math
from typing import List, Dict, Tuple, Literal, Any, Optional
from dataclasses import dataclass

# Importamos nuestros tipos (asegurate de tenerlos disponibles en core/types.py)
from core.services.types import Face3D, Vec3

# Importamos el GeometryGroup de group_classifier (usamos TYPE_CHECKING para evitar circular imports si es necesario)
from core.group_classifier import GeometryGroup, DEFAULT_MIN_REAL_AREA

# ============================================================================
# Mesh Splitter
#
# Corta un conjunto de caras 3D con un plano horizontal, produciendo dos
# conjuntos de caras: uno por encima y otro por debajo del plano.
# Útil para dividir muros que se extienden a través de varias losas.
# ============================================================================

UpAxis = Literal["Y", "Z"]


@dataclass
class SplitResult:
    above: List[Face3D]
    below: List[Face3D]


def lerp3(a: Vec3, b: Vec3, t: float) -> Vec3:
    return Vec3(
        x=a.x + (b.x - a.x) * t, y=a.y + (b.y - a.y) * t, z=a.z + (b.z - a.z) * t
    )


def get_up_val(v: Vec3, up: UpAxis) -> float:
    return v.y if up == "Y" else v.z


def clip_polygon(
    verts: List[Vec3], elevation: float, up: UpAxis, keep_above: bool, tolerance: float
) -> List[Vec3]:
    """
    Corta los vértices de un solo polígono en la elevación dada.
    Utiliza recorte estilo Sutherland-Hodgman contra un único plano.
    """
    result: List[Vec3] = []
    n = len(verts)
    if n < 3:
        return result

    for i in range(n):
        current = verts[i]
        nxt = verts[(i + 1) % n]
        d_curr = get_up_val(current, up) - elevation
        d_next = get_up_val(nxt, up) - elevation

        curr_inside = (d_curr >= -tolerance) if keep_above else (d_curr <= tolerance)
        next_inside = (d_next >= -tolerance) if keep_above else (d_next <= tolerance)

        if curr_inside:
            result.append(current)

        # La arista cruza el plano — agregar punto de intersección.
        if (d_curr > tolerance and d_next < -tolerance) or (
            d_curr < -tolerance and d_next > tolerance
        ):
            t = d_curr / (d_curr - d_next)
            result.append(lerp3(current, nxt, t))

    return result


def split_faces_at_plane(
    faces: List[Face3D], elevation: float, up: UpAxis = "Y", tolerance: float = 0.01
) -> SplitResult:
    """
    Corta un conjunto de caras por un plano horizontal.
    """
    above: List[Face3D] = []
    below: List[Face3D] = []

    for face in faces:
        dists = [get_up_val(v, up) - elevation for v in face.vertices]
        all_above = all(d >= -tolerance for d in dists)
        all_below = all(d <= tolerance for d in dists)

        if all_above and not all_below:
            above.append(face)
        elif all_below and not all_above:
            below.append(face)
        elif all_above and all_below:
            # La cara descansa completamente sobre el plano — incluir en ambos.
            above.append(face)
            below.append(face)
        else:
            # La cara cruza el plano — recortarla.
            above_verts = clip_polygon(face.vertices, elevation, up, True, tolerance)
            below_verts = clip_polygon(face.vertices, elevation, up, False, tolerance)

            # Usamos un truco simple para crear una nueva cara preservando propiedades viejas
            # (En el caso del TypeScript destructurabas properties, aquí las clonamos)
            if len(above_verts) >= 3:
                new_above = Face3D(
                    vertices=above_verts,
                    normal=face.normal,
                    inner_loops=[],  # Los innerLoops no se preservan en el recorte simple
                    panel_id=face.panel_id,
                )
                above.append(new_above)

            if len(below_verts) >= 3:
                new_below = Face3D(
                    vertices=below_verts,
                    normal=face.normal,
                    inner_loops=[],
                    panel_id=face.panel_id,
                )
                below.append(new_below)

    return SplitResult(above=above, below=below)


def split_wall_at_floors(
    faces: List[Face3D],
    floor_elevations: List[float],
    up: UpAxis = "Y",
    tolerance: float = 0.01,
) -> List[List[Face3D]]:
    """
    Corta las caras de un muro en múltiples elevaciones de piso.
    Retorna una lista de grupos de caras, uno por segmento entre cortes consecutivos.
    """
    if not floor_elevations:
        return [faces]

    # Encontrar extensión vertical del muro
    wall_min = float("inf")
    wall_max = float("-inf")
    for f in faces:
        for v in f.vertices:
            h = get_up_val(v, up)
            if h < wall_min:
                wall_min = h
            if h > wall_max:
                wall_max = h

    # Filtrar elevaciones que intersecan realmente este muro
    relevant = [
        e
        for e in floor_elevations
        if (e > wall_min + tolerance and e < wall_max - tolerance)
    ]

    if not relevant:
        return [faces]

    # Ordenar y cortar de abajo hacia arriba
    sorted_elevs = sorted(relevant)
    segments: List[List[Face3D]] = []
    remaining = faces

    for elev in sorted_elevs:
        res = split_faces_at_plane(remaining, elev, up, tolerance)
        if res.below:
            segments.append(res.below)
        remaining = res.above

    if remaining:
        segments.append(remaining)

    return segments


def collect_floor_planes(
    groups: List[GeometryGroup],
    override_map: Dict[int, str],
    faces: List[Face3D],
    up: UpAxis = "Y",
) -> List[float]:

    elevations: List[float] = []
    DEDUP_TOL = 0.10  # 10cm

    for group in groups:
        effective_cat = override_map.get(group.id, group.category)
        if effective_cat != "floor":
            continue

        ny = abs(get_up_val(group.representative_normal, up))
        if ny < 0.75:
            continue

        sum_y = 0.0
        count = 0
        for fi in group.face_indices:
            # Verificación en caso de índices fuera de rango o caras nulas
            if fi < 0 or fi >= len(faces):
                continue
            face = faces[fi]
            for v in face.vertices:
                sum_y += get_up_val(v, up)
                count += 1

        if count == 0:
            continue

        avg = sum_y / count

        # Deduplicar
        is_dup = any(abs(e - avg) < DEDUP_TOL for e in elevations)
        if not is_dup:
            elevations.append(avg)

    return sorted(elevations)


def subgroup_area(faces: List[Face3D]) -> float:
    total = 0.0
    for f in faces:
        verts = f.vertices
        if len(verts) < 3:
            continue
        sx, sy, sz = 0.0, 0.0, 0.0
        v0 = verts[0]
        for i in range(1, len(verts) - 1):
            e1x, e1y, e1z = verts[i].x - v0.x, verts[i].y - v0.y, verts[i].z - v0.z
            e2x, e2y, e2z = (
                verts[i + 1].x - v0.x,
                verts[i + 1].y - v0.y,
                verts[i + 1].z - v0.z,
            )
            sx += e1y * e2z - e1z * e2y
            sy += e1z * e2x - e1x * e2z
            sz += e1x * e2y - e1y * e2x
        total += 0.5 * math.sqrt(sx**2 + sy**2 + sz**2)
    return total


@dataclass
class SubgroupBounds:
    min_y: float
    max_y: float
    cx: float
    cy: float
    cz: float
    count: int


def subgroup_bounds(faces: List[Face3D], up: UpAxis) -> SubgroupBounds:
    min_y = float("inf")
    max_y = float("-inf")
    cx, cy, cz = 0.0, 0.0, 0.0
    count = 0

    for f in faces:
        for v in f.vertices:
            h = get_up_val(v, up)
            if h < min_y:
                min_y = h
            if h > max_y:
                max_y = h
            cx += v.x
            cy += v.y
            cz += v.z
            count += 1

    return SubgroupBounds(min_y, max_y, cx, cy, cz, count)


def merge_thin_segments(
    segments: List[List[Face3D]], min_area: float
) -> List[List[Face3D]]:
    """
    Fusiona segmentos finos producto de un corte en su vecino adyacente.
    """
    if len(segments) <= 1:
        return segments

    # Usamos diccionarios mutables en lugar de tuples para poder actualizar la información
    lst = [{"faces": seg, "area": subgroup_area(seg)} for seg in segments]

    while len(lst) > 1:
        # Encontrar el primero por debajo del área mínima
        idx = next((i for i, e in enumerate(lst) if e["area"] < min_area), -1)
        if idx == -1:
            break

        target = 0
        if idx == 0:
            target = 1  # sliver inferior: su único vecino es el de arriba
        elif idx == len(lst) - 1:
            target = idx - 1  # sliver superior: su único vecino es el de abajo
        else:
            # Interior: fundir en el vecino de mayor área (empate -> el de abajo)
            target = (
                idx - 1 if lst[idx - 1]["area"] >= lst[idx + 1]["area"] else idx + 1
            )

        lst[target]["faces"].extend(lst[idx]["faces"])
        lst[target]["area"] = subgroup_area(lst[target]["faces"])
        lst.pop(idx)

    return [e["faces"] for e in lst]


# ---------------------------------------------------------------------------
# Cortador principal
# ---------------------------------------------------------------------------

MIN_SPLIT_AREA = 0.01  # m^2 — fragmentos muy diminutos


@dataclass
class SlabPlane:
    elevation: float
    min_a: float
    max_a: float
    min_b: float
    max_b: float


def collect_slab_planes(
    groups: List[GeometryGroup],
    override_map: Dict[int, str],
    faces: List[Face3D],
    up: UpAxis,
) -> List[SlabPlane]:

    # Determinar qué propiedades de Vec3 representan el plano horizontal
    ax_a, ax_b = ("x", "z") if up == "Y" else ("x", "y")
    planes: List[SlabPlane] = []

    for group in groups:
        effective_cat = override_map.get(group.id, group.category)

        ny = abs(get_up_val(group.representative_normal, up))
        if ny < 0.75:
            continue

        is_floor = effective_cat == "floor"
        is_demoted_floor = (
            effective_cat == "discard" and group.original_category == "floor"
        )
        if not is_floor and not is_demoted_floor:
            continue

        sum_elev, count = 0.0, 0
        min_a, max_a = float("inf"), float("-inf")
        min_b, max_b = float("inf"), float("-inf")

        for fi in group.face_indices:
            if fi < 0 or fi >= len(faces):
                continue
            face = faces[fi]
            for v in face.vertices:
                sum_elev += get_up_val(v, up)
                count += 1
                a = getattr(v, ax_a)
                b = getattr(v, ax_b)
                if a < min_a:
                    min_a = a
                if a > max_a:
                    max_a = a
                if b < min_b:
                    min_b = b
                if b > max_b:
                    max_b = b

        if count > 0:
            planes.append(
                SlabPlane(
                    elevation=(sum_elev / count),
                    min_a=min_a,
                    max_a=max_a,
                    min_b=min_b,
                    max_b=max_b,
                )
            )

    return planes


def split_wall_groups_at_floors(
    faces: List[Face3D],
    groups: List[GeometryGroup],
    override_map: Dict[int, str],
    up: UpAxis = "Y",
    min_real_area: float = DEFAULT_MIN_REAL_AREA,
) -> Tuple[List[Face3D], List[GeometryGroup]]:

    slab_planes = collect_slab_planes(groups, override_map, faces, up)
    if not slab_planes:
        return faces, groups

    ax_a, ax_b = ("x", "z") if up == "Y" else ("x", "y")
    FOOTPRINT_TOL = 0.10
    DEDUP_TOL = 0.10

    new_faces = list(faces)
    new_groups: List[GeometryGroup] = []

    next_id = 0
    for g in groups:
        if g.id >= next_id:
            next_id = g.id + 1

    for group in groups:
        effective_cat = override_map.get(group.id, group.category)
        if effective_cat != "wall":
            new_groups.append(group)
            continue

        group_faces = [faces[fi] for fi in group.face_indices if 0 <= fi < len(faces)]

        # Footprint horizontal
        w_min_a, w_max_a = float("inf"), float("-inf")
        w_min_b, w_max_b = float("inf"), float("-inf")

        for f in group_faces:
            for v in f.vertices:
                a = getattr(v, ax_a)
                b = getattr(v, ax_b)
                if a < w_min_a:
                    w_min_a = a
                if a > w_max_a:
                    w_max_a = a
                if b < w_min_b:
                    w_min_b = b
                if b > w_max_b:
                    w_max_b = b

        tol = max(group.thickness or 0.0, FOOTPRINT_TOL)

        # Retener solo losas cuyo footprint se solape con este muro
        candidate_elevs: List[float] = []
        for sp in slab_planes:
            overlap_a = min(w_max_a, sp.max_a) - max(w_min_a, sp.min_a)
            overlap_b = min(w_max_b, sp.max_b) - max(w_min_b, sp.min_b)
            if overlap_a < -tol or overlap_b < -tol:
                continue
            candidate_elevs.append(sp.elevation)

        candidate_elevs.sort()
        elevations: List[float] = []
        for e in candidate_elevs:
            if not elevations or abs(e - elevations[-1]) > DEDUP_TOL:
                elevations.append(e)

        segments = split_wall_at_floors(group_faces, elevations, up)
        merged = merge_thin_segments(segments, min_real_area)

        if len(merged) <= 1:
            new_groups.append(group)
            continue

        for k, seg in enumerate(merged):
            if not seg:
                continue

            area = subgroup_area(seg)
            if area < MIN_SPLIT_AREA:
                continue

            face_indices: List[int] = []
            for face in seg:
                face_indices.append(len(new_faces))
                new_faces.append(face)

            b = subgroup_bounds(seg, up)
            centroid = (
                Vec3(b.cx / b.count, b.cy / b.count, b.cz / b.count)
                if b.count > 0
                else group.centroid
            )

            new_groups.append(
                GeometryGroup(
                    id=(group.id if k == 0 else next_id),
                    label=f"{group.label} ({k + 1}/{len(merged)})",
                    category=group.category,
                    original_category=group.original_category,
                    face_indices=face_indices,
                    total_area=area,
                    centroid=centroid,
                    orientation=group.orientation,
                    representative_normal=group.representative_normal,
                    thickness=group.thickness,
                    min_y=None if b.min_y == float("inf") else b.min_y,
                    max_y=None if b.max_y == float("-inf") else b.max_y,
                )
            )
            if k > 0:
                next_id += 1

    return new_faces, new_groups
