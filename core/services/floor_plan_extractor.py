import math
import numpy as np
from typing import List, Dict, Tuple, Literal, Set, Optional

# Importamos nuestros tipos y helpers vectoriales
from core.services.types import (
    Face3D,
    FloorPlan,
    FloorPlanSegment,
    Vec2,
    Vec3,
    cross,
    sub,
)

# Importamos el extractor de puertas
from core.services.floor_extractor import is_door_group, extract_doors_for_level

# ============================================================================
# Floor Plan Extractor — Cortes de sección horizontal
#
# Para cada nivel de piso detectado, corta el edificio con un plano horizontal
# a ~1 m sobre la losa, produciendo una vista en planta 2D.
# ============================================================================

CUT_HEIGHT = 1.0
HORIZONTAL_EPSILON = 0.25
VERTICAL_EPSILON = 0.20
BIN_SIZE = 0.3  # m — ancho de bin del histograma para detección de pisos
MIN_FLOOR_GAP = 2.0  # m — fusionar picos más cercanos que esto
PEAK_AREA_RATIO = 0.08  # los picos deben tener >= 8% del área del bin más alto

UpAxis = Literal["Y", "Z"]


def get_up(v: Vec3, up: UpAxis) -> float:
    return v.y if up == "Y" else v.z


def project_top_down(v: Vec3, up: UpAxis) -> Vec2:
    return Vec2(v.x, v.z) if up == "Y" else Vec2(v.x, v.y)


def face_area(face: Face3D) -> float:
    verts = face.vertices
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


def detect_floor_levels(faces: List[Face3D], up: UpAxis) -> List[float]:
    """Detecta las elevaciones de los pisos mediante un análisis de histograma de áreas."""
    elevations = []

    for face in faces:
        up_comp = abs(get_up(face.normal, up))
        if up_comp < 1.0 - HORIZONTAL_EPSILON:
            continue

        area = face_area(face)
        if area < 0.01:
            continue

        elev = sum(get_up(v, up) for v in face.vertices) / len(face.vertices)
        elevations.append({"elev": elev, "area": area})

    if not elevations:
        return []

    # Uso de Numpy para acelerar el cálculo de picos
    elev_vals = [e["elev"] for e in elevations]
    min_elev = min(elev_vals)
    max_elev = max(elev_vals)

    num_bins = max(1, math.ceil((max_elev - min_elev) / BIN_SIZE) + 1)
    bins = np.zeros(num_bins)

    for e in elevations:
        idx = min(math.floor((e["elev"] - min_elev) / BIN_SIZE), num_bins - 1)
        bins[idx] += e["area"]

    max_bin_area = np.max(bins)
    threshold = max_bin_area * PEAK_AREA_RATIO

    peak_elevations = []
    for i in range(num_bins):
        if bins[i] < threshold:
            continue

        is_local_max = (i == 0 or bins[i] >= bins[i - 1]) and (
            i == num_bins - 1 or bins[i] >= bins[i + 1]
        )

        if not is_local_max:
            continue

        bin_low = min_elev + i * BIN_SIZE
        bin_high = bin_low + BIN_SIZE

        sum_area = 0.0
        sum_weighted = 0.0
        for e in elevations:
            if bin_low <= e["elev"] < bin_high:
                sum_area += e["area"]
                sum_weighted += e["elev"] * e["area"]

        if sum_area > 0:
            peak_elevations.append({"elev": sum_weighted / sum_area, "area": sum_area})

    peak_elevations.sort(key=lambda x: x["elev"])

    levels: List[float] = []
    for peak in peak_elevations:
        if levels and (peak["elev"] - levels[-1]) < MIN_FLOOR_GAP:
            prev_idx = len(levels) - 1
            prev_peak = next(
                (
                    p
                    for p in peak_elevations
                    if abs(p["elev"] - levels[prev_idx]) < 1e-9
                ),
                None,
            )
            if prev_peak and peak["area"] > prev_peak["area"]:
                levels[prev_idx] = peak["elev"]
        else:
            levels.append(peak["elev"])

    return levels


def intersect_face_with_plane(
    face: Face3D, cut_elev: float, up: UpAxis
) -> List[Tuple[Vec3, Vec3]]:
    verts = face.vertices
    n = len(verts)
    if n < 3:
        return []

    dists = [get_up(v, up) - cut_elev for v in verts]
    intersections: List[Vec3] = []

    for i in range(n):
        j = (i + 1) % n
        di = dists[i]
        dj = dists[j]

        if abs(di) < 1e-9:
            intersections.append(verts[i])
        elif (di > 0) != (dj > 0):
            t = di / (di - dj)
            vi = verts[i]
            vj = verts[j]
            intersections.append(
                Vec3(
                    vi.x + t * (vj.x - vi.x),
                    vi.y + t * (vj.y - vi.y),
                    vi.z + t * (vj.z - vi.z),
                )
            )

    # Filtrar duplicados
    unique: List[Vec3] = []
    for pt in intersections:
        dup = False
        for u in unique:
            if (
                abs(pt.x - u.x) < 1e-6
                and abs(pt.y - u.y) < 1e-6
                and abs(pt.z - u.z) < 1e-6
            ):
                dup = True
                break
        if not dup:
            unique.append(pt)

    if len(unique) >= 2:
        return [(unique[0], unique[1])]
    return []


def to_floor_plan_segments(segments: List[Dict[str, Vec2]]) -> List[FloorPlanSegment]:
    return [FloorPlanSegment(a=s["a"], b=s["b"], is_interior=False) for s in segments]


def extract_with_axis(faces: List[Face3D], up: UpAxis) -> List[FloorPlan]:
    levels = detect_floor_levels(faces, up)
    if not levels:
        return []

    # --- Identificar grupos de puertas ---
    door_group_names = {
        face.panel_id
        for face in faces
        if face.panel_id and is_door_group(face.panel_id)
    }

    door_faces_by_group: Dict[str, List[Face3D]] = {}
    for face in faces:
        if face.panel_id and face.panel_id in door_group_names:
            if face.panel_id not in door_faces_by_group:
                door_faces_by_group[face.panel_id] = []
            door_faces_by_group[face.panel_id].append(face)

    # Caras verticales EXCLUYENDO puertas
    vertical_faces = [
        f
        for f in faces
        if abs(get_up(f.normal, up)) <= VERTICAL_EPSILON
        and (not f.panel_id or f.panel_id not in door_group_names)
    ]

    if not vertical_faces and not door_faces_by_group:
        return []

    plans: List[FloorPlan] = []

    for idx, floor_elev in enumerate(levels):
        cut_elev = floor_elev + CUT_HEIGHT

        # --- Corte de sección de muros ---
        raw_segments: List[Dict[str, Vec2]] = []

        for face in vertical_faces:
            ups = [get_up(v, up) for v in face.vertices]
            if min(ups) > cut_elev or max(ups) < cut_elev:
                continue

            segs = intersect_face_with_plane(face, cut_elev, up)
            for p1, p2 in segs:
                a = project_top_down(p1, up)
                b = project_top_down(p2, up)
                if abs(a.x - b.x) < 1e-6 and abs(a.y - b.y) < 1e-6:
                    continue
                raw_segments.append({"a": a, "b": b})

        # --- Puertas para este nivel ---
        raw_doors = extract_doors_for_level(door_faces_by_group, cut_elev, up)

        if not raw_segments and not raw_doors:
            continue

        # --- Bounding box ---
        all_pts = []
        for s in raw_segments:
            all_pts.extend([s["a"], s["b"]])
        for d in raw_doors:
            all_pts.extend([d.hinge, d.leaf_end])

        if not all_pts:
            continue

        min_x = min(p.x for p in all_pts)
        min_y = min(p.y for p in all_pts)
        max_x = max(p.x for p in all_pts)
        max_y = max(p.y for p in all_pts)

        shifted_segments = [
            {
                "a": Vec2(s["a"].x - min_x, s["a"].y - min_y),
                "b": Vec2(s["b"].x - min_x, s["b"].y - min_y),
            }
            for s in raw_segments
        ]

        shifted_doors = []
        for d in raw_doors:
            # Creamos una copia del objeto Door2D con coordenadas desplazadas
            d.hinge.x -= min_x
            d.hinge.y -= min_y
            d.leaf_end.x -= min_x
            d.leaf_end.y -= min_y
            shifted_doors.append(d)

        classified_segments = to_floor_plan_segments(shifted_segments)

        plans.append(
            FloorPlan(
                label=f"Piso {idx + 1}",
                segments=classified_segments,
                width=max_x - min_x,
                height=max_y - min_y,
                elevation=floor_elev,
                doors=shifted_doors if shifted_doors else None,
            )
        )

    return plans


def extract_floor_plans(
    faces: List[Face3D], up_axis: Optional[UpAxis] = None
) -> List[FloorPlan]:
    if not faces:
        return []

    if up_axis:
        return extract_with_axis(faces, up_axis)

    # Fallback: intentar con ambos ejes y elegir el que tenga más segmentos
    plans_z = extract_with_axis(faces, "Z")
    plans_y = extract_with_axis(faces, "Y")

    total_z = sum(len(p.segments) for p in plans_z)
    total_y = sum(len(p.segments) for p in plans_y)

    return plans_y if total_y > total_z else plans_z
