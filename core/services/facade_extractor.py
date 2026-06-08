import math
from typing import List, Dict, Tuple, Literal, Set

# Importamos los tipos y helpers matemáticos
from core.services.types import (
    Face3D,
    Facade,
    Loop2D,
    Vec2,
    Vec3,
    cross,
    dot,
    normalize,
    sub,
    vlength,
)

# ============================================================================
# Facade Extractor
#
# Toma Face3D[] y produce Facade[] — una vista de elevación por lado del edificio.
#
# Algoritmo:
#   1. Autodetecta el eje vertical (Y-up vs Z-up).
#   2. Filtra caras verticales.
#   3. Agrupa caras por dirección horizontal → grupos N/S/E/W.
#   4. Para cada grupo, recolecta aristas únicas del contorno para producir siluetas limpias.
#   5. Normaliza coordenadas para que (0,0) sea abajo a la izquierda.
# ============================================================================

VERTICAL_EPSILON = 0.20
DIRECTION_CLUSTER_THRESHOLD = 0.70

UpAxis = Literal["Y", "Z"]


def get_up_component(normal: Vec3, up: UpAxis) -> float:
    return normal.y if up == "Y" else normal.z


def get_up_vec(up: UpAxis) -> Vec3:
    return Vec3(0.0, 1.0, 0.0) if up == "Y" else Vec3(0.0, 0.0, 1.0)


def horizontal_dir(normal: Vec3, up: UpAxis) -> Vec3:
    if up == "Y":
        h = Vec3(normal.x, 0.0, normal.z)
    else:
        h = Vec3(normal.x, normal.y, 0.0)
    return normalize(h)


def compute_facade_axes(direction: Vec3, up: UpAxis) -> Tuple[Vec3, Vec3]:
    world_up = get_up_vec(up)
    u_axis = normalize(cross(world_up, direction))
    v_axis = world_up
    return u_axis, v_axis


def cluster_by_direction(faces: List[Face3D], up: UpAxis) -> List[Dict[str, any]]:
    clusters: List[Dict[str, any]] = []

    for face in faces:
        h_dir = horizontal_dir(face.normal, up)
        if vlength(h_dir) < 0.01:
            continue

        placed = False
        for cluster in clusters:
            if dot(h_dir, cluster["dir"]) > DIRECTION_CLUSTER_THRESHOLD:
                cluster["faces"].append(face)
                placed = True
                break

        if not placed:
            clusters.append({"dir": h_dir, "faces": [face]})

    return clusters


def direction_label(direction: Vec3, up: UpAxis) -> str:
    if up == "Y":
        angle = math.atan2(direction.x, direction.z)
    else:
        angle = math.atan2(direction.x, direction.y)

    deg = (
        (math.atan2(math.sin(angle), math.cos(angle)) * 180.0) / math.pi + 360.0
    ) % 360.0

    if deg < 45 or deg >= 315:
        return "Fachada Norte"
    if deg < 135:
        return "Fachada Este"
    if deg < 225:
        return "Fachada Sur"
    return "Fachada Oeste"


def round_coord(v: float) -> float:
    """Redondea una coordenada a una precisión fija para usar como clave de arista."""
    return round(v * 10000.0) / 10000.0


def edge_key(ax: float, ay: float, bx: float, by: float) -> str:
    a = f"{round_coord(ax)},{round_coord(ay)}"
    b = f"{round_coord(bx)},{round_coord(by)}"
    return f"{a}|{b}" if a < b else f"{b}|{a}"


def extract_with_axis(faces: List[Face3D], up: UpAxis) -> List[Facade]:
    # 1. Filtrar caras verticales.
    vertical_faces: List[Face3D] = [
        face
        for face in faces
        if abs(get_up_component(face.normal, up)) <= VERTICAL_EPSILON
    ]
    if not vertical_faces:
        return []

    # 2. Agrupar por dirección horizontal.
    clusters = cluster_by_direction(vertical_faces, up)

    # 3. Construir una Facade por cluster, manteniendo solo las caras más frontales.
    facades: List[Facade] = []

    for cluster in clusters:
        dir_vec: Vec3 = cluster["dir"]
        cluster_faces: List[Face3D] = cluster["faces"]

        depths_and_faces = []
        for face in cluster_faces:
            sum_depth = sum(dot(v, dir_vec) for v in face.vertices)
            depths_and_faces.append(
                {"depth": sum_depth / len(face.vertices), "face": face}
            )

        max_depth = max(d["depth"] for d in depths_and_faces)
        min_depth = min(d["depth"] for d in depths_and_faces)
        depth_range = max_depth - min_depth

        # Mantener caras dentro de una tolerancia de la superficie más frontal.
        depth_cutoff = min_depth if depth_range < 0.5 else max_depth - 0.5
        front_faces = [
            d["face"] for d in depths_and_faces if d["depth"] >= depth_cutoff
        ]

        if not front_faces:
            continue

        u_axis, v_axis = compute_facade_axes(dir_vec, up)

        # Proyectar caras frontales, recolectando aristas únicas para eliminar
        # líneas de triangulación internas.
        edge_counts: Dict[str, Dict[str, float]] = {}

        for face in front_faces:
            pts = [Vec2(dot(v, u_axis), dot(v, v_axis)) for v in face.vertices]

            # Recorrer aristas de este polígono
            n_pts = len(pts)
            for i in range(n_pts):
                j = (i + 1) % n_pts
                key = edge_key(pts[i].x, pts[i].y, pts[j].x, pts[j].y)

                if key in edge_counts:
                    del edge_counts[key]  # arista compartida → remover
                else:
                    edge_counts[key] = {
                        "ax": pts[i].x,
                        "ay": pts[i].y,
                        "bx": pts[j].x,
                        "by": pts[j].y,
                    }

        if not edge_counts:
            continue

        # Recolectar todos los puntos para el bounding box
        min_u = float("inf")
        max_u = float("-inf")
        min_v = float("inf")
        max_v = float("-inf")
        edges = []

        for edge in edge_counts.values():
            edges.append(edge)
            min_u = min(min_u, edge["ax"], edge["bx"])
            max_u = max(max_u, edge["ax"], edge["bx"])
            min_v = min(min_v, edge["ay"], edge["by"])
            max_v = max(max_v, edge["ay"], edge["by"])

        width = max_u - min_u
        height = max_v - min_v
        if width < 0.01 or height < 0.01:
            continue

        # Normalizar a origen (0,0) y construir segmentos Loop2D como líneas de 2 vértices.
        polygons = [
            Loop2D(
                vertices=[
                    Vec2(e["ax"] - min_u, e["ay"] - min_v),
                    Vec2(e["bx"] - min_u, e["by"] - min_v),
                ]
            )
            for e in edges
        ]

        label = direction_label(dir_vec, up)
        existing_labels = {f.label for f in facades}

        if label in existing_labels:
            n = 2
            while f"{label} {n}" in existing_labels:
                n += 1
            label = f"{label} {n}"

        facades.append(
            Facade(
                label=label,
                direction=dir_vec,
                polygons=polygons,
                width=width,
                height=height,
            )
        )

    # Ordenar fachadas
    order = {"Norte": 0, "Este": 1, "Sur": 2, "Oeste": 3}

    def sort_key(f: Facade) -> int:
        direction_word = f.label.split(" ")[-1]
        return order.get(direction_word, 99)

    facades.sort(key=sort_key)
    return facades


def detect_up_axis(faces: List[Face3D]) -> UpAxis:
    if not faces:
        return "Z"

    # --- Heurística 1: análisis de normales de caras ---
    y_area = 0.0
    z_area = 0.0

    for face in faces:
        n = face.normal
        verts = face.vertices
        if len(verts) < 3:
            continue

        e1 = sub(verts[1], verts[0])
        e2 = sub(verts[2], verts[0])
        cx = cross(e1, e2)
        area = vlength(cx) * 0.5
        if area < 1e-10:
            continue

        abs_y = abs(n.y)
        abs_z = abs(n.z)

        if abs_y > 0.9:
            y_area += area
        if abs_z > 0.9:
            z_area += area

    total_area = y_area + z_area
    if total_area > 0:
        diff = abs(y_area - z_area) / total_area
        if diff > 0.3:
            return "Y" if y_area > z_area else "Z"

    # --- Heurística 2: ratio del bounding-box ---
    min_y, max_y = float("inf"), float("-inf")
    min_z, max_z = float("inf"), float("-inf")

    limit = min(len(faces), 500)
    for i in range(limit):
        for v in faces[i].vertices:
            if v.y < min_y:
                min_y = v.y
            if v.y > max_y:
                max_y = v.y
            if v.z < min_z:
                min_z = v.z
            if v.z > max_z:
                max_z = v.z

    range_y = max_y - min_y
    range_z = max_z - min_z

    if range_z > 0 and (range_y / range_z) > 2.0:
        return "Z"
    if range_y > 0 and (range_z / range_y) > 2.0:
        return "Y"

    # --- Heurística 3: default ---
    return "Z"


def extract_facades(faces: List[Face3D], up_axis: UpAxis = None) -> List[Facade]:
    if not faces:
        return []

    up = up_axis if up_axis else detect_up_axis(faces)
    return extract_with_axis(faces, up)
