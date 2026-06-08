import math
import re
from typing import List, Optional, Dict, Literal

# Importamos los tipos y helpers matemáticos que creamos en types.py
from core.services.types import Face3D, Vec2, Vec3, Door2D, sub, cross

# ============================================================================
# Constantes
# ============================================================================

DOOR_NAME_PATTERN = re.compile(r"puerta|door|porta", re.IGNORECASE)
HINGE_RIGHT_PATTERN = re.compile(r"_der|_right|_R\b", re.IGNORECASE)

MAX_DOOR_THICKNESS = 0.30  # metros
MIN_DOOR_WIDTH = 0.40  # metros
MAX_DOOR_WIDTH = 3.0  # metros

UpAxis = Literal["Y", "Z"]

# ============================================================================
# Helpers
# ============================================================================


def is_door_group(name: str) -> bool:
    """Verifica si el nombre de un grupo OBJ parece ser un componente de puerta."""
    return bool(DOOR_NAME_PATTERN.search(name))


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


# ============================================================================
# Análisis de Puerta Individual
# ============================================================================


def analyze_door_group(
    group_name: str, faces: List[Face3D], cut_elev: float, up: UpAxis
) -> Optional[Door2D]:
    """
    Analiza un grupo de caras que representan una sola puerta y devuelve su
    representación 2D en planta, o `None` si la geometría no parece una puerta.
    """
    if not faces:
        return None

    # --- ¿La puerta cruza la elevación de corte? ---
    all_verts = [v for face in faces for v in face.vertices]
    elevations = [get_up(v, up) for v in all_verts]

    min_elev = min(elevations)
    max_elev = max(elevations)

    if min_elev > cut_elev or max_elev < cut_elev:
        return None

    # --- Proyectar todos los vértices al plano base ---
    pts_2d = [project_top_down(v, up) for v in all_verts]
    xs = [p.x for p in pts_2d]
    ys = [p.y for p in pts_2d]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    dx = max_x - min_x
    dy = max_y - min_y

    # --- Dirección del muro (eje largo) vs grosor (eje corto) ---
    if dx >= dy:
        # La puerta corre a lo largo de X
        width = dx
        thickness = dy
        mid_y = (min_y + max_y) / 2.0
        hinge_a = Vec2(min_x, mid_y)
        hinge_b = Vec2(max_x, mid_y)
    else:
        # La puerta corre a lo largo de Y
        width = dy
        thickness = dx
        mid_x = (min_x + max_x) / 2.0
        hinge_a = Vec2(mid_x, min_y)
        hinge_b = Vec2(mid_x, max_y)

    # --- Validación de dimensiones ---
    if thickness > MAX_DOOR_THICKNESS:
        return None
    if width < MIN_DOOR_WIDTH or width > MAX_DOOR_WIDTH:
        return None

    # --- Dirección de apertura desde la normal de la cara vertical más grande ---
    best_area = 0.0
    best_normal = None
    for face in faces:
        up_comp = abs(get_up(face.normal, up))
        if up_comp > 0.5:
            continue  # Saltar caras horizontales (piso/techo del marco)

        area = face_area(face)
        if area > best_area:
            best_area = area
            best_normal = face.normal

    if not best_normal:
        return None

    # Proyectar normal a 2D y normalizar
    raw_swing = project_top_down(best_normal, up)
    swing_len = math.sqrt(raw_swing.x**2 + raw_swing.y**2)
    if swing_len < 0.01:
        return None

    swing_dir = Vec2(raw_swing.x / swing_len, raw_swing.y / swing_len)

    # --- Selección de bisagra ---
    # Por defecto: hinge_a (extremo con coordenada mínima).
    # Sobrescribir si el nombre del grupo indica bisagra "derecha".
    hinge_right = bool(HINGE_RIGHT_PATTERN.search(group_name))
    hinge = hinge_b if hinge_right else hinge_a
    free_end = hinge_a if hinge_right else hinge_b

    # --- Calcular ángulos de arco DXF ---
    to_free = Vec2(free_end.x - hinge.x, free_end.y - hinge.y)
    wall_angle_deg = math.degrees(math.atan2(to_free.y, to_free.x))
    swing_angle_deg = math.degrees(math.atan2(swing_dir.y, swing_dir.x))

    # Producto cruzado 2D nos da el sentido de rotación
    cross_val = to_free.x * swing_dir.y - to_free.y * swing_dir.x

    if cross_val >= 0:
        # swing_dir es antihorario desde wall_dir -> el arco va antihorario de muro a swing
        start_angle = wall_angle_deg
        end_angle = swing_angle_deg
    else:
        # swing_dir es horario desde wall_dir -> invertir para que el arco DXF sea correcto
        start_angle = swing_angle_deg
        end_angle = wall_angle_deg

    # Normalizar a [0, 360)
    start_angle = (start_angle % 360 + 360) % 360
    end_angle = (end_angle % 360 + 360) % 360

    # --- Punto final de la hoja (posición completamente abierta) ---
    swing_rad = math.radians(swing_angle_deg)
    leaf_end = Vec2(
        hinge.x + width * math.cos(swing_rad), hinge.y + width * math.sin(swing_rad)
    )

    return Door2D(hinge, width, start_angle, end_angle, leaf_end)


# ============================================================================
# Extracción por Lote para un nivel de piso
# ============================================================================


def extract_doors_for_level(
    door_faces_by_group: Dict[str, List[Face3D]], cut_elev: float, up: UpAxis
) -> List[Door2D]:
    """
    Extrae entidades `Door2D` para una elevación de corte transversal dada.
    """
    doors = []
    for name, group_faces in door_faces_by_group.items():
        door = analyze_door_group(name, group_faces, cut_elev, up)
        if door:
            doors.append(door)

    return doors
