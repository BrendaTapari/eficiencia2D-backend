import math
from typing import List, Dict, Literal, Set, Optional
from dataclasses import dataclass

# Importamos nuestros tipos y helpers
from types import Face3D, Vec3, ElementFilter, sub, cross, vlength

# ============================================================================
# Geometry Classifier
#
# Clasifica las caras 3D en categorías arquitectónicas usando únicamente geometría
# (sin depender de nombres de grupos OBJ).
#
# Categorías:
#   floor   — pisos y techos (caras horizontales grandes)
#   wall    — todos los muros verticales
#   discard — zócalos, tiras de borde, caras diminutas
# ============================================================================

FaceCategory = Literal["floor", "wall", "discard"]


@dataclass
class ClassifiedFace:
    face: Face3D
    category: FaceCategory
    area: float
    centroid: Vec3


def face_area(f: Face3D) -> float:
    """Calcula el área de una Face3D mediante triangulación en abanico desde el vértice 0."""
    verts = f.vertices
    if len(verts) < 3:
        return 0.0

    area = 0.0
    for i in range(1, len(verts) - 1):
        e1 = sub(verts[i], verts[0])
        e2 = sub(verts[i + 1], verts[0])
        area += vlength(cross(e1, e2)) * 0.5
    return area


def face_centroid(f: Face3D) -> Vec3:
    """Calcula el centroide de una Face3D."""
    verts = f.vertices
    if not verts:
        return Vec3(0.0, 0.0, 0.0)

    sx, sy, sz = 0.0, 0.0, 0.0
    for v in verts:
        sx += v.x
        sy += v.y
        sz += v.z

    n = len(verts)
    return Vec3(sx / n, sy / n, sz / n)


DEFAULT_ELEMENT_FILTER = ElementFilter(floors=True, walls=True)


@dataclass
class FaceInfo:
    face: Face3D
    area: float
    centroid: Vec3
    orientation: Literal["horizontal", "vertical", "inclined"]


def classify_and_filter(
    faces: List[Face3D], filter_options: ElementFilter = DEFAULT_ELEMENT_FILTER
) -> List[Face3D]:
    """
    Clasifica todas las caras en el modelo y devuelve solo aquellas que coinciden con el filtro.
    Asume que el eje Y es el vertical (Up-Axis = Y).
    """
    if not faces:
        return []

    # --- Calcular área, centroide y clasificar orientación para cada cara ---
    HORIZONTAL_THRESHOLD = (
        0.98  # |normal.y| > esto → horizontal (rechaza techos de ~12°)
    )
    VERTICAL_THRESHOLD = 0.5  # |normal.y| < esto → vertical
    HEIGHT_BAND = 0.05  # tolerancia de 5cm para agrupar por nivel
    MIN_AREA = 1e-6  # saltar caras degeneradas

    infos: List[FaceInfo] = []

    for face in faces:
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
                face=face,
                area=area,
                centroid=face_centroid(face),
                orientation=orientation,
            )
        )

    # --- Verificar si el modelo es demasiado pequeño (< 1m en algún eje) ---
    min_x, max_x = float("inf"), float("-inf")
    min_z, max_z = float("inf"), float("-inf")

    for fi in infos:
        for v in fi.face.vertices:
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

    # Si el modelo es muy pequeño, saltar clasificación y devolver todo tal cual
    if range_x < 1.0 or range_z < 1.0:
        return faces

    # --- Paso 1: Histograma de área acumulada por banda de altura Y ---
    horizontals = [fi for fi in infos if fi.orientation == "horizontal"]

    # Agrupar por elevación (centroide Y) en bandas de ±5cm.
    level_groups: Dict[float, List[FaceInfo]] = {}

    for fi in horizontals:
        h = fi.centroid.y
        found_key = None
        for key in level_groups.keys():
            if abs(key - h) < HEIGHT_BAND:
                found_key = key
                break

        key = found_key if found_key is not None else h
        if key not in level_groups:
            level_groups[key] = []
        level_groups[key].append(fi)

    # Acumular área total por banda de altura
    band_area: Dict[float, float] = {}
    for key, group in level_groups.items():
        band_area[key] = sum(fi.area for fi in group)

    # --- Paso 2: Detectar niveles reales como picos del histograma ---
    max_band_area = max(band_area.values()) if band_area else 0.0

    # Una banda es un nivel real si su área acumulada excede este umbral
    LEVEL_THRESHOLD = max(1.0, 0.03 * max_band_area)

    real_level_keys: Set[float] = {
        key for key, area in band_area.items() if area >= LEVEL_THRESHOLD
    }

    # --- Paso 3: Clasificar cada cara horizontal ---
    # Usamos los id() de Python en lugar de los objetos mismos para Sets seguros
    floor_face_ids: Set[int] = set()
    discard_face_ids: Set[int] = set()

    for key, group in level_groups.items():
        if key in real_level_keys:
            for fi in group:
                floor_face_ids.add(id(fi.face))
        else:
            for fi in group:
                discard_face_ids.add(id(fi.face))

    # Si no se detectaron niveles reales, no descartar ninguna cara horizontal
    if not real_level_keys:
        for fi in horizontals:
            floor_face_ids.add(id(fi.face))
            discard_face_ids.discard(id(fi.face))

    # --- Clasificar caras verticales — todas se vuelven "muros" ---
    wall_face_ids: Set[int] = {
        id(fi.face) for fi in infos if fi.orientation == "vertical"
    }

    # --- Caras inclinadas → se descartan ---
    for fi in infos:
        if fi.orientation == "inclined":
            discard_face_ids.add(id(fi.face))

    # --- Aplicar filtro y retornar ---
    result: List[Face3D] = []

    for fi in infos:
        face_id = id(fi.face)

        if face_id in discard_face_ids:
            continue

        if face_id in floor_face_ids and not filter_options.floors:
            continue

        if face_id in wall_face_ids and not filter_options.walls:
            continue

        result.append(fi.face)

    return result
