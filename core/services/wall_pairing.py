import math
from typing import List, Set, Optional
from dataclasses import dataclass, field

# Importamos nuestros tipos y helpers
from types import Face3D, Vec3, dot, normalize, vlength

# ============================================================================
# Wall Thickness Pairing
#
# Identifica pares de grupos de caras verticales paralelas-opuestas que representan
# las dos "pieles" (caras) del mismo muro físico, y mide la distancia perpendicular
# entre ellas (el grosor del muro).
# ============================================================================

COPLANAR_NORMAL_DOT = 0.985
COPLANAR_D_TOLERANCE = 0.05    # 5cm de tolerancia en el plano
OPPOSITE_NORMAL_DOT = -0.985   # dot product < esto => normales opuestas
MAX_WALL_THICKNESS = 1.0       # rango de búsqueda para el "gemelo" (1m)
LATERAL_OVERLAP_FACTOR = 0.5   # cuánto desplazamiento lateral se tolera

@dataclass
class TwinCandidate:
    """Una región plana parametrizada (usada para comprobar el emparejamiento de muros delgados)."""
    normal: Vec3    # normal unitaria
    d: float        # desplazamiento del plano = dot(normal, punto_en_plano)
    centroid: Vec3
    extent: float   # dimensión máxima del bounding box (presupuesto de solapamiento lateral)

def are_thin_twins(
    a: TwinCandidate,
    b: TwinCandidate,
    thickness_threshold: float
) -> Optional[float]:
    """
    Si dos regiones planas son "gemelas delgadas" (paralelas con normales opuestas,
    distancia perpendicular por debajo de `thickness_threshold`, y solapándose lateralmente),
    devuelve la distancia perpendicular (grosor) en metros.
    Devuelve `None` si no lo son.
    """
    ndot = (a.normal.x * b.normal.x +
            a.normal.y * b.normal.y +
            a.normal.z * b.normal.z)
            
    if ndot > OPPOSITE_NORMAL_DOT:
        return None

    distance = abs(
        a.normal.x * b.centroid.x +
        a.normal.y * b.centroid.y +
        a.normal.z * b.centroid.z -
        a.d
    )
    
    if distance < 1e-4 or distance > thickness_threshold:
        return None

    dx = b.centroid.x - a.centroid.x
    dy = b.centroid.y - a.centroid.y
    dz = b.centroid.z - a.centroid.z
    
    nc = dx * a.normal.x + dy * a.normal.y + dz * a.normal.z
    lx = dx - nc * a.normal.x
    ly = dy - nc * a.normal.y
    lz = dz - nc * a.normal.z
    
    lateral_dist = math.sqrt(lx * lx + ly * ly + lz * lz)

    budget = (a.extent + b.extent) * 0.5 * LATERAL_OVERLAP_FACTOR
    
    if lateral_dist <= budget:
        return distance
    return None

@dataclass
class VerticalCluster:
    normal: Vec3
    d: float
    face_indices: List[int]
    centroid: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    extent: float = 0.0

def cluster_verticals(
    faces: List[Face3D],
    vertical_indices: List[int]
) -> List[VerticalCluster]:
    
    clusters: List[VerticalCluster] = []

    for i in vertical_indices:
        face = faces[i]
        if vlength(face.normal) < 0.01:
            continue
            
        n = normalize(face.normal)
        d = dot(n, face.vertices[0])

        placed = False
        for cl in clusters:
            if (dot(cl.normal, n) > COPLANAR_NORMAL_DOT and
                abs(d - cl.d) < COPLANAR_D_TOLERANCE):
                cl.face_indices.append(i)
                placed = True
                break
                
        if not placed:
            clusters.append(VerticalCluster(
                normal=n,
                d=d,
                face_indices=[i]
            ))

    # Calcular centroides y extensiones para cada cluster
    for cl in clusters:
        sx, sy, sz = 0.0, 0.0, 0.0
        count = 0
        min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
        max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
        
        for fi in cl.face_indices:
            for v in faces[fi].vertices:
                sx += v.x
                sy += v.y
                sz += v.z
                count += 1
                
                if v.x < min_x: min_x = v.x
                if v.y < min_y: min_y = v.y
                if v.z < min_z: min_z = v.z
                if v.x > max_x: max_x = v.x
                if v.y > max_y: max_y = v.y
                if v.z > max_z: max_z = v.z
                
        if count > 0:
            cl.centroid = Vec3(sx / count, sy / count, sz / count)
        cl.extent = max(max_x - min_x, max_y - min_y, max_z - min_z)

    return clusters

def find_twin_thickness(
    cluster: VerticalCluster,
    all_clusters: List[VerticalCluster]
) -> Optional[float]:
    
    best: Optional[float] = None
    
    for other in all_clusters:
        if other is cluster:
            continue
            
        if dot(cluster.normal, other.normal) > OPPOSITE_NORMAL_DOT:
            continue

        # Distancia perpendicular del centroide del 'otro' al plano del 'cluster'
        distance = abs(dot(cluster.normal, other.centroid) - cluster.d)
        if distance < 1e-4 or distance > MAX_WALL_THICKNESS:
            continue

        # Solapamiento lateral: proyectar el delta del centroide sobre el plano del 'cluster'
        dx = other.centroid.x - cluster.centroid.x
        dy = other.centroid.y - cluster.centroid.y
        dz = other.centroid.z - cluster.centroid.z
        
        normal_comp = (dx * cluster.normal.x + 
                       dy * cluster.normal.y + 
                       dz * cluster.normal.z)
                       
        lx = dx - normal_comp * cluster.normal.x
        ly = dy - normal_comp * cluster.normal.y
        lz = dz - normal_comp * cluster.normal.z
        
        lateral_dist = math.sqrt(lx * lx + ly * ly + lz * lz)

        budget = (cluster.extent + other.extent) * 0.5 * LATERAL_OVERLAP_FACTOR
        if lateral_dist > budget:
            continue

        if best is None or distance < best:
            best = distance
            
    return best

def find_thin_wall_faces(
    faces: List[Face3D],
    vertical_indices: List[int],
    thickness_threshold: float
) -> Set[int]:
    """
    Devuelve el conjunto de índices de caras verticales cuyo grosor de muro emparejado
    está por debajo del umbral dado (es decir, "muros delgados").
    """
    clusters = cluster_verticals(faces, vertical_indices)
    thin_faces: Set[int] = set()
    
    for cluster in clusters:
        thickness = find_twin_thickness(cluster, clusters)
        if thickness is not None and thickness < thickness_threshold:
            for fi in cluster.face_indices:
                thin_faces.add(fi)
                
    return thin_faces