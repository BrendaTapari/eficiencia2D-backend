from typing import List, Tuple, Literal, Optional
from dataclasses import dataclass

# Importamos nuestros tipos (asegurate de tenerlos en core/types.py y core/services/...)
from core.services.types import Face3D, Vec3, normalize
from core.group_classifier import GeometryGroup
from core.services.joint_detector import Joint

# ============================================================================
# Joint Topology Classifier
#
# Clasifica una unión muro-muro por DÓNDE se asienta la arista compartida en cada muro:
#   - L : ambos muros se encuentran en un EXTREMO (esquina exterior — visualmente prominente)
#   - T : un muro termina en el MEDIO del otro (el "tallo" de la T termina)
#   - X : ambos muros se cruzan en sus MEDIOS (cruce interior)
# ============================================================================

JointTopology = Literal["L", "T", "X", "unknown"]

@dataclass
class JointTopologyInfo:
    topology: JointTopology
    a_at_end: bool  # La arista compartida cae en el EXTREMO del muro A (vs. su medio)
    b_at_end: bool  # La arista compartida cae en el EXTREMO del muro B (vs. su medio)

# Una posición normalizada en [0,1] más cerca de un extremo que esto es un "extremo".
END_FRAC = 0.2
# Una relación de grosor menor a esta cuenta como una diferencia "grande" de grosor.
CRITICAL_RATIO = 0.6

def running_axis(normal: Vec3) -> Vec3:
    """
    Eje de recorrido horizontal de un muro vertical — perpendicular a su normal
    (horizontal) en el plano XZ. Un muro orientado Este/Oeste corre Norte-Sur y viceversa.
    """
    return normalize(Vec3(-normal.z, 0.0, normal.x))

def proj_onto(p: Vec3, axis: Vec3) -> float:
    """Proyecta un punto sobre un eje (horizontal) que pasa por el origen."""
    return p.x * axis.x + p.y * axis.y + p.z * axis.z

def span_along(faces: List[Face3D], face_indices: List[int], axis: Vec3) -> Tuple[float, float]:
    """Proyección min/max de los vértices de la huella de un muro sobre un eje de recorrido."""
    min_val = float('inf')
    max_val = float('-inf')
    
    for fi in face_indices:
        if fi < 0 or fi >= len(faces):
            continue
        face = faces[fi]
        for v in face.vertices:
            t = proj_onto(v, axis)
            if t < min_val: min_val = t
            if t > max_val: max_val = t
            
    return min_val, max_val

def wall_run_length(group: GeometryGroup, faces: List[Face3D]) -> float:
    axis = running_axis(group.representative_normal)
    min_val, max_val = span_along(faces, group.face_indices, axis)
    
    if min_val == float('inf') or max_val == float('-inf'):
        return 0.0
    return max_val - min_val

def edge_at_end(joint: Joint, group: GeometryGroup, faces: List[Face3D]) -> Optional[bool]:
    """Si el punto medio de la arista compartida se encuentra en un EXTREMO de la extensión del muro."""
    axis = running_axis(group.representative_normal)
    min_val, max_val = span_along(faces, group.face_indices, axis)
    
    if min_val == float('inf') or max_val == float('-inf'):
        return None
        
    length = max_val - min_val
    if length < 1e-6:
        return None  # footprint degenerado
        
    t = (proj_onto(joint.edge_mid, axis) - min_val) / length
    return t < END_FRAC or t > (1.0 - END_FRAC)

def classify_joint_topology(
    joint: Joint, g_a: GeometryGroup, g_b: GeometryGroup, faces: List[Face3D]
) -> JointTopologyInfo:
    """
    Clasifica una unión muro-muro como L / T / X utilizando la posición de la arista
    compartida dentro de la huella horizontal de cada muro.
    """
    a_end = edge_at_end(joint, g_a, faces)
    b_end = edge_at_end(joint, g_b, faces)

    if a_end is None or b_end is None:
        return JointTopologyInfo(
            topology="unknown", 
            a_at_end=bool(a_end), 
            b_at_end=bool(b_end)
        )

    if a_end and b_end:
        topology = "L"
    elif a_end or b_end:
        topology = "T"
    else:
        topology = "X"

    return JointTopologyInfo(topology=topology, a_at_end=a_end, b_at_end=b_end)

def is_critical_joint(topology: JointTopology, t_a: float, t_b: float) -> bool:
    """
    Una unión es "crítica" (vale la pena mostrarla al usuario) cuando su resolución
    afecta visiblemente el modelo ensamblado: esquinas exteriores en L, o uniones 
    donde un muro es mucho más grueso que el otro.
    """
    if topology == "L":
        return True
        
    lo = min(t_a, t_b)
    hi = max(t_a, t_b)
    
    if hi > 0.001 and (lo / hi) < CRITICAL_RATIO:
        return True
        
    return False