import math
from dataclasses import dataclass, field
from typing import List, Optional, Literal

# ============================================================================
# Tipos de geometría compartidos para el pipeline de procesamiento Eficiencia2D
# ============================================================================

@dataclass(slots=True)
class Vec3:
    x: float
    y: float
    z: float

@dataclass(slots=True)
class Vec2:
    x: float
    y: float

@dataclass
class Loop2D:
    """Un bucle cerrado de puntos 2D (límite exterior o contorno de polígono)."""
    vertices: List[Vec2]
    panel_id: Optional[str] = None

@dataclass(slots=True)
class Face3D:
    """Una cara 3D extraída del modelo fuente."""
    vertices: List[Vec3]
    normal: Vec3
    inner_loops: List[List[Vec3]]
    panel_id: Optional[str] = None

@dataclass(slots=True)
class IndexedFace3D(Face3D):
    """Face3D con los índices de vértices originales del OBJ para topología exacta."""
    vertex_indices: List[int] = field(default_factory=list)

def get_vertex_indices(face: Face3D) -> Optional[List[int]]:
    """Type guard: retorna los índices de vértices si están disponibles."""
    if isinstance(face, IndexedFace3D) and len(face.vertex_indices) == len(face.vertices):
        return face.vertex_indices
    return None

@dataclass
class Facade:
    """Una vista de elevación del edificio (N/S/E/O)."""
    label: str
    direction: Vec3
    polygons: List[Loop2D]
    width: float
    height: float

@dataclass
class FloorPlanSegment:
    """Un segmento de línea 2D en un plano de planta."""
    a: Vec2
    b: Vec2
    is_interior: bool

@dataclass
class Door2D:
    """Una puerta detectada, con su representación de arco de apertura en 2D."""
    hinge: Vec2         # Punto de pivote / bisagra (en coordenadas 2D, metros)
    width: float        # Ancho de la hoja de la puerta = radio del arco (metros)
    start_angle: float  # Ángulo de inicio del arco DXF en grados (CCW desde el eje +X)
    end_angle: float    # Ángulo final del arco DXF en grados (CCW desde el eje +X)
    leaf_end: Vec2      # Punto final de la hoja en la posición completamente abierta

@dataclass
class FloorPlan:
    """Una vista de corte horizontal en un nivel de piso específico."""
    label: str
    segments: List[FloorPlanSegment]
    width: float
    height: float
    elevation: float
    doors: Optional[List[Door2D]] = None

# Modos de descomposición para hojas de corte
DecompositionMode = Literal["detailed", "simple"]

@dataclass
class ElementFilter:
    """Qué categorías de elementos arquitectónicos incluir."""
    floors: bool
    walls: bool

@dataclass
class SheetConfig:
    """Dimensiones físicas de la hoja para el anidamiento (nesting) láser."""
    width_m: float
    height_m: float
    gap_m: float

@dataclass
class PipelineOptions:
    """Opciones que viajan a través de todo el pipeline."""
    scale_denom: float
    paper: str
    include_cutting_sheet: Optional[bool] = None
    decomposition_mode: Optional[DecompositionMode] = None
    element_filter: Optional[ElementFilter] = None
    sheet_config: Optional[SheetConfig] = None
    min_area_m2: Optional[float] = None

@dataclass
class OutputFile:
    """Un archivo generado listo para descarga."""
    name: str
    blob: bytes  # En Python usamos 'bytes' en lugar del 'Blob' del navegador web

# --- Helpers matemáticos de vectores ----------------------------------------

def vec3(x: float, y: float, z: float) -> Vec3:
    return Vec3(x, y, z)

def sub(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z)

def add(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z)

def scale_vec(v: Vec3, s: float) -> Vec3:
    return Vec3(v.x * s, v.y * s, v.z * s)

def dot(a: Vec3, b: Vec3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z

def cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    )

def vlength(v: Vec3) -> float:
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

def normalize(v: Vec3) -> Vec3:
    length = vlength(v)
    if length < 1e-12:
        return Vec3(0.0, 0.0, 0.0)
    return Vec3(v.x / length, v.y / length, v.z / length)
