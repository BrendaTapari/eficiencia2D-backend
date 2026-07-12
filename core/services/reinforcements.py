"""
Refuerzos estructurales: NERVIOS (cartelas) y COLUMNAS.

⚠️ Concepto (contrato v1): un refuerzo es un COMPONENTE NUEVO E INDEPENDIENTE. NO se
tocan las placas existentes (sin muescas en pared/piso). El refuerzo se agrega como una
pieza más al nesting / plancha / precio; el usuario lo pega a mano donde quiera. El
front sólo lo sugiere en el visor 3D. Para CORTAR alcanza el tamaño:
  - nervio: `size_m` (cateto del triángulo rectángulo).
  - columna: `size_m` (lado de sección) + `height_m` (alto), desplegada a plano.

`group_a/b`, `pos_t`, `position` son sólo pistas para el preview del front → aquí se
ignoran para el corte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from core.services.cutting_sheet import Edge2D
from core.services.types import Vec2

MIN_SIZE_M = 0.03
MAX_SIZE_M = 1.0
MIN_HEIGHT_M = 0.03
MAX_HEIGHT_M = 5.0
GLUE_TAB_M = 0.01  # pestaña de pegado de la columna


@dataclass
class ReinforcementPiece:
    """Pieza nueva de refuerzo lista para nestear (contorno 2D + pliegues opcionales)."""
    kind: str                 # "rib" | "column"
    ref_id: str
    width_m: float
    height_m: float
    edges: List[Edge2D] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Nervios (cartelas) = triángulo rectángulo plano
# ---------------------------------------------------------------------------


def parse_ribs(raw: Optional[List[dict]]) -> List[dict]:
    out: List[dict] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        try:
            size = float(item.get("size_m"))
        except (TypeError, ValueError):
            continue
        size = min(max(size, MIN_SIZE_M), MAX_SIZE_M)
        out.append({"id": str(item.get("id") or ""), "size_m": size})
    return out


def build_rib_piece(rib: dict) -> ReinforcementPiece:
    s = rib["size_m"]
    # Triángulo rectángulo de catetos s (se pega en la esquina; sin pestañas).
    pts = [(0.0, 0.0), (s, 0.0), (0.0, s)]
    edges = [Edge2D(a=Vec2(*pts[i]), b=Vec2(*pts[(i + 1) % len(pts)]))
             for i in range(len(pts))]
    return ReinforcementPiece(kind="rib", ref_id=rib["id"], width_m=s, height_m=s, edges=edges)


# ---------------------------------------------------------------------------
# Columnas = caja de sección cuadrada, desplegada a plano (tira de 4 caras)
# ---------------------------------------------------------------------------


def parse_columns(raw: Optional[List[dict]]) -> List[dict]:
    out: List[dict] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        try:
            size = float(item.get("size_m"))
            height = float(item.get("height_m"))
        except (TypeError, ValueError):
            continue
        size = min(max(size, MIN_SIZE_M), MAX_SIZE_M)
        height = min(max(height, MIN_HEIGHT_M), MAX_HEIGHT_M)
        out.append({"id": str(item.get("id") or ""), "size_m": size, "height_m": height})
    return out


def build_column_piece(col: dict) -> ReinforcementPiece:
    """Columna hueca de sección cuadrada `size` × alto `height`, DESPLEGADA a plano:
    tira de 4 caras (ancho 4·size) + pestaña de pegado, con líneas de pliegue (score)
    entre caras. Se corta plana y el usuario la pliega y pega en caja."""
    s = col["size_m"]
    h = col["height_m"]
    strip_w = 4.0 * s + GLUE_TAB_M
    # Contorno exterior (rectángulo).
    rect = [(0.0, 0.0), (strip_w, 0.0), (strip_w, h), (0.0, h)]
    edges = [Edge2D(a=Vec2(*rect[i]), b=Vec2(*rect[(i + 1) % len(rect)]))
             for i in range(len(rect))]
    # Líneas de pliegue (score → capa MARK_VECTOR/roja): entre las 4 caras y la pestaña.
    for k in (1, 2, 3, 4):
        x = s * k
        edges.append(Edge2D(a=Vec2(x, 0.0), b=Vec2(x, h), score=True))
    return ReinforcementPiece(kind="column", ref_id=col["id"], width_m=strip_w,
                              height_m=h, edges=edges)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def build_reinforcements(
    ribs: Optional[List[dict]], columns: Optional[List[dict]]
) -> List[ReinforcementPiece]:
    pieces: List[ReinforcementPiece] = []
    for rib in parse_ribs(ribs):
        pieces.append(build_rib_piece(rib))
    for col in parse_columns(columns):
        pieces.append(build_column_piece(col))
    return pieces
