"""
Patrones de flexión para superficies curvas (kerf bending + auxéticos).

Genera la geometría REAL de corte que permite doblar una plancha plana:
- **kerf**: filas/columnas de ranuras paralelas interrumpidas por ligamentos (puentes).
  La distancia entre columnas (`spacing_m`) fija el radio de doblez. Flexión en un eje.
- **auxético** (`rotating` / `reentrant` / `chiral`): teselado de celdas con ligamentos
  que, al expandirse, absorben doble curvatura. `spacing_m` = pitch de celda.

Coordenadas en METROS, en el marco local del panel de `project_faces_to_2d`
(u horizontal 0..width_m, v vertical 0..height_m), el mismo marco que usa `user_cuts`.
El módulo devuelve `Edge2D(flex=True)`; el llamador las agrega a las aristas del panel.
La geometría exacta de cada patrón la define el backend (el front sólo nombra el método).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Dict, List, Literal, Optional, Tuple

log = logging.getLogger(__name__)

# Tope de primitivas (ranuras/celdas) por panel. Evita que un panel grande con
# spacing chico genere cientos de miles de aristas (DXF/preview inusables). Si se
# excede, se sube el spacing efectivo (clamp) — respeta "descartar valores degenerados".
MAX_PRIMITIVES = 2500

from shapely import affinity
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
)
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union
from shapely.validation import make_valid

from core.services.cutting_sheet import Edge2D
from core.services.types import Vec2

FlexMethod = Literal["kerf", "auxetic_rotating", "auxetic_reentrant", "auxetic_chiral"]

# Límites de material (metros). Evitan patrones degenerados que romperían la plancha.
MIN_SPACING = 0.003        # separación mínima entre columnas / pitch de celda
MAX_SPACING = 0.08         # separación máxima razonable
MIN_LIGAMENT = 0.0008      # puente mínimo sin cortar
DEFAULT_LIGAMENT = 0.003
DEFAULT_KERF_WIDTH = 0.0015
MIN_KERF_WIDTH = 0.0004
EDGE_MARGIN = 0.004        # margen sin patrón contra el borde del panel


@dataclass
class FlexSpec:
    group_id: int
    method: FlexMethod
    spacing_m: float
    ligament_m: float = DEFAULT_LIGAMENT
    kerf_width_m: float = DEFAULT_KERF_WIDTH
    axis_deg: float = 0.0


_METHODS = ("kerf", "auxetic_rotating", "auxetic_reentrant", "auxetic_chiral")


def parse_flex(raw: Optional[List[dict]]) -> List[FlexSpec]:
    """Valida la lista cruda de specs (un objeto por grupo). Descarta inválidos."""
    if not raw:
        return []
    out: List[FlexSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        gid = item.get("group_id")
        method = item.get("method")
        if not isinstance(gid, int) or method not in _METHODS:
            continue
        try:
            spacing = float(item.get("spacing_m"))
        except (TypeError, ValueError):
            continue
        if not (spacing > 0):
            continue
        spacing = min(max(spacing, MIN_SPACING), MAX_SPACING)

        ligament = item.get("ligament_m")
        try:
            ligament = float(ligament) if ligament is not None else DEFAULT_LIGAMENT
        except (TypeError, ValueError):
            ligament = DEFAULT_LIGAMENT
        ligament = max(MIN_LIGAMENT, min(ligament, spacing * 0.8))

        kerf = item.get("kerf_width_m")
        try:
            kerf = float(kerf) if kerf is not None else DEFAULT_KERF_WIDTH
        except (TypeError, ValueError):
            kerf = DEFAULT_KERF_WIDTH
        kerf = max(MIN_KERF_WIDTH, min(kerf, spacing * 0.5))

        axis = item.get("axis_deg")
        try:
            axis = float(axis) if axis is not None else 0.0
        except (TypeError, ValueError):
            axis = 0.0

        out.append(
            FlexSpec(
                group_id=gid,
                method=method,  # type: ignore[arg-type]
                spacing_m=spacing,
                ligament_m=ligament,
                kerf_width_m=kerf,
                axis_deg=axis,
            )
        )
    return out


def build_flex_by_group(
    specs: List[FlexSpec],
    merge_target: Optional[Dict[int, int]] = None,
) -> Dict[int, FlexSpec]:
    """Un patrón por grupo (si llegan varios, gana el último). Remapea grupos fusionados."""
    merge_target = merge_target or {}
    by_group: Dict[int, FlexSpec] = {}
    for spec in specs:
        gid = merge_target.get(spec.group_id, spec.group_id)
        by_group[gid] = spec
    return by_group


# ---------------------------------------------------------------------------
# Polígono del panel (para recortar el patrón al contorno real, incluidos huecos)
# ---------------------------------------------------------------------------

_SNAP = 10_000


def _snap(v: float) -> float:
    return round(v * _SNAP) / _SNAP


def _edges_to_rings(edges: List[Edge2D]) -> List[List[Tuple[float, float]]]:
    """Reconstruye anillos cerrados a partir de aristas sueltas (contorno + huecos)."""
    adj: Dict[Tuple[float, float], List[Tuple[float, float]]] = {}

    def key(x: float, y: float) -> Tuple[float, float]:
        return (_snap(x), _snap(y))

    for e in edges:
        if getattr(e, "score", False) or getattr(e, "joint", False):
            continue
        ka, kb = key(e.a.x, e.a.y), key(e.b.x, e.b.y)
        if ka == kb:
            continue
        adj.setdefault(ka, []).append(kb)
        adj.setdefault(kb, []).append(ka)

    used: set = set()
    rings: List[List[Tuple[float, float]]] = []
    for start in list(adj.keys()):
        for first in adj[start]:
            ekey = (start, first) if start < first else (first, start)
            if ekey in used:
                continue
            ring = [start]
            used.add(ekey)
            prev, cur = start, first
            guard = 0
            while cur != start and guard < len(adj) * 4 + 8:
                guard += 1
                ring.append(cur)
                nxt = None
                for cand in adj.get(cur, []):
                    if cand == prev:
                        continue
                    ek = (cur, cand) if cur < cand else (cand, cur)
                    if ek in used:
                        continue
                    nxt = cand
                    used.add(ek)
                    break
                if nxt is None:
                    break
                prev, cur = cur, nxt
            if len(ring) >= 3 and cur == start:
                rings.append(ring)
    return rings


def _bbox(width_m: float, height_m: float) -> Polygon:
    return Polygon([(0, 0), (width_m, 0), (width_m, height_m), (0, height_m)])


def _panel_polygon(width_m: float, height_m: float, edges: List[Edge2D]):
    """Polígono(s) del MATERIAL real del panel: soporta VARIAS piezas separadas
    (paredes distintas de un grupo fusionado) y aberturas (ventanas/puertas).

    El patrón de flexión se recorta a este material → nunca cae en los huecos entre
    paredes ni sobre las aberturas ("exclusivamente dentro de los bordes de las paredes").

    Reconstrucción ROBUSTA con shapely: se nodan los segmentos del contorno
    (`unary_union`) y se arman las caras mínimas (`polygonize`); cada cara es material
    si está a profundidad IMPAR (regla par-impar) respecto de las caras que la contienen.
    Esto tolera contornos escalonados/en L, colineales y auto-contactos que el trazado
    manual de anillos podía no cerrar (cayendo antes al bbox y desbordando el patrón).
    """
    segs = []
    for e in edges:
        if getattr(e, "score", False) or getattr(e, "joint", False) or getattr(e, "flex", False):
            continue
        if (e.a.x - e.b.x) ** 2 + (e.a.y - e.b.y) ** 2 > 1e-12:
            segs.append(LineString([(e.a.x, e.a.y), (e.b.x, e.b.y)]))
    if not segs:
        return _bbox(width_m, height_m)

    try:
        faces = list(polygonize(unary_union(segs)))
    except Exception:
        faces = []
    faces = [f for f in faces if not f.is_empty and f.area > 1e-6]
    if not faces:
        return _bbox(width_m, height_m)

    # Anillos exteriores de todas las caras → clasificación par-impar por contención.
    ring_polys = []
    for f in faces:
        try:
            ring_polys.append(Polygon(f.exterior.coords))
        except Exception:
            ring_polys.append(f)

    material_faces = []
    for f in faces:
        rp = f.representative_point()
        depth = sum(1 for rpoly in ring_polys if rpoly.contains(rp))
        if depth % 2 == 1:            # profundidad impar = material (las caras ya traen sus huecos)
            material_faces.append(f)
    if not material_faces:
        return _bbox(width_m, height_m)

    material = unary_union(material_faces)
    return material if not material.is_empty else _bbox(width_m, height_m)


# ---------------------------------------------------------------------------
# Generadores de patrón (en el bbox [0,width]x[0,height], sin rotar)
# ---------------------------------------------------------------------------


def _kerf_slots(width_m: float, height_m: float, spec: FlexSpec) -> List[BaseGeometry]:
    """Living hinge por REMOCIÓN de ranuras rectangulares (huecos), peine interdigitado.

    El kerf bending real NO es un trazo: se REMUEVE material. Cada columna es un
    rectángulo abierto (hueco) que se corta y cae; el diente (`ligament`) entre columnas
    mantiene la integridad y el puente en un extremo (alternado por columna) forma la
    bisagra viva tipo acordeón. El sheet flexa alrededor del eje paralelo a las ranuras.

    - `pitch = spacing` (distancia entre columnas; única variable del usuario).
    - `slot_w = spacing − ligament` → al AUMENTAR spacing, el hueco se ensancha (se
      remueve más material). Coincide con la referencia física (pitch 3 mm, diente
      1.8 mm ⇒ ranura 1.2 mm).
    - Largo de ranura = alto de banda − puente; puente ≈12% del largo, alternando el
      extremo (columna par: puente arriba; impar: puente abajo).
    """
    spacing = spec.spacing_m
    ligament = spec.ligament_m
    slot_w = max(spacing - ligament, spacing * 0.25)  # ancho del hueco (crece con spacing)

    band = height_m - 2.0 * EDGE_MARGIN
    if band <= 0 or slot_w <= 0:
        return []
    bridge = min(max(band * 0.12, 0.002), band * 0.4)  # puente ~12% del largo
    slot_len = band - bridge
    if slot_len <= slot_w:
        return []

    slots: List[BaseGeometry] = []
    u = EDGE_MARGIN + spacing / 2.0
    col = 0
    while u <= width_m - EDGE_MARGIN:
        # Puente alternado: columna par arriba (ranura desde el margen inferior),
        # impar abajo (ranura hasta el margen superior). Peine interdigitado.
        if col % 2 == 0:
            v0 = EDGE_MARGIN
            v1 = EDGE_MARGIN + slot_len
        else:
            v1 = height_m - EDGE_MARGIN
            v0 = v1 - slot_len
        slots.append(
            Polygon(
                [
                    (u - slot_w / 2.0, v0),
                    (u + slot_w / 2.0, v0),
                    (u + slot_w / 2.0, v1),
                    (u - slot_w / 2.0, v1),
                ]
            )
        )
        u += spacing
        col += 1
    return slots


def _auxetic_rotating(width_m: float, height_m: float, spec: FlexSpec) -> List[BaseGeometry]:
    """Cuadrados rotatorios por REMOCIÓN de celdas: huecos rómbicos (cuadrados a 45°) en
    grilla de pitch `spacing`, dejando ligamentos en las esquinas que actúan de bisagra.

    Al remover los rombos, el material restante forma cuadrados unidos por las esquinas
    que rotan al traccionar → Poisson negativo (auxético). No son líneas: son huecos.
    """
    p = spec.spacing_m
    lig = spec.ligament_m
    half_diag = max((p - lig) / 2.0, p * 0.15)  # media diagonal del rombo removido
    holes: List[BaseGeometry] = []
    nx = int((width_m - 2 * EDGE_MARGIN) // p)
    ny = int((height_m - 2 * EDGE_MARGIN) // p)
    if nx < 1 or ny < 1:
        return holes
    ox = (width_m - nx * p) / 2.0
    oy = (height_m - ny * p) / 2.0

    for j in range(ny):
        for i in range(nx):
            cx = ox + i * p + p / 2.0
            cy = oy + j * p + p / 2.0
            # Rombo (cuadrado rotado 45°) centrado en la celda.
            holes.append(
                Polygon(
                    [
                        (cx, cy - half_diag),
                        (cx + half_diag, cy),
                        (cx, cy + half_diag),
                        (cx - half_diag, cy),
                    ]
                )
            )
    return holes


def _auxetic_reentrant(width_m: float, height_m: float, spec: FlexSpec) -> List[BaseGeometry]:
    """Honeycomb re-entrante: celdas hexagonales invertidas (paredes hacia adentro)."""
    p = spec.spacing_m
    lig = spec.ligament_m
    cells: List[BaseGeometry] = []
    # Celda re-entrante: hexágono con dos vértices empujados hacia adentro.
    cw = p                     # ancho de celda
    ch = p * 1.2               # alto de celda
    inset = min(p * 0.30, cw / 2 - lig)  # profundidad re-entrante
    if inset <= 0:
        return cells
    nx = int((width_m - 2 * EDGE_MARGIN) // cw)
    ny = int((height_m - 2 * EDGE_MARGIN) // ch)
    if nx < 1 or ny < 1:
        return cells
    ox = (width_m - nx * cw) / 2.0
    oy = (height_m - ny * ch) / 2.0

    def cell(cx: float, cy: float) -> Polygon:
        hw, hh = cw / 2.0, ch / 2.0
        return Polygon(
            [
                (cx - hw + inset, cy - hh),   # inf-izq (hacia adentro)
                (cx + hw - inset, cy - hh),   # inf-der (hacia adentro)
                (cx + hw, cy),                # der
                (cx + hw - inset, cy + hh),   # sup-der (hacia adentro)
                (cx - hw + inset, cy + hh),   # sup-izq (hacia adentro)
                (cx - hw, cy),                # izq
            ]
        )

    for j in range(ny):
        for i in range(nx):
            cx = ox + i * cw + cw / 2.0
            cy = oy + j * ch + ch / 2.0
            poly = cell(cx, cy).buffer(-lig / 2.0)
            if not poly.is_empty and poly.area > (lig * lig):
                cells.append(poly)
    return cells


def _auxetic_chiral(width_m: float, height_m: float, spec: FlexSpec) -> List[BaseGeometry]:
    """Quiral: nodos circulares con ligamentos tangentes (rotan al traccionar)."""
    p = spec.spacing_m
    lig = spec.ligament_m
    holes: List[BaseGeometry] = []
    r = max(p / 2.0 - lig, lig)      # radio del nodo circular
    nx = int((width_m - 2 * EDGE_MARGIN) // p)
    ny = int((height_m - 2 * EDGE_MARGIN) // p)
    if nx < 1 or ny < 1:
        return holes
    ox = (width_m - nx * p) / 2.0
    oy = (height_m - ny * p) / 2.0
    for j in range(ny):
        for i in range(nx):
            cx = ox + i * p + p / 2.0
            cy = oy + j * p + p / 2.0
            # Nodo circular; los ligamentos quirales quedan en el material entre nodos.
            circle = _circle(cx, cy, r, 20)
            holes.append(circle)
    return holes


def _circle(cx: float, cy: float, r: float, steps: int) -> Polygon:
    pts = [
        (cx + r * math.cos(2 * math.pi * k / steps), cy + r * math.sin(2 * math.pi * k / steps))
        for k in range(steps)
    ]
    return Polygon(pts)


# ---------------------------------------------------------------------------
# Conversión de geometría shapely -> Edge2D(flex=True)
# ---------------------------------------------------------------------------


def _ring_to_edges(coords: List[Tuple[float, float]]) -> List[Edge2D]:
    out: List[Edge2D] = []
    for i in range(len(coords) - 1):
        ax, ay = coords[i]
        bx, by = coords[i + 1]
        out.append(Edge2D(a=Vec2(ax, ay), b=Vec2(bx, by), flex=True))
    return out


def _geom_to_edges(geom: BaseGeometry) -> List[Edge2D]:
    edges: List[Edge2D] = []
    if geom.is_empty:
        return edges
    gt = geom.geom_type
    if gt == "Polygon":
        edges.extend(_ring_to_edges(list(geom.exterior.coords)))
        for interior in geom.interiors:
            edges.extend(_ring_to_edges(list(interior.coords)))
    elif gt == "MultiPolygon":
        for g in geom.geoms:
            edges.extend(_geom_to_edges(g))
    elif gt == "LineString":
        edges.extend(_ring_to_edges(list(geom.coords)))
    elif gt in ("MultiLineString", "GeometryCollection"):
        for g in geom.geoms:
            edges.extend(_geom_to_edges(g))
    return edges


_GENERATORS = {
    "kerf": _kerf_slots,
    "auxetic_rotating": _auxetic_rotating,
    "auxetic_reentrant": _auxetic_reentrant,
    "auxetic_chiral": _auxetic_chiral,
}


def apply_flex_to_panel(
    width_m: float,
    height_m: float,
    edges: List[Edge2D],
    spec: FlexSpec,
) -> List[Edge2D]:
    """Genera las aristas del patrón de flexión recortadas al contorno del panel.

    Devuelve SÓLO las aristas nuevas del patrón (flex=True); el contorno del panel
    (`edges`) no se modifica. El llamador hace `edges += apply_flex_to_panel(...)`.
    """
    if width_m < 3 * EDGE_MARGIN or height_m < 3 * EDGE_MARGIN:
        return []

    gen = _GENERATORS.get(spec.method)
    if gen is None:
        return []

    raw = gen(width_m, height_m, spec)
    if not raw:
        return []

    # Clamp de densidad POR CONTEO REAL: si el patrón excede el tope de primitivas
    # (agnóstico al método), sube el spacing efectivo y regenera una vez. Evita que un
    # panel grande con spacing chico produzca un DXF/preview inmanejable.
    if len(raw) > MAX_PRIMITIVES:
        factor = math.sqrt(len(raw) / MAX_PRIMITIVES)
        eff = spec.spacing_m * factor
        log.info(
            "flex: panel %.2fx%.2fm método %s spacing %.4f generó %d primitivas; "
            "spacing efectivo elevado a %.4f",
            width_m, height_m, spec.method, spec.spacing_m, len(raw), eff,
        )
        spec = replace(
            spec, spacing_m=eff, ligament_m=min(spec.ligament_m, eff * 0.8)
        )
        raw = gen(width_m, height_m, spec)
        if not raw:
            return []

    geom = unary_union(raw)

    # Rotar el patrón por axis_deg alrededor del centro del panel.
    if abs(spec.axis_deg) > 1e-6:
        geom = affinity.rotate(
            geom, spec.axis_deg, origin=(width_m / 2.0, height_m / 2.0), use_radians=False
        )

    # Recortar al contorno real del panel (respetando huecos y borde), con margen.
    panel = _panel_polygon(width_m, height_m, edges)
    try:
        safe = panel.buffer(-EDGE_MARGIN)
    except Exception:
        safe = panel
    if safe.is_empty:
        safe = panel
    try:
        clipped = geom.intersection(safe)
    except Exception:
        try:
            clipped = geom.intersection(panel)
        except Exception:
            return []

    return _geom_to_edges(clipped)
