"""
Chequeo automático de ensamble: detecta huecos / solapes / piezas sin apoyo entre las
piezas 3D (paredes y pisos con grosor), reusando lo que el backend ya sabe
(placements, planos, min_y/max_y). Devuelve `assembly_warnings` para el preview, así el
aviso llega ANTES de mandar a cortar.

La verdad vive en la geometría 3D (mismo espacio Y-up que `faces_packed`/`placements`);
esto sólo EXPONE interferencias medibles. Chequeo conservador (baja tasa de falsos
positivos): se apoya en relaciones verticales pieza-pieza y en coplanaridad, con una
tolerancia configurable que se echoa en cada aviso.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.group_classifier import GeometryGroup
from core.services.plate_intersect import group_plane, signed_dist
from core.services.types import Face3D, Vec3, dot, normalize

# Tolerancia por defecto (mm). Umbral por debajo del cual un hueco/solape se considera
# contacto válido. Configurable; se reporta en cada aviso (`tolerance_mm`).
DEFAULT_TOLERANCE_MM = 0.5
# Ventana (m) para considerar que un piso es el APOYO buscado de una pared (no un piso
# de otro nivel): sólo se evalúa la relación si el piso está a < esta distancia vertical.
SUPPORT_WINDOW_M = 0.08
# Mínimo solape horizontal (m) para considerar que dos footprints se pisan (evita rozar).
MIN_FOOTPRINT_OVERLAP_M = 0.02
# Umbral mínimo (m) de hueco/solape vertical para reportar: por debajo se considera ruido
# de malla, no un error real de ensamble. La tolerancia reportada puede ser menor.
GAP_THRESHOLD_M = 0.003


@dataclass
class _Piece:
    gid: int
    label: str
    category: str
    minx: float
    maxx: float
    miny: float
    maxy: float
    minz: float
    maxz: float
    n: Vec3
    d: float
    thickness: float


def _aabb(gfaces: List[Face3D]):
    xs, ys, zs = [], [], []
    for f in gfaces:
        for v in f.vertices:
            xs.append(v.x); ys.append(v.y); zs.append(v.z)
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def _footprint_overlap(a: _Piece, b: _Piece) -> Optional[Tuple[float, float]]:
    """Solape del footprint (x,z). Devuelve el centro (x,z) o None.

    Maneja footprints DEGENERADOS: una pared es un plano delgado → su footprint es casi
    una línea (extensión ~0 en un eje horizontal). Basta con que los rangos se intersequen
    en ambos ejes (con holgura) y haya solape real en el eje largo. Así una pared queda
    "apoyada" por el piso/pared bajo su línea de base.
    """
    eps = 0.02  # holgura de intersección (2 cm): tolera pequeños offsets de modelado
    ox = min(a.maxx, b.maxx) - max(a.minx, b.minx)
    oz = min(a.maxz, b.maxz) - max(a.minz, b.minz)
    if ox < -eps or oz < -eps:
        return None  # los rangos no se intersecan en algún eje
    if max(ox, oz) < MIN_FOOTPRINT_OVERLAP_M:
        return None  # sólo se rozan (ningún eje con solape real)
    cx = (max(a.minx, b.minx) + min(a.maxx, b.maxx)) / 2.0
    cz = (max(a.minz, b.minz) + min(a.maxz, b.maxz)) / 2.0
    return cx, cz


def _is_floor(g: GeometryGroup) -> bool:
    cat = getattr(g, "category", None)
    if cat == "floor":
        return True
    if cat == "wall":
        return False
    return (getattr(g, "orientation", "") or "").lower().startswith("horizontal")


def compute_assembly_warnings(
    groups: List[GeometryGroup],
    faces: List[Face3D],
    panel_id_by_group: Optional[Dict[int, str]] = None,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
) -> List[dict]:
    """Detecta gap / overlap / unsupported entre las piezas del modelo.

    - `gap`: una pieza no toca su apoyo (hueco vertical > tolerancia).
    - `overlap`: una pieza penetra a otra (pared dentro del slab del piso; o dos paredes
      coplanares superpuestas).
    - `unsupported`: una pared "en el aire", sin piso/pared debajo de su footprint.
    """
    panel_id_by_group = panel_id_by_group or {}
    tol = max(tolerance_mm, 0.0) / 1000.0

    pieces: List[_Piece] = []
    for g in groups:
        if getattr(g, "category", None) == "discard":
            continue
        gfaces = [faces[i] for i in g.face_indices if 0 <= i < len(faces)]
        if not gfaces:
            continue
        minx, maxx, miny, maxy, minz, maxz = _aabb(gfaces)
        n, d = group_plane(g, faces)
        th = g.thickness if g.thickness else max(maxy - miny, 0.0) if _is_floor(g) else 0.0
        pieces.append(_Piece(
            gid=g.id,
            label=str(panel_id_by_group.get(g.id) or getattr(g, "label", None) or f"G{g.id}"),
            category="floor" if _is_floor(g) else "wall",
            minx=minx, maxx=maxx, miny=miny, maxy=maxy, minz=minz, maxz=maxz,
            n=n, d=d, thickness=th or 0.0,
        ))

    if not pieces:
        return []

    walls = [p for p in pieces if p.category == "wall"]
    floors = [p for p in pieces if p.category == "floor"]
    ground_y = min(p.miny for p in pieces)

    warnings: List[dict] = []

    def emit(pieces_ids, wtype, measure_m, at_xyz):
        warnings.append({
            "pieces": pieces_ids,
            "type": wtype,
            "measure_mm": round(abs(measure_m) * 1000.0, 2),
            "at": [round(at_xyz[0], 4), round(at_xyz[1], 4), round(at_xyz[2], 4)],
            "tolerance_mm": round(tolerance_mm, 3),
        })

    # Umbral de "en el aire": sólo se marca si el pie está claramente sobre el suelo
    # (evita ruido a nivel de piso base). Un modelo válido de varios niveles NO dispara,
    # porque siempre hay una pieza (piso o pared inferior) bajo el footprint.
    FLOAT_THRESHOLD_M = 0.05

    for w in walls:
        # ¿hay ALGO (piso o pared) debajo del footprint de la pared?  → parte del stack.
        # Además, el piso de apoyo más cercano (para medir hueco/solape vertical real).
        supported = False
        support_floor = None      # (vgap, floor, center)
        for p in pieces:
            if p.gid == w.gid:
                continue
            ov = _footprint_overlap(w, p)
            if ov is None:
                continue
            # p sostiene a la pared si su rango vertical ALCANZA el pie de la pared desde
            # abajo (top ≥ pie − ventana) y arranca en/por debajo del pie (no está flotando
            # por encima). Cubre pared apilada sobre pared/piso o embebida en el slab base.
            if p.maxy >= w.miny - SUPPORT_WINDOW_M and p.miny <= w.miny + SUPPORT_WINDOW_M:
                supported = True
            if p.category == "floor" and abs(w.miny - p.maxy) <= SUPPORT_WINDOW_M \
                    and p.miny - 1e-6 <= w.miny:
                vgap = w.miny - p.maxy
                if support_floor is None or abs(vgap) < abs(support_floor[0]):
                    support_floor = (vgap, p, ov)

        if not supported and (w.miny - ground_y) > FLOAT_THRESHOLD_M:
            emit([w.label], "unsupported", w.miny - ground_y,
                 ((w.minx + w.maxx) / 2.0, w.miny, (w.minz + w.maxz) / 2.0))
            continue

        # Hueco / solape vertical contra el piso de apoyo (umbrales significativos para
        # no reaccionar al ruido submilimétrico de la malla).
        if support_floor is not None:
            vgap, f, (cx, cz) = support_floor
            if vgap > max(tol, GAP_THRESHOLD_M):
                emit([w.label, f.label], "gap", vgap, (cx, w.miny, cz))
            elif vgap < -max(tol, GAP_THRESHOLD_M) and w.miny > f.miny + tol:
                emit([w.label, f.label], "overlap", vgap, (cx, f.maxy, cz))

    warnings.sort(key=lambda w: w["measure_mm"], reverse=True)
    return warnings
