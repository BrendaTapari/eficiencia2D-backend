import math
from typing import List, Optional, Tuple
from dataclasses import dataclass

# Importamos nuestro tipo Vec2
from core.services.types import Vec2

# ============================================================================
# Sheet Nester — 2D Bin Packing para Corte Láser
#
# Utiliza el algoritmo de Rectángulos Máximos (MAXRECTS) con la heurística de
# "Mejor Ajuste de Lado Corto" (Best Short Side Fit).
# ============================================================================


@dataclass
class SheetConfig:
    width_m: float
    height_m: float
    gap_m: float


DEFAULT_SHEET = SheetConfig(width_m=1.0, height_m=0.6, gap_m=0.003)


@dataclass
class Edge:
    a: Vec2
    b: Vec2


@dataclass
class NestingPanel:
    id: str
    category: str  # "wall" | "floor"
    width_m: float
    height_m: float
    edges: List[Edge]


@dataclass
class PlacedNestingPanel:
    panel: NestingPanel
    x: float
    y: float
    rotated: bool
    effective_w: float
    effective_h: float


@dataclass
class NestingSheet:
    index: int
    panels: List[PlacedNestingPanel]
    utilization: float


@dataclass
class NestingResult:
    sheets: List[NestingSheet]
    config: SheetConfig
    scale_denom: float
    unplaced: List[NestingPanel]


# ---------------------------------------------------------------------------
# MAXRECTS
# ---------------------------------------------------------------------------


@dataclass
class FreeRect:
    x: float
    y: float
    w: float
    h: float


@dataclass
class SheetState:
    free_rects: List[FreeRect]
    panels: List[PlacedNestingPanel]


EPS = 0.0005


def find_best_rect(free_rects: List[FreeRect], pw: float, ph: float) -> Optional[dict]:
    """Best Short Side Fit — minimiza la dimensión sobrante más corta."""
    best = None
    for rect in free_rects:
        if pw <= rect.w + EPS and ph <= rect.h + EPS:
            left_h = rect.w - pw
            left_v = rect.h - ph
            short_side = min(left_h, left_v)
            long_side = max(left_h, left_v)
            score = short_side * 1000 + long_side

            if not best or score < best["score"]:
                best = {"x": rect.x, "y": rect.y, "score": score}

    return best


def prune_contained(rects: List[FreeRect]) -> List[FreeRect]:
    out: List[FreeRect] = []
    n = len(rects)

    for i in range(n):
        ri = rects[i]
        contained = False
        for j in range(n):
            if i == j:
                continue
            rj = rects[j]
            if (
                ri.x >= rj.x - EPS
                and ri.y >= rj.y - EPS
                and ri.x + ri.w <= rj.x + rj.w + EPS
                and ri.y + ri.h <= rj.y + rj.h + EPS
            ):
                contained = True
                break

        if not contained:
            out.append(ri)

    return out


def commit_placement(
    sheet: SheetState,
    panel: NestingPanel,
    px: float,
    py: float,
    rotated: bool,
    pw: float,
    ph: float,
    gap: float,
) -> None:

    sheet.panels.append(
        PlacedNestingPanel(
            panel=panel, x=px, y=py, rotated=rotated, effective_w=pw, effective_h=ph
        )
    )

    bx2 = px + pw + gap
    by2 = py + ph + gap
    next_rects: List[FreeRect] = []

    for rect in sheet.free_rects:
        rx2 = rect.x + rect.w
        ry2 = rect.y + rect.h

        # Si el rectángulo libre no se intersecta con el panel colocado, se conserva intacto
        if bx2 <= rect.x or px >= rx2 or by2 <= rect.y or py >= ry2:
            next_rects.append(rect)
            continue

        # Si hay intersección, el rectángulo libre se divide en hasta 4 sub-rectángulos nuevos
        if px > rect.x:
            next_rects.append(FreeRect(x=rect.x, y=rect.y, w=px - rect.x, h=rect.h))
        if bx2 < rx2:
            next_rects.append(FreeRect(x=bx2, y=rect.y, w=rx2 - bx2, h=rect.h))
        if py > rect.y:
            next_rects.append(FreeRect(x=rect.x, y=rect.y, w=rect.w, h=py - rect.y))
        if by2 < ry2:
            next_rects.append(FreeRect(x=rect.x, y=by2, w=rect.w, h=ry2 - by2))

    cleaned = [r for r in next_rects if r.w > 0.001 and r.h > 0.001]
    sheet.free_rects = prune_contained(cleaned)


def nest_panels(
    panels: List[NestingPanel], config: SheetConfig, scale_denom: float = 1.0
) -> NestingResult:

    width_m = config.width_m
    height_m = config.height_m
    gap_m = config.gap_m
    sheet_area = width_m * height_m

    # Ordenar los paneles por área de mayor a menor (Largest Area Fit First)
    sorted_panels = sorted(panels, key=lambda p: p.width_m * p.height_m, reverse=True)

    states: List[SheetState] = []
    unplaced: List[NestingPanel] = []

    for panel in sorted_panels:
        pw = panel.width_m
        ph = panel.height_m
        fits_n = pw <= width_m + EPS and ph <= height_m + EPS
        fits_r = ph <= width_m + EPS and pw <= height_m + EPS
        can_rotate = fits_r and abs(pw - ph) > 0.001

        if not fits_n and not fits_r:
            unplaced.append(panel)
            continue

        best_sheet = -1
        best_x, best_y, best_pw, best_ph = 0.0, 0.0, 0.0, 0.0
        best_rot = False
        best_score = float("inf")

        # Evaluar colocación en hojas existentes
        for si, state in enumerate(states):
            if fits_n:
                r = find_best_rect(state.free_rects, pw, ph)
                if r and r["score"] < best_score:
                    best_score = r["score"]
                    best_sheet = si
                    best_x, best_y = r["x"], r["y"]
                    best_pw, best_ph = pw, ph
                    best_rot = False

            if can_rotate:
                r = find_best_rect(state.free_rects, ph, pw)
                if r and r["score"] < best_score:
                    best_score = r["score"]
                    best_sheet = si
                    best_x, best_y = r["x"], r["y"]
                    best_pw, best_ph = ph, pw
                    best_rot = True

        # Si se encontró lugar en una hoja existente, colocarlo
        if best_sheet >= 0:
            commit_placement(
                states[best_sheet],
                panel,
                best_x,
                best_y,
                best_rot,
                best_pw,
                best_ph,
                gap_m,
            )
            continue

        # Si no hubo lugar, crear una hoja nueva
        new_sheet = SheetState(
            free_rects=[FreeRect(x=0.0, y=0.0, w=width_m, h=height_m)], panels=[]
        )

        placed = False
        if fits_n:
            r = find_best_rect(new_sheet.free_rects, pw, ph)
            if r:
                commit_placement(new_sheet, panel, r["x"], r["y"], False, pw, ph, gap_m)
                placed = True

        if not placed and can_rotate:
            r = find_best_rect(new_sheet.free_rects, ph, pw)
            if r:
                commit_placement(new_sheet, panel, r["x"], r["y"], True, ph, pw, gap_m)
                placed = True

        if placed:
            states.append(new_sheet)
        else:
            unplaced.append(panel)

    # Convertir los estados en resultados formateados
    sheets: List[NestingSheet] = []
    for i, s in enumerate(states):
        used_area = sum(p.effective_w * p.effective_h for p in s.panels)
        sheets.append(
            NestingSheet(index=i, panels=s.panels, utilization=used_area / sheet_area)
        )

    return NestingResult(
        sheets=sheets, config=config, scale_denom=scale_denom, unplaced=unplaced
    )


def rotate_edges(edges: List[Edge], original_w: float) -> List[Edge]:
    """
    Rota las coordenadas de las aristas 90 grados en sentido horario dentro de
    la caja delimitadora del panel. Un panel (width_m x height_m) se convierte
    en (height_m x width_m). El punto (x, y) se mapea a (y, original_w - x).
    """
    return [
        Edge(
            a=Vec2(x=e.a.y, y=original_w - e.a.x), b=Vec2(x=e.b.y, y=original_w - e.b.x)
        )
        for e in edges
    ]
