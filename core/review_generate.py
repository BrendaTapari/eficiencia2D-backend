"""
Generación de planos a partir del estado de revisión del frontend.

El upload/clasificación corre en parse_pipeline; acá se aplican overrides,
fusiones y decisiones muro-muro, se descompone por grupo y se exporta DXF/PDF.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from core.group_classifier import GeometryGroup
from core.pipeline import Phase1Result
from core.services.assembly_adjuster import (
    AdjustmentsResult,
    compute_adjustments,
    yield_by_pair,
)
from core.services.cutting_sheet import (
    Edge2D,
    Panel,
    apply_user_cuts,
    clip_panel_at_u,
    clip_panel_at_v,
    compute_group_placement,
    mirror_edges_horizontal,
    nested_sheets_to_dxf,
    orient_group_normals_outward,
    panel_cut_area_m2,
    project_faces_to_2d,
    shift_placement,
)
from core.services.plate_intersect import pair_key, resolve_plate_joints
from core.services.flex_bending import (
    apply_flex_to_panel,
    apply_mark_lines_to_panel,
    build_flex_by_group,
    parse_flex,
)
from core.services.reinforcements import build_reinforcements
from core.services.facade_extractor import extract_facades
from core.services.floor_plan_extractor import extract_floor_plans
from core.services.joint_detector import build_group_adjacency, detect_joints
from core.services.pdf_writer import generate_nesting_pdf, generate_pdf
from core.services.sheet_nester import (
    NestingPanel,
    NestingResult,
    SheetConfig,
    Vec2,
    Edge,
    nest_panels,
)
from core.services.types import OutputFile, PipelineOptions, Vec3, dot, normalize


# Una fusión sólo tiene sentido entre paredes COPLANARES (mismo plano). Si la
# selección por área agarró paredes en distinta dirección, fusionarlas crea un grupo
# de normales mezcladas que al proyectar produce "paredes fantasma". Guardas:
MERGE_PARALLEL_DOT = 0.95   # |n_m·n_s| > esto => paralelas/antiparalelas (mismo plano)
MERGE_PLANE_TOL = 0.30      # m: separación máxima de plano (tolera grosor/pieles)

# Área mínima de un panel 2D en la plancha (m² de edificio). Sólo descarta astillas
# degeneradas; es independiente del umbral de clasificación (opts.min_area_m2).
MIN_PANEL_AREA_M2 = 0.01


def _coplanar_for_merge(member: GeometryGroup, survivor: GeometryGroup) -> bool:
    ns = normalize(survivor.representative_normal)
    nm = normalize(member.representative_normal)
    if abs(dot(ns, nm)) < MERGE_PARALLEL_DOT:
        return False  # direcciones distintas -> no fusionar
    # mismo plano: distancia del centroide del miembro al plano del survivor
    return abs(dot(ns, member.centroid) - dot(ns, survivor.centroid)) <= MERGE_PLANE_TOL


# Tolerancia vertical para decidir si una losa cae en el límite entre dos tramos de
# muro (el pipeline parte los muros justo al nivel de la losa).
FLOOR_BAND_TOL_M = 0.05


def _floor_between(
    a: GeometryGroup, b: GeometryGroup, floors, faces
) -> Optional[GeometryGroup]:
    """Devuelve la losa que separa verticalmente a dos tramos de muro, si existe.

    El pipeline parte los muros al nivel de cada losa (split_wall_groups_at_floors)
    justamente para que la losa se inserte entre el tramo de abajo y el de arriba.
    Volver a fusionarlos produce una pieza que cruza ese nivel y el piso ya no entra:
    al armar la maqueta el entrepiso queda apoyado por encima en vez de encastrado.
    """
    if a.min_y is None or a.max_y is None or b.min_y is None or b.max_y is None:
        return None
    # Tramos apilados: uno termina donde empieza el otro.
    if a.max_y <= b.min_y + FLOOR_BAND_TOL_M:
        boundary = (a.max_y + b.min_y) / 2.0
    elif b.max_y <= a.min_y + FLOOR_BAND_TOL_M:
        boundary = (b.max_y + a.min_y) / 2.0
    else:
        return None  # se solapan en altura: no están en bandas distintas

    # Banda que ocupa el muro a lo largo de su propia normal en planta.
    import math as _math
    from core.services.mesh_splitter import SLAB_PENETRATION_TOL

    n = a.representative_normal
    n_len = _math.hypot(n.x, n.z)
    if n_len <= 1e-6:
        return None
    nx, nz = n.x / n_len, n.z / n_len
    offs = [
        v.x * nx + v.z * nz
        for g in (a, b)
        for fi in g.face_indices
        if 0 <= fi < len(faces)
        for v in faces[fi].vertices
    ]
    if not offs:
        return None
    w_lo, w_hi = min(offs), max(offs)

    for f in floors:
        if f.min_y is None or f.max_y is None:
            continue
        # La losa debe abarcar el nivel del límite entre los dos tramos.
        if not (f.min_y - FLOOR_BAND_TOL_M <= boundary <= f.max_y + FLOOR_BAND_TOL_M):
            continue
        # ...y ATRAVESAR el plano del muro. Mismo criterio que usa el splitter para
        # partirlo: si la losa sólo apoya contra la cara, el muro nunca se parte, así
        # que tampoco hay que impedir que el usuario lo fusione.
        f_offs = [
            v.x * nx + v.z * nz
            for fi in f.face_indices
            if 0 <= fi < len(faces)
            for v in faces[fi].vertices
        ]
        if not f_offs:
            continue
        if min(w_hi, max(f_offs)) - max(w_lo, min(f_offs)) > SLAB_PENETRATION_TOL:
            return f
    return None


def apply_merges(
    phase1: Phase1Result,
    merges: List[List[int]],
    overrides: Optional[Dict[int, str]] = None,
) -> Phase1Result:
    if not merges:
        return phase1

    overrides = overrides or {}
    group_by_id: Dict[int, GeometryGroup] = {g.id: g for g in phase1.groups}
    merged_ids: set[int] = set()
    skipped = 0
    floors = [
        g for g in phase1.groups
        if _effective_category(g, overrides) == "floor"
    ]
    blocked_by_floor: List[str] = []

    for merge_set in merges:
        # Categoría EFECTIVA (con overrides del usuario): un grupo rescatado de
        # "descartado" (p. ej. marco de ventana → Pared) debe poder fusionarse. Antes se
        # filtraba por la categoría original y la fusión hecha en el visor se perdía en
        # silencio → la plancha no reflejaba lo que mostraba el visor.
        members = [
            group_by_id[gid]
            for gid in merge_set
            if gid in group_by_id
            and _effective_category(group_by_id[gid], overrides) != "discard"
        ]
        if len(members) < 2:
            continue

        survivor = max(members, key=lambda g: g.total_area)
        # Sólo fusionar los miembros coplanares con el survivor; los que están en otra
        # dirección/plano se dejan como grupos sueltos (no se funden ni se descartan).
        members = [
            m for m in members
            if m.id == survivor.id or _coplanar_for_merge(m, survivor)
        ]
        skipped += len(merge_set) - len(members)

        # Tampoco se fusionan tramos separados por una LOSA: el pipeline los partió a
        # ese nivel para que el entrepiso se encastre entre ellos. Fusionarlos genera
        # una pieza que cruza el nivel de la losa y el piso deja de entrar (se apoya
        # por encima). Los tramos bloqueados quedan como piezas independientes.
        kept = [survivor]
        for m in members:
            if m.id == survivor.id:
                continue
            f = _floor_between(survivor, m, floors, phase1.faces)
            if f is None:
                kept.append(m)
            else:
                blocked_by_floor.append(
                    f"{m.label or m.id} / {survivor.label or survivor.id}"
                    f" (los separa {f.label or f.id})"
                )
        members = kept
        if len(members) < 2:
            continue

        combined_faces: List[int] = []
        total_area = 0.0
        cx = cy = cz = 0.0
        min_y = max_y = None

        for m in members:
            combined_faces.extend(m.face_indices)
            total_area += m.total_area
            cx += m.centroid.x * m.total_area
            cy += m.centroid.y * m.total_area
            cz += m.centroid.z * m.total_area
            if m.min_y is not None:
                min_y = m.min_y if min_y is None else min(min_y, m.min_y)
            if m.max_y is not None:
                max_y = m.max_y if max_y is None else max(max_y, m.max_y)
            if m.id != survivor.id:
                merged_ids.add(m.id)

        centroid = (
            Vec3(x=cx / total_area, y=cy / total_area, z=cz / total_area)
            if total_area > 0
            else survivor.centroid
        )
        group_by_id[survivor.id] = replace(
            survivor,
            face_indices=combined_faces,
            total_area=total_area,
            centroid=centroid,
            min_y=min_y,
            max_y=max_y,
        )

    if skipped:
        import logging
        logging.getLogger("eficiencia2d.pipeline").info(
            f"  apply_merges: {skipped} paredes no coplanares omitidas de la fusión"
        )

    warnings = list(phase1.warnings or [])
    if blocked_by_floor:
        import logging
        logging.getLogger("eficiencia2d.pipeline").info(
            f"  apply_merges: {len(blocked_by_floor)} fusiones bloqueadas por losa intermedia"
        )
        warnings.append(
            "No se fusionaron tramos de pared separados por una losa: el entrepiso "
            "debe encastrarse entre ellos y al unirlos dejaría de entrar. "
            + "; ".join(blocked_by_floor[:4])
            + ("…" if len(blocked_by_floor) > 4 else "")
        )

    new_groups = [
        group_by_id[g.id] for g in phase1.groups if g.id not in merged_ids
    ]
    joints = detect_joints(phase1.faces, new_groups)
    adj = compute_adjustments(joints, new_groups, None, phase1.faces)

    return replace(
        phase1,
        groups=new_groups,
        joints=joints,
        adjustments=adj.adjustments,
        wall_wall_joints=adj.wall_wall_joints,
        suggested_merges=[],
        warnings=warnings,
        # Los merges cambian qué grupos existen: la vecindad se reconstruye sobre las
        # juntas nuevas, no se arrastra la del modelo sin fusionar.
        group_adjacency=build_group_adjacency(joints),
    )


def _aplicar_muescas(edges, width_m, height_m, muescas):
    """Resta del material de la pieza una franja por cada vecino, acotada a su banda.

    Devuelve `(edges, width_m, height_m, offset_u, offset_v)` o None si no cambia nada.

    Cuando una muesca abarca toda la altura, el resultado es el recorte de borde de
    siempre y el ancho baja. Cuando abarca sólo una banda, la pieza conserva su ancho y
    queda con una entrada: es lo que hace falta cuando dos vecinos distintos llegan al
    mismo extremo a alturas distintas.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    material = _material_de_la_pieza(edges)
    if material is None or material.is_empty:
        return None

    quitar = []
    agregar = []
    for izquierda, prof, v_lo, v_hi in muescas:
        if v_hi - v_lo <= 0.001:
            continue
        if prof < -0.001:
            # AGRANDAR. Una esquina la cierran las dos piezas: la que cede va a la cara
            # interior de su vecina y la que pasa a la cara EXTERIOR de la que cede. Esa
            # segunda casi siempre exige crecer, porque en el modelo el muro llega hasta
            # la piel del macizo y a escalas gruesas eso queda corto: a 1:100 media placa
            # son 15 cm de edificio y el muro asoma 1 cm, faltan 14.
            #
            # Sin esto el ajuste salía negativo y se descartaba en silencio: la pieza
            # quedaba de eje a eje de sus vecinas, el único largo que no arma de ninguna
            # de las dos formas (sobran 3 mm para entrar entre ellas, faltan 2.8 para
            # pasarles por encima).
            ext = -prof
            lo = v_lo - 1e-6 if v_lo <= 1e-6 else v_lo
            hi = v_hi + 1e-6 if v_hi >= height_m - 1e-6 else v_hi
            if izquierda:
                agregar.append(Polygon([(-ext, lo), (1e-6, lo), (1e-6, hi), (-ext, hi)]))
            else:
                agregar.append(Polygon([
                    (width_m - 1e-6, lo), (width_m + ext, lo),
                    (width_m + ext, hi), (width_m - 1e-6, hi),
                ]))
            continue
        prof = min(prof, width_m - 0.01)
        if prof <= 0.001:
            continue
        # Se extiende un pelo fuera del panel en v para que la resta no deje una película
        # de material en los bordes por redondeo.
        lo = v_lo - 1e-6 if v_lo <= 1e-6 else v_lo
        hi = v_hi + 1e-6 if v_hi >= height_m - 1e-6 else v_hi
        if izquierda:
            quitar.append(Polygon([(-1e-6, lo), (prof, lo), (prof, hi), (-1e-6, hi)]))
        else:
            quitar.append(
                Polygon([(width_m - prof, lo), (width_m + 1e-6, lo),
                         (width_m + 1e-6, hi), (width_m - prof, hi)])
            )
    if not quitar and not agregar:
        return None

    resto = material
    if agregar:
        # El agregado se pega sólo si NO deja la pieza en más trozos de los que tenía: un
        # rectángulo que no toca el material sería una isla suelta en la plancha.
        pegado = unary_union([material] + agregar)
        antes = len(getattr(material, "geoms", [material]))
        if len(getattr(pegado, "geoms", [pegado])) <= antes:
            resto = pegado
    if quitar:
        resto = resto.difference(unary_union(quitar))
    if resto.is_empty or resto.area <= 1e-9:
        return None

    geoms = [g for g in getattr(resto, "geoms", [resto]) if getattr(g, "area", 0.0) > 1e-9]
    if not geoms:
        return None
    minx = min(g.bounds[0] for g in geoms)
    miny = min(g.bounds[1] for g in geoms)
    maxx = max(g.bounds[2] for g in geoms)
    maxy = max(g.bounds[3] for g in geoms)

    nuevas: List[Edge2D] = []
    for g in geoms:
        anillos = [(list(g.exterior.coords), False)]
        anillos += [(list(h.coords), True) for h in g.interiors]
        for coords, es_hueco in anillos:
            for i in range(len(coords) - 1):
                (ax, ay), (bx, by) = coords[i], coords[i + 1]
                # Los booleanos de shapely dejan puntos repetidos donde el rectángulo que
                # se une o se resta toca el contorno. Una arista de largo cero no corta
                # nada y rompe el cierre del polígono cuando se vuelve a leer.
                if abs(ax - bx) <= 1e-9 and abs(ay - by) <= 1e-9:
                    continue
                nuevas.append(
                    Edge2D(a=Vec2(ax - minx, ay - miny),
                           b=Vec2(bx - minx, by - miny), hole=es_hueco)
                )
    if not nuevas:
        return None
    return nuevas, maxx - minx, maxy - miny, minx, miny


def _material_de_la_pieza(edges: List[Edge2D]):
    """Polígono del MATERIAL de la pieza a partir de sus aristas (contorno menos huecos).

    Regla par-impar: una región contenida en un número par de otras es material, en un
    número impar es hueco. Vale para piezas con varias regiones sueltas y con aberturas
    anidadas, que es lo que aparece en cuanto el muro tiene ventanas.
    """
    from shapely.geometry import LineString, Point
    from shapely.ops import polygonize, unary_union

    segs = [
        LineString([(e.a.x, e.a.y), (e.b.x, e.b.y)])
        for e in edges
        if not getattr(e, "score", False) and not getattr(e, "flex", False)
    ]
    if not segs:
        return None
    caras = sorted(polygonize(unary_union(segs)), key=lambda p: p.area, reverse=True)
    if not caras:
        return None
    material = []
    for i, c in enumerate(caras):
        p = c.representative_point()
        dentro = sum(
            1 for j, o in enumerate(caras) if j != i and o.area > c.area and o.contains(p)
        )
        if dentro % 2 == 0:
            material.append(c)
    if not material:
        return None
    u = unary_union(material)
    return u if not u.is_empty else None


def _slot_dentro_del_material(edges, ax, ay, bx, by, ancho):
    """Anillos de la ranura, recortada al material de la pieza. Vacío si no la toca."""
    import math as _math

    from shapely.geometry import Polygon

    dx, dy = bx - ax, by - ay
    largo = _math.hypot(dx, dy)
    if largo < 1e-9 or ancho <= 0:
        return []
    ux, uy = dx / largo, dy / largo
    nx, ny = -uy, ux
    h = ancho / 2.0
    rect = Polygon([
        (ax + nx * h, ay + ny * h), (bx + nx * h, by + ny * h),
        (bx - nx * h, by - ny * h), (ax - nx * h, ay - ny * h),
    ])
    material = _material_de_la_pieza(edges)
    if material is None or not rect.is_valid or rect.is_empty:
        return []
    corte = rect.intersection(material)
    if corte.is_empty:
        return []
    geoms = list(getattr(corte, "geoms", [corte]))
    out = []
    for g in geoms:
        if getattr(g, "area", 0.0) <= 1e-12:
            continue
        x0, y0, x1, y1 = g.bounds
        # Una ranura más corta que ancha no sostiene nada: sale de un contacto rasante
        # entre dos piezas que apenas se rozan (segmentos de 0.15 mm en un modelo real) y
        # lo único que hace es debilitar la placa con un agujero inútil.
        if max(x1 - x0, y1 - y0) < ancho:
            continue
        out.append([(x, y) for x, y in g.exterior.coords])
    return out


def _anillo_a_edges(anillo, mark: bool) -> List[Edge2D]:
    """Anillo cerrado -> aristas de encastre (se cortan) o de apoyo (se graban)."""
    out: List[Edge2D] = []
    for i in range(len(anillo) - 1):
        a, b = anillo[i], anillo[i + 1]
        out.append(
            Edge2D(a=Vec2(a[0], a[1]), b=Vec2(b[0], b[1]),
                   score=mark, joint=not mark)
        )
    return out


def _effective_category(
    group: GeometryGroup, overrides: Dict[int, str]
) -> str:
    return overrides.get(group.id, group.category)


def _clip_segment_to_rect(
    ax: float, ay: float, bx: float, by: float, w: float, h: float
) -> Optional[Tuple[float, float, float, float]]:
    """Recorta el segmento al rectángulo [0,w]x[0,h] (Liang–Barsky).

    Devuelve None si queda totalmente fuera. Se usa para que las ranuras de encastre
    nunca sobresalgan de la pieza: el nesting ubica los paneles por su width×height, así
    que cualquier trazo fuera de ese rectángulo se dibuja encima del panel vecino.
    """
    # Tolerancia: permite que la ranura toque el borde sin perderse por redondeo.
    eps = 1e-9
    dx, dy = bx - ax, by - ay
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax), (dx, w - ax), (-dy, ay), (dy, h - ay)):
        if abs(p) < eps:
            if q < -eps:
                return None  # paralelo y fuera
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return None
            if r < t1:
                t1 = r
    if t1 - t0 < eps:
        return None
    return (ax + t0 * dx, ay + t0 * dy, ax + t1 * dx, ay + t1 * dy)


def _fit_panel_bbox(
    edges: List[Edge2D], width_m: float, height_m: float
) -> Tuple[List[Edge2D], float, float]:
    """Red de seguridad: garantiza que width×height contenga TODA la geometría dibujada.

    El nesting separa las piezas por su bounding box declarado; si alguna arista
    quedara fuera, la pieza invadiría a su vecina en la plancha pese a la brecha.
    Si algo se sale, se re-encuadra (traslada y agranda) en vez de recortar geometría.
    """
    if not edges:
        return edges, width_m, height_m
    xs = [c for e in edges for c in (e.a.x, e.b.x)]
    ys = [c for e in edges for c in (e.a.y, e.b.y)]
    min_x, max_x = min(xs), min(max(xs), float("inf"))
    max_x = max(xs)
    min_y, max_y = min(ys), max(ys)
    tol = 1e-9
    if min_x >= -tol and min_y >= -tol and max_x <= width_m + tol and max_y <= height_m + tol:
        return edges, width_m, height_m
    ox, oy = min(min_x, 0.0), min(min_y, 0.0)
    shifted = [
        Edge2D(
            a=Vec2(e.a.x - ox, e.a.y - oy),
            b=Vec2(e.b.x - ox, e.b.y - oy),
            hole=e.hole,
            joint=getattr(e, "joint", False),
            score=getattr(e, "score", False),
            flex=getattr(e, "flex", False),
        )
        for e in edges
    ]
    return shifted, max(width_m, max_x - ox), max(height_m, max_y - oy)


def _slot_edges(
    ax: float, ay: float, bx: float, by: float, width: float, mark: bool = False
) -> List[Edge2D]:
    """
    Ranura de encastre (Misión 1-C): rectángulo cerrado de ancho `width` centrado en
    el segmento (ax,ay)-(bx,by), en el marco 2D del panel. `width` = grosor de la placa
    cortante, de modo que la placa cortada quede con la abertura justa por donde aquella
    la atraviesa. Si el grosor es ~0 (desconocido) cae a una sola línea de junta.
    - `mark=False` (encastre físico): aristas joint=True → capa CUT_INTERIOR (se corta).
    - `mark=True` (apoyo pegado, kind="surface"): aristas score=True → capa MARK_VECTOR
      (rojo, se graba, NO se corta): el footprint donde apoya la otra placa.
    """
    import math

    dx, dy = bx - ax, by - ay
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-9:
        return []

    def _e(a, b):
        return Edge2D(a=a, b=b, score=True) if mark else Edge2D(a=a, b=b, joint=True)

    if width < 1e-4:
        return [_e(Vec2(ax, ay), Vec2(bx, by))]

    ux, uy = dx / seg_len, dy / seg_len
    px, py = -uy, ux  # perpendicular en el plano del panel
    hw = width / 2.0
    c1 = Vec2(ax + px * hw, ay + py * hw)
    c2 = Vec2(bx + px * hw, by + py * hw)
    c3 = Vec2(bx - px * hw, by - py * hw)
    c4 = Vec2(ax - px * hw, ay - py * hw)
    return [_e(c1, c2), _e(c2, c3), _e(c3, c4), _e(c4, c1)]


def decompose_panels_from_groups(
    phase1: Phase1Result,
    opts: PipelineOptions,
    overrides: Optional[Dict[int, str]] = None,
    wall_wall_decisions: Optional[Dict[int, int]] = None,
    marks: Optional[List[int]] = None,
    plate_joints: Optional[List] = None,
    user_cuts: Optional[List[dict]] = None,
    flex: Optional[List[dict]] = None,
    mark_lines: Optional[List[dict]] = None,
    ribs: Optional[List[dict]] = None,
    columns: Optional[List[dict]] = None,
    adj_result: Optional[AdjustmentsResult] = None,
) -> Tuple[List[Panel], List[Panel]]:
    overrides = overrides or {}
    marks_set = set(marks or [])  # ids de grupos cuyas aberturas se graban (no se cortan)
    min_area = opts.min_area_m2 if opts.min_area_m2 is not None else 0.01

    # Refuerzos (nervios + columnas): PIEZAS NUEVAS independientes. NO tocan las placas
    # existentes (sin muescas); sólo se agregan al nesting. El usuario las pega a mano.
    reinforcement_pieces = build_reinforcements(ribs, columns)

    # Cortes manuales por grupo (user_cuts): group_id -> [cut, ...] en el marco del panel.
    cuts_by_group: Dict[int, list] = {}
    for c in user_cuts or []:
        gid = c.get("group_id") if isinstance(c, dict) else getattr(c, "group_id", None)
        if gid is not None:
            cuts_by_group.setdefault(int(gid), []).append(c)

    # Patrón de flexión por grupo (flex): un FlexSpec por group_id (kerf / auxético).
    flex_by_group = build_flex_by_group(parse_flex(flex))

    # Líneas de marca rojas por grupo (mark_lines): group_id -> [polilínea, ...] en UV
    # del panel (metros). Se graban (score/MARK_VECTOR), no cortan.
    mark_lines_by_group: Dict[int, list] = {}
    for ml in mark_lines or []:
        if not isinstance(ml, dict):
            continue
        gid = ml.get("group_id")
        pts = ml.get("points")
        if gid is None or not isinstance(pts, list) or len(pts) < 2:
            continue
        poly = []
        for p in pts:
            try:
                poly.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError, IndexError):
                poly = []
                break
        if len(poly) >= 2:
            mark_lines_by_group.setdefault(int(gid), []).append(poly)

    # Ranuras de encastre por placa CORTADA (Misión 1): cut_id -> [(P_a, P_b, ancho), ...]
    # en 3D. `ancho` = grosor de la placa cortante (define el ancho de la ranura).
    joints_by_cut: Dict[int, list] = {}
    for pj in plate_joints or []:
        joints_by_cut.setdefault(pj.cut_id, []).append(
            (pj.a, pj.b, pj.width, getattr(pj, "kind", "slot"), pj.cutter_id)
        )

    # No hace falta sembrar las sugerencias: compute_adjustments ya resuelve por su
    # cuenta toda junta sin decisión del usuario (una sola fuente de verdad).
    # `adj_result` viene precalculado desde `_decompose`, que necesita las decisiones
    # ANTES para pasárselas a resolve_plate_joints. Se reutiliza tal cual: recalcularlo
    # acá abriría de nuevo la puerta a que el recorte y la ranura decidan distinto.
    if adj_result is None:
        adj_result = compute_adjustments(
            phase1.joints,
            phase1.groups,
            wall_wall_decisions,
            phase1.faces,
        )

    # El ajuste tiene dos componentes en unidades distintas y hay que sumarlas:
    #   `delta`       ya está en metros de EDIFICIO (el voladizo del modelo).
    #   `delta_plate` está en metros de PLANCHA (la media placa de MDF): se multiplica
    #                 por scale_denom para que, tras el 1/scale_denom del nesting, queden
    #                 los 1.5 mm físicos a cualquier escala.
    def _delta_m(adj) -> float:
        return adj.delta + getattr(adj, "delta_plate", 0.0) * (opts.scale_denom or 1.0)

    height_adj: Dict[int, float] = {}       # recorta la BASE del muro
    height_top_adj: Dict[int, float] = {}   # recorta la CIMA del muro (techo encima)
    width_adjs: Dict[int, list] = {}
    plane_adjs: Dict[int, list] = {}        # recorta la pieza en su plano (losa entre muros)
    group_by_id: Dict[int, GeometryGroup] = {g.id: g for g in phase1.groups}
    for adj in adj_result.adjustments:
        if adj.axis == "plane":
            plane_adjs.setdefault(adj.group_id, []).append(adj)
        elif adj.axis == "height":
            height_adj[adj.group_id] = height_adj.get(adj.group_id, 0.0) + _delta_m(adj)
        elif adj.axis == "height_top":
            # Antes caía en width_adjs y recortaba el ANCHO del panel en vez de su cima.
            height_top_adj[adj.group_id] = height_top_adj.get(adj.group_id, 0.0) + _delta_m(adj)
        else:
            width_adjs.setdefault(adj.group_id, []).append(adj)

    wall_panels: List[Panel] = []
    floor_panels: List[Panel] = []
    wall_count = floor_count = 0

    # Normal canónica hacia AFUERA por grupo: garantiza que el espejado deje el quemado
    # del láser del lado interior aunque el archivo traiga normales invertidas.
    oriented_normals = orient_group_normals_outward(phase1.groups)

    for group in phase1.groups:
        cat = _effective_category(group, overrides)
        if cat == "discard":
            continue

        is_floor = cat == "floor"
        faces = [phase1.faces[fi] for fi in group.face_indices if fi < len(phase1.faces)]
        if not faces:
            continue

        result = project_faces_to_2d(
            faces, oriented_normals.get(group.id, group.representative_normal), "Y"
        )
        if not result:
            continue

        width_m, height_m, edges = result.width_m, result.height_m, result.edges
        # Desplazamiento acumulado por los recortes de ensamble: cada clip re-normaliza
        # el panel a un origen nuevo, así que lo que se proyecte después en el marco
        # ORIGINAL (las ranuras de encastre) debe restarlo o se dibuja fuera de la pieza.
        clip_off_u = 0.0
        clip_off_v = 0.0
        # Umbral por grupo: el filtro predeterminado sigue en min_area (1 m²), PERO si
        # el usuario reclasificó EXPLÍCITAMENTE este grupo (override: p. ej. escalones o
        # marcos rescatados de "descartado" → piso/pared), debe aparecer en la plancha
        # aunque mida menos; para esos sólo se filtran astillas degeneradas (0.01 m²).
        group_min_area = MIN_PANEL_AREA_M2 if group.id in overrides else min_area
        if width_m * height_m < group_min_area:
            continue

        # Cortes manuales (user_cuts): se aplican en el marco de project_faces_to_2d
        # (el mismo del front), antes de espejar. Pueden partir el panel en varias
        # piezas y dejar huecos; la línea es marca de pliegue/score. Se omiten los
        # trims de ensamble y los encastres en paneles con cortes manuales.
        cuts = cuts_by_group.get(group.id)
        if cuts:
            pieces, score_lines = apply_user_cuts(width_m, height_m, edges, cuts)
            for piece_edges in pieces:
                mnu = min(min(e.a.x, e.b.x) for e in piece_edges)
                mnv = min(min(e.a.y, e.b.y) for e in piece_edges)
                mxu = max(max(e.a.x, e.b.x) for e in piece_edges)
                mxv = max(max(e.a.y, e.b.y) for e in piece_edges)
                pw, ph = mxu - mnu, mxv - mnv
                if pw * ph < group_min_area:
                    continue
                norm = [
                    Edge2D(a=Vec2(e.a.x - mnu, e.a.y - mnv),
                           b=Vec2(e.b.x - mnu, e.b.y - mnv),
                           hole=e.hole)
                    for e in piece_edges
                ]
                p_edges = mirror_edges_horizontal(norm, pw)
                for (sa, sb) in score_lines:
                    mx, my = (sa.x + sb.x) / 2.0, (sa.y + sb.y) / 2.0
                    if mnu - 1e-6 <= mx <= mxu + 1e-6 and mnv - 1e-6 <= my <= mxv + 1e-6:
                        p_edges.append(
                            Edge2D(
                                a=Vec2(pw - (sa.x - mnu), sa.y - mnv),
                                b=Vec2(pw - (sb.x - mnu), sb.y - mnv),
                                score=True,
                            )
                        )
                if is_floor:
                    floor_count += 1
                    floor_panels.append(
                        Panel(
                            id=f"B{floor_count}",
                            group_name=f"floor_{floor_count}",
                            category="floor",
                            floor_index=0,
                            width_m=pw,
                            height_m=ph,
                            edges=p_edges,
                            source_group_id=group.id,
                            is_mark=group.id in marks_set,
                        )
                    )
                else:
                    wall_count += 1
                    wall_panels.append(
                        Panel(
                            id=f"A{wall_count}",
                            group_name=f"wall_{wall_count}",
                            category="wall",
                            floor_index=0,
                            width_m=pw,
                            height_m=ph,
                            edges=p_edges,
                            source_group_id=group.id,
                            is_mark=group.id in marks_set,
                        )
                    )
            continue

        height_delta = height_adj.get(group.id, 0.0)
        if height_delta < 0 and not is_floor:
            strip = min(-height_delta, height_m - 0.01)
            if strip > 0.001:
                base_at_min_v = result.v_up >= 0
                clipped = (
                    clip_panel_at_v(edges, strip, True)
                    if base_at_min_v
                    else clip_panel_at_v(edges, height_m - strip, False)
                )
                if clipped:
                    width_m, height_m, edges = (
                        clipped["width_m"],
                        clipped["height_m"],
                        clipped["edges"],
                    )
                    clip_off_u += clipped.get("offset_u", 0.0)
                    clip_off_v += clipped.get("offset_v", 0.0)

        # Techo/losa encima del muro → recortar la CIMA del panel (borde opuesto a la base).
        top_delta = height_top_adj.get(group.id, 0.0)
        if top_delta < 0 and not is_floor:
            strip = min(-top_delta, height_m - 0.01)
            if strip > 0.001:
                base_at_min_v = result.v_up >= 0
                clipped = (
                    clip_panel_at_v(edges, height_m - strip, False)
                    if base_at_min_v
                    else clip_panel_at_v(edges, strip, True)
                )
                if clipped:
                    width_m, height_m, edges = (
                        clipped["width_m"],
                        clipped["height_m"],
                        clipped["edges"],
                    )
                    clip_off_u += clipped.get("offset_u", 0.0)
                    clip_off_v += clipped.get("offset_v", 0.0)

        # El recorte lateral es una MUESCA acotada a la banda donde el vecino existe, no
        # un recorte del borde entero.
        #
        # Antes se guardaba un solo recorte por borde, el mayor. Cuando dos vecinos
        # distintos llegan al MISMO extremo pero a alturas distintas —una fachada que
        # arriba topa con un muro y abajo con otro, separados 20 cm— ganaba el recorte
        # mayor y se aplicaba a toda la altura: la pieza perdía 4.00 mm de material donde
        # el vecino de abajo sí necesitaba que llegara. Medido: 9 de 22 bordes con varios
        # vecinos salían mal, contra 6 de 68 con uno solo.
        #
        # Restar rectángulos también resuelve de raíz la acumulación que motivó aquel
        # "un recorte por borde": restar dos veces la misma zona no acumula, y restar dos
        # zonas distintas es justamente lo que hace falta.
        muescas: List[Tuple[bool, float, float, float]] = []
        for w_adj in width_adjs.get(group.id, []):
            # No hay guarda de signo. La había —`delta >= 0 and delta_plate >= 0`— de
            # cuando un ajuste sólo podía achicar: si no achicaba, no servía. Desde que
            # una pieza puede ALARGARSE, esa condición tira justo los casos en que la
            # pieza queda más corta que media placa y hay que estirarla hasta la cara de
            # su vecina. Medido: descartaba 30 ajustes del muro pasante sobre los tres
            # modelos (20 en demo, 6 en bfe62e18, 4 en casa simple), el 100% de lo que
            # filtraba. El único descarte legítimo es el ajuste NULO, y ése lo hace el
            # `abs(recorte) <= 0.001` de abajo.
            if is_floor:
                continue
            recorte = -_delta_m(w_adj)
            # NEGATIVO = la pieza tiene que CRECER para llegar a la cara de su vecina.
            # Antes se descartaba acá y la pieza quedaba con el largo crudo del modelo.
            if abs(recorte) <= 0.001:
                continue
            j = phase1.joints[w_adj.joint_index]
            u_j = dot(j.edge_mid, result.u_axis) - result.origin_u
            izquierda = u_j < result.width_m / 2
            # Banda que ocupa el VECINO sobre el eje v del panel. Se mide proyectando sus
            # vértices, no tomando su altura en el mundo: así vale para paneles
            # inclinados, donde v no es la vertical.
            otro = group_by_id.get(w_adj.against_group_id)
            v_lo, v_hi = 0.0, height_m
            if otro is not None:
                proy = [
                    dot(v, result.v_axis) - result.origin_v - clip_off_v
                    for fi in otro.face_indices
                    if 0 <= fi < len(phase1.faces)
                    for v in phase1.faces[fi].vertices
                ]
                if proy:
                    v_lo, v_hi = max(min(proy), 0.0), min(max(proy), height_m)
            if v_hi - v_lo > 0.001:
                muescas.append((izquierda, recorte, v_lo, v_hi))

        if muescas:
            # El material se saca del extremo DONDE ESTÁ LA JUNTA. Las dos ramas del clip
            # estaban invertidas en su momento: la pieza salía con el largo correcto pero
            # desplazada el recorte entero, así que en la junta se metía dentro del vecino
            # y en el extremo libre no llegaba.
            recortado = _aplicar_muescas(edges, width_m, height_m, muescas)
            if recortado:
                edges, width_m, height_m, off_u, off_v = recortado
                clip_off_u += off_u
                clip_off_v += off_v
            else:
                # La muesca necesita el POLÍGONO del material y hay piezas cuyo contorno no
                # cierra: cuando una abertura llega al borde (una puerta que baja hasta el
                # piso) las jambas salen colapsadas y no hay cara que recortar. Ahí se cae
                # al recorte de borde entero de siempre, que sólo mira las aristas.
                #
                # Perder el recorte NO es una opción: al introducir la muesca, 4 piezas de
                # un modelo del corpus se quedaron sin ningún recorte —8.00 m en vez de
                # 7.40— y aparecieron 3 choques donde no había ninguno. Una muesca que no
                # se puede calcular tiene que degradar al comportamiento anterior, nunca a
                # no recortar.
                # Sólo los recortes: `clip_panel_at_u` corta, no agranda. Un ajuste
                # negativo se pierde acá, y por eso este camino es el respaldo y no el
                # principal.
                por_borde: Dict[bool, float] = {}
                for izquierda, prof, _v_lo, _v_hi in muescas:
                    if prof > por_borde.get(izquierda, 0.0):
                        por_borde[izquierda] = prof
                for izquierda, prof in por_borde.items():
                    strip = min(prof, width_m - 0.01)
                    if strip <= 0.001:
                        continue
                    clipped = (
                        clip_panel_at_u(edges, strip, True)
                        if izquierda
                        else clip_panel_at_u(edges, width_m - strip, False)
                    )
                    if clipped:
                        width_m = clipped["width_m"]
                        height_m = clipped["height_m"]
                        edges = clipped["edges"]
                        clip_off_u += clipped.get("offset_u", 0.0)
                        clip_off_v += clipped.get("offset_v", 0.0)

        # Recorte EN EL PLANO de la pieza contra un grupo continuo (axis="plane"): el caso
        # del entrepiso que tiene que ENTRAR entre dos muros que siguen de largo por
        # arriba y por abajo. El eje (u o v) y el lado salen de la normal de ese muro, no
        # de `v_up`, que en una losa no significa nada. Los pisos no recibían ningún
        # recorte, así que el entrepiso salía con el ancho que tiene entre las pieles del
        # modelo y al armar la maqueta no entraba.
        por_lado: Dict[Tuple[str, bool], float] = {}
        for p_adj in plane_adjs.get(group.id, []):
            recorte = -_delta_m(p_adj)
            if recorte <= 0.001:
                continue
            otro = group_by_id.get(p_adj.against_group_id)
            if otro is None:
                continue
            n_otro = normalize(otro.representative_normal)
            du = dot(n_otro, result.u_axis)
            dv = dot(n_otro, result.v_axis)
            eje = "u" if abs(du) >= abs(dv) else "v"
            # La normal del muro apunta hacia afuera: si va en el sentido creciente del
            # eje, el borde que toca ese muro es el ALTO.
            alto = (du if eje == "u" else dv) > 0
            clave = (eje, alto)
            if recorte > por_lado.get(clave, 0.0):
                por_lado[clave] = recorte

        for (eje, alto), recorte in sorted(por_lado.items()):
            limite = (width_m if eje == "u" else height_m) - 0.01
            strip = min(recorte, limite)
            if strip <= 0.001:
                continue
            if eje == "u":
                clipped = (
                    clip_panel_at_u(edges, width_m - strip, False)
                    if alto
                    else clip_panel_at_u(edges, strip, True)
                )
            else:
                clipped = (
                    clip_panel_at_v(edges, height_m - strip, False)
                    if alto
                    else clip_panel_at_v(edges, strip, True)
                )
            if clipped:
                width_m, height_m, edges = (
                    clipped["width_m"],
                    clipped["height_m"],
                    clipped["edges"],
                )
                clip_off_u += clipped.get("offset_u", 0.0)
                clip_off_v += clipped.get("offset_v", 0.0)

        edges = mirror_edges_horizontal(edges, width_m)

        # Misión 1 (C): si esta placa es la CORTADA/receptora de una junta, agregar la
        # RANURA de encastre (kind="slot", joint → CUT_INTERIOR, se corta) o el FOOTPRINT
        # de apoyo (kind="surface", score → MARK_VECTOR rojo, se graba, no se corta). El
        # segmento 3D se proyecta al marco del panel y se espeja igual que el contorno.
        # Se resta clip_off_u/v (desplazamiento de los recortes) y se descartan las
        # ranuras que caen fuera de la pieza ya recortada: si no, se dibujaban en el
        # marco viejo y quedaban fuera del bounding box del panel, invadiendo al vecino
        # en la plancha (el nesting sólo conoce width_m × height_m).
        slots_cortadas: List[int] = []
        for (pa, pb, slot_w, kind, cutter_id) in joints_by_cut.get(group.id, []):
            ua = dot(pa, result.u_axis) - result.origin_u - clip_off_u
            va = dot(pa, result.v_axis) - result.origin_v - clip_off_v
            ub = dot(pb, result.u_axis) - result.origin_u - clip_off_u
            vb = dot(pb, result.v_axis) - result.origin_v - clip_off_v
            # slot_w viene en metros de PLANCHA (es el material real). Se escala por
            # scale_denom para que tras el 1/scale_denom del nesting la ranura mida
            # exactamente lo que la placa que la atraviesa, en cualquier escala.
            slot_w_m = slot_w * (opts.scale_denom or 1.0)
            # La ranura es el RECTÁNGULO intersecado con el MATERIAL de la pieza.
            #
            # Antes se recortaba la RECTA contra el bounding box del panel achicado media
            # ranura. Eso tenía dos defectos, los dos visibles midiendo el DXF:
            #   - acortaba la ranura TAMBIÉN a lo largo: media ranura en cada punta. En un
            #     modelo real la ranura salía de 65.65mm donde la placa que tiene que
            #     entrar mide 70.00mm, o sea 4.35mm de menos, y no encastraba;
            #   - recortaba contra el BOUNDING BOX y no contra el contorno, así que en una
            #     pieza en L o con aberturas la ranura se dibujaba al aire, donde la pieza
            #     no tiene material.
            # Intersecar con el material resuelve las dos: conserva el largo donde hay
            # placa, no dibuja nada donde no la hay, y nunca asoma del contorno (así que
            # tampoco puede agrandar la pieza vía _fit_panel_bbox).
            trozos = _slot_dentro_del_material(
                edges, width_m - ua, va, width_m - ub, vb, slot_w_m
            )
            if not trozos:
                continue
            for anillo in trozos:
                edges.extend(_anillo_a_edges(anillo, mark=(kind == "surface")))
            # La ranura SOBREVIVIÓ: recién ahora es material que se quita de verdad. Las
            # que caen fuera de la pieza ya recortada no se registran, porque el tope ya
            # resolvió esa junta y la ranura sobraba.
            if kind != "surface":
                slots_cortadas.append(cutter_id)

        # Patrón de flexión (flex): se corta DENTRO del panel YA proyectado, trimado y
        # espejado — es decir, sobre EXACTAMENTE la misma silueta/orientación que sin
        # flex. Aplicar kerf sólo AGREGA huecos internos; no re-rota ni re-desarrolla el
        # panel (Problema 1 del contrato v3). spacing/ligament/kerf llegan en metros de
        # maqueta → se escalan por scale_denom (tras el 1/scale_denom del nesting quedan
        # con el tamaño físico pedido en la plancha).
        spec = flex_by_group.get(group.id)
        if spec is not None:
            scale = opts.scale_denom or 1.0
            spec_m = replace(
                spec,
                spacing_m=spec.spacing_m * scale,
                ligament_m=spec.ligament_m * scale,
                kerf_width_m=spec.kerf_width_m * scale,
            )
            edges = list(edges) + apply_flex_to_panel(width_m, height_m, edges, spec_m)

        # Líneas de marca ROJAS (grabado/score): se graban sobre el panel ya proyectado,
        # recortadas al material (contorno + aberturas). No cortan ni cambian la silueta.
        mlines = mark_lines_by_group.get(group.id)
        if mlines:
            edges = list(edges) + apply_mark_lines_to_panel(width_m, height_m, edges, mlines)

        # El bounding box declarado debe contener TODO lo dibujado: el nesting separa
        # las piezas por width×height, así que cualquier trazo fuera invadiría al vecino.
        edges, width_m, height_m = _fit_panel_bbox(edges, width_m, height_m)

        # Marco 3D de la pieza YA RECORTADA, para que el instructivo dibuje lo que se
        # corta y no la proyección cruda del modelo. Sale del MISMO cálculo que la
        # plancha: no hay una segunda ruta que pueda discrepar.
        marco = compute_group_placement(
            faces, oriented_normals.get(group.id, group.representative_normal), "Y"
        )
        if marco is not None:
            marco = shift_placement(
                marco, clip_off_u, clip_off_v, width_m, height_m,
                panel_cut_area_m2(edges, width_m, height_m),
            )

        if is_floor:
            floor_count += 1
            floor_panels.append(
                Panel(
                    id=f"B{floor_count}",
                    group_name=f"floor_{floor_count}",
                    category="floor",
                    floor_index=0,
                    width_m=width_m,
                    height_m=height_m,
                    edges=edges,
                    source_group_id=group.id,
                    is_mark=group.id in marks_set,
                    frame=marco,
                    slots_against=slots_cortadas,
                )
            )
        else:
            wall_count += 1
            wall_panels.append(
                Panel(
                    id=f"A{wall_count}",
                    group_name=f"wall_{wall_count}",
                    category="wall",
                    floor_index=0,
                    width_m=width_m,
                    height_m=height_m,
                    edges=edges,
                    source_group_id=group.id,
                    is_mark=group.id in marks_set,
                    frame=marco,
                    slots_against=slots_cortadas,
                )
            )

    # Refuerzos (nervios R#, columnas C#) como piezas propias → entran al nesting y al
    # precio. NO modifican ninguna placa existente. size_m/height_m vienen en metros de
    # MAQUETA (el material físico a cortar); los paneles del edificio están en metros de
    # edificio y el nesting los reduce por 1/scale_denom, así que la pieza se escala por
    # scale_denom para que en la plancha quede a su tamaño físico pedido.
    rscale = opts.scale_denom or 1.0
    rib_n = col_n = 0
    for piece in reinforcement_pieces:
        if piece.width_m * piece.height_m < 1e-6:
            continue
        if piece.kind == "column":
            col_n += 1
            pid = f"C{col_n}"
        else:
            rib_n += 1
            pid = f"R{rib_n}"
        wall_count += 1
        scaled_edges = [
            Edge2D(a=Vec2(e.a.x * rscale, e.a.y * rscale),
                   b=Vec2(e.b.x * rscale, e.b.y * rscale),
                   score=getattr(e, "score", False))
            for e in piece.edges
        ]
        wall_panels.append(
            Panel(
                id=pid,
                group_name=f"{piece.kind}_{pid}",
                category="wall",
                floor_index=0,
                width_m=piece.width_m * rscale,
                height_m=piece.height_m * rscale,
                edges=scaled_edges,
                source_group_id=-1,
                is_mark=False,
            )
        )

    return wall_panels, floor_panels


def _panels_to_nesting(panels: List[Panel], scale_denom: float) -> List[NestingPanel]:
    s = 1.0 / scale_denom
    out: List[NestingPanel] = []
    for p in panels:
        out.append(
            NestingPanel(
                id=p.id,
                category=p.category,
                width_m=p.width_m * s,
                height_m=p.height_m * s,
                edges=[
                    Edge(
                        a=Vec2(e.a.x * s, e.a.y * s),
                        b=Vec2(e.b.x * s, e.b.y * s),
                        hole=e.hole,
                        joint=getattr(e, "joint", False),
                        score=getattr(e, "score", False),
                        flex=getattr(e, "flex", False),
                    )
                    for e in p.edges
                ],
                is_mark=p.is_mark,
            )
        )
    return out


def _decompose(
    phase1: Phase1Result,
    opts: PipelineOptions,
    overrides: Optional[Dict[int, str]],
    wall_wall_decisions: Optional[Dict[int, int]],
    merges: Optional[List[List[int]]],
    marks: Optional[List[int]],
    user_cuts: Optional[List[dict]] = None,
    flex: Optional[List[dict]] = None,
    mark_lines: Optional[List[dict]] = None,
    ribs: Optional[List[dict]] = None,
    columns: Optional[List[dict]] = None,
) -> Tuple[Phase1Result, List[Panel], List[Panel], List]:
    """Aplica merges, resuelve encastres 3D y descompone a paneles 2D."""
    work = apply_merges(phase1, merges or [], overrides)

    # ORDEN IMPORTANTE: primero se decide quién cede en cada junta, y recién después se
    # resuelven las ranuras CONSUMIENDO esa decisión. Antes era al revés y cada una la
    # calculaba por su cuenta: la placa que se acortaba no era la que recibía la ranura en
    # 7 de 15 pares de un modelo real. Una sola decisión por junta, una sola vez.
    adj_result = compute_adjustments(
        work.joints, work.groups, wall_wall_decisions, work.faces
    )
    # Una decisión que no se pudo aplicar tiene que llegar al usuario. Es su elección
    # sobre quién cede en una esquina; si se descarta en silencio, la pieza sale sin el
    # recorte que pidió y no hay forma de enterarse antes de cortar el MDF.
    if adj_result.decisiones_ignoradas:
        work = replace(
            work,
            warnings=list(work.warnings or []) + adj_result.decisiones_ignoradas,
        )

    # Misión 1: resolver intersecciones placa-placa en 3D (encastres) sobre la
    # topología final (post-merges), antes de proyectar.
    # Sólo llevan ranura las juntas DETECTADAS que ningún tope resolvió. Las demás ya
    # están resueltas acortando una placa, y abrirles además una ranura sólo debilita la
    # pieza. El encaje losa-muro va por su propio camino (`_floor_slots`), porque ahí la
    # ranura no reemplaza a un tope: es lo único que sostiene la losa.
    sin_resolver = {
        pair_key(j.group_a, j.group_b)
        for j in adj_result.wall_wall_joints
        if j.yield_group_id is None
    }
    plate_joints = resolve_plate_joints(
        work.groups, work.faces,
        yield_by_pair=yield_by_pair(adj_result), sin_resolver=sin_resolver,
    )
    wall_panels, floor_panels = decompose_panels_from_groups(
        work, opts, overrides, wall_wall_decisions, marks, plate_joints,
        user_cuts, flex, mark_lines, ribs, columns,
        adj_result=adj_result,
    )
    return work, wall_panels, floor_panels, plate_joints


def panel_ids_by_group(
    wall_panels: List[Panel], floor_panels: List[Panel]
) -> Dict[int, str]:
    """Mapa group_id -> etiqueta de panel ("A1", "B2", ...) calculado por el back."""
    out: Dict[int, str] = {}
    for p in list(wall_panels) + list(floor_panels):
        gid = getattr(p, "source_group_id", None)
        if gid is not None:
            out[gid] = p.id
    return out


def compute_panel_id_by_group(
    phase1: Phase1Result,
    opts: Optional[PipelineOptions] = None,
    overrides: Optional[Dict[int, str]] = None,
) -> Dict[int, str]:
    """Etiquetas de panel por grupo (best-effort) para /upload y /recompute.

    Recibe los overrides para que las etiquetas coincidan con las piezas que
    realmente se van a cortar: sin ellos, un componente descartado por el usuario
    seguía recibiendo etiqueta y uno rescatado se quedaba sin ninguna.
    """
    if opts is None:
        opts = PipelineOptions(
            scale_denom=50.0, paper="A4", min_area_m2=None, sheet_config=None
        )
    try:
        _, wall_panels, floor_panels, _ = _decompose(
            phase1, opts, overrides, None, None, None
        )
        return panel_ids_by_group(wall_panels, floor_panels)
    except Exception:
        return {}


def _paper_sheet_candidates_m(paper_name: str):
    """Planchas candidatas (ancho, alto) en metros que entran A TAMAÑO REAL en el papel.

    Se descuenta exactamente lo que el PDF reserva: margen de página a los cuatro lados
    y el alto del rótulo "Plancha N". Antes se descontaban 10 mm fijos, que no alcanzaban,
    y el generador terminaba aplicando un "ajustar a la página" que encogía el dibujo
    ~15%: las piezas se habrían cortado más chicas de lo debido.

    Devuelve las dos orientaciones (apaisada y vertical) para que el nesting elija.
    """
    from core.services.pdf_writer import PAPERS, MM_TO_PT, PAGE_MARGIN_PT, SHEET_LABEL_SPACE_M

    p = PAPERS.get(paper_name, PAPERS["A4"])
    short = min(p["w"], p["h"]) / 1000.0
    long = max(p["w"], p["h"]) / 1000.0
    margin_m = PAGE_MARGIN_PT / MM_TO_PT / 1000.0  # pt -> mm -> m

    out = []
    for page_w, page_h in ((long, short), (short, long)):
        w = page_w - 2 * margin_m
        h = page_h - 2 * margin_m - SHEET_LABEL_SPACE_M
        if w > 0.05 and h > 0.05:
            out.append((w, h))
    return out or [(0.05, 0.05)]


def _nest_both(wall_np, floor_np, sheet_cfg: SheetConfig, scale: float):
    wn = nest_panels(wall_np, sheet_cfg, scale)
    fn = nest_panels(floor_np, sheet_cfg, scale)
    unplaced = len(wn.unplaced) + len(fn.unplaced)
    sheets = wn.sheets + fn.sheets
    util = sum(s.utilization for s in sheets) / len(sheets) if sheets else 0.0
    return wn, fn, unplaced, util


def compute_nesting(
    phase1: Phase1Result,
    opts: PipelineOptions,
    overrides: Optional[Dict[int, str]] = None,
    wall_wall_decisions: Optional[Dict[int, int]] = None,
    merges: Optional[List[List[int]]] = None,
    marks: Optional[List[int]] = None,
    user_cuts: Optional[List[dict]] = None,
    flex: Optional[List[dict]] = None,
    mark_lines: Optional[List[dict]] = None,
    ribs: Optional[List[dict]] = None,
    columns: Optional[List[dict]] = None,
) -> Tuple[Phase1Result, NestingResult, NestingResult, SheetConfig, Dict[int, str], List, Dict[str, Dict], List]:
    """Descompone y anida paneles. Compartido por /generate y /nesting-preview."""
    work, wall_panels, floor_panels, plate_joints = _decompose(
        phase1, opts, overrides, wall_wall_decisions, merges, marks,
        user_cuts, flex, mark_lines, ribs, columns,
    )

    sc = opts.sheet_config
    gap = sc.gap_m if sc else 0.003
    scale = opts.scale_denom
    page_mode = opts.page_mode or "one_per_sheet"
    wall_np = _panels_to_nesting(wall_panels, scale)
    floor_np = _panels_to_nesting(floor_panels, scale)

    # Plancha combinada: paredes y pisos juntos en las mismas planchas. Todo viaja por
    # wall_nesting (los ids A#/B#/R#/C# ya distinguen el tipo) y floor_nesting queda
    # vacío → el resto del flujo (serialización, DXF/PDF) no cambia.
    if opts.combine_sheets:
        wall_np = wall_np + floor_np
        floor_np = []

    if page_mode == "single_page":
        # Modo láser: plancha = sheet_config físico (comportamiento actual).
        sheet_cfg = SheetConfig(
            width_m=sc.width_m if sc else 1.0,
            height_m=sc.height_m if sc else 0.6,
            gap_m=gap,
        )
        wall_nesting = nest_panels(wall_np, sheet_cfg, scale)
        floor_nesting = nest_panels(floor_np, sheet_cfg, scale)
    else:
        # Modo cartón: plancha = papel − margen, auto-orientada (la que minimiza
        # piezas sin ubicar; desempate por mayor aprovechamiento).
        best = None  # (unplaced, -util, cfg, wn, fn)
        for (w, h) in _paper_sheet_candidates_m(opts.paper or "A4"):
            cfg = SheetConfig(width_m=w, height_m=h, gap_m=gap)
            wn, fn, unplaced, util = _nest_both(wall_np, floor_np, cfg, scale)
            key = (unplaced, -util)
            if best is None or key < best[0]:
                best = (key, cfg, wn, fn)
        _, sheet_cfg, wall_nesting, floor_nesting = best

    return (
        work,
        wall_nesting,
        floor_nesting,
        sheet_cfg,
        panel_ids_by_group(wall_panels, floor_panels),
        plate_joints,
        final_placements(wall_panels, floor_panels),
        # Fase D: ¿las piezas que se van a cortar arman el edificio? Se calcula acá, sobre
        # los paneles FINALES y con la escala ya elegida, porque la respuesta depende de
        # ambas cosas: a escalas gruesas la placa ocupa más edificio y aparecen choques
        # que a escalas finas no existen.
        _verificar(work, wall_panels, floor_panels, plate_joints, opts.scale_denom or 1.0),
    )


def _verificar(work, wall_panels, floor_panels, plate_joints, scale_denom):
    """Corre la verificación de ensamble, sin dejar que un fallo suyo tumbe la generación.

    Es un chequeo, no una etapa del cálculo: si se rompe, se pierde el aviso pero las
    piezas siguen saliendo. Lo contrario -que un verificador con un bug impida cortar-
    sería peor que no tenerlo.
    """
    try:
        from core.services.assembly_verify import verificar_ensamble
        from core.services.plate_intersect import pair_key

        detectadas = {pair_key(j.group_a, j.group_b) for j in work.joints}
        sin_resolver = {
            pair_key(j.group_a, j.group_b)
            for j in work.wall_wall_joints
            if j.yield_group_id is None
        }
        return verificar_ensamble(
            list(wall_panels) + list(floor_panels), plate_joints, scale_denom,
            detectadas, sin_resolver,
        )
    except Exception:
        import logging
        logging.getLogger("eficiencia2d.pipeline").exception("verificar_ensamble falló")
        return []


def final_placements(
    wall_panels: List[Panel], floor_panels: List[Panel]
) -> Dict[str, Dict]:
    """`group_id (str) -> marco de la pieza YA RECORTADA`, para el instructivo.

    Misma forma que `topology.placements` (origin / u_axis / v_axis / normal / width_m /
    height_m / mirrored) más `area_m2` y `panel_id`. La diferencia es que estas medidas
    son las que se van a cortar: `topology.placements` es la proyección cruda del modelo
    y en un modelo real 20 de 28 piezas salían más grandes ahí que en la plancha.
    """
    out: Dict[str, Dict] = {}
    for p in list(wall_panels) + list(floor_panels):
        if p.frame and p.source_group_id is not None and p.source_group_id >= 0:
            out[str(p.source_group_id)] = dict(p.frame, panel_id=p.id)
    return out


def placements_fieles(
    phase1: Phase1Result,
    scale_denom: float,
    overrides: Optional[Dict[int, str]] = None,
    wall_wall_decisions: Optional[Dict[int, int]] = None,
) -> Dict[str, Dict]:
    """Marco 3D de cada pieza YA RECORTADA, a la escala a la que se va a cortar.

    Es lo que el instructivo tiene que dibujar. `build_placements` proyecta el modelo
    crudo y por eso no conoce ningún recorte de encaje: sobre un modelo real dibujaba 19
    de 28 piezas más grandes de lo que se cortaba a 1:50, y 21 de 28 a 1:100, con excesos
    de hasta 17.8 mm de plancha. Con esa diferencia el instructivo no sirve para decidir
    si algo encaja, que es para lo único que existe.

    Depende de la escala y no puede no depender: los recortes valen 3 mm de MDF siempre,
    y esos 3 mm son 15 cm de edificio a 1:50 y 60 cm a 1:200.
    """
    opts = PipelineOptions(
        scale_denom=scale_denom, paper="A4", min_area_m2=1.0, sheet_config=None
    )
    _work, walls, floors, _pjs = _decompose(
        phase1, opts, overrides, wall_wall_decisions, None, None
    )
    return final_placements(walls, floors)


def generate_from_review(
    phase1: Phase1Result,
    opts: PipelineOptions,
    overrides: Optional[Dict[int, str]] = None,
    wall_wall_decisions: Optional[Dict[int, int]] = None,
    merges: Optional[List[List[int]]] = None,
    marks: Optional[List[int]] = None,
    user_cuts: Optional[List[dict]] = None,
    flex: Optional[List[dict]] = None,
    mark_lines: Optional[List[dict]] = None,
    ribs: Optional[List[dict]] = None,
    columns: Optional[List[dict]] = None,
) -> List[OutputFile]:
    work, wall_nesting, floor_nesting, sheet_cfg, _, _, _, _ = compute_nesting(
        phase1, opts, overrides, wall_wall_decisions, merges, marks,
        user_cuts, flex, mark_lines, ribs, columns,
    )
    scale = opts.scale_denom
    stem = work.stem

    files: List[OutputFile] = []
    paper_name = opts.paper or "A4"
    page_mode = opts.page_mode or "one_per_sheet"

    def add_nesting_outputs(nesting: NestingResult, label: str, prefix: str) -> None:
        if not nesting.sheets:
            return
        files.append(
            OutputFile(
                name=f"{stem}_{prefix}_con_referencias.dxf",
                blob=nested_sheets_to_dxf(nesting, True).encode("utf-8"),
            )
        )
        ref_pdf = generate_nesting_pdf(nesting, label, True, paper_name, page_mode)
        if ref_pdf:
            files.append(
                OutputFile(name=f"{stem}_{prefix}_con_referencias.pdf", blob=ref_pdf)
            )
        files.append(
            OutputFile(
                name=f"{stem}_{prefix}_corte.dxf",
                blob=nested_sheets_to_dxf(nesting, False).encode("utf-8"),
            )
        )
        cut_pdf = generate_nesting_pdf(nesting, label, False, paper_name, page_mode)
        if cut_pdf:
            files.append(OutputFile(name=f"{stem}_{prefix}_corte.pdf", blob=cut_pdf))

    if opts.combine_sheets:
        # Plancha combinada: todas las piezas viajan en wall_nesting.
        add_nesting_outputs(wall_nesting, "Piezas", "Piezas")
    else:
        add_nesting_outputs(wall_nesting, "Paredes", "Paredes")
        add_nesting_outputs(floor_nesting, "Pisos", "Pisos")

    facades = extract_facades(work.faces, "Y")
    floor_plans = extract_floor_plans(work.faces, "Y")
    plan_pdf = generate_pdf(facades, floor_plans, scale, opts.paper)
    if plan_pdf:
        files.append(OutputFile(name=f"{stem}_planos.pdf", blob=plan_pdf))

    return files
