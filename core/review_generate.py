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
from core.services.plate_intersect import (
    PLATE_THICKNESS_M,
    pair_key,
    resolve_plate_joints,
)
from core.services.recorte_csg import (
    aplicar_destinos,
    clave_de_recorte,
    construir_placa,
    contenido_del_grupo,
    destinos_de_ajustes,
    paralela_ya_resuelta_que_la_pisa,
    registrar_final,
    usar_caras,
)
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


def _poly_a_edges(poly, off_u: float, off_v: float) -> List[Edge2D]:
    """Aristas del polígono resultante, renormalizadas a su propio origen.

    Conserva la semántica de hueco: los anillos interiores salen con `hole=True` para que
    el emisor los mande a CUT_INTERIOR y no al contorno exterior.
    """
    out: List[Edge2D] = []
    for g in getattr(poly, "geoms", [poly]):
        if getattr(g, "area", 0.0) <= 1e-12:
            continue
        anillos = [(list(g.exterior.coords), False)]
        anillos += [(list(h.coords), True) for h in g.interiors]
        for coords, es_hueco in anillos:
            # Los booleanos de shapely dejan puntos repetidos donde el rectángulo toca el
            # contorno. Se quitan los PUNTOS repetidos y recién después se arman las
            # aristas entre los que quedan, encadenadas y cerrando contra el primero.
            #
            # Antes se salteaba la ARISTA de largo casi cero, y eso rompe el anillo: si
            # los dos puntos difieren en 1e-12, la arista anterior termina en uno y la
            # siguiente empieza en el otro. A la vista está cerrado, pero ya no comparten
            # nodo, `polygonize` no lo cierra y la pieza deja de ser un polígono para el
            # verificador, que la saltea sin avisar. Apareció al hacer crecer las losas:
            # 4 de 6 quedaban sin contorno en `Prueba app.obj`.
            pts: List[Tuple[float, float]] = []
            for (x, y) in coords:
                if pts and abs(pts[-1][0] - x) <= 1e-9 and abs(pts[-1][1] - y) <= 1e-9:
                    continue
                pts.append((x, y))
            while len(pts) > 1 and (
                abs(pts[0][0] - pts[-1][0]) <= 1e-9 and abs(pts[0][1] - pts[-1][1]) <= 1e-9
            ):
                pts.pop()
            if len(pts) < 3:
                continue
            for i in range(len(pts)):
                (ax, ay), (bx, by) = pts[i], pts[(i + 1) % len(pts)]
                out.append(
                    Edge2D(a=Vec2(ax - off_u, ay - off_v),
                           b=Vec2(bx - off_u, by - off_v), hole=es_hueco)
                )
    return out


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
    # `polygonize` ya devuelve cada cara CON sus huecos: el contorno del muro llega como
    # "muro menos ventana" y la ventana como otra cara. Para contar en cuántas regiones
    # está metida una cara hay que mirar el CONTORNO EXTERIOR de las otras: con la cara
    # entera, el punto de la ventana no está "dentro" del muro (cae en su hueco), la
    # ventana contaba como material y la unión la rellenaba. Toda abertura cerrada (una
    # ventana, un hueco de escalera) desaparecía de cualquier pieza que pasara por el
    # recorte de encaje. Medido: demo.obj a 1:100 cortaba un muro de 16 ventanas sin
    # ninguna.
    from shapely.geometry import Polygon as _Poligono

    llenas = [_Poligono(c.exterior) for c in caras]
    material = []
    for i, c in enumerate(caras):
        p = c.representative_point()
        dentro = sum(
            1 for j, o in enumerate(llenas) if j != i and o.area > c.area and o.contains(p)
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
    cache: Optional[Dict] = None,
) -> Tuple[List[Panel], List[Panel]]:
    """Piezas 2D de cada grupo, ya recortadas.

    `cache` es un diccionario que el llamador conserva entre llamadas sobre la MISMA
    topología y escala (el resolvedor arma las piezas varias veces cambiando sólo quién
    cede). Guarda lo que no depende de esas decisiones: las placas de cada grupo y, dentro
    de ellas, las huellas de sus vecinas.
    """
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

    # Placas para el motor booleano: marco 3D + contorno 2D de cada grupo, una sola vez.
    usar_caras(phase1.faces)
    # Las placas se conservan POR CONTENIDO (las coordenadas de las caras del grupo y su
    # normal), no por id ni por índices de cara: así sobreviven de un pedido al siguiente
    # dentro de una sesión. Ni el id ni los índices alcanzan: dos fusiones distintas
    # producen un grupo nuevo con EL MISMO id y EL MISMO índice de cara (el siguiente
    # libre) y geometría distinta. Confundirlos cortaba la pieza de la otra fusión.
    por_firma: Optional[Dict] = None
    recortes: Optional[Dict] = None
    if cache is not None:
        por_firma = cache.setdefault("placas", {})
        recortes = cache.setdefault("recortes", {})
    placas_csg: Dict[int, object] = {}
    for _g in phase1.groups:
        if _effective_category(_g, overrides) == "discard":
            continue
        _n = oriented_normals.get(_g.id, _g.representative_normal)
        # Sin el id: cuando el usuario fusiona dos piezas, los grupos que se funden solos
        # por escala reciben OTRO id (el siguiente libre se corre) y son exactamente las
        # mismas piezas. Con el id en la firma se recalculaban unas 85 piezas por una
        # fusión que toca 4.
        _firma = (
            contenido_del_grupo(_g, phase1.faces, cache),
            round(_n.x, 6), round(_n.y, 6), round(_n.z, 6),
        )
        _pl = por_firma.get(_firma, False) if por_firma is not None else False
        if _pl is False:
            _pl = construir_placa(_g, phase1.faces, _n)
            if _pl is not None:
                _pl.firma = _firma
            if por_firma is not None:
                por_firma[_firma] = _pl
        elif _pl is not None:
            # Misma placa que en un armado anterior: se olvida lo que SÍ depende de las
            # decisiones (el contorno resuelto) y se conserva lo que no (las huellas).
            _pl.poly_final = None
            _pl.crecio = False
            _pl.group_id = _g.id      # mismo contenido, pero el id es el de ESTE armado
        if _pl is not None:
            placas_csg[_g.id] = _pl


    # Las placas no cambian durante el armado (sólo su contorno resuelto): la lista, el
    # nombre de cada una y la identidad del conjunto se calculan una vez.
    todas_las_placas = list(placas_csg.values())
    firma_de_placa = {pl.group_id: pl.firma for pl in todas_las_placas}
    universo = hash(tuple(pl.firma for pl in todas_las_placas)) if cache is not None else None
    bordes: Dict = cache.setdefault("bordes", {}) if cache is not None else {}

    cancelar = cache.get("cancelar") if cache else None
    for group in phase1.groups:
        if cancelar is not None:
            cancelar()          # cálculo en segundo plano: se abandona si llega un pedido
        cat = _effective_category(group, overrides)
        if cat == "discard":
            continue

        is_floor = cat == "floor"
        faces = [phase1.faces[fi] for fi in group.face_indices if fi < len(phase1.faces)]
        if not faces:
            continue

        # La proyección cruda y el marco 3D dependen sólo de las caras del grupo y de su
        # normal: no de la escala, ni de quién cede, ni de lo que haya decidido el
        # usuario. Eran la mitad del costo de cada armado y el resolvedor arma varias
        # veces por pedido. Se recuerdan por contenido, igual que las placas. Nadie los
        # modifica: todo lo que sigue trabaja sobre copias.
        normal_g = oriented_normals.get(group.id, group.representative_normal)
        if cache is not None:
            firma_g = (
                contenido_del_grupo(group, phase1.faces, cache),
                round(normal_g.x, 9), round(normal_g.y, 9), round(normal_g.z, 9),
            )
            proyecciones = cache.setdefault("proyecciones", {})
            result = proyecciones.get(firma_g, False)
            if result is False:
                result = project_faces_to_2d(faces, normal_g, "Y")
                proyecciones[firma_g] = result
        else:
            firma_g = None
            result = project_faces_to_2d(faces, normal_g, "Y")
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

        # ---------------------------------------------------------------
        # DESTINOS + BOOLEANO. Una sola pasada, sin análisis de casos.
        #
        # Reemplaza la cadena procedural que aplicaba en secuencia el recorte de base, el
        # de cima, las muescas laterales y el recorte en el plano — cada uno con su propio
        # criterio de lado y arrastrando `clip_off_u/v` entre pasos. Esa cadena fue la
        # causa directa de tres defectos medidos: la decisión izquierda/derecha se tomaba
        # en un marco desactualizado, el respaldo por recorte de borde perdía todo ajuste
        # que pidiera ALARGAR (porque `clip_panel_at_u` sólo sabe cortar), y había que
        # distinguir a mano si la junta caía en el extremo o en la panza.
        #
        # Ahora cada ajuste se traduce UNA vez a una coordenada objetivo en el marco crudo,
        # y el motor booleano la ejecuta: resta lo que sobra, une lo que falta. Las dos
        # operaciones se hacen siempre y la que no corresponde queda vacía sola.
        ajustes_pieza = (
            [a for a in adj_result.adjustments if a.group_id == group.id]
        )
        placa = placas_csg.get(group.id)
        espesor_edif = PLATE_THICKNESS_M * (opts.scale_denom or 1.0)
        # También entra la pieza SIN ajustes si una paralela ya resuelta creció: ese
        # material está tomado y hay que cedérselo aunque a ésta nadie le pida nada.
        if placa is not None and (
            ajustes_pieza
            or paralela_ya_resuelta_que_la_pisa(
                placa, todas_las_placas, espesor_edif
            )
        ):
            destinos = destinos_de_ajustes(
                ajustes_pieza, placa, opts.scale_denom or 1.0, group_by_id
            )
            # Cede = el ajuste de ancho le QUITA la media placa (la que pasa la suma).
            cede_ante = {
                a.against_group_id for a in ajustes_pieza
                if a.axis == "width" and getattr(a, "delta_plate", 0.0) < 0
                and a.against_group_id is not None
            }
            obstaculos = todas_las_placas
            clave = None
            if recortes is not None:
                clave = clave_de_recorte(
                    placa, destinos, obstaculos, espesor_edif, cede_ante,
                    universo=universo, firma_de=firma_de_placa,
                )
                res_csg = recortes.get(clave, False)
                if res_csg is False:
                    res_csg = aplicar_destinos(
                        placa, destinos, obstaculos=obstaculos,
                        espesor_m=espesor_edif, cede_ante=cede_ante,
                    )
                    recortes[clave] = res_csg
            else:
                res_csg = aplicar_destinos(
                    placa, destinos, obstaculos=obstaculos,
                    espesor_m=espesor_edif, cede_ante=cede_ante,
                )
            if res_csg is not None:
                poly_csg, width_m, height_m, off_u, off_v = res_csg
                registrar_final(placa, poly_csg)
                if clave is not None:
                    # Mismo recorte, mismos bordes. La lista no se toca: lo que sigue
                    # (espejado) crea bordes nuevos.
                    edges = bordes.get(clave)
                    if edges is None:
                        edges = bordes[clave] = _poly_a_edges(poly_csg, off_u, off_v)
                else:
                    edges = _poly_a_edges(poly_csg, off_u, off_v)
                clip_off_u += off_u
                clip_off_v += off_v

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
        if firma_g is not None:
            marcos = cache.setdefault("marcos", {})
            marco = marcos.get(firma_g, False)
            if marco is False:
                marco = compute_group_placement(faces, normal_g, "Y")
                marcos[firma_g] = marco
        else:
            marco = compute_group_placement(faces, normal_g, "Y")
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


def _vanos_tapados_por_dentro(g, faces, normal, cache: Optional[Dict]) -> int:
    """Cuántas aberturas de la cara principal de un grupo quedan tapadas por OTRAS caras
    del mismo grupo.

    Un vidrio modelado dentro del espesor del muro queda agrupado con el muro desde la
    clasificación: nunca llega a ser un componente aparte. La pieza se arma con todas las
    caras del grupo, así que la ventana sale cerrada, y sin esto nadie se entera.

    Se separan las caras por su distancia al plano; la tanda de mayor área es la cara del
    muro, y se cuenta cuántos de SUS huecos caen sobre material del conjunto.
    """
    from shapely.affinity import translate
    from shapely.geometry import Polygon

    from core.services.cutting_sheet import _edges_to_polygon

    clave = None
    if cache is not None:
        clave = (
            contenido_del_grupo(g, faces, cache),
            round(normal.x, 6), round(normal.y, 6), round(normal.z, 6),
        )
        vistos = cache.setdefault("vanos_tapados", {})
        if clave in vistos:
            return vistos[clave]
    tapados = 0
    try:
        n = normalize(normal)
        tandas: Dict[float, list] = {}
        for i in g.face_indices:
            f = faces[i]
            if not f.vertices:
                continue
            tandas.setdefault(round(dot(f.vertices[0], n), 3), []).append(f)
        if len(tandas) >= 2:
            proyectadas = [
                (r, _edges_to_polygon(r.edges))
                for r in (project_faces_to_2d(fs, n, "Y") for fs in tandas.values()) if r
            ]
            proyectadas = [(r, p) for r, p in proyectadas if p is not None and not p.is_empty]
            todo = project_faces_to_2d([faces[i] for i in g.face_indices], n, "Y")
            if proyectadas and todo:
                r0, cara = max(proyectadas, key=lambda t: t[1].area)
                material = _edges_to_polygon(todo.edges)
                du, dv = r0.origin_u - todo.origin_u, r0.origin_v - todo.origin_v
                for parte in getattr(cara, "geoms", [cara]):
                    for hueco in parte.interiors:
                        vano = translate(Polygon(hueco), du, dv)
                        if vano.area > 1e-6 and material.intersection(vano).area >= 0.5 * vano.area:
                            tapados += 1
    except Exception:
        tapados = 0
    if clave is not None:
        cache["vanos_tapados"][clave] = tapados
    return tapados


def _avisar_vanos_cerrados(
    work: Phase1Result, en_aberturas: Dict[int, int],
    overrides: Optional[Dict[int, str]] = None, cache: Optional[Dict] = None,
) -> Phase1Result:
    """Avisa qué aberturas salen CERRADAS porque el modelo trae algo adentro.

    Una ventana vacía se corta abierta. Una ventana con un vidrio o una carpintería
    modelados en el plano del muro se corta cerrada: ese elemento ocupa la misma placa que
    el muro. Puede ser un componente aparte que se funde con el muro (`en_aberturas`, de
    `detectar_placas_superpuestas`) o caras que ya vienen agrupadas con él
    (`_vanos_tapados_por_dentro`). El usuario tiene que saberlo.
    """
    overrides = overrides or {}
    by_id = {g.id: g for g in work.groups}
    por_muro: Dict[int, int] = {}
    for gid, muro in en_aberturas.items():
        por_muro[muro] = por_muro.get(muro, 0) + 1
    normales = orient_group_normals_outward(work.groups)
    for g in work.groups:
        if _effective_category(g, overrides) == "discard" or len(g.face_indices) < 2:
            continue
        n = _vanos_tapados_por_dentro(
            g, work.faces, normales.get(g.id, g.representative_normal), cache
        )
        if n:
            por_muro[g.id] = por_muro.get(g.id, 0) + n
    if not por_muro:
        return work
    total = sum(por_muro.values())
    detalle = "; ".join(
        f"{by_id[m].label or m}: {n}"
        for m, n in sorted(por_muro.items(), key=lambda t: -t[1])[:4]
    ) + ("…" if len(por_muro) > 4 else "")
    aviso = (
        f"{total} abertura(s) salen CERRADAS: el modelo tiene algo dentro del vano "
        "(vidrios, carpinterías, paños) y eso ocupa la misma placa que el muro. Para que "
        "salgan abiertas, descartá esos elementos en el visor o quitalos del modelo. "
        f"{detalle}"
    )
    return replace(work, warnings=list(work.warnings or []) + [aviso])


def _una_placa_por_lugar(
    work: Phase1Result,
    scale_denom: float,
    overrides: Optional[Dict[int, str]],
    wall_wall_decisions: Optional[Dict[int, int]],
    cache: Optional[Dict] = None,
) -> Tuple[Phase1Result, Optional[Dict[int, int]]]:
    """Funde los grupos paralelos que a esta escala son la misma placa de MDF.

    Ver `core.services.placas_superpuestas`. Devuelve la topología fundida y las
    decisiones del usuario TRADUCIDAS a ella.

    Por qué hay que traducir las decisiones: viajan como {índice de junta -> grupo que
    cede} y el índice es posicional. Fundir grupos reconstruye la lista de juntas, así que
    el mismo índice pasa a nombrar otra junta. Acá se conoce la lista de antes y la de
    después, entonces la decisión se pasa a {par de grupos -> quién cede}, que es lo único
    que significa algo, y se vuelve a ubicar por par en la lista nueva. Si la decisión era
    sobre una piel que se fundió, vale para la pieza que la absorbió.
    """
    from core.services.placas_superpuestas import detectar_placas_superpuestas

    overrides = overrides or {}
    en_aberturas: Dict[int, int] = {}
    try:
        conjuntos = detectar_placas_superpuestas(
            work.groups, work.faces, scale_denom,
            lambda g: _effective_category(g, overrides),
            orient_group_normals_outward(work.groups),
            cache=cache,
            en_aberturas=en_aberturas,
        )
    except Exception:
        import logging
        logging.getLogger("eficiencia2d.pipeline").exception(
            "detectar_placas_superpuestas falló; se sigue sin fundir"
        )
        return work, wall_wall_decisions
    work = _avisar_vanos_cerrados(work, en_aberturas, overrides, cache)
    if not conjuntos:
        return work, wall_wall_decisions

    fundido = apply_merges(work, conjuntos, overrides)
    vivos = {g.id for g in fundido.groups}
    by_id = {g.id: g for g in work.groups}
    # Quién absorbió a quién. `apply_merges` puede negarse a fundir un miembro (lo separa
    # una losa): ése sigue vivo y se representa a sí mismo.
    absorbe: Dict[int, int] = {}
    nombres: List[str] = []
    for ids in conjuntos:
        sobrevivientes = [i for i in ids if i in vivos]
        if not sobrevivientes:
            continue
        principal = max(sobrevivientes, key=lambda i: by_id[i].total_area)
        idos = [i for i in ids if i not in vivos]
        for i in idos:
            absorbe[i] = principal
        if idos:
            nombres.append(
                f"{by_id[principal].label or principal} absorbe "
                + ", ".join(str(by_id[i].label or i) for i in idos)
            )
    if not absorbe:
        return fundido, wall_wall_decisions

    avisos = list(fundido.warnings or [])
    espesor_cm = PLATE_THICKNESS_M * scale_denom * 100.0
    avisos.append(
        f"A escala 1:{scale_denom:g} la placa de {PLATE_THICKNESS_M * 1000:g} mm ocupa "
        f"{espesor_cm:.0f} cm de edificio: {len(absorbe)} superficie(s) paralela(s) que "
        "en el modelo están más cerca que eso se cortan como UNA sola pieza con su "
        "vecina (no hay lugar físico para dos). "
        + "; ".join(nombres[:4]) + ("…" if len(nombres) > 4 else "")
    )

    decisiones = None
    if wall_wall_decisions:
        nuevo_indice: Dict[Tuple[int, int], int] = {}
        for ji, j in enumerate(fundido.joints):
            nuevo_indice.setdefault(pair_key(j.group_a, j.group_b), ji)
        decisiones = {}
        for ji, cede in wall_wall_decisions.items():
            if not (0 <= ji < len(work.joints)):
                continue
            j = work.joints[ji]
            a = absorbe.get(j.group_a, j.group_a)
            b = absorbe.get(j.group_b, j.group_b)
            if a == b:
                continue  # la junta era entre dos pieles de la misma placa: ya no existe
            destino = nuevo_indice.get(pair_key(a, b))
            if destino is None:
                avisos.append(
                    f"La decisión sobre la unión {by_id[j.group_a].label or j.group_a} / "
                    f"{by_id[j.group_b].label or j.group_b} no se pudo conservar al "
                    "fundir superficies superpuestas: se resolvió automáticamente."
                )
                continue
            decisiones.setdefault(destino, absorbe.get(cede, cede))

    return replace(fundido, warnings=avisos, absorbed_groups=absorbe), decisiones


def _fusiones_pendientes_del_usuario(
    work: Phase1Result,
    merges: Optional[List[List[int]]],
    overrides: Optional[Dict[int, str]],
    wall_wall_decisions: Optional[Dict[int, int]],
) -> Tuple[Phase1Result, Optional[Dict[int, int]]]:
    """Segunda oportunidad para las fusiones que pidió el usuario y no se aplicaron.

    Las fusiones del usuario se aplican PRIMERO, sobre los grupos del modelo. Ahí dos
    tramos de muro separados por una losa no se unen (el entrepiso dejaría de entrar). Pero
    después corre la fusión automática por escala, y uno de esos tramos puede quedar
    fundido con una piel paralela que SÍ cruza la losa: la pieza resultante ya atraviesa
    ese nivel, y la guarda que frenó la fusión dejó de tener sentido para ella.

    El caso concreto: el reporte marcaba dos piezas montadas una sobre otra y ofrecía
    "cortarlas como una sola pieza". El usuario lo elegía, la fusión se frenaba en el
    primer paso por la losa, y el problema volvía idéntico. Medido en un modelo real: 5 de
    las 7 fusiones ofrecidas no cambiaban nada.

    Acá se vuelven a intentar, ya sobre la topología de esta escala, las fusiones cuyos
    miembros siguen siendo grupos distintos. Las guardas de `apply_merges` se aplican de
    nuevo, ahora contra las piezas que de verdad se van a cortar.
    """
    if not merges:
        return work, wall_wall_decisions
    absorbe = work.absorbed_groups or {}
    vivos = {g.id for g in work.groups}
    pendientes: List[List[int]] = []
    for conjunto in merges:
        actuales = sorted({absorbe.get(int(g), int(g)) for g in conjunto} & vivos)
        if len(actuales) >= 2:
            pendientes.append(actuales)
    if not pendientes:
        return work, wall_wall_decisions

    antes = work
    avisos_antes = list(work.warnings or [])
    fundido = apply_merges(replace(work, warnings=[]), pendientes, overrides)
    vivos_despues = {g.id for g in fundido.groups}
    por_id = {g.id: g for g in antes.groups}
    nuevos: Dict[int, int] = {}
    for conjunto in pendientes:
        quedan = [g for g in conjunto if g in vivos_despues]
        if not quedan:
            continue
        principal = max(quedan, key=lambda g: por_id[g].total_area)
        for g in conjunto:
            if g not in vivos_despues:
                nuevos[g] = principal
    if not nuevos:
        # No se pudo fundir nada tampoco ahora: se conserva la topología que había (con
        # sus avisos) en vez de pagar las juntas recalculadas por nada.
        return work, wall_wall_decisions

    # El aviso del primer intento ("no se fusionaron tramos separados por una losa") ya
    # no es cierto para lo que se acaba de fundir.
    resueltos = {str(por_id[g].label or g) for g in nuevos}
    avisos = [
        a for a in avisos_antes
        if not (a.startswith("No se fusionaron tramos") and any(r in a for r in resueltos))
    ] + list(fundido.warnings or [])
    absorbidos = dict(absorbe)
    for viejo, nuevo in list(absorbidos.items()):
        absorbidos[viejo] = nuevos.get(nuevo, nuevo)

    decisiones = None
    if wall_wall_decisions:
        indice: Dict[Tuple[int, int], int] = {}
        for ji, j in enumerate(fundido.joints):
            indice.setdefault(pair_key(j.group_a, j.group_b), ji)
        decisiones = {}
        for ji, cede in wall_wall_decisions.items():
            if not (0 <= ji < len(antes.joints)):
                continue
            j = antes.joints[ji]
            a, b = nuevos.get(j.group_a, j.group_a), nuevos.get(j.group_b, j.group_b)
            destino = indice.get(pair_key(a, b)) if a != b else None
            if destino is not None:
                decisiones.setdefault(destino, nuevos.get(cede, cede))
    return replace(fundido, warnings=avisos, absorbed_groups=absorbidos), decisiones


def _decisiones_por_par_a_indice(
    work: Phase1Result,
    wall_wall_decisions: Optional[Dict[int, int]],
    decisiones_por_par: Optional[List[dict]],
) -> Tuple[Phase1Result, Optional[Dict[int, int]]]:
    """Suma a las decisiones por índice las que llegan por PAR de grupos.

    `{group_a, group_b, yield_group_id}` es la forma estable de decir quién cede: no
    depende de en qué posición de la lista quedó la junta después de una fusión, una
    división o un cambio de escala. El índice posicional se sigue aceptando para no romper
    al front que ya lo manda; si llegan las dos para la misma junta, manda la del par.

    Un id de un grupo que a esta escala se fundió en otro vale para el que lo absorbió.
    """
    if not decisiones_por_par:
        return work, wall_wall_decisions
    absorbe = work.absorbed_groups or {}
    indice: Dict[Tuple[int, int], int] = {}
    for ji, j in enumerate(work.joints):
        indice.setdefault(pair_key(j.group_a, j.group_b), ji)

    out = dict(wall_wall_decisions or {})
    avisos: List[str] = []
    for d in decisiones_por_par:
        try:
            a = int(d["group_a"]); b = int(d["group_b"]); cede = int(d["yield_group_id"])
        except (KeyError, TypeError, ValueError):
            continue
        a, b, cede = absorbe.get(a, a), absorbe.get(b, b), absorbe.get(cede, cede)
        if a == b:
            continue   # las dos son hoy la misma placa: ya no hay junta
        ji = indice.get(pair_key(a, b))
        if ji is None or cede not in (a, b):
            avisos.append(
                f"La decisión sobre la unión entre los grupos {a} y {b} no se pudo "
                "aplicar: esa unión no existe con la clasificación actual. Se resolvió "
                "automáticamente."
            )
            continue
        out[ji] = cede
    if avisos:
        work = replace(work, warnings=list(work.warnings or []) + avisos)
    return work, out


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
    decisiones_por_par: Optional[List[dict]] = None,
    resolver: bool = True,
    pista: Optional[Dict] = None,
    sesion: Optional[Dict] = None,
) -> Tuple[Phase1Result, List[Panel], List[Panel], List]:
    """Aplica merges, resuelve encastres 3D y descompone a paneles 2D.

    `sesion` es un diccionario que el llamador conserva entre pedidos sobre el MISMO
    modelo, escala y cortes del usuario (ver `nueva_sesion`). Con ella una acción del
    usuario deja de ser "recalcular el modelo": se reusan las placas, el recorte de cada
    pieza cuyos datos no cambiaron y lo ya medido entre piezas que siguen igual. El
    resultado es idéntico al del cálculo completo. Además fija las ETIQUETAS: una pieza conserva la suya
    aunque otras se fusionen o se descarten. Sin sesión todo se calcula de cero y las
    etiquetas son correlativas, como siempre.

    `pista` es el `resolucion` de una corrida ANTERIOR sobre el mismo modelo y la misma
    escala (lo conserva la API entre un pedido y el siguiente). El resolvedor arranca de
    lo que ya había encontrado en vez de redescubrirlo: después de que el usuario toca un
    botón, casi todo el modelo sigue igual.

    Con `resolver` (por defecto) las uniones muro-muro que el usuario NO decidió pasan por
    `_resolver_uniones`, que prueba la decisión contraria donde la sugerida no cierra y se
    queda con la que el verificador mide mejor. Lo que resolvió queda en
    `work.resolucion`.
    """
    cache_de_armado: Dict = sesion.setdefault("armado", {}) if sesion is not None else {}
    work = apply_merges(phase1, merges or [], overrides)
    work, wall_wall_decisions = _una_placa_por_lugar(
        work, opts.scale_denom or 1.0, overrides, wall_wall_decisions,
        cache=cache_de_armado if sesion is not None else None,
    )
    work, wall_wall_decisions = _fusiones_pendientes_del_usuario(
        work, merges, overrides, wall_wall_decisions
    )
    work, wall_wall_decisions = _decisiones_por_par_a_indice(
        work, wall_wall_decisions, decisiones_por_par
    )

    # Con sesión NO se usa pista. La pista hace que el resolvedor parta de lo que resolvió
    # en otro estado, y con eso el resultado depende del camino: medido sobre un modelo de
    # 295 piezas, la misma fusión daba 208 piezas verificadas calculada de cero y 206
    # viniendo del estado anterior. Un resultado que depende del orden de los clics no se
    # puede guardar ni prever, y la plancha podría no ser la que el usuario revisó. La
    # sesión consigue la velocidad de otra manera (no recalcula las piezas que no
    # cambian) y devuelve EXACTAMENTE lo mismo que el cálculo completo.
    if sesion is not None:
        pista = None

    # Etiquetas de referencia: las que tiene cada pieza SIN acciones del usuario. Si este
    # pedido no trae fusiones ni descartes, salen de este mismo armado; si trae, hay que
    # calcularlas una vez (queda guardado en la sesión).
    etiquetas_base: Optional[Dict[int, List[str]]] = None
    if sesion is not None:
        etiquetas_base = sesion.get("etiquetas")
        if etiquetas_base is None and (merges or overrides):
            _w, m0, p0, _pj = _decompose(
                phase1, opts, None, None, None, marks, user_cuts, flex, mark_lines,
                ribs, columns, resolver=False,
                sesion={"armado": cache_de_armado, "etiquetas": {}},
            )
            etiquetas_base = _etiquetas_de(m0, p0)
            sesion["etiquetas"] = etiquetas_base

    def armar(decisiones: Optional[Dict[int, int]]):
        # Un cálculo en segundo plano se abandona si llega un pedido del usuario: quien
        # conserva la sesión deja acá una función que levanta una excepción en ese caso.
        cancelar = cache_de_armado.get("cancelar")
        if cancelar is not None:
            cancelar()
        """Decisiones -> piezas. Es la parte que depende de quién cede; todo lo de
        arriba (fusiones, traducción de decisiones) se hace una sola vez."""
        # ORDEN IMPORTANTE: primero se decide quién cede en cada junta, y recién después
        # se resuelven las ranuras CONSUMIENDO esa decisión. Antes era al revés y cada una
        # la calculaba por su cuenta: la placa que se acortaba no era la que recibía la
        # ranura en 7 de 15 pares de un modelo real. Una sola decisión por junta.
        adj = compute_adjustments(work.joints, work.groups, decisiones, work.faces)
        # Sólo llevan ranura las juntas DETECTADAS que ningún tope resolvió. Las demás ya
        # están resueltas acortando una placa, y abrirles además una ranura sólo debilita
        # la pieza. El encaje losa-muro va por su propio camino (`_floor_slots`), porque
        # ahí la ranura no reemplaza a un tope: es lo único que sostiene la losa.
        sin_resolver = {
            pair_key(j.group_a, j.group_b)
            for j in adj.wall_wall_joints
            if j.yield_group_id is None
        }
        pjs = resolve_plate_joints(
            work.groups, work.faces,
            yield_by_pair=yield_by_pair(adj), sin_resolver=sin_resolver,
        )
        muros, pisos = decompose_panels_from_groups(
            work, opts, overrides, decisiones, marks, pjs,
            user_cuts, flex, mark_lines, ribs, columns,
            adj_result=adj, cache=cache_de_armado,
        )
        if etiquetas_base:
            _aplicar_etiquetas(muros, pisos, etiquetas_base)
        return adj, muros, pisos, pjs

    resolucion = None
    uniones_medidas = None
    resultado = None
    if resolver:
        try:
            resultado, resolucion, uniones_medidas = _resolver_uniones(
                work, opts.scale_denom or 1.0, wall_wall_decisions, armar, pista,
                cache_de_medicion=(
                    sesion.setdefault("medicion", {}) if sesion is not None else None
                ),
            )
        except Exception:
            import logging
            logging.getLogger("eficiencia2d.pipeline").exception(
                "_resolver_uniones falló; se sigue con las decisiones sugeridas"
            )
            resultado = None
    if resultado is None:
        resultado = armar(wall_wall_decisions)
        resolucion = uniones_medidas = None
    adj_result, wall_panels, floor_panels, plate_joints = resultado
    if sesion is not None:
        if sesion.get("etiquetas") is None:
            # Pedido sin acciones del usuario: sus etiquetas SON las de referencia.
            sesion["etiquetas"] = _etiquetas_de(wall_panels, floor_panels)

    # Sólo informa: no cambia ninguna pieza. Ver `_absorbidos_completos`.
    work = replace(work, absorbed_groups=_absorbidos_completos(phase1, work, overrides))

    # Una decisión que no se pudo aplicar tiene que llegar al usuario. Es su elección
    # sobre quién cede en una esquina; si se descarta en silencio, la pieza sale sin el
    # recorte que pidió y no hay forma de enterarse antes de cortar el MDF.
    if adj_result.decisiones_ignoradas:
        work = replace(
            work,
            warnings=list(work.warnings or []) + adj_result.decisiones_ignoradas,
        )
    # Las juntas muro-muro de `work` pasan a ser las de ESTA corrida, con la decisión que
    # de verdad se aplicó en cada una (la del usuario, la sugerida o la que eligió el
    # resolvedor). Las que traía eran las de `apply_merges`, calculadas sin decisiones:
    # quien mirara ahí quién cede veía una respuesta que no era la que se cortó.
    work = replace(
        work, wall_wall_joints=adj_result.wall_wall_joints, resolucion=resolucion,
        uniones_medidas=uniones_medidas,
        pares_decididos=sorted({
            pair_key(work.joints[ji].group_a, work.joints[ji].group_b)
            for ji in (wall_wall_decisions or {})
            if 0 <= ji < len(work.joints)
        }),
    )
    aviso = _aviso_piezas_mas_finas_que_la_placa(
        wall_panels + floor_panels, opts.scale_denom or 1.0
    )
    if aviso:
        work = replace(work, warnings=list(work.warnings or []) + [aviso])
    return work, wall_panels, floor_panels, plate_joints


# Tope de rondas de "probar lo contrario y conservar lo que mejora". Cada ronda cuesta
# hasta dos armados completos de piezas. Se repite hasta que no quede ninguna unión con
# problema sin probar: así el resultado es un punto fijo, y el pedido siguiente (con la
# pista) no encuentra candidatas viejas que volver a probar. Con el tope en 2 quedaban
# candidatas sin probar y cada clic del usuario pagaba una ronda entera por ellas.
RONDAS_DEL_RESOLVEDOR = 5

_RANGO_ESTADO = {"no_arma": 0, "revisar": 1, "ok": 2}


def _pares_a_verificar(work) -> Tuple[set, set]:
    """`(previstos, de_apoyo)`: qué encuentros existen en el modelo. Ver
    `verificacion_uniones.verificar_uniones`."""
    from core.services.assembly_adjuster import (
        floor_wall_encounters,
        wall_end_fit_adjustments,
    )

    previstos = {pair_key(j.group_a, j.group_b) for j in work.joints}
    apoyos = {
        pair_key(a.group_id, a.against_group_id)
        for a in wall_end_fit_adjustments(work.groups, work.faces)
        if a.against_group_id is not None
    } | {
        pair_key(f.id, w.id)
        for f, w, _s in floor_wall_encounters(work.groups, work.faces)
    }
    return previstos, apoyos


def _puntaje(uniones) -> Tuple[int, int, float]:
    """Qué tan mal está un conjunto de uniones; menor es mejor.

    En orden: las que no arman, las que no están bien (para revisar, incluidas las que
    sólo se tocan de canto) y, a igualdad de eso, los milímetros acumulados de desvío.

    Una unión de canto cuenta igual que una para revisar, a propósito. Con un escalón
    propio y más barato, el resolvedor cambiaba una luz de medio milímetro por una esquina
    abierta de tres: técnicamente "subía de categoría" y a la vista quedaba peor. Medido en
    demo.obj a 1:50: 5 inversiones aceptadas para pasar de 3 luces a 2, con una esquina
    abierta más y dos bordes de esquina peor.
    """
    no_arma = otras = 0
    mm = 0.0
    for u in uniones:
        if u.estado == "ok":
            continue
        mm += u.medida_mm
        if u.estado == "no_arma":
            no_arma += 1
        else:
            otras += 1
    return (no_arma, otras, round(mm, 3))


def _absorbidos_completos(
    original: Phase1Result, work: Phase1Result, overrides: Optional[Dict[int, str]],
) -> Dict[int, int]:
    """`id de grupo sin pieza propia -> id del grupo en cuya pieza quedó`, para TODAS las
    fusiones: las automáticas por escala y las que pidió el usuario.

    Antes sólo figuraban las automáticas. Cuando el usuario fusionaba cuatro tramos de una
    fachada en el visor, la pieza resultante conservaba el id de uno y los otros tres
    simplemente dejaban de aparecer en `placements`, sin nada que dijera dónde habían ido:
    para el instructivo eran piezas desaparecidas. Es sólo información para quien dibuja;
    no interviene en el cálculo de ninguna pieza.
    """
    absorbe = dict(work.absorbed_groups or {})
    vivos = {g.id for g in work.groups}
    originales = {g.id: g for g in original.groups}
    duenio: Dict[int, int] = {}
    for g in work.groups:
        o = originales.get(g.id)
        if o is None or len(g.face_indices) != len(o.face_indices):
            for fi in g.face_indices:       # sólo los grupos que sumaron caras de otro
                duenio[fi] = g.id
    for g in original.groups:
        if g.id in vivos or g.id in absorbe or not g.face_indices:
            continue
        if _effective_category(g, overrides or {}) == "discard":
            continue
        d = duenio.get(g.face_indices[0])
        if d is not None and d != g.id:
            absorbe[g.id] = d
    return absorbe


def nueva_sesion() -> Dict:
    """Sesión vacía para `_decompose`. Vale para UN modelo, UNA escala y unos cortes
    manuales dados: quien la conserva es responsable de no mezclarla con otros."""
    return {}


def _etiquetas_de(muros: List[Panel], pisos: List[Panel]) -> Dict[int, List[str]]:
    """`id de grupo -> etiquetas de sus piezas`, en el orden en que salieron."""
    out: Dict[int, List[str]] = {}
    for p in list(muros) + list(pisos):
        if p.source_group_id is not None and p.source_group_id >= 0:
            out.setdefault(p.source_group_id, []).append(p.id)
    return out


def _aplicar_etiquetas(muros: List[Panel], pisos: List[Panel], base: Dict[int, List[str]]) -> None:
    """Le devuelve a cada pieza la etiqueta que tenía en el armado de referencia.

    Las etiquetas son correlativas (A1, A2, …) en el orden en que salen las piezas. Si el
    usuario descarta una pieza o fusiona dos, todas las que venían después se corrían un
    lugar: la que estaba mirando como A84 pasaba a ser A83, en el visor y en la plancha.

    Acá cada grupo conserva la etiqueta de referencia. La pieza que desaparece deja un
    hueco en la numeración. Una pieza que en la referencia no existía (un descartado que
    el usuario rescata, un grupo que cambió de categoría) recibe un número nuevo, después
    del último usado, para no chocar con ninguna.
    """
    ultimo = {"A": 0, "B": 0}
    for etiquetas in base.values():
        for e in etiquetas:
            if e[:1] in ultimo and e[1:].isdigit():
                ultimo[e[:1]] = max(ultimo[e[:1]], int(e[1:]))
    for lista, prefijo in ((muros, "A"), (pisos, "B")):
        vistos: Dict[int, int] = {}
        for p in lista:
            gid = p.source_group_id
            if gid is None or gid < 0:
                continue                      # refuerzos (R#, C#): numeración propia
            propias = [e for e in base.get(gid, []) if e.startswith(prefijo)]
            k = vistos.get(gid, 0)
            vistos[gid] = k + 1
            if k < len(propias):
                p.id = propias[k]
            else:
                ultimo[prefijo] += 1
                p.id = f"{prefijo}{ultimo[prefijo]}"


def _resolver_uniones(
    work, scale_denom, decisiones_usuario, armar, pista=None, cache_de_medicion=None,
):
    """Prueba la decisión contraria donde la sugerida no cierra, y conserva lo que mejora.

    Por qué existe
    --------------
    Quién cede en cada unión muro-muro lo sugiere una heurística (posición de la junta,
    espesores, topología). Acierta casi siempre, pero cuando no acierta la unión queda
    pisándose o con luz, y hasta ahora la única salida era que una persona lo notara en
    el visor y lo invirtiera a mano, unión por unión. En un modelo de 300 piezas eso no
    lo hace nadie con una entrega encima.

    Invertir una decisión sólo mueve UN extremo de dos piezas, así que el efecto es local:
    se pueden invertir muchas a la vez, medir cada una por separado y quedarse con las
    que mejoraron. El juez es el verificador exacto por unión, no una regla.

    Cómo decide
    -----------
    1. Mide las piezas con las decisiones de partida.
    2. Toma las uniones que fallan, tienen quién ceda y el usuario NO decidió.
    3. Arma de nuevo con todas ésas invertidas y mide.
    4. Conserva una inversión sólo si las uniones de SUS dos piezas quedaron mejor.
    5. Arma con las conservadas. Si el conjunto no mejora al de partida, descarta todo.

    Nunca toca una decisión del usuario. Sin `pista`, nunca devuelve algo peor que las
    decisiones sugeridas.

    Con `pista`
    -----------
    `pista` es el `resolucion` de una corrida anterior del mismo modelo a la misma escala.
    Se parte de las inversiones que esa corrida ya había encontrado y no se vuelve a
    probar lo que ya probó sin mejora. Es lo que hace que apretar un botón en el visor
    cueste un armado y no cinco: después de fusionar o descartar una pieza, el resto del
    modelo sigue igual. Una inversión heredada cuya unión ahora falla vuelve a ser
    candidata, así que la pista se corrige sola. Lo que se pierde es la comparación contra
    las decisiones sugeridas, que ya no se arman.

    Devuelve `(piezas, resolucion, uniones)`; `resolucion` es None si no había nada que
    probar ni que heredar.
    """
    from core.services.verificacion_uniones import verificar_uniones

    previstos, apoyos = _pares_a_verificar(work)
    if cache_de_medicion is None:
        cache_de_medicion = {}

    def medir(resultado):
        _adj, muros, pisos, _pjs = resultado
        return verificar_uniones(
            list(muros) + list(pisos), scale_denom, previstos, apoyos,
            cache=cache_de_medicion,
        )

    fijas = {
        pair_key(work.joints[ji].group_a, work.joints[ji].group_b)
        for ji in (decisiones_usuario or {})
        if 0 <= ji < len(work.joints)
    }
    indice: Dict[Tuple[int, int], int] = {}
    for ji, j in enumerate(work.joints):
        indice.setdefault(pair_key(j.group_a, j.group_b), ji)
    vivos = {g.id for g in work.groups}

    decisiones = dict(decisiones_usuario or {})
    invertidas: Dict[Tuple[int, int], int] = {}
    probadas: set = set()
    heredadas: set = set()
    # Cómo quedó cada unión cuando se aceptó su inversión: (rango de estado, mm). Una
    # heredada sólo se vuelve a probar si HOY está peor que entonces. Sin esto volvía a
    # ser candidata toda inversión que hubiera mejorado la unión sin dejarla perfecta, o
    # sea casi todas, y cada pedido pagaba una ronda entera por ellas.
    al_aceptar: Dict[Tuple[int, int], Tuple[int, float]] = {}
    if pista:
        for clave, valor in (pista.get("estado_de_invertidas") or {}).items():
            try:
                a, b = (int(x) for x in str(clave).split(","))
                al_aceptar[pair_key(a, b)] = (int(valor[0]), float(valor[1]))
            except (ValueError, TypeError, IndexError):
                continue
        for d in pista.get("invertidas", []):
            try:
                k = pair_key(int(d["group_a"]), int(d["group_b"]))
                g = int(d["yield_group_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if k in fijas or k not in indice or g not in k or g not in vivos:
                continue
            invertidas[k] = g
            decisiones[indice[k]] = g
        heredadas = set(invertidas)
        for par in pista.get("probadas_sin_mejora", []):
            try:
                k = pair_key(int(par[0]), int(par[1]))
            except (IndexError, TypeError, ValueError):
                continue
            if k in indice and k not in fijas:
                probadas.add(k)

    actual = armar(decisiones)
    uniones = medir(actual)
    puntaje_inicial = _puntaje(uniones)
    rondas = 0

    for _ in range(RONDAS_DEL_RESOLVEDOR):
        cede = {
            pair_key(j.group_a, j.group_b): j.yield_group_id
            for j in actual[0].wall_wall_joints
            if j.yield_group_id is not None
        }
        candidatas: Dict[Tuple[int, int], int] = {}
        for u in uniones:
            if u.estado == "ok" or u.tipo == "superpuestas":
                continue
            k = pair_key(u.grupo_a, u.grupo_b)
            if k in fijas or k in probadas or k not in cede or k not in indice:
                continue
            if k in heredadas:
                antes = al_aceptar.get(k)
                ahora = (_RANGO_ESTADO[u.estado], u.medida_mm)
                if antes is not None and not (
                    ahora[0] < antes[0] or ahora[1] > antes[1] + 0.05
                ):
                    continue        # está como cuando se la aceptó: no hay nada nuevo
            candidatas[k] = u.grupo_b if cede[k] == u.grupo_a else u.grupo_a
        if not candidatas:
            break
        rondas += 1
        probadas |= set(candidatas)

        prueba = armar({**decisiones, **{indice[k]: g for k, g in candidatas.items()}})
        uniones_prueba = medir(prueba)

        def local(us, k):
            a, b = k
            return _puntaje([u for u in us if u.grupo_a in (a, b) or u.grupo_b in (a, b)])

        mejoran = {
            k: g for k, g in candidatas.items()
            if local(uniones_prueba, k) < local(uniones, k)
        }
        if not mejoran:
            continue    # ninguna sirvió; quedan anotadas como probadas y no hay más
        if len(mejoran) == len(candidatas):
            nuevo, uniones_nuevo = prueba, uniones_prueba
        else:
            nuevo = armar({**decisiones, **{indice[k]: g for k, g in mejoran.items()}})
            uniones_nuevo = medir(nuevo)
        if _puntaje(uniones_nuevo) >= _puntaje(uniones):
            continue    # por separado mejoraban, juntas no: se queda con lo que tenía
        actual, uniones = nuevo, uniones_nuevo
        decisiones.update({indice[k]: g for k, g in mejoran.items()})
        for k, g in mejoran.items():
            if k in heredadas:
                # Era una inversión heredada y hubo que deshacerla: vuelve a la sugerida.
                invertidas.pop(k, None)
                heredadas.discard(k)
            else:
                invertidas[k] = g

    if not probadas and not invertidas:
        return actual, None, uniones
    por_par = {pair_key(u.grupo_a, u.grupo_b): u for u in uniones}
    estado_de_invertidas = {}
    for k in invertidas:
        u = por_par.get(k)
        estado_de_invertidas[f"{k[0]},{k[1]}"] = (
            [_RANGO_ESTADO[u.estado], round(u.medida_mm, 3)] if u is not None
            else [_RANGO_ESTADO["ok"], 0.0]
        )
    resolucion = {
        "rondas": rondas,
        "invertidas": [
            {"group_a": a, "group_b": b, "yield_group_id": g}
            for (a, b), g in sorted(invertidas.items())
        ],
        # Uniones donde se probó invertir y NO mejoró: ofrecerle eso al usuario como
        # arreglo sería mandarlo a probar algo que ya se sabe que no sirve.
        "probadas_sin_mejora": sorted([a, b] for (a, b) in probadas - set(invertidas)),
        "puntaje_antes": list(puntaje_inicial),
        "puntaje_despues": list(_puntaje(uniones)),
        "con_pista": bool(pista),
        # Para la pista del pedido siguiente; ver `al_aceptar`.
        "estado_de_invertidas": estado_de_invertidas,
    }
    return actual, resolucion, uniones


def _aviso_piezas_mas_finas_que_la_placa(panels: List[Panel], scale_denom: float):
    """Texto de aviso si alguna pieza mide, de ancho o de alto, menos que el espesor.

    Una tira más angosta que los 3 mm de la placa no se puede armar parada: es más fina
    que el canto del material con el que se corta. Casi siempre es un elemento del modelo
    que a esta escala queda por debajo de lo que el MDF puede representar (un zócalo, un
    antepecho, el canto de algo). Hasta ahora salía a la plancha sin ningún aviso.

    No se la quita: decidir que sobra es del usuario. Se le dice cuáles son y cuánto
    miden, en centímetros de edificio y en milímetros de plancha.
    """
    limite = PLATE_THICKNESS_M * scale_denom
    finas = [
        p for p in panels
        if p.frame and min(p.width_m, p.height_m) < limite - 1e-9
    ]
    if not finas:
        return None
    finas.sort(key=lambda p: min(p.width_m, p.height_m))

    def _una(p: Panel) -> str:
        chico, grande = sorted((p.width_m, p.height_m))
        return (
            f"{p.id} ({grande:.2f} m × {chico * 100:.0f} cm de edificio = "
            f"{chico / scale_denom * 1000:.1f} mm de plancha)"
        )

    return (
        f"{len(finas)} pieza(s) miden menos que el espesor de la placa "
        f"({PLATE_THICKNESS_M * 1000:g} mm = {limite * 100:.0f} cm de edificio a "
        f"1:{scale_denom:g}) y no se pueden armar como placa: "
        + "; ".join(_una(p) for p in finas[:5])
        + ("…" if len(finas) > 5 else "")
        + ". Conviene descartarlas, engrosarlas en el modelo o usar una escala más fina."
    )


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
        # Sin resolvedor: las etiquetas dependen de QUÉ piezas hay, no de quién cede en
        # cada unión, y esto corre en cada subida.
        _, wall_panels, floor_panels, _ = _decompose(
            phase1, opts, overrides, None, None, None, resolver=False
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
    decisiones_por_par: Optional[List[dict]] = None,
    pista: Optional[Dict] = None,
    sesion: Optional[Dict] = None,
) -> Tuple[Phase1Result, NestingResult, NestingResult, SheetConfig, Dict[int, str], List, Dict[str, Dict], List]:
    """Descompone y anida paneles. Compartido por /generate y /nesting-preview.

    El `work` que devuelve trae en `verificacion` el reporte por unión de las piezas.
    """
    work, wall_panels, floor_panels, plate_joints = _decompose(
        phase1, opts, overrides, wall_wall_decisions, merges, marks,
        user_cuts, flex, mark_lines, ribs, columns,
        decisiones_por_par=decisiones_por_par, pista=pista, sesion=sesion,
    )
    work = replace(
        work,
        verificacion=reporte_de_verificacion(
            work, wall_panels, floor_panels, opts.scale_denom or 1.0
        ),
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
        final_placements(wall_panels, floor_panels, opts.scale_denom or 1.0),
        # Fase D: ¿las piezas que se van a cortar arman el edificio? Se calcula acá, sobre
        # los paneles FINALES y con la escala ya elegida, porque la respuesta depende de
        # ambas cosas: a escalas gruesas la placa ocupa más edificio y aparecen choques
        # que a escalas finas no existen.
        _verificar(work, wall_panels, floor_panels, plate_joints, opts.scale_denom or 1.0),
    )


def compute_verificacion(
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
    decisiones_por_par: Optional[List[dict]] = None,
    pista: Optional[Dict] = None,
    sesion: Optional[Dict] = None,
) -> Tuple[Phase1Result, Dict[int, str], List, Dict[str, Dict]]:
    """Lo que cambia cuando el usuario toma una decisión, y nada más.

    Es `compute_nesting` sin acomodar las piezas en planchas y sin el verificador viejo
    por muestreo: quien está resolviendo un nudo necesita ver las piezas y el reporte, no
    la plancha. Devuelve `(work con .verificacion, etiqueta por grupo, encastres,
    placements)`.
    """
    work, wall_panels, floor_panels, plate_joints = _decompose(
        phase1, opts, overrides, wall_wall_decisions, merges, marks,
        user_cuts, flex, mark_lines, ribs, columns,
        decisiones_por_par=decisiones_por_par, pista=pista, sesion=sesion,
    )
    escala = opts.scale_denom or 1.0
    work = replace(
        work,
        verificacion=reporte_de_verificacion(work, wall_panels, floor_panels, escala),
    )
    return (
        work,
        panel_ids_by_group(wall_panels, floor_panels),
        plate_joints,
        final_placements(wall_panels, floor_panels, escala),
    )


def reporte_de_verificacion(work, wall_panels, floor_panels, scale_denom) -> Optional[Dict]:
    """Reporte por unión de las piezas que se van a cortar (ver `reporte_verificacion`).

    Igual que `_verificar`, no puede tumbar la generación: si falla devuelve None y el
    consumidor lo trata como "sin verificar", que no es lo mismo que "verificado".
    """
    try:
        from core.services.reporte_verificacion import construir_reporte
        from core.services.verificacion_uniones import verificar_uniones

        piezas = list(wall_panels) + list(floor_panels)
        uniones = getattr(work, "uniones_medidas", None)
        if uniones is None:
            # No pasó por el resolvedor (o falló): se mide acá. Si pasó, ésta es la misma
            # medición con la que el resolvedor eligió, y repetirla era un 20% del tiempo.
            previstos, apoyos = _pares_a_verificar(work)
            uniones = verificar_uniones(piezas, scale_denom, previstos, apoyos)
        cede = {
            pair_key(j.group_a, j.group_b): j.yield_group_id
            for j in work.wall_wall_joints
            if j.yield_group_id is not None
        }
        por_id = {g.id: g for g in work.groups}
        losas = [g for g in work.groups if g.category == "floor"]

        def puede_fusionar(a: int, b: int) -> bool:
            """Las MISMAS guardas que `apply_merges`: si acá da False, allá no las une."""
            ga, gb = por_id.get(a), por_id.get(b)
            if ga is None or gb is None:
                return False
            mayor, menor = (ga, gb) if ga.total_area >= gb.total_area else (gb, ga)
            return (
                _coplanar_for_merge(menor, mayor)
                and _floor_between(mayor, menor, losas, work.faces) is None
            )

        return construir_reporte(
            piezas, uniones, scale_denom, cede,
            {g.id: g.category for g in work.groups},
            puede_fusionar=puede_fusionar,
            resolucion=work.resolucion,
            pares_decididos_por_el_usuario={tuple(p) for p in (work.pares_decididos or [])},
            espesor_modelo={g.id: float(g.thickness or 0.0) for g in work.groups},
        )
    except Exception:
        import logging
        logging.getLogger("eficiencia2d.pipeline").exception("reporte_de_verificacion falló")
        return None


class EnsambleNoArma(Exception):
    """Se pidió bloquear la generación si hay uniones que no arman, y las hay."""

    def __init__(self, reporte: Dict):
        super().__init__("La maqueta tiene uniones que no arman.")
        self.reporte = reporte


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
    wall_panels: List[Panel],
    floor_panels: List[Panel],
    scale_denom: float = 1.0,
) -> Dict[str, Dict]:
    """Todo lo que hace falta para dibujar la maqueta EXACTAMENTE como se va a cortar.

    `group_id (str) -> pieza`, con el marco 3D de la pieza YA RECORTADA más su CONTORNO
    de corte. Con eso el visor no tiene que juntar datos de dos endpoints ni deducir nada:

        world = origin + u·u_axis + v·v_axis

    vale tal cual para cada punto de `outline`, sin compensar el espejado (viene horneado
    en el marco, por eso `mirrored` es False).

    Por qué el CONTORNO y no sólo el bounding box: una pieza con una muesca lateral, una
    abertura o un faldón inclinado NO es su rectángulo. Dibujarla como caja orientada
    —que es lo que hacía el instructivo— muestra material donde no lo hay y esconde los
    huecos por donde entra la pieza vecina. El bounding box de un faldón triangular tiene
    el doble de superficie que el faldón.

    `thickness_m` viene en metros de EDIFICIO, ya multiplicado por la escala: la placa
    mide 3 mm de plancha siempre, pero eso son 0.15 m a 1:50 y 0.30 m a 1:200. Extruir la
    pieza por ese espesor es lo que hace visibles los choques que el espesor provoca.
    """
    out: Dict[str, Dict] = {}
    for p in list(wall_panels) + list(floor_panels):
        if not p.frame or p.source_group_id is None or p.source_group_id < 0:
            continue
        out[str(p.source_group_id)] = dict(
            p.frame,
            panel_id=p.id,
            category=p.category,
            thickness_m=PLATE_THICKNESS_M * (scale_denom or 1.0),
            outline=[
                {
                    "a": {"x": e.a.x, "y": e.a.y},
                    "b": {"x": e.b.x, "y": e.b.y},
                    "hole": bool(e.hole),
                }
                for e in p.edges
                if not getattr(e, "score", False)
            ],
        )
    return out


def placements_fieles(
    phase1: Phase1Result,
    scale_denom: float,
    overrides: Optional[Dict[int, str]] = None,
    wall_wall_decisions: Optional[Dict[int, int]] = None,
    absorbidos_out: Optional[Dict[int, int]] = None,
    decisiones_por_par: Optional[List[dict]] = None,
    pista: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """Marco 3D de cada pieza YA RECORTADA, a la escala a la que se va a cortar.

    Si se pasa `absorbidos_out`, se llena con los grupos que a esta escala no tienen
    pieza propia (`id absorbido -> id del grupo que lo incluye`).

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
    work, walls, floors, _pjs = _decompose(
        phase1, opts, overrides, wall_wall_decisions, None, None,
        decisiones_por_par=decisiones_por_par, pista=pista,
    )
    if absorbidos_out is not None:
        absorbidos_out.update(work.absorbed_groups or {})
    return final_placements(walls, floors, scale_denom)


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
    decisiones_por_par: Optional[List[dict]] = None,
    bloquear_si_no_arma: bool = False,
    pista: Optional[Dict] = None,
    sesion: Optional[Dict] = None,
) -> List[OutputFile]:
    work, wall_nesting, floor_nesting, sheet_cfg, _, _, _, _ = compute_nesting(
        phase1, opts, overrides, wall_wall_decisions, merges, marks,
        user_cuts, flex, mark_lines, ribs, columns,
        decisiones_por_par=decisiones_por_par, pista=pista, sesion=sesion,
    )
    # Pedido original del usuario: si una unión no tiene solución, frenar ANTES de
    # entregar el archivo de corte, y sólo seguir si quien genera lo fuerza a conciencia.
    # Es opcional (lo pide el front) para no dejar sin poder generar a quien todavía usa
    # un visor que no muestra el reporte.
    if bloquear_si_no_arma and (work.verificacion or {}).get("veredicto") == "no_arma":
        raise EnsambleNoArma(work.verificacion)
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
