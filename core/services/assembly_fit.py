"""
Verificación de ensamble sobre las PIEZAS FINALES (armar la maqueta antes de cortar).

La diferencia con assembly_check es qué se mide:

  - assembly_check trabaja sobre la geometría del MODELO (los grupos del archivo 3D).
  - acá se trabaja sobre las piezas tal como van a salir de la cortadora: contorno ya
    recortado por los encastres, ya espejado, ubicado en su posición del edificio.

Es la comprobación que hoy sólo se hace a mano, pegando el MDF: dos placas que se
atraviesan (sobra material) o una junta que queda abierta (falta material).

Método: cada pieza es una placa plana. Dos placas de planos distintos sólo pueden
tocarse a lo largo de la recta donde se cortan sus planos. Se calcula esa recta, se
mira qué tramos de ella caen DENTRO de cada pieza y se comparan:

  - tramo compartido por ambas  -> se atraviesan (la longitud es cuánto)
  - ningún tramo compartido pero sí cercano -> junta abierta

No hace falta construir sólidos: para placas planas el problema se reduce a intervalos
sobre una recta, que es exacto y barato.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from core.services.types import Vec3, cross, dot, normalize, vlength

# Dos planos con |n_a·n_b| por encima de esto se consideran paralelos: no se cortan en
# una recta (o son la misma placa vista dos veces).
PARALLEL_DOT = 0.999
# Solape mínimo para reportar que dos piezas se atraviesan. Por debajo es el contacto
# normal de dos placas que se tocan de canto.
MIN_OVERLAP_M = 0.005


def _panel_polygon(panel):
    """Contorno exterior de la pieza en sus coordenadas locales (u, v)."""
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union

    segs = [
        LineString([(e.a.x, e.a.y), (e.b.x, e.b.y)])
        for e in panel.edges
        # Las líneas de grabado y las de encastre no son contorno de material.
        if not getattr(e, "score", False) and not getattr(e, "flex", False)
        and (abs(e.a.x - e.b.x) > 1e-12 or abs(e.a.y - e.b.y) > 1e-12)
    ]
    if len(segs) < 3:
        return None
    try:
        polys = list(polygonize(unary_union(segs)))
    except Exception:
        return None
    if not polys:
        return None
    # El contorno exterior es el polígono de mayor área.
    return max(polys, key=lambda p: p.area)


def _line_intervals_inside(poly, p0_u: float, p0_v: float, du: float, dv: float):
    """Tramos [t0, t1] de la recta (p0 + t·d) que caen dentro del polígono."""
    from shapely.geometry import LineString

    if poly is None or poly.is_empty:
        return []
    # Se extiende la recta bien más allá de la pieza y se recorta contra ella.
    minx, miny, maxx, maxy = poly.bounds
    span = math.hypot(maxx - minx, maxy - miny) + 1.0
    a = (p0_u - du * span, p0_v - dv * span)
    b = (p0_u + du * span, p0_v + dv * span)
    try:
        inter = poly.intersection(LineString([a, b]))
    except Exception:
        return []
    if inter.is_empty:
        return []

    geoms = getattr(inter, "geoms", [inter])
    out: List[Tuple[float, float]] = []
    for g in geoms:
        coords = list(getattr(g, "coords", []))
        if len(coords) < 2:
            continue
        ts = [(c[0] - p0_u) * du + (c[1] - p0_v) * dv for c in coords]
        lo, hi = min(ts), max(ts)
        if hi - lo > 1e-9:
            out.append((lo, hi))
    return out


def _overlap_length(ia, ib) -> float:
    """Longitud total compartida por dos conjuntos de intervalos."""
    total = 0.0
    for a0, a1 in ia:
        for b0, b1 in ib:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                total += hi - lo
    return total


def _crosses_interior(poly, p0_u, p0_v, du, dv, ia, ib) -> bool:
    """¿El tramo compartido pasa por el INTERIOR de la pieza, y no por su borde?"""
    from shapely.geometry import Point

    for a0, a1 in ia:
        for b0, b1 in ib:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi - lo <= 1e-9:
                continue
            tm = (lo + hi) / 2.0
            pt = Point(p0_u + du * tm, p0_v + dv * tm)
            # Distancia al contorno: ~0 significa que la junta corre por el borde de
            # la pieza (contacto de canto, correcto).
            if poly.exterior.distance(pt) > MIN_OVERLAP_M:
                return True
    return False


def check_piece_collisions(
    panels: List, tolerance_m: float = MIN_OVERLAP_M
) -> List[Dict]:
    """Piezas que se atraviesan una vez armadas. Devuelve una lista de hallazgos.

    Cada hallazgo: {pieces, type: "interpenetra", measure_mm, at:[x,y,z]}.
    `measure_mm` es cuánto material comparten a lo largo de la junta: es exactamente
    lo que sobra y hay que recortar para que las dos piezas entren.
    """
    prepared = []
    for p in panels:
        fr = getattr(p, "frame", None)
        if not fr:
            continue
        poly = _panel_polygon(p)
        if poly is None or poly.area <= 0:
            continue
        prepared.append((p, fr, poly))

    findings: List[Dict] = []

    for i in range(len(prepared)):
        pa, fa, ga = prepared[i]
        na, da = fa["normal"], fa["d"]
        for j in range(i + 1, len(prepared)):
            pb, fb, gb = prepared[j]
            nb, db = fb["normal"], fb["d"]

            if abs(dot(na, nb)) > PARALLEL_DOT:
                continue  # planos paralelos: no se cortan en una recta

            # Recta de intersección de los dos planos.
            d_dir = cross(na, nb)
            if vlength(d_dir) < 1e-9:
                continue
            d_dir = normalize(d_dir)
            # Punto de la recta: se resuelve n_a·X = d_a, n_b·X = d_b, d_dir·X = 0.
            denom = dot(cross(na, nb), d_dir)
            if abs(denom) < 1e-12:
                continue
            t1 = cross(nb, d_dir)
            t2 = cross(d_dir, na)
            p0 = Vec3(
                (da * t1.x + db * t2.x) / denom,
                (da * t1.y + db * t2.y) / denom,
                (da * t1.z + db * t2.z) / denom,
            )

            def to_local(fr, pt: Vec3):
                o = fr["origin"]
                r = Vec3(pt.x - o.x, pt.y - o.y, pt.z - o.z)
                return dot(r, fr["u_dir"]), dot(r, fr["v_dir"])

            au, av = to_local(fa, p0)
            adu, adv = dot(d_dir, fa["u_dir"]), dot(d_dir, fa["v_dir"])
            bu, bv = to_local(fb, p0)
            bdu, bdv = dot(d_dir, fb["u_dir"]), dot(d_dir, fb["v_dir"])

            ia = _line_intervals_inside(ga, au, av, adu, adv)
            if not ia:
                continue
            ib = _line_intervals_inside(gb, bu, bv, bdu, bdv)
            if not ib:
                continue

            ov = _overlap_length(ia, ib)
            if ov <= tolerance_m:
                continue

            # Distinguir TOCARSE de ATRAVESARSE. Dos placas que se encuentran de canto
            # (una pared parada sobre un piso, dos paredes en esquina) comparten toda
            # esa línea de contacto de forma legítima: ahí la recta corre por el BORDE
            # de al menos una de las piezas. Sólo hay atravesamiento si la recta pasa
            # por el INTERIOR de ambas, que es cuando el material se pisa de verdad.
            if not (
                _crosses_interior(ga, au, av, adu, adv, ia, ib)
                and _crosses_interior(gb, bu, bv, bdu, bdv, ia, ib)
            ):
                continue

            # Punto medio del tramo compartido, para ubicar el problema en el visor.
            mids = [
                (max(a0, b0) + min(a1, b1)) / 2.0
                for a0, a1 in ia
                for b0, b1 in ib
                if min(a1, b1) > max(a0, b0)
            ]
            tm = sum(mids) / len(mids) if mids else 0.0
            at = Vec3(
                p0.x + d_dir.x * tm, p0.y + d_dir.y * tm, p0.z + d_dir.z * tm
            )

            findings.append({
                "pieces": [pa.id, pb.id],
                "type": "interpenetra",
                "measure_mm": round(ov * 1000.0, 2),
                "at": [round(at.x, 4), round(at.y, 4), round(at.z, 4)],
            })

    findings.sort(key=lambda f: f["measure_mm"], reverse=True)
    return findings
