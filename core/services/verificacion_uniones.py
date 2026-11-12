"""
Verificación POR UNIÓN: de cada encuentro entre dos piezas, ¿encaja, se pisa o queda luz?

Por qué existe
--------------
`assembly_verify.verificar_ensamble` responde una sola pregunta —¿hay dos piezas ocupando
el mismo material?— y la responde por muestreo. Sirve como alarma, pero no alcanza para
decirle a alguien "esto arma": tiene tres puntos ciegos, los tres medidos.

1. **No ve las luces.** Una pieza que queda corta no choca con nada, así que pasa como
   correcta. Es el defecto que más se nota al armar y el único que no se informaba.
2. **Muestrea.** Reparte puntos en una grilla y mira si caen dentro de la otra placa. Un
   solape más fino que la grilla no se ve, y el resultado cambia si cambia el tamaño de la
   pieza: al hacer crecer una losa "aparecieron" dos choques que ya estaban.
3. **Excusa por lista.** Si la pieza tiene una ranura contra la vecina, el par no se mira.
   Nadie comprueba que la ranura esté donde tiene que estar ni que mida lo que tiene que
   medir.

Acá cada par se mide EXACTO y se informa qué pasa en él, no sólo si falla.

Cómo mide
---------
Dos placas no paralelas se cortan a lo largo de una recta. Parado sobre esa recta, en cada
punto `τ` hay dos números que lo dicen todo:

    q_a  = hasta dónde llega el material de A, medido como distancia al plano medio de B
    q_b  = lo mismo para B respecto del plano medio de A

La placa B ocupa la franja `|q_a| <= t/2` y la placa A la franja `|q_b| <= t/2`. Entonces:

- **penetración** de A en B a esa altura: cuánto del material de A cae dentro de la franja;
- **choque**: las dos se meten una en la otra a la vez → `min(pen_a, pen_b)`. Una ranura
  bien hecha deja `pen = 0` del lado de la pieza ranurada, sin ninguna lista de excusas:
  el material simplemente no está;
- **luz**: ninguna toca la franja de la otra → distancia que les falta;
- **cobertura**: en una unión sana UNA de las dos atraviesa todo el espesor de la otra (el
  muro pasante de una T, la que tapa en una esquina, la losa bajo el muro). Si ninguna lo
  hace queda un canto a la vista o medio apoyo en el aire.

Los contornos son polígonos, así que esos números son lineales a tramos en `τ`: se evalúan
en el medio de cada tramo entre vértices y el resultado es exacto, sin grilla.

Las placas paralelas no se cortan: se apilan. Ahí la cuenta es directa —separación entre
planos menor que una placa y contornos que se pisan— y sí ve el caso que al verificador
anterior se le escapaba (separación entre media y una placa).

Qué NO promete
--------------
Mide las piezas tal como salen a la plancha, con el espesor NOMINAL de la placa. No sabe si
el MDF real mide 3.0 mm ni cuánto se come el láser: eso lo dice la probeta de calibración
(`core.services.probeta`), no un cálculo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from core.services.cutting_sheet import Panel, _edges_to_polygon
from core.services.plate_intersect import PLATE_THICKNESS_M, pair_key

# Umbrales en milímetros de PLANCHA. Hasta `TOL_OK_MM` es ruido de corte (orden del kerf).
# Entre eso y `TOL_REVISAR_MM` la unión se arma forzando o lijando, o queda una luz que se
# ve: hay que mirarla. Por encima no arma. El segundo valor es un cuarto de placa y está
# puesto a criterio, SIN calibrar contra maquetas cortadas.
TOL_OK_MM = 0.2
TOL_REVISAR_MM = 0.75

# Normales más paralelas que esto: las placas se apilan, no se cruzan.
COS_PARALELAS = 0.99
# Hasta qué distancia de la franja de la vecina se mira el contorno, en espesores de placa.
ALCANCE_PLACAS = 1.0

ESTADOS = ("ok", "revisar", "no_arma")


@dataclass
class Union:
    """Lo que pasa en el encuentro entre dos piezas."""

    pieza_a: str
    pieza_b: str
    grupo_a: int
    grupo_b: int
    # "encaja" | "choque" | "luz" | "sin_cubrir" | "superpuestas"
    tipo: str
    # "ok" | "revisar" | "no_arma"
    estado: str
    # Cuánto se pisan / cuánto falta, en mm de PLANCHA. 0 si encaja.
    medida_mm: float
    # La unión existe en el modelo (junta detectada o apoyo muro-losa). Un choque entre
    # dos piezas que el modelo no une es otro tipo de problema: nadie lo resolvió porque
    # nadie sabía que se tocaban.
    prevista: bool
    # Largo del contacto en metros de EDIFICIO, y dónde está, para llevar la cámara.
    largo_m: float = 0.0
    punto: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    segmento: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    )
    # Qué pieza es la que sobra o falta, cuando se puede decir (None si es simétrico).
    pieza_que_ajusta: Optional[str] = None


@dataclass
class _Placa:
    id: str
    gid: int
    categoria: str
    poly: object
    o: object
    u: object
    v: object
    n: object
    d: float
    lo: object = None
    hi: object = None
    verts: object = None


def _estado(medida_mm: float, tope: str = "no_arma") -> str:
    if medida_mm <= TOL_OK_MM + 1e-6:
        return "ok"
    if medida_mm <= TOL_REVISAR_MM + 1e-6 or tope == "revisar":
        return "revisar"
    return "no_arma"


def _placas(panels: Iterable[Panel]) -> List[_Placa]:
    import numpy as np

    out: List[_Placa] = []
    for p in panels:
        if not p.frame or p.source_group_id is None or p.source_group_id < 0:
            continue  # refuerzos sueltos: se pegan a mano, no tienen lugar fijo
        poly = _edges_to_polygon([e for e in p.edges if not getattr(e, "flex", False)])
        if poly is None or poly.is_empty or poly.area <= 1e-9:
            continue
        m = p.frame
        o = np.array([m["origin"]["x"], m["origin"]["y"], m["origin"]["z"]], dtype=float)
        u = np.array([m["u_axis"]["x"], m["u_axis"]["y"], m["u_axis"]["z"]], dtype=float)
        v = np.array([m["v_axis"]["x"], m["v_axis"]["y"], m["v_axis"]["z"]], dtype=float)
        n = np.cross(u, v)
        ln = float(np.linalg.norm(n))
        if ln < 1e-9:
            continue
        n = n / ln
        pts = []
        for g in getattr(poly, "geoms", [poly]):
            for x, y in g.exterior.coords:
                pts.append(o + u * x + v * y)
        if not pts:
            continue
        verts = np.array(pts)
        out.append(_Placa(
            id=p.id, gid=p.source_group_id, categoria=p.category, poly=poly,
            o=o, u=u, v=v, n=n, d=float(n @ o),
            lo=verts.min(axis=0), hi=verts.max(axis=0), verts=verts,
        ))
    return out


def _en_tau_q(pl: _Placa, tau, n_otra, d_otra):
    """Contorno de `pl` en coordenadas (τ a lo largo de la recta de corte, q = distancia
    con signo al plano medio de la otra placa). El mapa es afín: sigue siendo un polígono."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    def mapa(x: float, y: float) -> Tuple[float, float]:
        w = pl.o + pl.u * x + pl.v * y
        return (float(w @ tau), float(w @ n_otra) - d_otra)

    partes = []
    for g in getattr(pl.poly, "geoms", [pl.poly]):
        try:
            p = Polygon(
                [mapa(x, y) for x, y in g.exterior.coords],
                [[mapa(x, y) for x, y in h.coords] for h in g.interiors],
            )
        except Exception:
            continue
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_empty:
            partes.append(p)
    if not partes:
        return None
    return unary_union(partes) if len(partes) > 1 else partes[0]


def _intervalos(poly, tau: float, alto: float) -> List[Tuple[float, float]]:
    """Tramos de q donde hay material sobre la vertical τ."""
    from shapely.geometry import LineString

    corte = LineString([(tau, -alto), (tau, alto)]).intersection(poly)
    out: List[Tuple[float, float]] = []
    for g in getattr(corte, "geoms", [corte]):
        cs = list(getattr(g, "coords", []))
        if len(cs) >= 2:
            qs = [c[1] for c in cs]
            out.append((min(qs), max(qs)))
    return out


def _dentro_y_distancia(ivs: List[Tuple[float, float]], h: float) -> Tuple[float, float]:
    """(cuánto material cae dentro de la franja [-h, h], distancia del material a ella)."""
    dentro = 0.0
    dist = math.inf
    for lo, hi in ivs:
        dentro += max(0.0, min(hi, h) - max(lo, -h))
        if hi < -h:
            dist = min(dist, -h - hi)
        elif lo > h:
            dist = min(dist, lo - h)
        else:
            dist = 0.0
    return dentro, dist


def _vertices_tau(poly) -> List[float]:
    out: List[float] = []
    for g in getattr(poly, "geoms", [poly]):
        if not hasattr(g, "exterior"):
            continue
        out.extend(x for x, _y in g.exterior.coords)
        for hh in g.interiors:
            out.extend(x for x, _y in hh.coords)
    return out


def _medir_cruce(
    a: _Placa, b: _Placa, espesor: float, prevista: bool, tol_contacto: float = 1e-9
) -> Optional[dict]:
    """Mediciones del encuentro entre dos placas NO paralelas, en metros de edificio.

    Devuelve choque, luz, contacto y cobertura con el tramo de la recta donde ocurre cada
    uno, más `armar` para construir la `Union` una vez decidido el tipo. None si las dos
    placas nunca tienen material a la vez sobre la recta de corte."""
    import numpy as np
    from shapely.geometry import box

    tau = np.cross(a.n, b.n)
    seno = float(np.linalg.norm(tau))
    if seno < 1e-6:
        return None
    tau = tau / seno
    h = espesor / 2.0
    alcance = h + ALCANCE_PLACAS * espesor

    pa = _en_tau_q(a, tau, b.n, b.d)
    pb = _en_tau_q(b, tau, a.n, a.d)
    if pa is None or pb is None:
        return None
    # Sólo importa lo que está cerca de la franja de la otra.
    lejos = 1e6
    try:
        pa = pa.intersection(box(-lejos, -alcance, lejos, alcance))
        pb = pb.intersection(box(-lejos, -alcance, lejos, alcance))
    except Exception:
        return None
    if pa.is_empty or pb.is_empty or pa.area <= 1e-12 or pb.area <= 1e-12:
        return None

    t0 = max(pa.bounds[0], pb.bounds[0])
    t1 = min(pa.bounds[2], pb.bounds[2])
    if t1 - t0 <= 1e-9:
        return None

    cortes = sorted({t0, t1, *(
        t for t in _vertices_tau(pa) + _vertices_tau(pb) if t0 < t < t1
    )})

    choque = 0.0
    tramo_choque = None
    luz = math.inf
    tramo_luz = None
    contacto = 0.0
    tramo_contacto: Optional[Tuple[float, float]] = None
    deficit_pesado: List[Tuple[float, float, int]] = []   # (déficit, largo, quién cubre)

    for i in range(len(cortes) - 1):
        largo = cortes[i + 1] - cortes[i]
        if largo <= 1e-9:
            continue
        medio = (cortes[i] + cortes[i + 1]) / 2.0
        ia = _intervalos(pa, medio, alcance + 1.0)
        ib = _intervalos(pb, medio, alcance + 1.0)
        if not ia or not ib:
            continue
        den_a, dis_a = _dentro_y_distancia(ia, h)
        den_b, dis_b = _dentro_y_distancia(ib, h)
        c = min(den_a, den_b)
        if c > choque:
            choque, tramo_choque = c, (cortes[i], cortes[i + 1])
        g = math.hypot(dis_a, dis_b)
        if g < luz:
            luz, tramo_luz = g, (cortes[i], cortes[i + 1])
        if g <= tol_contacto:
            contacto += largo
            tramo_contacto = (
                (cortes[i], cortes[i + 1]) if tramo_contacto is None
                else (min(tramo_contacto[0], cortes[i]), max(tramo_contacto[1], cortes[i + 1]))
            )
            cubre = max(den_a, den_b)
            deficit_pesado.append((max(0.0, espesor - cubre), largo, 0 if den_a >= den_b else 1))

    if tramo_luz is None:
        return None  # nunca hay material de las dos a la vez sobre la recta

    def lugar(tramo):
        # Punto de la recta de corte: combinación de las dos normales, más τ.
        cosf = float(a.n @ b.n)
        den = 1.0 - cosf * cosf
        p0 = ((a.d - b.d * cosf) * a.n + (b.d - a.d * cosf) * b.n) / den
        e0, e1 = p0 + tramo[0] * tau, p0 + tramo[1] * tau
        m = (e0 + e1) / 2.0
        return (
            tuple(float(x) for x in m),
            (tuple(float(x) for x in e0), tuple(float(x) for x in e1)),
        )

    def armar(tipo, medida_m, tramo, estado, ajusta=None):
        punto, seg = lugar(tramo)
        return Union(
            pieza_a=a.id, pieza_b=b.id, grupo_a=a.gid, grupo_b=b.gid,
            tipo=tipo, estado=estado, medida_mm=medida_m, prevista=prevista,
            largo_m=float(tramo[1] - tramo[0]) if tipo != "encaja" else float(contacto),
            punto=punto, segmento=seg, pieza_que_ajusta=ajusta,
        )

    return {
        "choque": choque, "tramo_choque": tramo_choque,
        "luz": luz, "tramo_luz": tramo_luz,
        "contacto": contacto, "tramo_contacto": tramo_contacto,
        "deficit": deficit_pesado, "armar": armar,
    }


def _medir_paralelas(a: _Placa, b: _Placa, espesor: float, prevista: bool, a_mm: float) -> Optional[Union]:
    """Dos placas paralelas montadas una sobre otra."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    sep = abs(float(a.n @ b.o) - a.d)
    if sep >= espesor:
        return None

    def mapa(x: float, y: float) -> Tuple[float, float]:
        w = b.o + b.u * x + b.v * y - a.o
        return (float(w @ a.u), float(w @ a.v))

    partes = []
    for g in getattr(b.poly, "geoms", [b.poly]):
        try:
            p = Polygon(
                [mapa(x, y) for x, y in g.exterior.coords],
                [[mapa(x, y) for x, y in hh.coords] for hh in g.interiors],
            )
        except Exception:
            continue
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_empty:
            partes.append(p)
    if not partes:
        return None
    try:
        comun = a.poly.intersection(unary_union(partes))
    except Exception:
        return None
    # Tiene que ser un solape de SUPERFICIE: dos tramos coplanares que se tocan de canto
    # comparten una línea, no área. Se pide una franja de al menos una placa de ancho.
    if comun.is_empty or comun.area <= espesor * espesor:
        return None
    medida_mm = (espesor - sep) * a_mm
    if medida_mm <= TOL_OK_MM:
        return None
    c = comun.representative_point()
    w = a.o + a.u * c.x + a.v * c.y
    punto = tuple(float(x) for x in w)
    chica = a if a.poly.area <= b.poly.area else b
    return Union(
        pieza_a=a.id, pieza_b=b.id, grupo_a=a.gid, grupo_b=b.gid,
        tipo="superpuestas", estado=_estado(medida_mm), medida_mm=medida_mm,
        prevista=prevista, largo_m=float(math.sqrt(comun.area)),
        punto=punto, segmento=(punto, punto), pieza_que_ajusta=chica.id,
    )


def verificar_uniones(
    panels: List[Panel],
    scale_denom: float,
    pares_previstos: Optional[Set[Tuple[int, int]]] = None,
    pares_de_apoyo: Optional[Set[Tuple[int, int]]] = None,
) -> List[Union]:
    """Estado de cada unión entre las piezas que se van a cortar.

    `pares_previstos` son `pair_key(gid_a, gid_b)` de los encuentros que existen en el
    modelo (juntas detectadas y apoyos muro-losa). Para ésos se informa también si queda
    luz o si falta cubrir; para el resto de los pares sólo se informa si chocan, porque
    dos piezas que el modelo no une no tienen por qué tocarse.

    `pares_de_apoyo` son encuentros muro-losa deducidos por cercanía (cajas que coinciden
    en planta), no por una junta del modelo. Se miden igual, pero de ellos NO se informa
    luz: que un muro no toque una losa que tiene cerca puede ser diseño (un vacío, un
    balcón), y avisarlo sería ruido.

    Devuelve TODAS las uniones evaluadas, incluidas las que encajan: el que consume esto
    tiene que poder decir "287 de 295 verificadas", no sólo listar las que fallan.
    """
    pares_previstos = pares_previstos or set()
    pares_de_apoyo = (pares_de_apoyo or set()) - pares_previstos
    escala = scale_denom or 1.0
    espesor = PLATE_THICKNESS_M * escala          # metros de EDIFICIO
    a_mm = 1000.0 / escala                        # de metros de edificio a mm de plancha
    margen = espesor * (0.5 + ALCANCE_PLACAS)

    pz = _placas(panels)
    out: List[Union] = []
    for i in range(len(pz)):
        A = pz[i]
        for j in range(i + 1, len(pz)):
            B = pz[j]
            if ((A.hi + margen) < (B.lo - margen)).any() or ((B.hi + margen) < (A.lo - margen)).any():
                continue
            clave = pair_key(A.gid, B.gid)
            de_apoyo = clave in pares_de_apoyo
            prevista = clave in pares_previstos or de_apoyo
            if abs(float(A.n @ B.n)) > COS_PARALELAS:
                u = _medir_paralelas(A, B, espesor, prevista, a_mm)
                if u is not None:
                    out.append(u)
                continue

            # "Se tocan" con la misma tolerancia con la que después se juzga la luz: la
            # holgura del kerf en una ranura deja las caras a centésimas, no a cero.
            m = _medir_cruce(A, B, espesor, prevista, TOL_OK_MM / a_mm)
            if m is None:
                continue
            armar = m["armar"]
            choque_mm = m["choque"] * a_mm
            if choque_mm > TOL_OK_MM:
                out.append(armar("choque", choque_mm, m["tramo_choque"], _estado(choque_mm)))
                continue
            if not prevista:
                continue        # no chocan y no tenían por qué tocarse: no es una unión
            luz_mm = m["luz"] * a_mm
            if m["tramo_contacto"] is None or luz_mm > TOL_OK_MM:
                if de_apoyo:
                    continue
                out.append(armar("luz", luz_mm, m["tramo_luz"], _estado(luz_mm)))
                continue
            # Se tocan sin pisarse. ¿Alguna atraviesa el espesor de la otra?
            total = sum(l for _d, l, _q in m["deficit"]) or 1.0
            sin_cubrir = sum(l for d, l, _q in m["deficit"] if d * a_mm > TOL_OK_MM)
            if sin_cubrir > 0.5 * total:
                # déficit típico: el de mayor largo acumulado
                peor = max(m["deficit"], key=lambda x: x[1])
                quien = (A, B)[peor[2]]
                out.append(armar(
                    "sin_cubrir", peor[0] * a_mm, m["tramo_contacto"],
                    _estado(peor[0] * a_mm, tope="revisar"), ajusta=quien.id,
                ))
            else:
                out.append(armar("encaja", 0.0, m["tramo_contacto"], "ok"))
    return out
