"""
Sospechas: huecos que el modelo deja en la maqueta.

Por qué existe
--------------
El verificador de uniones contesta "¿las piezas encajan entre sí tal como las une el
modelo?". Parte de un supuesto: el modelo está bien hecho. Por eso no puede ver un muro que
no llega a la esquina: para él, dos piezas que el modelo no une no tienen por qué tocarse
(podría ser un vano). Se armaron a propósito dos edificios de prueba con 16 errores de
modelado y el veredicto fue "LISTO" en los 16.

Midiendo qué pasa con cada error en las piezas cortadas aparecieron tres casos:

* el motor lo CORRIGE solo (un muro levantado unos centímetros se estira hasta la losa,
  dos pieles duplicadas se cortan como una);
* el motor lo REPRODUCE tal cual y las piezas encajan (un muro corrido 10 cm hacia
  adentro es un retiro posible; sólo quien diseñó sabe si era la idea);
* queda un HUECO en la maqueta: dos piezas que casi se tocan y no se tocan.

Esto busca el tercero, que es el único que se ve en la maqueta terminada y el único que se
puede afirmar sin conocer la intención: un hueco de pocos centímetros entre dos piezas que
se encuentran, en un lugar que ninguna otra pieza ocupa. No cambia ninguna pieza ni el
veredicto de encaje: es una pregunta ("¿tienen que encontrarse?") con lugar y medida.

Cómo se mide
------------
Para cada par de piezas NO paralelas, sobre la recta donde se cruzan sus planos, se recorre
el encuentro tramo a tramo (las mismas coordenadas que usa el verificador: τ a lo largo de
la recta, q distancia al plano de la otra) y en cada tramo se mide la separación entre el
material de las dos. Se mide por tramo y no la distancia mínima porque un muro puede tocar
la losa en las puntas y quedar 30 cm en el aire en el medio.

Un tramo cuenta como hueco si la separación está entre `sep_min` y `SEPARACION_MAXIMA_M` y
el punto medio del hueco no lo ocupa ninguna otra pieza. Lo segundo es lo que evita los
falsos avisos: el muro de planta baja y el de planta alta están a una placa de distancia,
pero en el medio está la losa.

Calibración: 0 avisos en los cuatro modelos correctos de prueba (casa, edificio con
aberturas y con vidrios, a 1:50 y 1:100). No está calibrado contra maquetas cortadas.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from core.services.plate_intersect import PLATE_THICKNESS_M

# Separación desde la que dos piezas "no se tocan": 1 cm de edificio, o 0.2 mm de plancha
# (la tolerancia "ok" del verificador) si a esa escala es más.
SEPARACION_MINIMA_M = 0.01
TOLERANCIA_OK_MM = 0.2
# Hasta dónde una separación parece un error y no un vano. Una puerta mide 70 cm o más.
SEPARACION_MAXIMA_M = 0.35
# Largo mínimo del hueco a lo largo del encuentro, en placas. En las esquinas el recorte
# de encaje deja vacíos del largo de UNA placa (la muesca de un muro donde pasa el otro):
# aparecen igual en un modelo correcto y no son un error de modelado. Un error de modelado
# corre a lo largo del borde: la altura del muro, el largo de la losa.
LARGO_MINIMO_PLACAS = 2.0
# Y nunca menos que esto, en metros de edificio. En un modelo real las muescas de cada
# encuentro en T quedan pegadas unas a otras y suman más de dos placas; un error de
# modelado deja un hueco de la altura de un muro o del largo de un borde.
LARGO_MINIMO_M = 0.5
COS_PARALELAS = 0.95


def _en_mundo(a, b, tau, t: float, q: float, en_a: bool):
    """Punto de la placa (a si `en_a`, si no b) a coordenada τ = t y distancia q al plano
    de la otra. Es la inversa del mapa de `verificacion_uniones._en_tau_q`."""
    import numpy as np

    propia, otra = (a, b) if en_a else (b, a)
    M = np.vstack([tau, otra.n, propia.n])
    rhs = np.array([t, q + otra.d, propia.d])
    try:
        return np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None


class _Indice:
    """Cajas de todas las placas en dos arreglos, para filtrar de una vez cuáles pueden
    contener un punto (era un recorrido en Python por las 300 piezas en cada consulta)."""

    def __init__(self, placas):
        import numpy as np

        self.placas = placas
        self.lo = np.array([p.lo for p in placas], dtype=float).reshape(-1, 3)
        self.hi = np.array([p.hi for p in placas], dtype=float).reshape(-1, 3)


def _ocupado(p, indice: "_Indice", espesor: float, excepto, tol: float) -> bool:
    import numpy as np
    import shapely

    h = espesor / 2.0
    cerca = np.nonzero(
        ((p >= indice.lo - espesor - tol) & (p <= indice.hi + espesor + tol)).all(axis=1)
    )[0]
    for k in cerca:
        C = indice.placas[k]
        if C.id in excepto:
            continue
        rel = p - C.o
        pn = float(p @ C.n - C.d)
        if abs(pn) > h + tol:
            continue
        if C.poly.distance(shapely.Point(float(rel @ C.u), float(rel @ C.v))) <= tol:
            return True
    return False


def _tramos_candidatos(a, b, espesor: float, sep_min: float) -> List[Tuple[float, float, float, object]]:
    """Tramos del encuentro entre dos placas no paralelas donde queda una separación
    sospechosa: `(t_ini, t_fin, separación, centro del hueco)`. Depende SÓLO de las dos
    placas, así que se puede recordar por sus firmas; si el hueco está ocupado por otra
    pieza se decide después (`_huecos_del_cruce`)."""
    import numpy as np
    from shapely.geometry import box

    from core.services.verificacion_uniones import (
        _dentro_y_distancia,
        _en_tau_q,
        _intervalos,
        _vertices_tau,
    )

    tau = np.cross(a.n, b.n)
    seno = float(np.linalg.norm(tau))
    if seno < 1e-6:
        return []
    tau = tau / seno
    h = espesor / 2.0
    alcance = h + SEPARACION_MAXIMA_M + sep_min
    # Se recorta un poco más allá del alcance para poder medir el ESPESOR del material
    # que está del otro lado del hueco (ver más abajo).
    recorte = alcance + 2.0 * espesor
    pa = _en_tau_q(a, tau, b.n, b.d)
    pb = _en_tau_q(b, tau, a.n, a.d)
    if pa is None or pb is None:
        return []
    try:
        pa = pa.intersection(box(-1e6, -recorte, 1e6, recorte))
        pb = pb.intersection(box(-1e6, -recorte, 1e6, recorte))
    except Exception:
        return []
    if pa.is_empty or pb.is_empty:
        return []
    t0 = max(pa.bounds[0], pb.bounds[0])
    t1 = min(pa.bounds[2], pb.bounds[2])
    if t1 - t0 <= 1e-9:
        return []
    cortes = sorted({t0, t1, *(t for t in _vertices_tau(pa) + _vertices_tau(pb) if t0 < t < t1)})

    tramos: List[Tuple[float, float, float, object]] = []   # (t_ini, t_fin, separación, punto)
    for i in range(len(cortes) - 1):
        ini, fin = cortes[i], cortes[i + 1]
        if fin - ini <= 1e-9:
            continue
        medio = (ini + fin) / 2.0
        ia = _intervalos(pa, medio, recorte + 1.0)
        ib = _intervalos(pb, medio, recorte + 1.0)
        if not ia or not ib:
            continue
        _da, dis_a = _dentro_y_distancia(ia, h)
        _db, dis_b = _dentro_y_distancia(ib, h)
        g = math.hypot(dis_a, dis_b)
        if not (sep_min < g <= SEPARACION_MAXIMA_M):
            continue
        # Lo que queda del otro lado del hueco tiene que ser un BORDE de pieza, no una tira:
        # al menos una placa de material. Si del otro lado hay una franja de un centímetro,
        # el "hueco" es una abertura (una puerta con umbral, una ventana baja con su
        # antepecho fino), que es diseño. Medido: en un modelo real eran la mitad de los
        # avisos. En un muro corto o levantado, del otro lado está el muro entero.
        cerca_a = min(ia, key=lambda iv: min(abs(iv[0]), abs(iv[1])))
        cerca_b = min(ib, key=lambda iv: min(abs(iv[0]), abs(iv[1])))
        if min(cerca_a[1] - cerca_a[0], cerca_b[1] - cerca_b[0]) < espesor - 1e-9:
            continue
        # Dónde está el hueco: entre el material de a más cercano a la franja de b y el de
        # b más cercano a la franja de a.
        qa = min((q for iv in ia for q in iv), key=lambda q: abs(q))
        qb = min((q for iv in ib for q in iv), key=lambda q: abs(q))
        xa = _en_mundo(a, b, tau, medio, qa, True)
        xb = _en_mundo(a, b, tau, medio, qb, False)
        if xa is None or xb is None:
            continue
        tramos.append((ini, fin, g, (xa + xb) / 2.0))
    return tramos


def _largo_minimo(espesor: float) -> float:
    return max(LARGO_MINIMO_PLACAS * espesor, LARGO_MINIMO_M)


def _corridas(tramos):
    """La corrida más larga de tramos CONTIGUOS: tres muescas de esquina sueltas no son un
    borde abierto aunque sumen más que el mínimo."""
    corridas: List[list] = []
    for t in tramos:
        if corridas and abs(corridas[-1][-1][1] - t[0]) <= 1e-6:
            corridas[-1].append(t)
        else:
            corridas.append([t])
    return max(corridas, key=lambda c: c[-1][1] - c[0][0])


def _corrida_mas_larga(tramos) -> float:
    if not tramos:
        return 0.0
    c = _corridas(tramos)
    return c[-1][1] - c[0][0]


def _huecos_del_cruce(a, b, indice, espesor: float, sep_min: float, candidatos=None) -> Optional[Dict]:
    """El peor hueco entre dos placas no paralelas, o None si no hay."""
    if candidatos is None:
        candidatos = _tramos_candidatos(a, b, espesor, sep_min)
    # Ocupar el hueco sólo puede ACORTAR los tramos: si ni con todos los candidatos se
    # llega al largo mínimo, no hace falta preguntar por las vecinas (es lo caro).
    if _corrida_mas_larga(candidatos) < _largo_minimo(espesor) + 1e-6:
        return None
    tramos = [
        t for t in candidatos
        if not _ocupado(t[3], indice, espesor, {a.id, b.id}, sep_min)
    ]
    if not tramos:
        return None
    mejor = _corridas(tramos)
    largo = mejor[-1][1] - mejor[0][0]
    if largo < _largo_minimo(espesor) + 1e-6:
        return None
    peor = max(mejor, key=lambda x: x[2])
    return {"separacion_m": peor[2], "largo_m": largo, "punto": peor[3]}


def detectar_huecos(piezas, escala: float, cache: Optional[Dict] = None) -> List[Dict]:
    """Huecos entre piezas que casi se tocan, del más grande al más chico.

    `piezas` son los `Panel` que se van a cortar (con `frame`), como los recibe el
    verificador de uniones.

    `cache` (opcional) recuerda lo medido entre cada par por la firma de las dos piezas.
    Si el hueco está ocupado por otra pieza se vuelve a preguntar siempre, pero eso es
    barato y sólo hace falta en los pocos pares con un hueco candidato. Sin esto, en un
    modelo de 300 piezas, eran 2 a 3 s más por clic."""
    import numpy as np

    from core.services.verificacion_uniones import _placas

    escala = escala or 1.0
    espesor = PLATE_THICKNESS_M * escala
    sep_min = max(SEPARACION_MINIMA_M, TOLERANCIA_OK_MM / 1000.0 * escala)
    a_mm = 1000.0 / escala
    placas = _placas(piezas, cache.setdefault("placas", {}) if cache is not None else None)
    margen = SEPARACION_MAXIMA_M + espesor
    indice = _Indice(placas)
    medidos = cache.setdefault("pares", {}) if cache is not None else None
    if medidos is not None and len(medidos) > 200_000:
        medidos.clear()
    out: List[Dict] = []
    for i in range(len(placas)):
        A = placas[i]
        for j in range(i + 1, len(placas)):
            B = placas[j]
            if abs(float(A.n @ B.n)) > COS_PARALELAS:
                continue
            if ((A.hi + margen) < (B.lo - margen)).any() or ((B.hi + margen) < (A.lo - margen)).any():
                continue
            if medidos is not None:
                clave = (A.firma, B.firma, round(escala, 6))
                cand = medidos.get(clave)
                if cand is None:
                    cand = medidos[clave] = _tramos_candidatos(A, B, espesor, sep_min)
            else:
                cand = _tramos_candidatos(A, B, espesor, sep_min)
            h = _huecos_del_cruce(A, B, indice, espesor, sep_min, cand) if cand else None
            if h is None:
                continue
            sep, largo = h["separacion_m"], h["largo_m"]
            p = h["punto"]
            out.append({
                "tipo": "hueco",
                "piezas": [A.id, B.id],
                "grupos": [A.gid, B.gid],
                "medida_cm_edificio": round(sep * 100.0, 1),
                "medida_mm": round(sep * a_mm, 2),
                "largo_cm_edificio": round(largo * 100.0, 1),
                "punto": {"x": round(float(p[0]), 3), "y": round(float(p[1]), 3),
                          "z": round(float(p[2]), 3)},
                "texto": (
                    f"{A.id} y {B.id} casi se tocan: queda un hueco de {sep * 100:.0f} cm "
                    f"({sep * a_mm:.1f} mm en la maqueta) a lo largo de {largo * 100:.0f} cm."
                ),
                "pregunta": (
                    "¿Tienen que encontrarse? Si es un error del modelo (un muro corto, una "
                    "esquina abierta, un muro que no apoya), corregilo en el modelo: en la "
                    "maqueta queda ese hueco."
                ),
            })
    out.sort(key=lambda s: (-s["medida_cm_edificio"] * s["largo_cm_edificio"]))
    for k, s in enumerate(out):
        s["id"] = k
    return out
