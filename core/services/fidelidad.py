"""
Fidelidad al modelo: ¿las piezas que se cortan reproducen lo que hay en el .obj?

Por qué existe
--------------
Hasta acá cada cambio del motor se juzgaba contra la línea base del propio motor (cuántas
uniones encajan). Eso dice si las piezas encajan ENTRE SÍ, no si son el edificio del
modelo: una placa de losa corrida 23 cm hacia abajo puede encajar perfecto con muros que
también se corrieron. Y pasó: se probaron tres arreglos para las losas y cada uno mejoraba
una escala y empeoraba la otra, sin forma de saber cuál era el correcto.

Esto mide contra el modelo mismo, sin pasar por ninguna decisión del clasificador:

* `Solido`: ¿un punto del espacio está DENTRO del material del modelo? Se responde con la
  malla cruda del .obj.
* `plano_en_el_material`: la placa de cada pieza tiene que pasar por material del modelo.
  Una placa cuyo plano medio cae en el aire está mal ubicada, encaje o no con sus vecinas.
* `muros_y_losas_representados`: cada cara de un muro o una losa del modelo tiene que
  estar cubierta por alguna pieza.

Es lento a propósito: lo que importa es que sea exacto.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Desplazamiento de cada consulta para no pegar justo en una arista o un vértice. Los
# modelos de arquitectura tienen las coordenadas en números redondos y un rayo paralelo a
# un eje que pasa por una arista la cuenta en los dos triángulos que la comparten: la
# paridad sale mal y un punto en el aire aparece "dentro". Valores chicos, distintos por
# eje y sin relación entre sí.
_SACUDIDA = np.array([3.1415e-5, 2.7182e-5, 1.4142e-5])
# Tope de elementos (puntos × triángulos) por operación, para acotar la memoria.
_MAX_ELEMENTOS = 2_000_000


# Hasta dónde se busca la otra lámina de un elemento abierto (ver `Solido.entre_laminas`).
ENTRE_LAMINAS_M = 0.10


@dataclass
class Solido:
    """El material del modelo, para preguntar si un punto está adentro.

    Se tiran tres rayos (uno por eje) desde el punto y se cuenta cuántas caras cruza cada
    uno: impar = adentro. Se decide por mayoría de los tres, así una malla con algún
    agujero o una cara de más no da vuelta el resultado. Dos sólidos que se tocan
    comparten una cara dos veces (una de cada uno): el rayo la cruza dos veces y la
    paridad no cambia, que es lo correcto.
    """

    tri: np.ndarray                         # (n, 3, 3)
    _indices: Dict[int, Dict[Tuple[int, int], np.ndarray]] = field(default_factory=dict)
    celda: float = 0.5

    @classmethod
    def desde_caras(cls, caras, celda: float = 0.5) -> "Solido":
        T = []
        for f in caras:
            v = [(p.x, p.y, p.z) for p in f.vertices]
            for k in range(1, len(v) - 1):
                T.append((v[0], v[k], v[k + 1]))
        s = cls(tri=np.asarray(T, dtype=float).reshape(-1, 3, 3), celda=celda)
        for eje in range(3):
            s._indexar(eje)
        return s

    def _indexar(self, eje: int) -> None:
        """Triángulos por celda de la grilla en el plano perpendicular al eje del rayo."""
        a, b = [k for k in range(3) if k != eje]
        lo = self.tri.min(axis=1)
        hi = self.tri.max(axis=1)
        c = self.celda
        ia0 = np.floor(lo[:, a] / c).astype(np.int64)
        ia1 = np.floor(hi[:, a] / c).astype(np.int64)
        ib0 = np.floor(lo[:, b] / c).astype(np.int64)
        ib1 = np.floor(hi[:, b] / c).astype(np.int64)
        na = ia1 - ia0 + 1
        nb = ib1 - ib0 + 1
        cuantas = na * nb
        t = np.repeat(np.arange(len(self.tri), dtype=np.int64), cuantas)
        # Posición de cada entrada dentro del rectángulo de celdas de su triángulo.
        inicio = np.repeat(np.cumsum(cuantas) - cuantas, cuantas)
        k = np.arange(len(t), dtype=np.int64) - inicio
        nb_t = nb[t]
        ci = ia0[t] + k // nb_t
        cj = ib0[t] + k % nb_t
        orden = np.lexsort((cj, ci))
        ci, cj, t = ci[orden], cj[orden], t[orden]
        corte = np.nonzero((np.diff(ci) != 0) | (np.diff(cj) != 0))[0] + 1
        bordes = np.concatenate([[0], corte, [len(t)]])
        self._indices[eje] = {
            (int(ci[bordes[m]]), int(cj[bordes[m]])): t[bordes[m]:bordes[m + 1]]
            for m in range(len(bordes) - 1)
        }

    def _cruces(self, p: np.ndarray, eje: int) -> int:
        a, b = [k for k in range(3) if k != eje]
        c = self.celda
        cand = self._indices[eje].get((int(math.floor(p[a] / c)), int(math.floor(p[b] / c))))
        if cand is None or not len(cand):
            return 0
        T = self.tri[cand]
        # Intersección exacta del rayo p + t·e_eje (t > 0) con cada triángulo, en el plano
        # (a, b): coordenadas baricéntricas del punto proyectado.
        x0, y0 = T[:, 0, a], T[:, 0, b]
        x1, y1 = T[:, 1, a], T[:, 1, b]
        x2, y2 = T[:, 2, a], T[:, 2, b]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        ok = np.abs(den) > 1e-14
        den = np.where(ok, den, 1.0)
        l0 = ((y1 - y2) * (p[a] - x2) + (x2 - x1) * (p[b] - y2)) / den
        l1 = ((y2 - y0) * (p[a] - x2) + (x0 - x2) * (p[b] - y2)) / den
        l2 = 1.0 - l0 - l1
        dentro = ok & (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        z = l0 * T[:, 0, eje] + l1 * T[:, 1, eje] + l2 * T[:, 2, eje]
        return int((dentro & (z > p[eje])).sum())

    def contiene(self, p) -> bool:
        q = np.asarray(p, dtype=float) + _SACUDIDA
        votos = sum(self._cruces(q, e) % 2 for e in range(3))
        return votos >= 2

    def contiene_varios(self, puntos) -> np.ndarray:
        """`contiene` para muchos puntos a la vez: se agrupan por celda de la grilla y cada
        grupo se cruza con sus triángulos candidatos en una sola operación. Mismo
        resultado que llamar a `contiene` punto por punto."""
        P = np.asarray(puntos, dtype=float).reshape(-1, 3) + _SACUDIDA
        if not len(P):
            return np.zeros(0, dtype=bool)
        votos = np.zeros(len(P), dtype=np.int64)
        for eje in range(3):
            votos += self._cruces_varios(P, eje) % 2
        return votos >= 2

    def _coeficientes(self, eje: int, celda):
        """Los triángulos de una columna de la grilla, ya preparados para el cruce. Se
        guardan: cada consulta volvía a juntar y a preparar los mismos (en el modelo
        grande, miles de triángulos por columna)."""
        cache = self.__dict__.setdefault("_pre", {})
        k = (eje, celda)
        if k in cache:
            return cache[k]
        cand = self._indices[eje].get(celda)
        if cand is None or not len(cand):
            cache[k] = None
            return None
        a, b = [q for q in range(3) if q != eje]
        T = self.tri[cand]
        x0, y0 = T[:, 0, a][None, :], T[:, 0, b][None, :]
        x1, y1 = T[:, 1, a][None, :], T[:, 1, b][None, :]
        x2, y2 = T[:, 2, a][None, :], T[:, 2, b][None, :]
        z0, z1, z2 = T[:, 0, eje][None, :], T[:, 1, eje][None, :], T[:, 2, eje][None, :]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        ok = np.abs(den) > 1e-14
        den = np.where(ok, den, 1.0)
        cache[k] = r = (cand, x0, y0, x1, y1, x2, y2, z0, z1, z2, den, ok)
        return r

    def _cruces_varios(self, P: np.ndarray, eje: int) -> np.ndarray:
        a, b = [k for k in range(3) if k != eje]
        c = self.celda
        ci = np.floor(P[:, a] / c).astype(np.int64)
        cj = np.floor(P[:, b] / c).astype(np.int64)
        out = np.zeros(len(P), dtype=np.int64)
        orden = np.lexsort((cj, ci))
        ci_o, cj_o = ci[orden], cj[orden]
        corte = np.nonzero((np.diff(ci_o) != 0) | (np.diff(cj_o) != 0))[0] + 1
        bordes = np.concatenate([[0], corte, [len(orden)]])
        indice = self._indices[eje]
        for m in range(len(bordes) - 1):
            sel = orden[bordes[m]:bordes[m + 1]]
            celda = (int(ci_o[bordes[m]]), int(cj_o[bordes[m]]))
            pre = self._coeficientes(eje, celda)
            if pre is None:
                continue
            cand, x0, y0, x1, y1, x2, y2, z0, z1, z2, den, ok = pre
            # En bloques: una celda densa del modelo cruzada con miles de puntos no puede
            # ser una sola matriz (medido: se quedaba sin memoria).
            bloque = max(1, _MAX_ELEMENTOS // max(len(cand), 1))
            for i0 in range(0, len(sel), bloque):
                s_ = sel[i0:i0 + bloque]
                pa = P[s_, a][:, None]
                pb = P[s_, b][:, None]
                l0 = ((y1 - y2) * (pa - x2) + (x2 - x1) * (pb - y2)) / den
                l1 = ((y2 - y0) * (pa - x2) + (x0 - x2) * (pb - y2)) / den
                l2 = 1.0 - l0 - l1
                dentro = ok & (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
                z = l0 * z0 + l1 * z1 + l2 * z2
                out[s_] = (dentro & (z > P[s_, eje][:, None])).sum(axis=1)
        return out

    def distancia_a_caras(self, p, hasta: float = 0.05) -> float:
        """Distancia del punto a la cara más cercana del modelo, si es menor que `hasta`
        (si no, devuelve `hasta`). Para las superficies de una sola piel, que no encierran
        volumen: una placa sobre ellas está bien ubicada aunque no esté "adentro" de nada."""
        p = np.asarray(p, dtype=float)
        c = self.celda
        cand = []
        for eje in range(3):
            a, b = [k for k in range(3) if k != eje]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    t = self._indices[eje].get((int(math.floor(p[a] / c)) + di, int(math.floor(p[b] / c)) + dj))
                    if t is not None:
                        cand.append(t)
        if not cand:
            return hasta
        idx = np.unique(np.concatenate(cand))
        T = self.tri[idx]
        lo, hi = T.min(axis=1), T.max(axis=1)
        cerca = ((p >= lo - hasta) & (p <= hi + hasta)).all(axis=1)
        if not cerca.any():
            return hasta
        d = _distancia_punto_triangulos(p, T[cerca])
        return float(min(hasta, d.min()))

    def _triangulos_cerca(self, p, hasta: float):
        c = self.celda
        cand = []
        for eje in range(3):
            a, b = [k for k in range(3) if k != eje]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    t = self._indices[eje].get(
                        (int(math.floor(p[a] / c)) + di, int(math.floor(p[b] / c)) + dj)
                    )
                    if t is not None:
                        cand.append(t)
        if not cand:
            return None
        T = self.tri[np.unique(np.concatenate(cand))]
        lo, hi = T.min(axis=1), T.max(axis=1)
        cerca = ((p >= lo - hasta) & (p <= hi + hasta)).all(axis=1)
        return T[cerca] if cerca.any() else None

    def entre_laminas(self, p, n, hasta: float = ENTRE_LAMINAS_M) -> bool:
        """¿Hay una cara del modelo a menos de `hasta` a cada lado del punto, a lo largo de
        `n`? Es el material de un elemento que el modelo trae como dos láminas sueltas,
        sin cerrar (un tabique de 5 cm hecho con dos caras): no encierra volumen y la
        paridad de rayos lo ve como aire, pero una placa entre las dos láminas está bien
        ubicada. Medido en `Prueba app.obj`: 3 de las 6 piezas "fuera del material" eran
        esto, con su plano a 2.5 cm de cada cara."""
        p = np.asarray(p, dtype=float); n = np.asarray(n, dtype=float)
        T = self._triangulos_cerca(p, hasta)
        if T is None:
            return False
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        e1, e2 = b - a, c - a
        h = np.cross(n, e2)
        det = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(det) > 1e-12
        if not ok.any():
            return False
        f = 1.0 / det[ok]
        s_ = p - a[ok]
        u = f * np.einsum("ij,ij->i", s_, h[ok])
        q = np.cross(s_, e1[ok])
        v = f * (q @ n)
        t = f * np.einsum("ij,ij->i", e2[ok], q)
        dentro = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        t = t[dentro]
        return bool(((t > 0) & (t <= hasta)).any() and ((t < 0) & (t >= -hasta)).any())

    def en_material_o_superficie(self, puntos, tol: float = 0.02, normal=None) -> np.ndarray:
        P = np.asarray(puntos, dtype=float).reshape(-1, 3)
        out = self.contiene_varios(P)
        for k in np.nonzero(~out)[0]:
            out[k] = self.distancia_a_caras(P[k], tol) < tol or (
                normal is not None and self.entre_laminas(P[k], normal)
            )
        return out


def _distancia_punto_triangulos(p: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Distancia exacta de un punto a cada triángulo (Ericson, Real-Time Collision
    Detection, 5.1.5), vectorizada."""
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    q = np.empty_like(a)
    # Por defecto: proyección dentro del triángulo.
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-18, 1e-18, den)
    v = vb / den
    w = vc / den
    q[:] = a + ab * v[:, None] + ac * w[:, None]
    m = (d1 <= 0) & (d2 <= 0)
    q[m] = a[m]
    m2 = (d3 >= 0) & (d4 <= d3)
    q[m2] = b[m2]
    m3 = (d6 >= 0) & (d5 <= d6)
    q[m3] = c[m3]
    e = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~m & ~m2 & ~m3
    t = np.where(e, d1 / np.where(d1 - d3 == 0, 1, d1 - d3), 0)
    q[e] = (a + ab * t[:, None])[e]
    e2 = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~m & ~m2 & ~m3 & ~e
    t2 = np.where(e2, d2 / np.where(d2 - d6 == 0, 1, d2 - d6), 0)
    q[e2] = (a + ac * t2[:, None])[e2]
    e3 = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~m & ~m2 & ~m3 & ~e & ~e2
    den3 = (d4 - d3) + (d5 - d6)
    t3 = np.where(e3, (d4 - d3) / np.where(den3 == 0, 1, den3), 0)
    q[e3] = (b + (c - b) * t3[:, None])[e3]
    return np.linalg.norm(q - p, axis=1)


# ---------------------------------------------------------------------------
# Muestras sobre las piezas
# ---------------------------------------------------------------------------

def _muestras_de_pieza(P, paso: float) -> np.ndarray:
    """Puntos del plano medio de la pieza, sobre su material (no en las aberturas), en una
    grilla de `paso` metros. `P` es la placa del verificador (`verificacion_uniones._placas`)."""
    import shapely

    minx, miny, maxx, maxy = P.poly.bounds
    xs = np.arange(minx + paso / 2, maxx, paso)
    ys = np.arange(miny + paso / 2, maxy, paso)
    if not len(xs) or not len(ys):
        c = P.poly.representative_point()
        xs, ys = np.array([c.x]), np.array([c.y])
    gx, gy = np.meshgrid(xs, ys)
    pts2 = np.column_stack([gx.ravel(), gy.ravel()])
    dentro = shapely.contains_xy(P.poly, pts2[:, 0], pts2[:, 1])
    pts2 = pts2[dentro]
    if not len(pts2):
        c = P.poly.representative_point()
        pts2 = np.array([[c.x, c.y]])
    return P.o + np.outer(pts2[:, 0], P.u) + np.outer(pts2[:, 1], P.v)


@dataclass
class PlanoEnMaterial:
    pieza: str
    grupo: int
    muestras: int
    en_material: int
    # Si el plano está fuera del material: cuánto habría que moverlo (a lo largo de la
    # normal) para que la mayoría de sus puntos caiga adentro. None si no se encontró.
    corrimiento_m: Optional[float] = None

    @property
    def fraccion(self) -> float:
        return self.en_material / max(self.muestras, 1)


def plano_en_el_material(
    piezas, solido: Solido, escala: float, paso: Optional[float] = None,
    buscar_corrimiento: bool = True,
) -> List[PlanoEnMaterial]:
    """Para cada pieza: qué parte de su plano medio cae dentro del material del modelo.

    Una pieza que representa bien un muro o una losa tiene su plano medio adentro del
    sólido en casi toda su superficie. Las partes que crecen para tapar una esquina caen
    dentro del muro vecino, así que también cuentan como material.
    """
    from core.services.plate_intersect import PLATE_THICKNESS_M
    from core.services.verificacion_uniones import _placas

    escala = escala or 1.0
    espesor = PLATE_THICKNESS_M * escala
    paso = paso or max(0.10, espesor / 2.0)
    out: List[PlanoEnMaterial] = []
    for P in _placas(piezas):
        m = _muestras_de_pieza(P, paso)
        dentro = solido.en_material_o_superficie(m, normal=P.n)
        r = PlanoEnMaterial(P.id, P.gid, len(m), int(dentro.sum()))
        if buscar_corrimiento and r.fraccion < 0.5:
            r.corrimiento_m = _mejor_corrimiento(m, P.n, solido, espesor)
        out.append(r)
    return out


def _mejor_corrimiento(muestras, n, solido: Solido, espesor: float) -> Optional[float]:
    """Corrimiento a lo largo de la normal (hasta ±60 cm) que deja la mayor parte de las
    muestras dentro del material."""
    sub = muestras[:: max(1, len(muestras) // 40)]
    mejor, mejor_f = None, 0.0
    for d in np.arange(-0.60, 0.6001, 0.01):
        f = solido.en_material_o_superficie(sub + n * d, normal=n).mean()
        if f > mejor_f + 1e-9 or (abs(f - mejor_f) <= 1e-9 and mejor is not None and abs(d) < abs(mejor)):
            mejor, mejor_f = float(d), f
    return mejor if mejor_f >= 0.5 else None


# ---------------------------------------------------------------------------
# Muros y losas del modelo representados
# ---------------------------------------------------------------------------

@dataclass
class CaraSinPieza:
    grupo: int
    etiqueta: str
    punto: Tuple[float, float, float]
    area_m2: float


def muros_y_losas_representados(
    grupos, caras, piezas, escala: float, categoria_de, paso: float = 0.25,
) -> Tuple[float, List[CaraSinPieza]]:
    """Qué parte de las caras de muros y losas del modelo tiene una pieza encima.

    Cada cara se muestrea en una grilla; un punto está representado si alguna pieza
    PARALELA lo cubre en su plano (a menos de media placa del borde) y está a menos de
    media placa más el espesor del modelo (hasta 60 cm) de su plano. Devuelve
    `(fracción del área representada, lista de lugares sin pieza)`.
    """
    import shapely

    from core.services.plate_intersect import PLATE_THICKNESS_M
    from core.services.verificacion_uniones import _placas

    escala = escala or 1.0
    espesor = PLATE_THICKNESS_M * escala
    placas = _placas(piezas)
    N = np.array([P.n for P in placas]) if placas else np.zeros((0, 3))
    total = cubierta = 0.0
    faltan: List[CaraSinPieza] = []
    for g in grupos:
        if categoria_de(g) not in ("wall", "floor"):
            continue
        for i in g.face_indices:
            f = caras[i]
            V = np.array([(p.x, p.y, p.z) for p in f.vertices], dtype=float)
            if len(V) < 3:
                continue
            n = np.array([f.normal.x, f.normal.y, f.normal.z], dtype=float)
            ln = np.linalg.norm(n)
            if ln < 1e-9:
                continue
            n /= ln
            area = 0.0
            for k in range(1, len(V) - 1):
                area += np.linalg.norm(np.cross(V[k] - V[0], V[k + 1] - V[0])) / 2.0
            if area < 1e-4:
                continue
            pts = _muestras_de_cara(V, n, paso)
            paralelas = [k for k in range(len(placas)) if abs(float(N[k] @ n)) > 0.95]
            paralelas_set = set(paralelas)
            cub = np.zeros(len(pts), dtype=bool)
            for k in paralelas:
                P = placas[k]
                if (pts.max(axis=0) < P.lo - 0.7).any() or (pts.min(axis=0) > P.hi + 0.7).any():
                    continue
                rel = pts - P.o
                dn = np.abs(pts @ P.n - P.d)
                cerca = dn <= espesor / 2.0 + 0.6
                if not cerca.any():
                    continue
                d2 = shapely.distance(P.poly, shapely.points(np.column_stack([rel @ P.u, rel @ P.v])))
                cub |= cerca & (d2 <= espesor / 2.0 + 0.01)
            # También la representa una pieza NO paralela que la contiene en su volumen: el
            # canto de una losa (cara vertical, en el plano del muro, que el clasificador
            # junta con el muro) lo cubre la placa de la losa, que llega hasta la cara
            # exterior. Sin esto, cortar el muro bajo la losa —que es lo fiel al modelo—
            # contaba como 1.4% del modelo sin representar en las 9 casas de prueba.
            if not cub.all():
                for k in range(len(placas)):
                    if k in paralelas_set:
                        continue
                    P = placas[k]
                    if (pts.max(axis=0) < P.lo - 0.1).any() or (pts.min(axis=0) > P.hi + 0.1).any():
                        continue
                    rel = pts - P.o
                    dentro = np.abs(pts @ P.n - P.d) <= espesor / 2.0 + 0.01
                    if not dentro.any():
                        continue
                    d2 = shapely.distance(P.poly, shapely.points(np.column_stack([rel @ P.u, rel @ P.v])))
                    cub |= dentro & (d2 <= 0.01)
            total += area
            cubierta += area * cub.mean()
            if not cub.all():
                q = pts[~cub].mean(axis=0)
                faltan.append(CaraSinPieza(g.id, g.label, tuple(float(x) for x in q), area * (1 - cub.mean())))
    return (cubierta / total if total else 1.0), faltan


def _muestras_de_cara(V: np.ndarray, n: np.ndarray, paso: float) -> np.ndarray:
    import shapely
    from shapely.geometry import Polygon

    u = V[1] - V[0]
    u /= np.linalg.norm(u)
    w = np.cross(n, u)
    p2 = np.column_stack([(V - V[0]) @ u, (V - V[0]) @ w])
    poly = Polygon(p2).buffer(0)
    if poly.is_empty:
        return V.mean(axis=0, keepdims=True)
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx + paso / 2, maxx, paso)
    ys = np.arange(miny + paso / 2, maxy, paso)
    if len(xs) and len(ys):
        gx, gy = np.meshgrid(xs, ys)
        q = np.column_stack([gx.ravel(), gy.ravel()])
        q = q[shapely.contains_xy(poly, q[:, 0], q[:, 1])]
    else:
        q = np.zeros((0, 2))
    if not len(q):
        c = poly.representative_point()
        q = np.array([[c.x, c.y]])
    return V[0] + np.outer(q[:, 0], u) + np.outer(q[:, 1], w)
