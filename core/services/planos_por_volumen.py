"""
Dónde va la placa de una losa, según el VOLUMEN del modelo.

Por qué existe
--------------
El plano de cada placa se calculaba con las caras del grupo: el medio entre la más baja y
la más alta. En una losa eso falla cuando el clasificador le agrega caras de otros
elementos: en un modelo real, la cara de abajo de las vigas (a 2.62 m, colgadas bajo una
losa de 2.92 a 3.12) quedaba en el grupo de la losa y la placa terminaba en 2.875, dentro
de la viga y en el aire. Todos los muros que llegaban a la losa quedaban separados.

Mirando caras sueltas no hay regla que distinga "la cara de abajo de la losa" de "la cara
de abajo de una viga pegada a la losa": en la malla son un solo sólido en T. Mirando el
volumen sí: se recorre la planta de la losa y, a cada altura, se mide qué parte de ella
es material. La losa ocupa toda la planta; la viga, una franja. La placa va en el medio
de la capa que ocupa la planta.

Sólo para LOSAS. En los muros las caras sí definen bien el plano (y los tramos de un
mismo muro se alinean aparte, ver `plate_intersect.alinear_tramos`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Parte de la planta que tiene que ser material para que esa altura cuente como "losa".
COBERTURA_LOSA = 0.5
# Para mover la placa al volumen, el volumen tiene que ser claro: la capa ocupa al menos
# esta parte de la planta. Una losa que el modelo trae en parte como sólido y en parte
# como una sola superficie (medido en demo.obj: 52 %) conserva el plano de sus caras: en
# la parte que es sólo superficie, el medio del sólido queda en el aire.
COBERTURA_SOLIDA = 0.9
# Resolución en altura.
PASO_ALTURA_M = 0.005
# Cuánto más allá de las caras del grupo se busca la capa de material.
BUSQUEDA_M = 0.6
# Muestras en planta.
PASO_PLANTA_M = 0.25
MAX_MUESTRAS = 400


def _planta(grupo, caras):
    """La planta del grupo: unión de sus caras horizontales proyectadas."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    polys = []
    for i in grupo.face_indices:
        f = caras[i]
        if abs(f.normal.y) < 0.9 or len(f.vertices) < 3:
            continue
        try:
            p = Polygon([(v.x, v.z) for v in f.vertices]).buffer(0)
        except Exception:
            continue
        if not p.is_empty:
            polys.append(p)
    return unary_union(polys) if polys else None


def capa_de_losa(grupo, caras, solido) -> Optional[Tuple[float, float, float]]:
    """`(y_abajo, y_arriba, cobertura)` de la capa de material que ocupa la planta de la
    losa, o None si no se encuentra (malla abierta, losa de una sola cara sin volumen)."""
    import shapely

    planta = _planta(grupo, caras)
    if planta is None or planta.area <= 1e-6:
        return None
    minx, minz, maxx, maxz = planta.bounds
    paso = max(PASO_PLANTA_M, np.sqrt(planta.area / MAX_MUESTRAS))
    xs = np.arange(minx + paso / 2, maxx, paso)
    zs = np.arange(minz + paso / 2, maxz, paso)
    if not len(xs) or not len(zs):
        return None
    gx, gz = np.meshgrid(xs, zs)
    q = np.column_stack([gx.ravel(), gz.ravel()])
    q = q[shapely.contains_xy(planta.buffer(-0.05), q[:, 0], q[:, 1])]
    if len(q) < 3:
        return None
    if len(q) > MAX_MUESTRAS:
        q = q[np.linspace(0, len(q) - 1, MAX_MUESTRAS).astype(int)]
    ys_caras = [v.y for i in grupo.face_indices for v in caras[i].vertices]
    # Se busca bastante más allá de las caras del grupo: una losa que llega con una sola
    # cara (la de arriba) tiene el resto de su espesor DEBAJO de esa cara.
    y0, y1 = min(ys_caras) - BUSQUEDA_M, max(ys_caras) + BUSQUEDA_M
    niveles = sorted({
        round(float(np.mean([v.y for v in caras[i].vertices])), 3)
        for i in grupo.face_indices if abs(caras[i].normal.y) >= 0.9
    })
    alturas = np.arange(y0, y1 + 1e-9, PASO_ALTURA_M)
    pts = np.column_stack([
        np.tile(q[:, 0], len(alturas)), np.repeat(alturas, len(q)), np.tile(q[:, 1], len(alturas)),
    ])
    cobertura = solido.contiene_varios(pts).reshape(len(alturas), len(q)).mean(axis=1)
    llena = cobertura >= COBERTURA_LOSA
    if not llena.any():
        return None
    # Capas contiguas. Vale sólo una capa que CONTENGA alguna cara horizontal del propio
    # grupo (con el rango amplio podría aparecer otra losa cercana); entre ésas, la que
    # más material junta (cobertura × espesor).
    def propia(k, j):
        lo = alturas[k] - PASO_ALTURA_M - 0.01
        hi = alturas[j] + PASO_ALTURA_M + 0.01
        return any(lo <= y <= hi for y in niveles)

    mejor, mejor_peso = None, -1.0
    k = 0
    while k < len(alturas):
        if not llena[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(alturas) and llena[j + 1]:
            j += 1
        peso = float(cobertura[k:j + 1].sum())
        if propia(k, j) and peso > mejor_peso:
            mejor, mejor_peso = (k, j), peso
        k = j + 1
    if mejor is None:
        return None
    k, j = mejor
    if k == 0 or j == len(alturas) - 1:
        # La capa llega al borde de lo que se buscó: el material sigue de largo (la losa
        # es la tapa de un bloque macizo, p. ej.). No es una losa definida y el "medio" de
        # ese recorte no significa nada: medido en demo.obj, mandaba placas 30 cm abajo.
        # Queda el plano de sus caras.
        return None
    abajo = float(alturas[k] - PASO_ALTURA_M / 2)
    arriba = float(alturas[j] + PASO_ALTURA_M / 2)
    return abajo, arriba, float(cobertura[k:j + 1].mean())


def planos_de_losas(grupos, caras, categoria_de, solido) -> Dict[int, float]:
    """`{id de losa: altura del plano medio de su placa}` según el volumen. Sólo las losas
    donde el volumen da una capa clara; las demás quedan con el plano de sus caras."""
    out: Dict[int, float] = {}
    for g in grupos:
        if categoria_de(g) != "floor" or abs(g.representative_normal.y) < 0.9:
            continue
        c = capa_de_losa(g, caras, solido)
        if c is None:
            continue
        out[g.id] = (c[0] + c[1]) / 2.0
    return out


# ---------------------------------------------------------------------------
# Memoria por modelo: el sólido se arma una vez por modelo y la capa de cada losa, una vez
# por contenido de la losa. Las acciones del usuario casi nunca tocan las losas.
# ---------------------------------------------------------------------------

_SOLIDOS: Dict[int, tuple] = {}
_CAPAS: Dict[tuple, Optional[tuple]] = {}


def solido_del_modelo(raw_faces, eje: str = "Y"):
    """El sólido del modelo en el sistema de ejes de los grupos.

    Se arma con las caras ORIGINALES del .obj (`Phase1Result.raw_faces`), no con las de
    trabajo: el sistema parte las caras de los muros a la altura de cada losa y esa
    partición agrega caras que rompen el conteo de cruces (medido en el edificio de prueba:
    1 acierto de 4 con las partidas, 4 de 4 con las originales). Si el modelo venía con Z
    hacia arriba se le aplica la misma rotación que a las de trabajo."""
    from core.services.fidelidad import Solido

    e = _SOLIDOS.get(id(raw_faces))
    if e is not None and e[0] is raw_faces and e[1] == eje:
        return e[2]
    if len(_SOLIDOS) >= 2:
        _SOLIDOS.clear()
        _CAPAS.clear()
    caras = raw_faces
    if eje == "Z":
        from core.pipeline import _rotate_z_to_y

        caras = [_rotate_z_to_y(f) for f in raw_faces]
    s = Solido.desde_caras(caras)
    _SOLIDOS[id(raw_faces)] = (raw_faces, eje, s)
    return s


def planos_de_losas_del_modelo(grupos, caras, raw_faces, categoria_de, cache=None, eje: str = "Y") -> Dict[int, float]:
    """`planos_de_losas` con memoria por modelo y por contenido de cada losa."""
    from core.services.recorte_csg import contenido_del_grupo

    solido = solido_del_modelo(raw_faces, eje)
    out: Dict[int, float] = {}
    for g in grupos:
        if categoria_de(g) != "floor" or abs(g.representative_normal.y) < 0.9:
            continue
        clave = (id(raw_faces), contenido_del_grupo(g, caras, cache))
        if clave not in _CAPAS:
            _CAPAS[clave] = capa_de_losa(g, caras, solido)
        c = _CAPAS[clave]
        if c is not None and c[2] >= COBERTURA_SOLIDA:
            out[g.id] = (c[0] + c[1]) / 2.0
    return out


# ---------------------------------------------------------------------------
# Cualquier orientación (muros incluidos)
# ---------------------------------------------------------------------------

def capa_de_placa(grupo, caras, solido, n) -> Optional[Tuple[float, float, float]]:
    """Como `capa_de_losa` pero a lo largo de una normal cualquiera `n` (Vec3):
    `(d_abajo, d_arriba, cobertura)` en coordenadas d = n·x."""
    import shapely
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    nn = np.array([n.x, n.y, n.z], dtype=float)
    nn /= np.linalg.norm(nn)
    # Marco del plano: u horizontal (o x si la placa es horizontal), w = n × u.
    arriba = np.array([0.0, 1.0, 0.0])
    u = np.cross(arriba, nn)
    if np.linalg.norm(u) < 1e-6:
        u = np.array([1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    w = np.cross(nn, u)
    polys, ds = [], []
    for i in grupo.face_indices:
        f = caras[i]
        fn = np.array([f.normal.x, f.normal.y, f.normal.z])
        if len(f.vertices) < 3 or abs(float(fn @ nn)) < 0.9:
            continue
        V = np.array([(v.x, v.y, v.z) for v in f.vertices])
        ds.append(float(V[0] @ nn))
        try:
            pg = Polygon(np.column_stack([V @ u, V @ w])).buffer(0)
        except Exception:
            continue
        if not pg.is_empty:
            polys.append(pg)
    if not polys:
        return None
    planta = unary_union(polys)
    if planta.area <= 1e-6:
        return None
    minx, miny, maxx, maxy = planta.bounds
    paso = max(PASO_PLANTA_M, np.sqrt(planta.area / MAX_MUESTRAS))
    xs = np.arange(minx + paso / 2, maxx, paso)
    ys = np.arange(miny + paso / 2, maxy, paso)
    if not len(xs) or not len(ys):
        return None
    gx, gy = np.meshgrid(xs, ys)
    q = np.column_stack([gx.ravel(), gy.ravel()])
    q = q[shapely.contains_xy(planta.buffer(-0.05), q[:, 0], q[:, 1])]
    if len(q) < 3:
        return None
    if len(q) > MAX_MUESTRAS:
        q = q[np.linspace(0, len(q) - 1, MAX_MUESTRAS).astype(int)]
    base = np.outer(q[:, 0], u) + np.outer(q[:, 1], w)
    niveles = sorted({round(d, 3) for d in ds})
    d0, d1 = min(ds) - BUSQUEDA_M, max(ds) + BUSQUEDA_M
    alturas = np.arange(d0, d1 + 1e-9, PASO_ALTURA_M)
    pts = (base[None, :, :] + alturas[:, None, None] * nn[None, None, :]).reshape(-1, 3)
    cobertura = solido.contiene_varios(pts).reshape(len(alturas), len(base)).mean(axis=1)
    llena = cobertura >= COBERTURA_LOSA
    mejor, mejor_peso = None, -1.0
    k = 0
    while k < len(alturas):
        if not llena[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(alturas) and llena[j + 1]:
            j += 1
        lo = alturas[k] - PASO_ALTURA_M - 0.01
        hi = alturas[j] + PASO_ALTURA_M + 0.01
        peso = float(cobertura[k:j + 1].sum())
        if any(lo <= y <= hi for y in niveles) and peso > mejor_peso:
            mejor, mejor_peso = (k, j), peso
        k = j + 1
    if mejor is None:
        return None
    k, j = mejor
    if k == 0 or j == len(alturas) - 1:
        # La capa llega al borde de lo que se buscó: el material sigue de largo (la losa
        # es la tapa de un bloque macizo, p. ej.). No es una losa definida y el "medio" de
        # ese recorte no significa nada: medido en demo.obj, mandaba placas 30 cm abajo.
        # Queda el plano de sus caras.
        return None
    return (float(alturas[k] - PASO_ALTURA_M / 2), float(alturas[j] + PASO_ALTURA_M / 2),
            float(cobertura[k:j + 1].mean()))


def planos_por_volumen_del_modelo(
    grupos, caras, raw_faces, categoria_de, normales, cache=None, eje: str = "Y",
    categorias=("floor",),
) -> Dict[int, Tuple[Tuple[float, float, float], float]]:
    """`{id: (normal de referencia, d)}` del plano de la placa de cada grupo de esas
    categorías, según su volumen, donde el sólido es claro."""
    from core.services.recorte_csg import contenido_del_grupo

    solido = solido_del_modelo(raw_faces, eje)
    out = {}
    for g in grupos:
        if categoria_de(g) not in categorias:
            continue
        n = normales.get(g.id, g.representative_normal)
        clave = (id(raw_faces), contenido_del_grupo(g, caras, cache),
                 round(n.x, 6), round(n.y, 6), round(n.z, 6))
        if clave not in _CAPAS:
            _CAPAS[clave] = capa_de_placa(g, caras, solido, n)
        c = _CAPAS[clave]
        if c is not None and c[2] >= COBERTURA_SOLIDA:
            out[g.id] = ((float(n.x), float(n.y), float(n.z)), (c[0] + c[1]) / 2.0)
    return out
