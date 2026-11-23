"""
Recorte por resta booleana: la pieza es lo que queda después de sacarle sus vecinas.

Por qué existe
--------------
Todo el encaje se venía calculando por análisis de casos: si la junta cae en el extremo o
en la panza, si hay uno o varios vecinos, si el muro cede o pasa, qué eje sobra, cuánto se
acorta. Cada rama es una fórmula escrita a mano, y cada fórmula fue —una por una— un
defecto medido: el recorte al extremo equivocado, la contraparte que nunca se emitía, la
guarda que descartaba 30 ajustes, la ranura de 7.50 m para una pared de 0.83 m.

Acá no hay ramas. La pieza es su contorno menos el volumen que ocupan las vecinas que la
invaden, y la geometría decide sola qué forma tiene el resultado: si la vecina la cruza por
el medio queda un agujero, si la toca en la esquina se lleva la esquina, y si la invaden
tres a distintas alturas queda un escalonado exacto. Es lo que haría una fresadora.

Por qué NO hace falta un motor 3D
---------------------------------
La propuesta natural sería extruir cada muro a sólido y usar OpenCASCADE/CGAL/trimesh. No
se puede y no hace falta:

- **No se puede sobre la malla del modelo.** Medido sobre los tres modelos del corpus: de
  90 grupos, CERO tienen malla cerrada. Los booleanos de malla exigen sólidos watertight;
  con entrada abierta fallan o devuelven basura en silencio. Y 52 de esos 90 no tienen
  espesor detectado — son paredes de una sola cara, sin volumen que restar.
- **No hace falta.** Todas las piezas que cortamos son PLACAS PLANAS. El resultado de un
  booleano 3D, proyectado de vuelta al plano de la pieza, es exactamente un booleano 2D.
  Basta calcular la HUELLA que la losa vecina deja sobre el plano de esta pieza y restarla
  con shapely, que ya es el motor real del pipeline.

La huella se calcula exacto, sin aproximar. La vecina B es su contorno `P_B` en su plano,
extruido el espesor de placa a cada lado. Intersecarlo con el plano de A da, en el marco
de A, la intersección de dos regiones:

1. una FRANJA — `|n_B · w(u,v) − d_B| <= t/2` es lineal en (u, v);
2. la PREIMAGEN de `P_B` — el mapa de (u,v) al marco de B es afín, así que la preimagen es
   `P_B` transformado por una matriz de 2x2 y un desplazamiento.

Las dos son polígonos y shapely las corta sin error de discretización.

Qué resuelve y qué NO — medido, no supuesto
--------------------------------------------
Sobre los tres modelos del corpus a 1:50 y 1:100, separando cada junta por el rol de la
pieza:

                              la pieza CEDE        la pieza PASA
    resta booleana (esto)    148 / 14  → 91%       9 / 155 →  5%
    pipeline actual          110 / 58  → 65%      82 /  78 → 51%

La resta es netamente mejor donde hay que SACAR material —91% contra 65%— y estructuralmente
incapaz donde hay que AGREGARLO. No es un defecto de la implementación: una diferencia sólo
puede quitar. La pieza que pasa de largo por una esquina tiene que llegar hasta la cara
EXTERIOR de su vecina, y en el modelo llega hasta la piel del muro macizo, que a escalas
gruesas queda corta (a 1:100 media placa son 15 cm de edificio y el muro asoma 1 cm).

Por eso esto no reemplaza al grafo de restricciones: lo complementa. El grafo dice a dónde
tiene que llegar cada borde —en los dos roles, incluido cuando hay que crecer— y esto lo
EJECUTA sin análisis de casos: diferencia donde el destino queda por dentro del contorno
crudo, unión donde queda por fuera.

Unidades
--------
El espesor a extruir es `PLATE_THICKNESS_M * scale_denom`, en metros de EDIFICIO. NO son
3 mm en las coordenadas del modelo: la placa mide 3 mm de plancha, que a 1:100 son 30 cm de
edificio. Extruir a 0.003 en el espacio del modelo daría una lámina cien veces más fina de
la que se va a cortar. Es la misma confusión de unidades que causó la mitad de los defectos
de este pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.group_classifier import GeometryGroup
from core.services.plate_intersect import PLATE_THICKNESS_M, mid_plane_offset_faces
from core.services.types import Face3D, Vec3, dot, normalize

# Dos planos con normales más paralelas que esto no se cortan de forma útil: la preimagen
# degenera y la "huella" sería toda la pieza.
MIN_SENO = 0.05


@dataclass
class Placa:
    """Una pieza plana con su marco 3D y su contorno 2D, lista para el booleano."""

    group_id: int
    o: Vec3          # punto 3D que proyecta a (0,0) del marco
    u: Vec3
    v: Vec3
    n: Vec3
    d: float         # plano medio: n·x = d
    poly: object     # shapely Polygon/MultiPolygon en el marco (u, v)
    width_m: float
    height_m: float
    # Signo de v respecto del "arriba" del mundo: dice cuál de los dos bordes en v es la
    # BASE de la pieza. Sin esto no se sabe si el recorte de apoyo va abajo o arriba.
    v_up: float = 1.0
    # Contorno YA RESUELTO (mismo marco que `poly`), una vez que la pieza pasó por
    # `aplicar_destinos`. Lo consultan las piezas PARALELAS que se procesan después: la
    # que ya creció ocupa ese material y la siguiente tiene que respetarlo.
    poly_final: object = None
    # True si `poly_final` tiene material que el contorno crudo no tenía.
    crecio: bool = False
    # Huellas ya calculadas de las vecinas NO paralelas sobre este plano, por
    # `(id de la vecina, espesor, acotada)`. Dependen sólo de los contornos crudos, no de
    # quién cede en cada unión, así que valen entre un armado y el siguiente del
    # resolvedor. Sin esto se recalculaban 106.000 por armado en un modelo de 300 piezas.
    huellas_cache: Dict = None
    # Vecinas que pueden afectar el recorte de esta placa, por conjunto de placas (ver
    # `clave_de_recorte`).
    vecinas_cache: Dict = None
    # Identidad de la placa por su contenido (qué caras tiene y con qué normal). La usa
    # quien conserva placas y recortes entre un pedido y el siguiente.
    firma: object = None
    # Capa que el grupo ocupa EN EL MODELO a lo largo de `n`: (mín, máx) de n·x de sus
    # vértices. Dice qué otras pieles caen dentro del espesor real del muro.
    capa: Optional[Tuple[float, float]] = None


def contenido_del_grupo(g, faces, cache: Optional[Dict]) -> int:
    """Hash de las coordenadas de todas las caras del grupo.

    Recorrer las caras de todo el modelo en cada armado costaría más que lo que ahorra,
    así que se recuerda por OBJETO grupo. El objeto se guarda junto al hash (mientras esté
    guardado, su `id()` no puede pasar a otro objeto) y se comprueba que sus caras de los
    extremos sigan siendo los mismos objetos: los grupos que no cambian entre un pedido y
    el siguiente son literalmente los mismos objetos del modelo en memoria.
    """
    fi = g.face_indices
    n = len(fi)
    primera = faces[fi[0]] if n else None
    ultima = faces[fi[-1]] if n else None
    vistos = cache.setdefault("contenidos", {}) if cache is not None else None
    if vistos is not None:
        v = vistos.get(id(g))
        if v is not None and v[0] is g and v[1] == n and v[2] is primera and v[3] is ultima:
            return v[4]
    coords = []
    for i in fi:
        f = faces[i]
        for p in f.vertices:
            coords.append((p.x, p.y, p.z))
        for anillo in f.inner_loops or ():
            coords.append(None)
            for p in anillo:
                coords.append((p.x, p.y, p.z))
        coords.append(None)
    h = hash(tuple(coords))
    if vistos is not None:
        if len(vistos) > 200_000:
            vistos.clear()
        vistos[id(g)] = (g, n, primera, ultima, h)
    return h


def construir_placa(
    grupo: GeometryGroup,
    faces: List[Face3D],
    normal: Optional[Vec3] = None,
) -> Optional[Placa]:
    """Marco y contorno de un grupo, en el mismo espacio que usa la plancha.

    `normal` tiene que ser la MISMA que usa el pipeline al proyectar
    (`orient_group_normals_outward` sobre TODOS los grupos). No se puede recalcular acá
    con un solo grupo: esa función decide el sentido "hacia afuera" contra el centroide
    del edificio, y con un grupo solo ese grupo es su propio centroide, así que la normal
    nunca se invierte. El resultado era un marco distinto al de la plancha para las piezas
    cuya normal viene dada vuelta en el archivo — y por lo tanto un `frame` corrido, que
    es lo que consumen las ranuras, el instructivo y toda la verificación.
    """
    from core.review_generate import _material_de_la_pieza
    from core.services.cutting_sheet import (
        orient_group_normals_outward,
        project_faces_to_2d,
    )

    gfaces = [faces[i] for i in grupo.face_indices if 0 <= i < len(faces)]
    if not gfaces:
        return None
    if normal is None:
        normal = orient_group_normals_outward([grupo]).get(
            grupo.id, grupo.representative_normal
        )
    n = normalize(normal)
    res = project_faces_to_2d(gfaces, n, "Y")
    if res is None:
        return None
    poly = _material_de_la_pieza(res.edges)
    if poly is None or poly.is_empty:
        return None

    d = mid_plane_offset_faces(gfaces, n)
    proy = [n.x * q.x + n.y * q.y + n.z * q.z for f in gfaces for q in f.vertices]
    ou, ov = res.origin_u, res.origin_v
    u, v = res.u_axis, res.v_axis
    o = Vec3(
        ou * u.x + ov * v.x + d * n.x,
        ou * u.y + ov * v.y + d * n.y,
        ou * u.z + ov * v.z + d * n.z,
    )
    return Placa(
        group_id=grupo.id, o=o, u=u, v=v, n=n, d=d,
        poly=poly, width_m=res.width_m, height_m=res.height_m,
        v_up=res.v_up, capa=(min(proy), max(proy)),
    )


_SIN_HUELLA = object()


def huella_cacheada(a: Placa, b: Placa, espesor_m: float, acotada: bool = False):
    """`huella`, recordando el resultado en `a`. Sólo para pares NO paralelos: entre
    paralelas lo que importa es el contorno ya resuelto de la otra, que sí cambia."""
    if a.huellas_cache is None:
        a.huellas_cache = {}
    # Por CONTENIDO de la otra placa cuando se conoce: las placas viven de un pedido al
    # siguiente y un mismo id puede ser, en otro pedido, otra pieza (dos fusiones
    # distintas crean un grupo con el mismo id). Con el id solo, la vecina de una pieza
    # fusionada se recortaba contra la huella de la fusión ANTERIOR.
    clave = (
        b.firma if b.firma is not None else b.group_id, round(espesor_m, 9), acotada,
    )
    h = a.huellas_cache.get(clave, _SIN_HUELLA)
    if h is _SIN_HUELLA:
        h = huella(a, b, espesor_m, acotada)
        a.huellas_cache[clave] = h
    return h


def huella(a: Placa, b: Placa, espesor_m: float, acotada: bool = False):
    """Región del marco de `a` que ocupa la placa `b` extruida `espesor_m`.

    Devuelve un polígono shapely en coordenadas (u, v) de `a`, o None si `b` no llega a
    cortar el plano de `a`.

    Es el booleano 3D hecho exacto en 2D: la intersección de la franja donde el plano de
    `a` cae dentro del espesor de `b`, con la preimagen del contorno de `b`.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    # Si los planos son casi paralelos no hay corte: o son la misma placa o están una al
    # lado de la otra. Sin esto la preimagen degenera y "la huella" sería la pieza entera.
    cos = abs(dot(a.n, b.n))
    if cos > (1.0 - MIN_SENO):
        return None

    # --- (1) la FRANJA: |n_b · w(u,v) − d_b| <= t/2, lineal en (u, v) ---
    ka = dot(b.n, a.u)
    kb = dot(b.n, a.v)
    kc = dot(b.n, a.o) - b.d
    if abs(ka) < 1e-12 and abs(kb) < 1e-12:
        return None
    t2 = espesor_m / 2.0
    # La franja se materializa como un rectángulo gigante rotado, acotado a la pieza.
    holgura = max(a.width_m, a.height_m) * 2.0 + espesor_m
    # Dirección de la franja en el marco de a: perpendicular al gradiente (ka, kb).
    import math

    g = math.hypot(ka, kb)
    gx, gy = ka / g, kb / g           # normal de la franja, unitaria
    px, py = -gy, gx                  # a lo largo de la franja
    # Distancia con signo del origen del marco a la línea central de la franja.
    c0 = -kc / g
    ancho = t2 / g                    # media franja, medida en el marco de a
    esquinas = [
        (gx * (c0 - ancho) + px * (-holgura), gy * (c0 - ancho) + py * (-holgura)),
        (gx * (c0 + ancho) + px * (-holgura), gy * (c0 + ancho) + py * (-holgura)),
        (gx * (c0 + ancho) + px * (holgura), gy * (c0 + ancho) + py * (holgura)),
        (gx * (c0 - ancho) + px * (holgura), gy * (c0 - ancho) + py * (holgura)),
    ]
    franja = Polygon(esquinas)
    if not franja.is_valid or franja.is_empty:
        return None

    # --- (2) la PREIMAGEN del contorno de b, por el mapa afín (u,v) -> (s,r) ---
    m11, m12 = dot(b.u, a.u), dot(b.u, a.v)
    m21, m22 = dot(b.v, a.u), dot(b.v, a.v)
    det = m11 * m22 - m12 * m21
    if abs(det) < 1e-9:
        # El mapa no es invertible: el plano de a corta al de b en una recta y el contorno
        # de b no acota nada sobre esa recta. La franja sola es la mejor cota disponible.
        #
        # OJO, medido: esto pasa SIEMPRE que los planos son perpendiculares —el encuentro
        # normal— así que la huella de casi toda vecina es una franja INFINITA, no acotada
        # por su contorno. Se probó acotarla (cortando la recta de intersección con el
        # contorno de b, con y sin una placa de margen por punta) y el invariante de
        # esquina EMPEORÓ en `Prueba app.obj`: de 544/252 a 531/265 a 1:50 y de 548/230 a
        # 543/235 a 1:100, con 2 choques más. La franja infinita funciona como un
        # sustituto conservador de "la vecina todavía va a crecer por acá". Es una
        # aproximación que acierta por la razón equivocada; queda así hasta entender qué
        # bloqueos de más son los que hacen falta.
        #
        # Quien necesite la cota exacta la pide con `acotada=True` (hoy, sólo el alcance
        # de las losas: ahí la franja infinita hacía que un muro de OTRO piso, en el mismo
        # plano que el que apoya, tapara la pisada entera).
        if acotada:
            return _huella_perpendicular(a, b, (gx, gy), (px, py), c0, ancho, holgura)
        preimagen = None
    else:
        s0 = dot(b.u, a.o) - dot(b.u, b.o)
        r0 = dot(b.v, a.o) - dot(b.v, b.o)
        inv11, inv12 = m22 / det, -m12 / det
        inv21, inv22 = -m21 / det, m11 / det

        def a_marco_de_a(s: float, r: float) -> Tuple[float, float]:
            ds, dr = s - s0, r - r0
            return (inv11 * ds + inv12 * dr, inv21 * ds + inv22 * dr)

        trozos = []
        for g_ in getattr(b.poly, "geoms", [b.poly]):
            if getattr(g_, "area", 0.0) <= 1e-12:
                continue
            ext = [a_marco_de_a(s, r) for s, r in g_.exterior.coords]
            huecos = [
                [a_marco_de_a(s, r) for s, r in h.coords] for h in g_.interiors
            ]
            try:
                p = Polygon(ext, huecos)
                if p.is_valid and not p.is_empty:
                    trozos.append(p)
            except Exception:
                continue
        preimagen = unary_union(trozos) if trozos else None

    reg = franja if preimagen is None else franja.intersection(preimagen)
    if reg.is_empty or reg.area <= 1e-12:
        return None
    return reg


def _huella_perpendicular(a: Placa, b: Placa, g, p, c0: float, ancho: float, holgura: float):
    """Huella de `b` sobre el plano de `a` cuando el mapa entre marcos es singular.

    `g` es la normal unitaria de la franja en el marco de `a`, `p` su dirección, `c0` la
    distancia del origen a su línea central y `ancho` la media franja. Se recorre la línea
    central con el parámetro τ, se la lleva al marco de `b` —donde es una recta— y se la
    corta con el contorno de `b`: los tramos de τ que caen adentro son donde `b` existe.
    """
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union

    gx, gy = g
    px, py = p

    def en_b(tau: float) -> Tuple[float, float]:
        qu, qv = gx * c0 + px * tau, gy * c0 + py * tau
        w = Vec3(
            a.o.x + qu * a.u.x + qv * a.v.x - b.o.x,
            a.o.y + qu * a.u.y + qv * a.v.y - b.o.y,
            a.o.z + qu * a.u.z + qv * a.v.z - b.o.z,
        )
        return (dot(w, b.u), dot(w, b.v))

    s0, r0 = en_b(-holgura)
    s1, r1 = en_b(holgura)
    if ((s1 - s0) ** 2 + (r1 - r0) ** 2) ** 0.5 < 1e-9:
        return None
    try:
        corte = LineString([(s0, r0), (s1, r1)]).intersection(b.poly)
    except Exception:
        return None
    if corte.is_empty:
        return None

    # De un punto de la recta en el marco de b, de vuelta a τ (la recta es afín en τ).
    ds, dr = (s1 - s0) / (2.0 * holgura), (r1 - r0) / (2.0 * holgura)
    k2 = ds * ds + dr * dr

    def tau_de(s: float, r: float) -> float:
        return -holgura + ((s - s0) * ds + (r - r0) * dr) / k2

    trozos = []
    for tramo in getattr(corte, "geoms", [corte]):
        cs = list(getattr(tramo, "coords", []))
        if len(cs) < 2:
            continue
        ts = [tau_de(s, r) for s, r in cs]
        t0, t1 = min(ts), max(ts)
        if t1 - t0 <= 1e-9:
            continue
        trozos.append(Polygon([
            (gx * (c0 - ancho) + px * t0, gy * (c0 - ancho) + py * t0),
            (gx * (c0 + ancho) + px * t0, gy * (c0 + ancho) + py * t0),
            (gx * (c0 + ancho) + px * t1, gy * (c0 + ancho) + py * t1),
            (gx * (c0 - ancho) + px * t1, gy * (c0 - ancho) + py * t1),
        ]))
    if not trozos:
        return None
    reg = unary_union(trozos)
    return None if reg.is_empty or reg.area <= 1e-12 else reg


# Dos placas con normales más paralelas que esto están "en el mismo plano" a los efectos
# de ocupar material: no se cortan, se APILAN.
COS_PARALELAS = 0.99


def distancia_entre_planos(a: Placa, b: Placa) -> Optional[float]:
    """Separación entre los planos medios de dos placas paralelas; None si no lo son."""
    if abs(dot(a.n, b.n)) < COS_PARALELAS:
        return None
    return abs(dot(a.n, b.o) - dot(a.n, a.o))


def proyeccion_paralela(a: Placa, b: Placa, poly=None):
    """Contorno de `b` llevado al marco (u, v) de `a`, para dos placas PARALELAS.

    `poly` permite proyectar otra región del marco de `b` en vez de su contorno crudo
    (p. ej. su contorno ya resuelto).

    Es el caso que `huella` deja afuera a propósito: dos planos paralelos no se cortan,
    así que no hay franja. Pero si están más cerca que una placa ocupan el mismo MDF, y la
    región compartida es simplemente el contorno de una visto desde el marco de la otra.
    Entre planos paralelos ese mapa es una rotación más un desplazamiento: exacto.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if abs(dot(a.n, b.n)) < COS_PARALELAS:
        return None
    # El resultado depende sólo de las dos placas y de la región proyectada: se recuerda
    # en `a`, por contenido de `b` (ver `huella_cacheada`). Una pieza con varias vecinas
    # paralelas repetía esta proyección en cada armado: eran 10.000 llamadas por clic.
    clave = None
    if b.firma is not None:
        clave = ("paralela", b.firma, None if poly is None else hash(poly.wkb))
        if a.huellas_cache is None:
            a.huellas_cache = {}
        guardada = a.huellas_cache.get(clave, _SIN_HUELLA)
        if guardada is not _SIN_HUELLA:
            return guardada
        if len(a.huellas_cache) > 4000:
            a.huellas_cache.clear()
    resultado = _proyectar_paralela(a, b, poly, Polygon, unary_union)
    if clave is not None:
        a.huellas_cache[clave] = resultado
    return resultado


def _proyectar_paralela(a: Placa, b: Placa, poly, Polygon, unary_union):
    bu_au, bu_av = dot(b.u, a.u), dot(b.u, a.v)
    bv_au, bv_av = dot(b.v, a.u), dot(b.v, a.v)
    rel = Vec3(b.o.x - a.o.x, b.o.y - a.o.y, b.o.z - a.o.z)
    ou, ov = dot(rel, a.u), dot(rel, a.v)

    def al_marco_de_a(s: float, r: float) -> Tuple[float, float]:
        return (ou + s * bu_au + r * bv_au, ov + s * bu_av + r * bv_av)

    trozos = []
    fuente = b.poly if poly is None else poly
    for g_ in getattr(fuente, "geoms", [fuente]):
        if getattr(g_, "area", 0.0) <= 1e-12 or not hasattr(g_, "exterior"):
            continue
        try:
            p = Polygon(
                [al_marco_de_a(s, r) for s, r in g_.exterior.coords],
                [[al_marco_de_a(s, r) for s, r in h.coords] for h in g_.interiors],
            )
        except Exception:
            continue
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_empty:
            trozos.append(p)
    return unary_union(trozos) if trozos else None


def registrar_final(placa: Placa, poly=None) -> None:
    """Anota que la placa ya se resolvió, con su contorno final (el crudo si no cambió).

    A partir de acá ocupa ese material para las paralelas que vienen después.
    """
    placa.poly_final = placa.poly if poly is None else poly
    try:
        placa.crecio = (
            poly is not None and poly.difference(placa.poly).area > 1e-9
        )
    except Exception:
        placa.crecio = False


def paralela_ya_resuelta_que_la_pisa(placa: Placa, otras, espesor_m: float) -> bool:
    """¿Alguna placa paralela YA RESUELTA creció sobre material que ésta podría ocupar?

    Es la pregunta barata que decide si una pieza sin ningún ajuste tiene que pasar igual
    por `aplicar_destinos`.
    """
    for otra in otras:
        if otra is placa or not otra.crecio:
            continue
        sep = distancia_entre_planos(placa, otra)
        if sep is not None and sep < espesor_m:
            return True
    return False


def recortar(
    placa: Placa,
    invasoras: List[Placa],
    scale_denom: float,
):
    """El contorno de `placa` menos el volumen de las placas que la invaden.

    Una sola resta, sin ramas: no hay que saber si la vecina la toca en el extremo o en la
    panza, ni cuántas son, ni qué eje sobra. Restar dos veces la misma zona tampoco
    acumula, que era el motivo del viejo "un recorte por borde".
    """
    from shapely.ops import unary_union

    espesor = PLATE_THICKNESS_M * scale_denom
    quitar = []
    for otra in invasoras:
        if otra.group_id == placa.group_id:
            continue
        h = huella(placa, otra, espesor)
        if h is not None:
            quitar.append(h)
    if not quitar:
        return placa.poly
    resto = placa.poly.difference(unary_union(quitar))
    return None if resto.is_empty else resto


# ---------------------------------------------------------------------------
# De ajustes a DESTINOS, y de destinos a geometría
#
# Ésta es la unión de las dos piezas del refactor: el grafo dice a dónde tiene que llegar
# cada borde y el booleano lo ejecuta. Reemplaza la cadena procedural que aplicaba
# `height` → `height_top` → muescas de `width` → `plane` en secuencia, arrastrando
# `clip_off_u/v` entre pasos.
# ---------------------------------------------------------------------------


@dataclass
class Destino:
    """A qué coordenada del marco CRUDO tiene que llegar un borde de la pieza.

    No es un delta: no depende de cuánto mida hoy la pieza ni de qué se aplicó antes. Que
    caiga por dentro del contorno significa recortar y por fuera significa crecer, y el
    ejecutor no necesita distinguirlos.
    """

    eje: str            # "u" | "v"
    lado: str           # "bajo" | "alto"
    objetivo_m: float
    banda: Optional[Tuple[float, float]] = None    # tramo del otro eje; None = borde entero
    # Contra QUIÉN es este destino. Al crecer, la placa de esa vecina NO cuenta como
    # obstáculo: llegar hasta su cara exterior es justamente lo que se está pidiendo.
    contra_group_id: Optional[int] = None
    motivo: str = ""
    # Destino de ALCANCE: `objetivo_m` es la coordenada del plano medio del muro,
    # `alcance_m` la media placa y `banda` lo que el muro ocupa a lo largo. La pieza tiene
    # que cubrir ese rectángulo donde ya llega hasta él; nunca se recorta (`lado` se
    # ignora).
    alcance_m: Optional[float] = None


def destinos_de_ajustes(
    ajustes,
    marco,
    scale_denom: float,
    group_by_id: Dict[int, GeometryGroup],
) -> List[Destino]:
    """Traduce los `DimensionAdjustment` de un grupo a destinos absolutos.

    Los ajustes traen CUÁNTO moverse; acá se decide UNA SOLA VEZ contra qué borde, y se
    convierte en una coordenada. Antes esa decisión estaba repetida en cuatro bloques
    distintos de `decompose_panels_from_groups`, cada uno con su propio criterio de lado y
    su propia forma de acumular desplazamientos.
    """
    out: List[Destino] = []
    base_en_v_min = marco.v_up >= 0

    for a in ajustes:
        if a.axis == "reach":
            d = _destino_de_alcance(a, marco, scale_denom, group_by_id)
            if d is not None:
                out.append(d)
            continue
        recorte = -(a.delta + getattr(a, "delta_plate", 0.0) * scale_denom)
        if abs(recorte) <= 0.001:
            continue

        if a.axis == "height":
            eje, lado = "v", ("bajo" if base_en_v_min else "alto")
        elif a.axis == "height_top":
            eje, lado = "v", ("alto" if base_en_v_min else "bajo")
        elif a.axis == "plane":
            otro = group_by_id.get(a.against_group_id)
            if otro is None:
                continue
            n = normalize(otro.representative_normal)
            du, dv = dot(n, marco.u), dot(n, marco.v)
            eje = "u" if abs(du) >= abs(dv) else "v"
            # La normal del muro apunta hacia afuera: si va en el sentido creciente del
            # eje, el borde de la losa que lo toca es el ALTO.
            lado = "alto" if (du if eje == "u" else dv) > 0 else "bajo"
        else:                                   # "width"
            otro = group_by_id.get(a.against_group_id)
            if otro is None:
                continue
            n = normalize(otro.representative_normal)
            if abs(dot(marco.u, n)) < 0.5:
                continue
            # De qué lado del panel cae el plano del vecino. Se mide en el marco CRUDO, que
            # es donde vive el destino: no hay desplazamientos acumulados que restar.
            t = (mid_plane_offset_group(otro) - dot(marco.o, n)) / dot(marco.u, n)
            eje = "u"
            lado = "bajo" if t < marco.width_m / 2.0 else "alto"

        largo = marco.width_m if eje == "u" else marco.height_m
        objetivo = recorte if lado == "bajo" else largo - recorte
        out.append(
            Destino(
                eje=eje,
                lado=lado,
                objetivo_m=objetivo,
                banda=_banda(a, marco, eje, group_by_id),
                contra_group_id=getattr(a, "against_group_id", None),
                motivo=a.reason,
            )
        )
    # NO se deduplica por borde. Se intentó quedarse con el destino "más exigente"
    # cuando dos compiten por el mismo borde, y es peor: los destinos de ancho contra
    # vecinas distintas suelen tener banda None (el vecino abarca toda la altura), así que
    # "se pisan" siempre y sobrevive uno solo. Eso es exactamente el viejo "un recorte por
    # borde, el mayor" que las muescas habían arreglado. Medido: el invariante de esquina
    # cayó de 177/151 a 117/211.
    #
    # Aplicarlos en secuencia es correcto porque cada uno lleva SU borde a SU objetivo
    # sobre SU banda: donde las bandas no se pisan no interfieren, y donde sí, la resta de
    # la más exigente sobrevive a la unión de la otra (la unión sólo llega hasta donde hay
    # material adyacente).
    return out


def _destino_de_alcance(a, marco: "Placa", scale_denom: float, group_by_id) -> Optional[Destino]:
    """Destino para que una losa llegue a la cara exterior de un muro que apoya en ella.

    Sólo para muros alineados con uno de los ejes del marco de la losa: la banda es un
    rectángulo en ese marco. Un muro oblicuo queda sin tratar (limitación conocida).
    """
    otro = group_by_id.get(getattr(a, "against_group_id", None))
    if otro is None:
        return None
    n = normalize(otro.representative_normal)
    du, dv = dot(n, marco.u), dot(n, marco.v)
    if max(abs(du), abs(dv)) < 0.95:
        return None
    eje = "u" if abs(du) >= abs(dv) else "v"
    k = du if eje == "u" else dv
    coord = (mid_plane_offset_group(otro) - dot(marco.o, n)) / k
    media = getattr(a, "delta_plate", 0.0) * scale_denom
    if media <= 0:
        return None
    # Banda: lo que el muro ocupa a lo largo del borde, sin estirarla. Se probó sumarle
    # media placa por punta para cubrir el cuadradito de la esquina y deja "orejas": un
    # tabique que llega a una fachada pasante le saca a la losa una pestaña por fuera.
    otro_eje = marco.v if eje == "u" else marco.u
    orig = dot(marco.o, otro_eje)
    proy = [
        dot(v, otro_eje) - orig
        for fi in otro.face_indices
        if 0 <= fi < len(_FACES_CACHE[0])
        for v in _FACES_CACHE[0][fi].vertices
    ]
    if not proy:
        return None
    return Destino(
        eje=eje, lado="auto", objetivo_m=coord,
        banda=(min(proy), max(proy)),
        contra_group_id=otro.id, motivo=a.reason, alcance_m=media,
    )


def mid_plane_offset_group(grupo: GeometryGroup) -> float:
    """`d` del plano medio del grupo sobre su propia normal, con las caras de la corrida."""
    from core.services.plate_intersect import mid_plane_offset

    return mid_plane_offset(
        grupo, _FACES_CACHE[0], normalize(grupo.representative_normal)
    )


# Las caras del modelo, para no arrastrarlas por toda la firma. Se setea por corrida.
#
# Es POR HILO. Era una variable del módulo, compartida: con dos cálculos a la vez (el del
# pedido y el que prevé en segundo plano el efecto de cada botón, o dos pedidos atendidos
# en hilos distintos) uno leía las caras del modelo del OTRO y recortaba mal sin ningún
# aviso. Lo encontró una prueba que fallaba sólo según el orden: una pieza de 7.575 m
# salía de 7.875 m.
class _CarasDeLaCorrida:
    def __init__(self) -> None:
        import threading

        self._por_hilo = threading.local()

    def __getitem__(self, _i: int) -> List[Face3D]:
        return getattr(self._por_hilo, "caras", [])

    def __setitem__(self, _i: int, caras: List[Face3D]) -> None:
        self._por_hilo.caras = caras


_FACES_CACHE = _CarasDeLaCorrida()


def usar_caras(faces: List[Face3D]) -> None:
    _FACES_CACHE[0] = faces


def _banda(a, marco: "Placa", eje: str, group_by_id) -> Optional[Tuple[float, float]]:
    """Franja del OTRO eje donde la restricción aplica, si el vecino no la abarca entera.

    Es lo que separa una muesca de un recorte de borde entero: dos vecinos que llegan al
    mismo extremo a alturas distintas dejan cada uno su franja, en vez de que gane el mayor.
    """
    otro = group_by_id.get(getattr(a, "against_group_id", None))
    if otro is None or eje != "u":
        return None
    ejeb, orig, largo = (marco.v, dot(marco.o, marco.v), marco.height_m)
    proy = [
        dot(v, ejeb) - orig
        for fi in otro.face_indices
        if 0 <= fi < len(_FACES_CACHE[0])
        for v in _FACES_CACHE[0][fi].vertices
    ]
    if not proy:
        return None
    lo, hi = max(min(proy), 0.0), min(max(proy), largo)
    if hi - lo <= 1e-6 or hi - lo >= largo - 1e-6:
        return None
    return (lo, hi)


_INDICE_DE_PLACAS: Dict[int, tuple] = {}


def _coplanar_con(gid, otro_id, placas, espesor_m: float) -> bool:
    """¿`gid` es otra piel del mismo muro que `otro_id`?

    Mismo plano (normales paralelas) y planos medios más cerca que una placa. No alcanza
    con comparar ids: el modelo trae cada muro como dos grupos, uno por piel.
    """
    if otro_id is None:
        return False
    # Índice por id, una vez por lista de placas: buscar recorriendo la lista eran
    # 240.000 búsquedas lineales por armado en el modelo grande.
    indice = _INDICE_DE_PLACAS.get(id(placas))
    if indice is None or indice[0] is not placas or indice[1] != len(placas):
        indice = (placas, len(placas), {p.group_id: p for p in placas})
        _INDICE_DE_PLACAS.clear()
        _INDICE_DE_PLACAS[id(placas)] = indice
    a = indice[2].get(gid)
    b = indice[2].get(otro_id)
    if a is None or b is None:
        return False
    c = dot(a.n, b.n)
    if abs(c) < 0.99:
        return False
    if abs(abs(a.d) - abs(b.d)) <= max(espesor_m, 1e-6) + 1e-4:
        return True
    # Una piel suelta cae DENTRO del espesor que el muro tiene en el modelo, a cualquier
    # escala. Comparar sólo contra una placa dependía de la escala: la piel de un muro de
    # 30 cm queda a 15 cm de su plano medio, que a 1:50 es exactamente una placa y el
    # redondeo la dejaba afuera. Medido: esa piel bloqueaba el crecimiento de los muros
    # que llegan en T y quedaban 7.5 cm cortos (luz de 1.5 mm en la plancha).
    s = 1.0 if c > 0 else -1.0
    for x, y, sy in ((a, b, s), (b, a, s)):
        if y.capa is None:
            continue
        dx = x.d * sy                      # plano medio de x en el marco de y
        lo, hi = y.capa
        if lo - 1e-3 <= dx <= hi + 1e-3:
            return True
    return False


def clave_de_recorte(
    placa: "Placa", destinos, obstaculos, espesor_m: float, cede_ante=None,
    universo=None, firma_de=None,
):
    """Todo de lo que depende el resultado de `aplicar_destinos` para esta placa.

    Dos llamadas con la misma clave dan el mismo contorno, así que el segundo se puede
    reusar. Es lo que permite que, cuando el usuario descarta o fusiona UNA pieza, se
    recalculen las vecinas afectadas y no las 300 del modelo.

    Qué entra en la clave, y por qué alcanza:

    - la placa (por contenido) y sus destinos;
    - las vecinas ante las que cede;
    - las vecinas NO paralelas cuya huella cae cerca de donde la pieza puede crecer. Las
      huellas sólo se usan para recortar lo que se agrega, y lo que se agrega nunca va más
      allá del destino más lejano; una huella fuera de ese margen no puede cambiar nada;
    - las vecinas PARALELAS a menos de una placa, con su contorno YA RESUELTO: si una
      creció distinto, esta pieza tiene que cederle otra cosa.

    Si falta algo acá el error es silencioso —se reusa una pieza que debía cambiar—, por
    eso `test_api.test_la_memoria_entre_pedidos_no_cambia_ningun_resultado` (y la del
    modelo grande) comparan el resultado con memoria contra el cálculo completo.
    """
    pb = placa.poly.bounds
    margen = 2.0 * espesor_m
    for d in destinos:
        lo, hi = (pb[0], pb[2]) if d.eje == "u" else (pb[1], pb[3])
        alcance = d.alcance_m or 0.0
        fuera = max(0.0, lo - (d.objetivo_m - alcance), (d.objetivo_m + alcance) - hi)
        margen = max(margen, fuera + 2.0 * espesor_m)

    # Qué vecinas pueden importar (las que cortan el plano de esta placa y las paralelas
    # a menos de una placa) depende sólo de las placas, no de las decisiones: se calcula
    # una vez por conjunto de placas (`universo`) y se recuerda en la placa. Sin esto se
    # recorrían las 300 vecinas de cada una de las 300 piezas en cada armado.
    relevantes = None
    if universo is not None:
        if placa.vecinas_cache is None:
            placa.vecinas_cache = {}
        relevantes = placa.vecinas_cache.get((universo, round(espesor_m, 9)))
    if relevantes is None:
        relevantes = []
        for otra in obstaculos or []:
            if otra is placa:
                continue
            h = huella_cacheada(placa, otra, espesor_m)
            if h is None:
                sep = distancia_entre_planos(placa, otra)
                if sep is not None and sep < espesor_m:
                    relevantes.append((otra, None))
            else:
                relevantes.append((otra, h.bounds))
        if universo is not None:
            if len(placa.vecinas_cache) > 16:
                placa.vecinas_cache.clear()
            placa.vecinas_cache[(universo, round(espesor_m, 9))] = relevantes

    cerca = []
    for otra, hb in relevantes:
        if hb is None:
            # De una paralela ya resuelta entra su contorno ENTERO. Se probó dejar sólo
            # lo que creció (que es lo que se le cede) para recalcular menos piezas, y el
            # banco de pruebas mostró piezas reusadas que debían cambiar. Revertido.
            cerca.append((
                otra.firma,
                None if otra.poly_final is None else hash(otra.poly_final.wkb),
                otra.crecio,
            ))
            continue
        if (hb[0] > pb[2] + margen or hb[2] < pb[0] - margen
                or hb[1] > pb[3] + margen or hb[3] < pb[1] - margen):
            continue
        cerca.append(otra.firma)
    # Las otras piezas se nombran por CONTENIDO, no por id: el id de una pieza fusionada
    # se repite entre pedidos con otra geometría.
    if firma_de is None:
        firma_de = {o.group_id: o.firma for o in (obstaculos or [])}

    def quien(gid):
        return firma_de.get(gid, gid)

    return (
        placa.firma,
        tuple(
            (d.eje, d.lado, round(d.objetivo_m, 9), d.banda, quien(d.contra_group_id),
             d.alcance_m)
            for d in destinos
        ),
        tuple(sorted((quien(g) for g in (cede_ante or ())), key=repr)),
        # Conjunto y no lista: el ORDEN de las vecinas cambia cuando el usuario fusiona
        # dos piezas (la pieza nueva va al final de la lista) y no cambia el resultado.
        frozenset(cerca),
        round(espesor_m, 9),
    )


BORDE_TOL_M = 1e-4
RESTITUIR_DENTRO = True


def _cerrar_ranuras_finas(poly, espesor_m: float):
    """Cierra las ranuras y huecos del contorno más angostos que media placa.

    A la escala de corte no existen: 5 cm de edificio son 0.5 mm de MDF a 1:100, menos que
    el corte del láser, y nadie los puede armar. Pero partían la pieza: un muro del modelo
    traía una ranura vertical de 5.6 cm a toda la altura entre su punta y el resto, y al
    recortarlo contra el vecino la punta quedaba suelta, se descartaba por astilla y el
    muro terminaba 13.9 cm antes de la cara contra la que tenía que apoyar.

    Cierre morfológico con esquinas rectas: no toca nada de una placa o más de ancho
    (ventanas, muescas de encastre, vanos).
    """
    if poly is None or poly.is_empty or espesor_m <= 0:
        return poly
    r = espesor_m / 4.0
    try:
        cerrado = poly.buffer(r, join_style=2, mitre_limit=10.0).buffer(
            -r, join_style=2, mitre_limit=10.0
        )
    except Exception:
        return poly
    cerrado = _solo_poligonos(cerrado)
    if cerrado is None or cerrado.is_empty:
        return poly
    # Sólo agrega lo que estaba entre partes de la pieza; nunca la agranda por fuera.
    return _solo_poligonos(poly.union(cerrado.intersection(poly.envelope)))


def _agujeros(poly):
    """Los agujeros interiores de la pieza (ventanas), como una sola geometría, o None."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    partes = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
    hs = [Polygon(r) for p in partes if p.geom_type == "Polygon" for r in p.interiors]
    return unary_union(hs) if hs else None


def _tramos_del_borde(poly, eje: str, lado: str, actual: float, b0: float, b1: float,
                      hondo: float = BORDE_TOL_M):
    """Intervalos, sobre el OTRO eje, donde la pieza tiene material cerca de su borde.

    Se mira una franja de `hondo` hacia adentro del borde `actual`: lo que el polígono
    ocupa ahí es el canto que crece. Devuelve [(t0, t1), ...] dentro de la banda [b0, b1].
    """
    from shapely.geometry import box

    t = hondo
    if eje == "u":
        franja = box(actual, b0, actual + t, b1) if lado == "bajo" else box(actual - t, b0, actual, b1)
    else:
        franja = box(b0, actual, b1, actual + t) if lado == "bajo" else box(b0, actual - t, b1, actual)
    borde = _solo_poligonos(poly.intersection(franja))
    if borde is None or borde.is_empty:
        return []
    partes = list(borde.geoms) if hasattr(borde, "geoms") else [borde]
    ivs = sorted(
        ((p.bounds[1], p.bounds[3]) if eje == "u" else (p.bounds[0], p.bounds[2]))
        for p in partes if not p.is_empty
    )
    out: list = []
    for a, b in ivs:
        if out and a <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return [(a, b) for a, b in out if b - a > 1e-9]


def _solo_poligonos(geom):
    """La parte de ÁREA de una geometría: descarta puntos y líneas sueltas.

    Una intersección o diferencia de shapely entre regiones que se tocan de canto devuelve
    una GeometryCollection con segmentos además de polígonos. Unida al contorno de la
    pieza lo convierte en algo que ya no se puede trazar como anillos cerrados, y la pieza
    deja de ser un polígono para todo lo que viene después (plancha, instructivo y, lo
    peor, el verificador, que la saltea en silencio).
    """
    from shapely.ops import unary_union

    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    partes = [
        g for g in getattr(geom, "geoms", [])
        if g.geom_type in ("Polygon", "MultiPolygon") and g.area > 1e-12
    ]
    from shapely.geometry import Polygon
    return unary_union(partes) if partes else Polygon()


# Una lengüeta más angosta que esto (fracción del espesor de placa) no es material: es el
# resto de una resta. 1/15 de placa son 0.2 mm de plancha, el orden del kerf.
FRACCION_ASTILLA = 1.0 / 15.0


def _sin_astillas(poly, espesor_m: float):
    """Quita del contorno las tiras más finas que el kerf y las islas que dejan.

    Cuando una pieza crece hacia su vecina y una tercera placa ocupa casi todo ese
    lugar, de lo agregado sobrevive una tira de un milímetro de edificio —centésimas de
    plancha—, a veces pegada al cuerpo y a veces suelta. No es material que se pueda
    cortar: el láser quema más que eso. Pero viaja a la plancha como un contorno aparte,
    agranda el bounding box de la pieza y, peor, hace que quien reconstruye la pieza desde
    sus aristas vea dos caras y se quede con una. Medido: 101 de las 296 piezas de
    `Prueba app.obj` a 1:50 tenían al menos una.

    Es una apertura morfológica (achicar y volver a agrandar con esquinas en inglete):
    deja idéntico todo lo que mide más que el umbral y borra lo que mide menos. Después
    se descartan las islas chicas que hayan quedado separadas del cuerpo.
    """
    if poly is None or poly.is_empty or espesor_m <= 0:
        return poly
    r = espesor_m * FRACCION_ASTILLA / 2.0
    try:
        limpio = poly.buffer(-r, join_style=2).buffer(r, join_style=2)
        # La apertura sólo puede QUITAR; el inglete puede asomar un pelo en esquinas agudas.
        limpio = limpio.intersection(poly)
    except Exception:
        return poly
    limpio = _solo_poligonos(limpio)
    if limpio is None or limpio.is_empty:
        return poly
    partes = [g for g in getattr(limpio, "geoms", [limpio]) if g.area > 1e-12]
    if len(partes) > 1:
        mayor = max(g.area for g in partes)
        partes = [g for g in partes if g.area >= 0.2 * mayor]
        from shapely.ops import unary_union
        limpio = unary_union(partes) if len(partes) > 1 else partes[0]
    return limpio


def aplicar_destinos(
    placa: "Placa",
    destinos: List[Destino],
    obstaculos: Optional[List["Placa"]] = None,
    espesor_m: float = 0.0,
    cede_ante: Optional[set] = None,
):
    """Lleva cada borde a su destino con dos booleanos, sin ninguna rama.

    `cede_ante` son los ids de las vecinas ante las que ESTA pieza cede. Nunca se crece
    dentro de su placa, pida lo que pida otro destino (ver más abajo).

    Para cada destino:
      - RESTA lo que quede del lado prohibido del objetivo (si sobra material)
      - UNE la franja que va del material actual hasta el objetivo (si falta)

    Las dos operaciones se hacen siempre; la que no corresponde queda vacía sola. Ahí está
    la diferencia con la cadena vieja, que tenía que saber de antemano si el ajuste
    acortaba o alargaba —y descartaba los que alargaban, porque `clip_panel_at_u` sólo
    sabe cortar.

    Devuelve `(poly, width_m, height_m, offset_u, offset_v)` o None.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union

    poly = placa.poly
    if poly is None or poly.is_empty:
        return None
    G = max(placa.width_m, placa.height_m) * 4.0 + 1.0

    # Material que YA ocupan otras piezas sobre este plano. Crecer hasta la cara de la
    # vecina es correcto, pero crecer DENTRO de una tercera no: son dos placas de 3 mm
    # peleando el mismo MDF y la maqueta no arma. Se calcula una vez por pieza —no depende
    # del destino— y se resta de cada agregado ANTES de unirlo.
    #
    # Sin esto, al habilitar el crecimiento los choques subían en todos los modelos: en un
    # archivo del corpus de 0 a 15 a 1:100. No era un defecto del crecimiento sino su
    # consecuencia directa: antes el 100% de los alargues se descartaba en silencio, así
    # que el conflicto no existía porque tampoco existía la solución.
    huellas: Dict[int, object] = {}
    ocupadas: List[object] = []
    if obstaculos and espesor_m > 0:
        for otra in obstaculos:
            if otra.group_id == placa.group_id:
                continue
            h = huella_cacheada(placa, otra, espesor_m)
            if h is None:
                # Paralela y a menos de una placa: no la CORTA, pero ocupa el mismo MDF.
                # Es el caso de una fachada que el modelo trae partida en dos tramos
                # coplanares con un tabique llegando a la costura: cada tramo trata el
                # encuentro como una esquina propia y crece para tapar el canto del
                # tabique, o sea que crecen UNO DENTRO DEL OTRO. Medido: 4 de los 10
                # choques de demo.obj a 1:100 eran esto.
                #
                # La que se procesó ANTES ya tiene su contorno resuelto y es ése el que
                # ocupa material; lo que CRECIÓ más allá de su contorno crudo se le resta
                # a ésta, para que el segundo tramo apoye contra el primero en vez de
                # quedarse los dos cortos. Así los dos bordes terminan a media placa del
                # plano medio del tabique: uno lo tapa y el otro hace tope contra el que
                # lo tapa.
                sep = distancia_entre_planos(placa, otra)
                # La que todavía NO se procesó no bloquea: si bloquearan las dos con su
                # contorno crudo, ninguna taparía el canto del tabique. Pasa la primera
                # (el orden es el de los grupos: mayor área primero) y cede la segunda.
                if sep is not None and sep < espesor_m and otra.poly_final is not None:
                    h = proyeccion_paralela(placa, otra, otra.poly_final)
                    if h is not None:
                        # Sólo se cede lo que la otra CRECIÓ. Se probó ceder también todo
                        # solape a menos de media placa (una hoja que monta sobre su
                        # marco) y el resultado fue mixto: 2 choques menos en demo a 1:50,
                        # pero 1 más y 6 bordes de esquina peor en el modelo grande. Se
                        # revirtió; ese solape marginal queda para el verificador.
                        cruda = proyeccion_paralela(placa, otra)
                        crecido = h if cruda is None else h.difference(cruda)
                        if not crecido.is_empty and crecido.area > 1e-12:
                            ocupadas.append(crecido)
            if h is not None:
                huellas[otra.group_id] = h

    if ocupadas:
        resto = poly.difference(unary_union(ocupadas))
        if resto.is_empty:
            return None
        if not destinos and poly.difference(resto).area <= 1e-12:
            return None        # nada que ceder y nada que ajustar: la pieza queda igual
        poly = resto
    elif not destinos:
        return None

    apoyados = {d.contra_group_id for d in destinos if d.alcance_m is not None}
    cede_ante = cede_ante or set()
    poly = _cerrar_ranuras_finas(poly, espesor_m)
    original = poly        # el contorno del modelo, antes de cualquier destino

    for d in destinos:
        if poly.is_empty:
            return None
        # Banda sobre el OTRO eje. Sin banda declarada, la restricción vale para el borde
        # entero — pero "entero" es la extensión de ESTA pieza, no el infinito auxiliar:
        # el rectángulo que se UNE para crecer usa esta misma banda, y con ±G la pieza se
        # extendía hasta el borde auxiliar. Medido: un panel de 7.666 m salía de 39.444.
        pb = poly.bounds
        if d.banda:
            b0, b1 = d.banda
        else:
            b0, b1 = (pb[1], pb[3]) if d.eje == "u" else (pb[0], pb[2])

        def caja(lo, hi):
            return box(lo, b0, hi, b1) if d.eje == "u" else box(b0, lo, b1, hi)

        if d.alcance_m is not None:
            # ALCANCE: la losa tiene que cubrir la PISADA de la placa del muro —el
            # rectángulo de una placa de ancho centrado en su plano medio, a lo largo de
            # lo que el muro ocupa— allí donde el muro apoya en su borde. Sólo agrega.
            #
            # Se mira la pisada y no "el borde de la losa" porque en una planta que no es
            # un rectángulo el borde donde apoya el muro es LOCAL: no coincide con el del
            # bounding box, ni siquiera con el de la franja del muro.
            c, m = d.objetivo_m, d.alcance_m
            pisada = caja(c - m, c + m)
            falta = pisada.difference(poly)
            if falta.is_empty or falta.area <= 1e-12:
                continue                      # muro interior: ya está todo cubierto
            if pisada.intersection(poly).area <= 1e-9:
                # La pisada no toca la losa: el muro está AL LADO, separado, no apoyado
                # en el borde. Agregar la parte "cercana" dejaría una tira suelta, sin
                # unir a la pieza, metida en la placa del muro.
                continue
            # Sólo lo que queda pegado a la losa, a no más de una placa: si la pisada
            # cruza un vacío (un muro sobre el hueco de una escalera) no se le tiende un
            # puente por debajo.
            falta = _solo_poligonos(falta.intersection(poly.buffer(2.0 * m)))
            # No se crece DENTRO de otra placa: un muro pasante que corre junto al borde
            # ocupa ese lugar. Los muros que apoyan en esta misma losa no cuentan —la
            # losa pasa por debajo de ellos, es justo lo que se pide— y son los que
            # tienen su propio destino de alcance.
            otras = []
            if not falta.is_empty:
                for otra in obstaculos or []:
                    if otra.group_id == placa.group_id or otra.group_id in apoyados:
                        continue
                    h = huella_cacheada(placa, otra, espesor_m, acotada=True)
                    if h is not None:
                        otras.append(h)
            if otras and not falta.is_empty:
                falta = _solo_poligonos(falta.difference(unary_union(otras)))
            if not falta.is_empty and falta.area > 1e-12:
                poly = _solo_poligonos(unary_union([poly, falta]))
            continue

        # 1) SACAR lo que pasa del objetivo.
        prohibido = caja(-G, d.objetivo_m) if d.lado == "bajo" else caja(d.objetivo_m, G)
        poly = poly.difference(prohibido)
        if poly.is_empty:
            return None

        # 2) AGREGAR lo que falta, desde donde hoy termina el material hasta el objetivo.
        franja = poly.intersection(caja(-G, G))
        if not franja.is_empty:
            bb = franja.bounds
            actual = (bb[0] if d.eje == "u" else bb[1]) if d.lado == "bajo" else (
                bb[2] if d.eje == "u" else bb[3]
            )
            # Sólo se agrega si el objetivo queda MÁS AFUERA que el material actual. Si
            # queda más adentro, el recorte del paso 1 ya lo resolvió y no hay nada que
            # unir. Sin esta guarda, `box()` acepta coordenadas invertidas y arma un
            # rectángulo hacia el lado equivocado: medido, una pieza de 7.666 m salía de
            # 39.444 porque el agregado se extendía hasta el borde auxiliar `G`.
            crece = (
                d.objetivo_m < actual - 1e-12 if d.lado == "bajo"
                else d.objetivo_m > actual + 1e-12
            )
            if not crece:
                continue
            # Qué se agrega:
            #
            # - DENTRO del contorno original de la pieza (lo que otro destino recortó y éste
            #   vuelve a pedir) se restituye el material del modelo, con sus vanos. Tender
            #   la franja llena rellenaba ventanas y puertas. La primera vez que se probó
            #   salió peor, pero por otra causa: destinos espurios que recortaban metros de
            #   muro (ver `assembly_adjuster.cruza_de_verdad`); sin ellos, mide mejor.
            # - MÁS ALLÁ del contorno original, un alargue CORTO (hasta media placa: el
            #   ajuste a la cara de la losa o del vecino) crece sólo donde el borde original
            #   tiene material. Un muro con un vacío que llega al borde (el
            #   muro levantado 30 cm sobre la losa salvo en las esquinas) no tiene borde en
            #   ese tramo: tender la franja de lado a lado le cerraba el vacío y lo dejaba
            #   como agujero interior. Medido en `casa_error_flotante` (el hueco de 30 cm
            #   desaparecía) y en demo.obj (una losa en L colgaba una tira de 4.6 m sobre
            #   el vacío). "En el borde" es a menos de media placa: un escalón de pocos cm
            #   (ruido del modelo) crece entero.
            hondo = max(BORDE_TOL_M, espesor_m / 2.0)
            partes_falta = []
            banda_orig = original.intersection(caja(-G, G))
            borde_orig = None
            if not banda_orig.is_empty:
                ob = banda_orig.bounds
                borde_orig = (ob[0] if d.eje == "u" else ob[1]) if d.lado == "bajo" else (
                    ob[2] if d.eje == "u" else ob[3]
                )
            if borde_orig is None:
                lim = d.objetivo_m
            elif d.lado == "bajo":
                lim = max(d.objetivo_m, borde_orig)
            else:
                lim = min(d.objetivo_m, borde_orig)
            if (lim < actual - 1e-12) if d.lado == "bajo" else (lim > actual + 1e-12):
                dentro_ = caja(lim, actual) if d.lado == "bajo" else caja(actual, lim)
                if RESTITUIR_DENTRO:
                    # Dentro del contorno original se restituye el material del MODELO, con
                    # sus vanos: tender la franja llena rellenaba ventanas y puertas.
                    dentro_ = _solo_poligonos(dentro_.intersection(original))
                if dentro_ is not None and not dentro_.is_empty:
                    partes_falta.append(dentro_)
            alargue = (
                0.0 if borde_orig is None
                else (borde_orig - d.objetivo_m if d.lado == "bajo" else d.objetivo_m - borde_orig)
            )
            if alargue > hondo:
                # Un alargue largo (cerrar una junta a otra pieza) sigue tendiendo la franja
                # de lado a lado: restringirlo también se midió peor en el modelo grande
                # (menos caras representadas, más uniones que no arman), por decisiones de
                # junta que cambian en cascada. Queda pendiente.
                partes_falta.append(
                    caja(d.objetivo_m, borde_orig) if d.lado == "bajo"
                    else caja(borde_orig, d.objetivo_m)
                )
            elif alargue > 1e-12:
                tramos = _tramos_del_borde(original, d.eje, d.lado, borde_orig, b0, b1, hondo)
                fin = borde_orig + hondo if d.lado == "bajo" else borde_orig - hondo
                for t0, t1 in tramos:
                    falta_t = (
                        (box(d.objetivo_m, t0, fin, t1) if d.lado == "bajo"
                         else box(fin, t0, d.objetivo_m, t1)) if d.eje == "u" else
                        (box(t0, d.objetivo_m, t1, fin) if d.lado == "bajo"
                         else box(t0, fin, t1, d.objetivo_m))
                    )
                    agujeros = _agujeros(original)
                    partes_falta.append(
                        falta_t if agujeros is None else falta_t.difference(agujeros)
                    )
            if not partes_falta:
                continue
            falta = unary_union(partes_falta)
            # No bloquean: ni la vecina de ESTE destino —crecer hasta su cara exterior es
            # exactamente lo que se pide— ni ninguna placa COPLANAR con ella. Un muro del
            # modelo tiene dos pieles, así que llega como dos grupos con el mismo plano:
            # excluir sólo a la nombrada dejaba a su gemela tapando la misma franja, y el
            # crecimiento se perdía. Medido: 26 bordes del rol "pasa" quedaban a 0.00 mm
            # del plano medio en vez de a 1.50, o sea sin crecer nada.
            #
            # PERO una placa ante la que esta pieza CEDE bloquea siempre, sea coplanar
            # con quien sea. El caso: un tabique llega en T a un muro que el modelo trae
            # partido en dos tramos coplanares. Contra un tramo cede y contra el otro
            # "pasa", así que el segundo destino lo hace crecer justo dentro del tramo
            # ante el que cedía, y la exclusión por coplanar de arriba le abría la puerta.
            # Medido: 11 de las 15 uniones en T/L "resueltas" que igual chocaban en
            # `Prueba app.obj` a 1:100 terminaban con el borde sobre el plano medio del
            # muro en vez de a media placa.
            otras = [
                h for gid, h in huellas.items()
                if gid != d.contra_group_id
                and (
                    gid in cede_ante
                    or not _coplanar_con(gid, d.contra_group_id, obstaculos, espesor_m)
                )
            ]
            if otras:
                # Sólo las que tocan lo que falta. Unir las cien franjas de la planta
                # para descontarle tres a un rectángulo de 15 cm era el 40% del tiempo de
                # armado. Se sigue restando la UNIÓN (de las cercanas) y no una por una:
                # da el mismo polígono, pero restar de a una cambiaba algún vértice en la
                # décima cifra y con él el estado de una o dos uniones.
                fb = falta.bounds
                cerca = [
                    h for h in otras
                    if not (
                        h.bounds[0] > fb[2] or h.bounds[2] < fb[0]
                        or h.bounds[1] > fb[3] or h.bounds[3] < fb[1]
                    )
                ]
                if cerca:
                    falta = falta.difference(unary_union(cerca))
            if not falta.is_empty and falta.area > 1e-12:
                poly = unary_union([poly, falta])

    poly = _sin_astillas(poly, espesor_m)
    if poly is None or poly.is_empty or poly.area <= 1e-9:
        return None
    minx, miny, maxx, maxy = poly.bounds
    return poly, maxx - minx, maxy - miny, minx, miny
