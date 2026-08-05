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
        v_up=res.v_up,
    )


def huella(a: Placa, b: Placa, espesor_m: float):
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


def mid_plane_offset_group(grupo: GeometryGroup) -> float:
    """`d` del plano medio del grupo sobre su propia normal, con las caras de la corrida."""
    from core.services.plate_intersect import mid_plane_offset

    return mid_plane_offset(
        grupo, _FACES_CACHE[0], normalize(grupo.representative_normal)
    )


# Las caras del modelo, para no arrastrarlas por toda la firma. Se setea por corrida.
_FACES_CACHE: List[List[Face3D]] = [[]]


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


def _coplanar_con(gid, otro_id, placas, espesor_m: float) -> bool:
    """¿`gid` es otra piel del mismo muro que `otro_id`?

    Mismo plano (normales paralelas) y planos medios más cerca que una placa. No alcanza
    con comparar ids: el modelo trae cada muro como dos grupos, uno por piel.
    """
    if otro_id is None:
        return False
    a = next((p for p in placas if p.group_id == gid), None)
    b = next((p for p in placas if p.group_id == otro_id), None)
    if a is None or b is None:
        return False
    if abs(dot(a.n, b.n)) < 0.99:
        return False
    return abs(abs(a.d) - abs(b.d)) <= max(espesor_m, 1e-6)


def aplicar_destinos(
    placa: "Placa",
    destinos: List[Destino],
    obstaculos: Optional[List["Placa"]] = None,
    espesor_m: float = 0.0,
):
    """Lleva cada borde a su destino con dos booleanos, sin ninguna rama.

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
    if obstaculos and espesor_m > 0:
        for otra in obstaculos:
            if otra.group_id == placa.group_id:
                continue
            h = huella(placa, otra, espesor_m)
            if h is not None:
                huellas[otra.group_id] = h

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
            falta = (
                caja(d.objetivo_m, actual) if d.lado == "bajo"
                else caja(actual, d.objetivo_m)
            )
            # No bloquean: ni la vecina de ESTE destino —crecer hasta su cara exterior es
            # exactamente lo que se pide— ni ninguna placa COPLANAR con ella. Un muro del
            # modelo tiene dos pieles, así que llega como dos grupos con el mismo plano:
            # excluir sólo a la nombrada dejaba a su gemela tapando la misma franja, y el
            # crecimiento se perdía. Medido: 26 bordes del rol "pasa" quedaban a 0.00 mm
            # del plano medio en vez de a 1.50, o sea sin crecer nada.
            otras = [
                h for gid, h in huellas.items()
                if gid != d.contra_group_id
                and not _coplanar_con(gid, d.contra_group_id, obstaculos, espesor_m)
            ]
            if otras:
                falta = falta.difference(unary_union(otras))
            if not falta.is_empty and falta.area > 1e-12:
                poly = unary_union([poly, falta])

    if poly.is_empty or poly.area <= 1e-9:
        return None
    minx, miny, maxx, maxy = poly.bounds
    return poly, maxx - minx, maxy - miny, minx, miny
