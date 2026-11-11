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
            h = huella(placa, otra, espesor_m)
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
                    h = huella(placa, otra, espesor_m, acotada=True)
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
                falta = falta.difference(unary_union(otras))
            if not falta.is_empty and falta.area > 1e-12:
                poly = unary_union([poly, falta])

    if poly.is_empty or poly.area <= 1e-9:
        return None
    minx, miny, maxx, maxy = poly.bounds
    return poly, maxx - minx, maxy - miny, minx, miny
