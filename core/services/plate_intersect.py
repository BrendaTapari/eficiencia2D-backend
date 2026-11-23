"""
Resolución de intersecciones entre placas en 3D ANTES de proyectar (Misión 1).

Cuando dos placas volumétricas se cruzan (T / cruz), proyectar y unir siluetas pierde
la "línea de encastre" (junta transversal interior). Acá la resolvemos en 3D:

  A. Clasificación punto-plano (de Berg cap. 12, construcción de BSP).
  B. Intersección paramétrica arista-plano -> segmento de encastre 3D.
  C. Jerarquía placa cortante / cortada (reusa assembly_adjuster.choose_wall_wall_yielder)
     y partición (split) de la placa que cede.

Robustez: una sola ecuación de plano para clasificar y cortar; `s` clampeado a 0 en ε
(ε ligado a la grilla de soldado de topology); broad-phase por AABB para evitar O(n²).
Los booleanos 2D siguen en shapely.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.services.types import Face3D, Vec3, cross, dot, normalize, sub, vlength
from core.group_classifier import GeometryGroup

# ε de clasificación, ligado a la grilla de soldado (topology.WELD_TOL=1e-4).
EPS = 1e-4

POS = 1
NEG = -1
ON = 0


# ---------------------------------------------------------------------------
# A. Clasificación punto-plano
# ---------------------------------------------------------------------------


def plane_of(face: Face3D) -> Tuple[Vec3, float]:
    """Plano n·x = d de una cara (n normalizada, d = n·v0)."""
    n = normalize(face.normal)
    return n, dot(n, face.vertices[0])


def mid_plane_offset(group: GeometryGroup, faces: List[Face3D], n: Vec3) -> float:
    """Offset del plano MEDIO del grupo a lo largo de `n`: el punto medio entre sus dos
    pieles, o sea (min + max) / 2 de las proyecciones.

    FUENTE ÚNICA. Es la misma referencia que usa el recorte del tope
    (`assembly_adjuster._mid_plane_offset`, que delega acá) y la que usa la ranura
    (`group_plane`). Tenerlas separadas era un defecto real: `group_plane` promediaba los
    vértices y `_mid_plane_offset` tomaba el punto medio, así que la ranura se ubicaba en
    un plano y el recorte se calculaba contra otro. Diferencia medida en un modelo real:
    hasta 8.4 mm de edificio.

    NO es el promedio de los vértices: ese valor se sesga hacia la piel que tenga más
    vértices (más teselada, o con aberturas) y da un plano que no es el medio. Con un muro
    de 20 cm el promedio daba 0.0667 en vez de 0.100. Es la placa la que va en el plano
    medio, así que el punto medio geométrico es la referencia correcta.
    """
    return mid_plane_offset_faces(
        [faces[fi] for fi in group.face_indices if 0 <= fi < len(faces)], n
    )


def mid_plane_offset_faces(group_faces: List[Face3D], n: Vec3) -> float:
    """Igual que `mid_plane_offset` pero sobre una lista de caras ya resuelta, para los
    consumidores que no tienen el GeometryGroup a mano (p. ej. el marco de proyección
    que usa el instructivo).

    Si el grupo es un TRAMO de un muro alineado con otros (ver `alinear_tramos`), devuelve
    el plano común de ese muro en vez del suyo."""
    if group_faces:
        alineado = _PLANOS_ALINEADOS.buscar(group_faces, n)
        if alineado is not None:
            return alineado
    vals = [dot(n, v) for f in group_faces for v in f.vertices]
    if not vals:
        return 0.0
    if NIVEL_MINIMO_FRACCION > 0:
        r = _extension_por_niveles(group_faces, n)
        if r is not None:
            return (r[0] + r[1]) / 2.0
    return (min(vals) + max(vals)) / 2.0


# Parte del área de las caras PARALELAS que tiene que tener un nivel (un plano a lo largo
# de la normal) para contar en el espesor del grupo. Ver `_extension_por_niveles`.
NIVEL_MINIMO_FRACCION = 0.05


_MEMO_NIVELES: Dict[tuple, tuple] = {}
_MAX_MEMO_NIVELES = 200_000


def _extension_por_niveles(group_faces: List[Face3D], n: Vec3):
    """Memoria de `_extension_por_niveles_calculada`. El plano medio se pide decenas de
    miles de veces por armado para los mismos grupos (medido: un clic pasaba de 4 a 50 s
    sin esto). La clave es la identidad de las caras —verificada, no sólo su id—, la
    normal y el sólido en uso."""
    if not group_faces:
        return None
    solido = solido_actual()
    k = (id(group_faces[0]), id(group_faces[-1]), len(group_faces),
         round(n.x, 6), round(n.y, 6), round(n.z, 6), id(solido))
    e = _MEMO_NIVELES.get(k)
    if e is not None and e[0] is group_faces[0] and e[1] is group_faces[-1] and e[2] is solido:
        return e[3]
    r = _extension_por_niveles_calculada(group_faces, n)
    if len(_MEMO_NIVELES) >= _MAX_MEMO_NIVELES:
        _MEMO_NIVELES.clear()
    _MEMO_NIVELES[k] = (group_faces[0], group_faces[-1], solido, r)
    return r


def _extension_por_niveles_calculada(group_faces: List[Face3D], n: Vec3):
    """(mín, máx) a lo largo de `n` de los niveles de cara que pesan en el grupo.

    El plano medio salía del vértice más bajo y el más alto a lo largo de la normal, así
    que cualquier cara suelta lo movía: una piel de muro de 5 m² con un retorno de 5 cm de
    ancho (1% del área) cinco centímetros hacia afuera tenía su placa corrida 2.5 cm, fuera
    del muro. Acá el espesor lo definen los NIVELES de las caras paralelas (agrupadas al
    milímetro) que tienen al menos `NIVEL_MINIMO_FRACCION` del área. None si no hay caras
    paralelas con área.

    Un nivel chico igual cuenta si entre él y los que pesan hay MATERIAL del modelo (se le
    pregunta al sólido): es la otra piel de un muro de verdad, tapada casi entera por los
    muros que llegan a ella. Medido: un muro de 30 cm con el 4% del área en una piel
    perdía su espesor y quedaba con la placa en la otra cara. Una tira suelta tiene aire
    detrás. Sin sólido no se descarta ningún nivel (el plano de siempre)."""
    niveles: Dict[float, float] = {}
    caras_grandes: List[tuple] = []
    total = 0.0
    for f in group_faces:
        fn = f.normal
        if abs(fn.x * n.x + fn.y * n.y + fn.z * n.z) < 0.95 or len(f.vertices) < 3:
            continue
        vs = f.vertices
        area = 0.0
        for i in range(1, len(vs) - 1):
            ax, ay, az = vs[i].x - vs[0].x, vs[i].y - vs[0].y, vs[i].z - vs[0].z
            bx, by, bz = vs[i + 1].x - vs[0].x, vs[i + 1].y - vs[0].y, vs[i + 1].z - vs[0].z
            cx, cy, cz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
            area += 0.5 * (cx * cx + cy * cy + cz * cz) ** 0.5
        if area <= 0:
            continue
        d = round(sum(dot(n, v) for v in vs) / len(vs), 3)
        niveles[d] = niveles.get(d, 0.0) + area
        total += area
        cx_ = sum(v.x for v in vs) / len(vs); cy_ = sum(v.y for v in vs) / len(vs)
        cz_ = sum(v.z for v in vs) / len(vs)
        caras_grandes.append(((cx_, cy_, cz_), area, d))
    if total <= 0:
        return None
    pesan = [d for d, a in niveles.items() if a >= NIVEL_MINIMO_FRACCION * total]
    if not pesan or len(pesan) == len(niveles):
        return (min(pesan), max(pesan)) if pesan else None
    solido = solido_actual()
    if solido is None:
        return None
    import numpy as np

    lo, hi = min(pesan), max(pesan)
    nn = np.array([n.x, n.y, n.z])
    # Se pregunta frente a las caras GRANDES (el cuerpo del grupo), no frente a la cara
    # chica: una tira en la punta de un muro cae dentro del muro perpendicular que llega
    # ahí, y preguntar en ella siempre respondía "material".
    grandes = [(c, a) for c, a, d_ in sorted(caras_grandes, key=lambda t: -t[1]) if d_ in pesan][:5]
    for d in niveles:
        if d in pesan or lo <= d <= hi:
            continue
        borde = lo if d < lo else hi
        pts = np.array([
            np.array(c) + nn * ((borde + d) / 2.0 - float(np.dot(nn, c))) for c, _a in grandes
        ])
        if len(pts) and solido.contiene_varios(pts).mean() > 0.5:
            lo, hi = min(lo, d), max(hi, d)
    return lo, hi


# ---------------------------------------------------------------------------
# Tramos de un mismo muro: un solo plano de placa
#
# Un muro del modelo puede venir en tramos de distinto espesor —25 cm de un lado de un
# encuentro y 20 del otro, con la cara exterior a ras—. Cada tramo es un grupo, y cada
# grupo pone su placa en SU plano medio: las placas de un mismo muro quedan corridas la
# mitad de la diferencia de espesor. Un muro que llega a ese encuentro no puede terminar
# contra las dos: queda una luz contra una o se pisa con la otra. Medido en un modelo
# real a 1:100: 13 de los 43 problemas atribuidos al modelo eran esto.
#
# Los tramos se reconocen por la geometría (paralelos, uno a continuación del otro, con
# una cara a ras y planos medios a menos de media placa) y comparten el plano medio
# ponderado por área. El registro vale para el hilo que arma las piezas y se reemplaza en
# cada armado; guarda las caras mismas, así que nunca confunde un grupo de otro modelo.
# ---------------------------------------------------------------------------

class _Registro:
    def __init__(self) -> None:
        import threading

        self._local = threading.local()

    def fijar(self, planos) -> None:
        self._local.planos = planos

    def actual(self):
        return getattr(self._local, "planos", None)

    def buscar(self, group_faces, n) -> Optional[float]:
        planos = getattr(self._local, "planos", None)
        if not planos:
            return None
        e = planos.get((id(group_faces[0]), id(group_faces[-1]), len(group_faces)))
        if e is None or e[0] is not group_faces[0] or e[1] is not group_faces[-1]:
            return None
        nr, d = e[2], e[3]
        c = n.x * nr[0] + n.y * nr[1] + n.z * nr[2]
        if abs(c) < 0.99:
            return None
        return d * (1.0 if c > 0 else -1.0)


_PLANOS_ALINEADOS = _Registro()


def fijar_planos_de_losas(grupos, faces, planos: Dict[int, float]) -> None:
    """Agrega al registro del armado el plano de la placa de cada losa calculado por su
    VOLUMEN (ver `planos_por_volumen`). Se suma a los planos de los tramos alineados."""
    actual = dict(_PLANOS_ALINEADOS.actual() or {})
    por_id = {g.id: g for g in grupos}
    for gid, y in planos.items():
        g = por_id.get(gid)
        if g is None or not g.face_indices:
            continue
        caras = [faces[i] for i in g.face_indices if 0 <= i < len(faces)]
        if not caras:
            continue
        actual[(id(caras[0]), id(caras[-1]), len(caras))] = (caras[0], caras[-1], (0.0, 1.0, 0.0), y)
    _PLANOS_ALINEADOS.fijar(actual)


def fijar_planos_por_volumen(grupos, faces, planos) -> None:
    """Como `fijar_planos_de_losas` pero con la normal de cada grupo:
    `{id: ((nx, ny, nz), d)}`."""
    actual = dict(_PLANOS_ALINEADOS.actual() or {})
    por_id = {g.id: g for g in grupos}
    for gid, (n, d) in planos.items():
        g = por_id.get(gid)
        if g is None or not g.face_indices:
            continue
        caras = [faces[i] for i in g.face_indices if 0 <= i < len(faces)]
        if not caras:
            continue
        k = (id(caras[0]), id(caras[-1]), len(caras))
        if k in actual:
            continue                # un tramo alineado conserva el plano de su muro
        actual[k] = (caras[0], caras[-1], tuple(n), d)
    _PLANOS_ALINEADOS.fijar(actual)


def olvidar_alineacion() -> None:
    """Al empezar un armado: hasta que se alinee, cada grupo usa su propio plano."""
    _PLANOS_ALINEADOS.fijar({})
    _SOLIDO_DEL_ARMADO.fijar(None)
    _CEDER.fijar(None)
    _PLANOS_DEFINITIVOS.fijar(None)


# Si los planos de este armado ya son los definitivos (tramos alineados, losas por
# volumen). Antes, las fusiones calculan ajustes con planos provisorios que después se
# descartan; las reglas que consultan el sólido (viga, cruce) no tienen que pagar ahí: eran
# la mitad del costo de un clic de fusión.
_PLANOS_DEFINITIVOS = _Registro()


def marcar_planos_definitivos() -> None:
    _PLANOS_DEFINITIVOS.fijar(True)


def planos_definitivos() -> bool:
    return bool(_PLANOS_DEFINITIVOS.actual())


# Un cálculo en segundo plano deja acá la función que lo interrumpe si llegó un pedido
# del usuario. Por hilo: el pedido del usuario nunca se interrumpe a sí mismo. La llaman
# los bucles largos de otros módulos (los ajustes recorren miles de juntas y muros).
_CEDER = _Registro()


def fijar_ceder(fn) -> None:
    _CEDER.fijar(fn)


def ceder() -> None:
    fn = _CEDER.actual()
    if fn is not None:
        fn()


# El sólido del modelo (ver `fidelidad.Solido`) para las decisiones del armado que
# necesitan saber si entre dos piezas hay MATERIAL en el modelo (una viga entre un muro y
# la losa). Por hilo, como los planos.
_SOLIDO_DEL_ARMADO = _Registro()


def fijar_solido(solido) -> None:
    _SOLIDO_DEL_ARMADO.fijar(solido)


def solido_actual():
    return _SOLIDO_DEL_ARMADO.actual()


def plano_alineado(group: GeometryGroup, faces: List[Face3D], n: Vec3) -> Optional[float]:
    """El plano común del muro al que pertenece este tramo, o None si no se alineó.
    Para incluirlo en las claves de lo que se guarda entre un armado y el siguiente."""
    fi = group.face_indices
    if not fi or fi[0] >= len(faces) or fi[-1] >= len(faces):
        return None
    caras = [faces[fi[0]]] + [None] * (len(fi) - 2) + [faces[fi[-1]]] if len(fi) > 1 else [faces[fi[0]]]
    return _PLANOS_ALINEADOS.buscar(caras, n)

# Cara a ras: misma posición a menos de esto.
A_RAS_M = 0.002
# Hasta dónde dos tramos se consideran uno a continuación del otro.
SEPARACION_DE_TRAMOS_M = 0.5


def alinear_tramos(grupos, faces, categoria_de, espesor_placa_m: float) -> Dict[int, float]:
    """Fija el plano común de los tramos de cada muro para los armados de este hilo.

    Devuelve `{id de grupo: plano}` de los grupos que se movieron (vacío si ninguno), para
    que quien guarda piezas entre pedidos lo incluya en su clave."""
    import numpy as np

    murales = []
    for g in grupos:
        if categoria_de(g) != "wall" or not g.face_indices:
            continue
        n = normalize(g.representative_normal)
        if abs(n.y) > 0.3:
            continue
        caras = [faces[i] for i in g.face_indices if 0 <= i < len(faces)]
        if not caras:
            continue
        P = np.array([(v.x, v.y, v.z) for f in caras for v in f.vertices], dtype=float)
        nn = np.array([n.x, n.y, n.z])
        pr = P @ nn
        if pr.max() - pr.min() < 0.01:
            # Una sola piel, sin espesor: su placa va sobre esa cara y no hay "tramo de
            # otro espesor" que corregir.
            continue
        murales.append({
            "g": g, "caras": caras, "n": nn, "lo": float(pr.min()), "hi": float(pr.max()),
            "min": P.min(axis=0), "max": P.max(axis=0), "area": float(g.total_area or 0.0),
        })
    padre = list(range(len(murales)))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    limite = 0.5 * espesor_placa_m
    for i in range(len(murales)):
        A = murales[i]
        for j in range(i + 1, len(murales)):
            B = murales[j]
            c = float(A["n"] @ B["n"])
            if abs(c) < 0.995:
                continue
            blo, bhi = (B["lo"], B["hi"]) if c > 0 else (-B["hi"], -B["lo"])
            if abs((A["lo"] + A["hi"]) / 2 - (blo + bhi) / 2) > limite:
                continue
            if abs((A["lo"] + A["hi"]) - (blo + bhi)) < 2 * A_RAS_M:
                continue                       # ya están en el mismo plano
            if not (abs(A["lo"] - blo) < A_RAS_M or abs(A["hi"] - bhi) < A_RAS_M):
                continue                       # sin cara a ras: no es el mismo muro
            # Uno a continuación del otro: las cajas están a menos de lo que mide el muro
            # que suele separarlos (el tabique que llega al encuentro, hasta 50 cm). Con
            # el límite en una placa, a 1:50 un tabique de 15 cm ya los separaba.
            sep = np.maximum(A["min"] - B["max"], B["min"] - A["max"])
            if (sep > max(espesor_placa_m, SEPARACION_DE_TRAMOS_M)).any():
                continue
            # LADO A LADO en la misma planta, no apilados: comparten al menos la mitad de
            # la altura del más bajo. Dos muros de plantas distintas tienen una losa en el
            # medio y no necesitan el mismo plano; alinearlos movía la placa de muros que
            # no tenían ningún problema (medido en demo.obj: tres piezas corridas 1.8 cm,
            # y todas las que llegaban a ellas quedaban mal).
            alto = min(A["max"][1] - A["min"][1], B["max"][1] - B["min"][1])
            comun = min(A["max"][1], B["max"][1]) - max(A["min"][1], B["min"][1])
            if alto <= 0.0 or comun < 0.5 * alto:
                continue
            padre[raiz(i)] = raiz(j)
    grupos_de: Dict[int, list] = {}
    for i in range(len(murales)):
        grupos_de.setdefault(raiz(i), []).append(murales[i])
    planos = {}
    movidos: Dict[int, float] = {}
    for miembros in grupos_de.values():
        if len(miembros) < 2:
            continue
        ref = miembros[0]["n"]
        total = sum(max(m["area"], 1e-9) for m in miembros)
        d = 0.0
        for m in miembros:
            s = 1.0 if float(m["n"] @ ref) > 0 else -1.0
            d += s * (m["lo"] + m["hi"]) / 2.0 * max(m["area"], 1e-9) / total
        for m in miembros:
            caras = m["caras"]
            planos[(id(caras[0]), id(caras[-1]), len(caras))] = (
                caras[0], caras[-1], tuple(float(x) for x in ref), d,
            )
            movidos[m["g"].id] = round(d * (1.0 if float(m["n"] @ ref) > 0 else -1.0), 9)
    _PLANOS_ALINEADOS.fijar(planos)
    return movidos


def group_plane(group: GeometryGroup, faces: List[Face3D]) -> Tuple[Vec3, float]:
    """Plano representativo del grupo: normal representativa + plano MEDIO entre pieles."""
    n = normalize(group.representative_normal)
    return n, mid_plane_offset(group, faces, n)


def signed_dist(v: Vec3, n: Vec3, d: float) -> float:
    return dot(n, v) - d


def classify_vertex(v: Vec3, n: Vec3, d: float, eps: float = EPS) -> int:
    s = signed_dist(v, n, d)
    if s > eps:
        return POS
    if s < -eps:
        return NEG
    return ON


def classify_polygon(verts: List[Vec3], n: Vec3, d: float, eps: float = EPS) -> str:
    """'pos' | 'neg' | 'coplanar' | 'spanning' del polígono contra el plano (n,d)."""
    has_pos = has_neg = False
    for v in verts:
        c = classify_vertex(v, n, d, eps)
        if c == POS:
            has_pos = True
        elif c == NEG:
            has_neg = True
    if has_pos and has_neg:
        return "spanning"
    if has_pos:
        return "pos"
    if has_neg:
        return "neg"
    return "coplanar"


# ---------------------------------------------------------------------------
# B. Intersección paramétrica arista-plano -> puntos -> segmento
# ---------------------------------------------------------------------------


def edge_plane_point(va: Vec3, vb: Vec3, n: Vec3, d: float) -> Optional[Vec3]:
    """Punto de cruce de la arista (va,vb) con el plano (n,d), o None si ~paralela."""
    sa = signed_dist(va, n, d)
    sb = signed_dist(vb, n, d)
    if abs(sa - sb) < 1e-12:  # arista (casi) paralela al plano -> no se fuerza
        return None
    t = sa / (sa - sb)
    if t < -1e-9 or t > 1.0 + 1e-9:
        return None
    return Vec3(
        va.x + t * (vb.x - va.x),
        va.y + t * (vb.y - va.y),
        va.z + t * (vb.z - va.z),
    )


def face_plane_crossings(verts: List[Vec3], n: Vec3, d: float, eps: float = EPS) -> List[Vec3]:
    """Puntos donde el contorno de la cara cruza el plano (n,d)."""
    out: List[Vec3] = []
    m = len(verts)
    for i in range(m):
        va, vb = verts[i], verts[(i + 1) % m]
        ca, cb = classify_vertex(va, n, d, eps), classify_vertex(vb, n, d, eps)
        if ca == ON:
            out.append(va)
        elif ca != cb and cb != ON:
            p = edge_plane_point(va, vb, n, d)
            if p is not None:
                out.append(p)
    return out


# Espesor físico de la placa de MDF, en metros de plancha. Fuente única para todo lo que
# depende del MATERIAL (ranuras, topes), a diferencia del espesor que trae el modelo 3D,
# que es una cota del edificio.
PLATE_THICKNESS_M = 0.003
# Holgura por el kerf del láser: la ranura se corta un pelo más ancha que la placa para
# que entre a presión. CALIBRAR con la cortadora real antes de producción.
KERF_CLEARANCE_M = 0.0001
# Por encima de este |cos| entre normales dos placas son casi coplanares: se solapan, no
# se cruzan, y no hay nada que encastrar. Es el complemento de MIN_JOINT_ANGLE_DEG, que
# usa el detector de uniones: mantener los dos umbrales iguales evita que una unión sea
# "real" para una etapa e inexistente para la otra.
MAX_COPLANAR_DOT = math.cos(math.radians(20.0))


@dataclass
class PlateJoint:
    """Junta transversal entre dos placas (la futura línea de encastre)."""
    cutter_id: int          # placa que queda intacta
    cut_id: int             # placa que cede (recibe la ranura) / cuya cara recibe el apoyo
    a: Vec3                 # extremos del segmento de encastre en 3D
    b: Vec3
    # Ancho de la ranura en metros de PLANCHA (físicos). Por la ranura pasa una placa
    # real de MDF, así que su ancho lo fija el material, NO el espesor que el muro tenga
    # en el modelo: ese es una cota del edificio y al escalar daba una ranura distinta en
    # cada escala (10mm a 1:20, 2mm a 1:100 para una placa de 3mm; y 0 si la malla no
    # traía espesor, con lo cual no había ranura). Se multiplica por scale_denom al
    # descomponer, igual que los recortes de ensamble.
    width: float
    kind: str = "slot"      # "slot" = encastre físico (se corta) | "surface" = apoyo (se graba)


def _segment_from_points(pts: List[Vec3]) -> Optional[Tuple[Vec3, Vec3]]:
    """Extremos del conjunto de puntos colineales: los 2 más alejados entre sí."""
    if len(pts) < 2:
        return None
    # dirección dominante: del primero al más lejano
    best = None
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            dd = (pts[i].x - pts[j].x) ** 2 + (pts[i].y - pts[j].y) ** 2 + (pts[i].z - pts[j].z) ** 2
            if best is None or dd > best[0]:
                best = (dd, pts[i], pts[j])
    if best is None or best[0] < 1e-12:
        return None
    return best[1], best[2]


def plate_joint_segment(
    cut: GeometryGroup, cutter: GeometryGroup, faces: List[Face3D], eps: float = EPS
) -> Optional[Tuple[Vec3, Vec3]]:
    """
    Segmento de encastre = por dónde la placa `cut` (B) atraviesa el plano de `cutter`
    (A), recortado a la extensión (AABB) de A. Devuelve los extremos 3D o None.
    """
    n, d = group_plane(cutter, faces)
    crossings: List[Vec3] = []
    for fi in cut.face_indices:
        if 0 <= fi < len(faces):
            f = faces[fi]
            if classify_polygon(f.vertices, n, d, eps) == "spanning":
                crossings.extend(face_plane_crossings(f.vertices, n, d, eps))
    if len(crossings) < 2:
        return None

    # Recortar al footprint de A (su AABB): el encastre sólo existe donde A existe.
    amnx = amny = amnz = float("inf")
    amxx = amxy = amxz = float("-inf")
    for fi in cutter.face_indices:
        if 0 <= fi < len(faces):
            for v in faces[fi].vertices:
                amnx = min(amnx, v.x); amxx = max(amxx, v.x)
                amny = min(amny, v.y); amxy = max(amxy, v.y)
                amnz = min(amnz, v.z); amxz = max(amxz, v.z)
    tol = 0.05
    inside = [
        p for p in crossings
        if amnx - tol <= p.x <= amxx + tol
        and amny - tol <= p.y <= amxy + tol
        and amnz - tol <= p.z <= amxz + tol
    ]
    return _segment_from_points(inside if len(inside) >= 2 else crossings)


# ---------------------------------------------------------------------------
# C. Jerarquía (placa cortante vs cortada) + broad-phase
# ---------------------------------------------------------------------------


def _aabb(group: GeometryGroup, faces: List[Face3D]):
    mnx = mny = mnz = float("inf")
    mxx = mxy = mxz = float("-inf")
    for fi in group.face_indices:
        if 0 <= fi < len(faces):
            for v in faces[fi].vertices:
                mnx = min(mnx, v.x); mxx = max(mxx, v.x)
                mny = min(mny, v.y); mxy = max(mxy, v.y)
                mnz = min(mnz, v.z); mxz = max(mxz, v.z)
    return (mnx, mxx, mny, mxy, mnz, mxz)


def _aabb_overlap(a, b, tol: float = 0.05) -> bool:
    return (
        a[0] - tol <= b[1] and b[0] - tol <= a[1]
        and a[2] - tol <= b[3] and b[2] - tol <= a[3]
        and a[4] - tol <= b[5] and b[4] - tol <= a[5]
    )


def pair_key(a: int, b: int) -> Tuple[int, int]:
    """Clave canónica de un par de grupos, independiente del orden."""
    return (a, b) if a <= b else (b, a)


def resolve_plate_joints(
    groups: List[GeometryGroup],
    faces: List[Face3D],
    eps: float = EPS,
    yield_by_pair: Optional[Dict[Tuple[int, int], int]] = None,
    sin_resolver: Optional[set] = None,
) -> List[PlateJoint]:
    """
    Encuentra placas (paredes) que se cruzan en 3D y devuelve las juntas de encastre.
    Broad-phase por AABB (evita O(n²) real: sólo clasifica pares con AABB que solapa).

    `yield_by_pair` es la decisión YA TOMADA por `compute_adjustments`
    (`pair_key(a,b) -> id del grupo que cede`), y manda. Es obligatorio pasarla desde el
    pipeline: sin ella esta función recalculaba la jerarquía por su cuenta, con
    `topo_info=None`, con los espesores crudos (sin el fallback de 3 mm) y **sin ver
    `wall_wall_decisions`**, o sea ignorando la elección del usuario. Las dos decisiones
    se contradecían en 7 de 15 pares de un modelo real: la placa que se acortaba no era la
    que recibía la ranura, así que se quitaba material de un lado y se abría la ranura del
    otro. Ese era el fallo físico del par 255/261.

    El fallback local sólo actúa cuando el par no tiene decisión: cruces en medio de AMBOS
    muros, que `compute_adjustments` deja deliberadamente sin recorte
    (`suggested_yield_group_id = None`) porque el encastre ahí es la ranura, no el tope.
    """
    from core.services.assembly_adjuster import choose_wall_wall_yielder

    yield_by_pair = yield_by_pair or {}
    walls = [g for g in groups if g.category == "wall"]
    boxes = {g.id: _aabb(g, faces) for g in walls}
    by_id = {g.id: g for g in walls}

    joints: List[PlateJoint] = []
    for i in range(len(walls)):
        ga = walls[i]
        for j in range(i + 1, len(walls)):
            gb = walls[j]
            # Dos placas se atraviesan salvo que sean casi coplanares. El umbral es el
            # MISMO que usa el resto del pipeline para considerar que dos muros forman
            # una unión real (MIN_JOINT_ANGLE_DEG). Antes era `ndot > 0.5`, o sea 60°, y
            # eso abría un hueco de 40°: `compute_adjustments` no recorta los cruces en
            # medio de ambos muros —delega en la ranura— pero la ranura no se generaba
            # por debajo de 60°, así que toda unión oblicua en X quedaba sin recorte Y sin
            # encastre. Medido en un modelo real: dos cruces a 53° que se pisaban.
            ndot = abs(dot(normalize(ga.representative_normal), normalize(gb.representative_normal)))
            if ndot > MAX_COPLANAR_DOT:
                continue
            if not _aabb_overlap(boxes[ga.id], boxes[gb.id]):
                continue

            # ¿alguna atraviesa el plano de la otra?
            na, da = group_plane(ga, faces)
            nb, db = group_plane(gb, faces)
            a_spans_b = any(
                classify_polygon(faces[fi].vertices, nb, db, eps) == "spanning"
                for fi in ga.face_indices if 0 <= fi < len(faces)
            )
            b_spans_a = any(
                classify_polygon(faces[fi].vertices, na, da, eps) == "spanning"
                for fi in gb.face_indices if 0 <= fi < len(faces)
            )
            if not (a_spans_b or b_spans_a):
                continue

            # Si un TOPE ya resolvió la junta, no lleva ranura: acortar una de las dos
            # placas alcanza, y abrir además una ranura sólo debilita la pieza y deja un
            # agujero a la vista.
            #
            # Antes esto no se decidía en ningún lado: se generaba la ranura para todo par
            # que se atravesara y se descartaba más tarde, por accidente, porque tras el
            # recorte la recta caía fuera del panel. En cuanto se corrigió ese recorte
            # —que acortaba la ranura 4 mm y no la dejaba encastrar— las ranuras espurias
            # empezaron a dibujarse. La decisión ahora es explícita.
            #
            # NO se exige que la junta esté DETECTADA: el caso que la ranura resuelve por
            # excelencia —dos muros que se cruzan por el medio de ambos— suele no compartir
            # vértices, así que `detect_joints` no lo ve y `wall_wall_joints` ni lo lista.
            # Exigir junta detectada mataba justamente ese caso.
            if pair_key(ga.id, gb.id) in yield_by_pair:
                continue

            # Sin tope que la resuelva: hace falta ranura. Quién la recibe se decide con
            # la misma jerarquía que usa el recorte.
            t_a = ga.thickness or 0.0
            t_b = gb.thickness or 0.0
            yid = choose_wall_wall_yielder(ga, gb, t_a, t_b, None, faces)
            cut = by_id.get(yid, gb)
            cutter = ga if cut is gb else gb
            # Ancho físico: la placa que atraviesa, más la holgura del kerf para que
            # entre a presión y no floja. En una unión OBLICUA la placa se presenta de
            # costado y su sección aparente es mayor: hay que abrir la ranura
            # `espesor / sen(θ)`, el mismo factor 1/sen(θ) con el que se estiran los
            # recortes. Con una ranura de 3.1 mm recta, una placa que cruza a 53° no
            # entra.
            from core.services.assembly_adjuster import oblique_trim_factor
            ang = math.degrees(math.acos(max(-1.0, min(1.0, ndot))))
            width = (PLATE_THICKNESS_M + KERF_CLEARANCE_M) * oblique_trim_factor(ang)

            seg = plate_joint_segment(cut, cutter, faces, eps)
            if seg is None:
                continue
            joints.append(
                PlateJoint(cutter_id=cutter.id, cut_id=cut.id, a=seg[0], b=seg[1],
                           width=width, kind="slot")
            )

    joints.extend(_floor_slots(groups, faces, eps))

    # NOTA: las "marcas de apoyo" (surface) se detectan en resolve_support_joints, pero NO
    # se agregan acá: inundaban la lámina de corte con rectángulos/diagonales rojos y
    # ensuciaban plate_joints (que el visor 3D usa para las intersecciones). Si se quieren
    # exponer, van en un array/entregable propio, no mezcladas en plate_joints.
    return joints


def _floor_slots(
    groups: List[GeometryGroup], faces: List[Face3D], eps: float = EPS
) -> List[PlateJoint]:
    """Ranura PASANTE en el muro para que la losa se apoye.

    Un entrepiso que pasa entre dos muros continuos entraba sin chocar pero sin nada que
    lo sostuviera: de 21 ranuras de un modelo real, CERO involucraban una losa, porque
    `resolve_plate_joints` sólo mira pares muro-muro. La losa quedaba encajada a fricción
    y al armar la maqueta se cae.

    El muro recibe una ranura horizontal al nivel de la losa (`cut_id` = muro) y la losa
    la atraviesa (`cutter_id` = losa). El par sale de `floor_wall_encounters`, la misma
    función que decide el recorte de la losa: una sola decisión, dos consumidores.
    """
    from core.services.assembly_adjuster import floor_wall_encounters

    out: List[PlateJoint] = []
    width = PLATE_THICKNESS_M + KERF_CLEARANCE_M
    for f, w, _s in floor_wall_encounters(groups, faces):
        seg = plate_joint_segment(w, f, faces, eps)
        if seg is None:
            continue
        out.append(
            PlateJoint(cutter_id=f.id, cut_id=w.id, a=seg[0], b=seg[1],
                       width=width, kind="slot")
        )
    return out


def resolve_support_joints(
    groups: List[GeometryGroup], faces: List[Face3D], eps: float = EPS
) -> List[PlateJoint]:
    """Apoyos placa↔placa: una placa (piso/estante) se APOYA contra la cara de una pared
    (contacto butt, perpendicular, sin atravesarla) → PlateJoint kind="surface". El
    footprint se graba en rojo sobre la cara de la pared receptora (cut_id).
    """
    considered = [g for g in groups if getattr(g, "category", None) in ("wall", "floor")]
    boxes = {g.id: _aabb(g, faces) for g in considered}
    contact_eps = 0.012  # ~12 mm: tolerancia de contacto cara-arista
    out: List[PlateJoint] = []

    for i in range(len(considered)):
        for j in range(i + 1, len(considered)):
            A, B = considered[i], considered[j]
            nA = normalize(A.representative_normal)
            nB = normalize(B.representative_normal)
            if abs(dot(nA, nB)) > 0.5:
                continue  # no perpendiculares
            if not _aabb_overlap(boxes[A.id], boxes[B.id]):
                continue

            # receptor (cut) = la PARED (normal horizontal); el que se apoya (cutter) = el otro.
            A_wall = abs(nA.y) < 0.5
            B_wall = abs(nB.y) < 0.5
            if A_wall == B_wall:
                if not A_wall:
                    continue  # dos horizontales: no es apoyo pared-piso
                receiver, rester = A, B  # dos paredes ⊥: la cara de A recibe a B
            else:
                receiver, rester = (A, B) if A_wall else (B, A)

            n, d = group_plane(receiver, faces)
            # el que se apoya NO debe atravesar (eso sería slot); debe TOCAR la cara.
            spans = any(
                classify_polygon(faces[fi].vertices, n, d, eps) == "spanning"
                for fi in rester.face_indices if 0 <= fi < len(faces)
            )
            if spans:
                continue
            onplane = [
                v for fi in rester.face_indices if 0 <= fi < len(faces)
                for v in faces[fi].vertices if abs(signed_dist(v, n, d)) < contact_eps
            ]
            if len(onplane) < 2:
                continue

            # recortar la arista de contacto al footprint de la pared receptora
            rb = boxes[receiver.id]
            tol = 0.05
            inside = [
                p for p in onplane
                if rb[0] - tol <= p.x <= rb[1] + tol
                and rb[2] - tol <= p.y <= rb[3] + tol
                and rb[4] - tol <= p.z <= rb[5] + tol
            ]
            seg = _segment_from_points(inside if len(inside) >= 2 else onplane)
            if seg is None:
                continue
            width = rester.thickness or 0.006  # grosor del que se apoya (default visible)
            out.append(PlateJoint(cutter_id=rester.id, cut_id=receiver.id,
                                  a=seg[0], b=seg[1], width=width, kind="surface"))
    return out
