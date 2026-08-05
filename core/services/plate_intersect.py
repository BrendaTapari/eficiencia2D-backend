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
    que usa el instructivo)."""
    vals = [dot(n, v) for f in group_faces for v in f.vertices]
    if not vals:
        return 0.0
    return (min(vals) + max(vals)) / 2.0


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
    from core.services.assembly_adjuster import (
        floor_wall_encounters,
        walls_born_on_floors,
    )

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

    # Y la mitad que faltaba: la ranura EN LA LOSA para el muro que NACE sobre ella.
    #
    # En el edificio ese muro simplemente apoya, y el pipeline lo trataba así: le
    # recortaba media placa la base para que se apoyara sobre la cara de la losa. Pero
    # apoyar no sujeta nada. Medido sobre los tres modelos: 27 ranuras y NINGUNA en un
    # piso. Los muros de planta baja llegaban a la plancha sin ningún encastre con su
    # base — se paran a tope sobre una placa de 3 mm y se caen.
    #
    # Acá la losa recibe la ranura (`cut_id` = losa) y el muro la atraviesa
    # (`cutter_id` = muro). La otra mitad —que el muro baje hasta la cara inferior de la
    # losa en vez de apoyarse arriba— la hace `compute_adjustments` a partir de la misma
    # lista, para que las dos no puedan discrepar.
    for f, w in walls_born_on_floors(groups, faces):
        seg = plate_joint_segment(f, w, faces, eps)
        if seg is None:
            continue
        out.append(
            PlateJoint(cutter_id=w.id, cut_id=f.id, a=seg[0], b=seg[1],
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
