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


def group_plane(group: GeometryGroup, faces: List[Face3D]) -> Tuple[Vec3, float]:
    """Plano representativo del grupo: normal representativa + d medio sobre sus caras."""
    n = normalize(group.representative_normal)
    s = 0.0
    cnt = 0
    for fi in group.face_indices:
        if 0 <= fi < len(faces):
            for v in faces[fi].vertices:
                s += dot(n, v)
                cnt += 1
    d = s / cnt if cnt else 0.0
    return n, d


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


@dataclass
class PlateJoint:
    """Junta transversal entre dos placas (la futura línea de encastre)."""
    cutter_id: int          # placa que queda intacta
    cut_id: int             # placa que cede (recibe la ranura) / cuya cara recibe el apoyo
    a: Vec3                 # extremos del segmento de encastre en 3D
    b: Vec3
    width: float            # ancho de la ranura = grosor de la placa cortante
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


def resolve_plate_joints(
    groups: List[GeometryGroup], faces: List[Face3D], eps: float = EPS
) -> List[PlateJoint]:
    """
    Encuentra placas (paredes) que se cruzan en 3D y devuelve las juntas de encastre.
    Broad-phase por AABB (evita O(n²) real: sólo clasifica pares con AABB que solapa).
    Jerarquía (quién cede) vía assembly_adjuster.choose_wall_wall_yielder.
    """
    from core.services.assembly_adjuster import choose_wall_wall_yielder

    walls = [g for g in groups if g.category == "wall"]
    boxes = {g.id: _aabb(g, faces) for g in walls}
    by_id = {g.id: g for g in walls}

    joints: List[PlateJoint] = []
    for i in range(len(walls)):
        ga = walls[i]
        for j in range(i + 1, len(walls)):
            gb = walls[j]
            # sólo placas ~perpendiculares pueden atravesarse (no coplanares ni paralelas)
            ndot = abs(dot(normalize(ga.representative_normal), normalize(gb.representative_normal)))
            if ndot > 0.5:
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

            # Jerarquía: el yielder es la placa CORTADA (recibe la ranura).
            t_a = ga.thickness or 0.0
            t_b = gb.thickness or 0.0
            yid = choose_wall_wall_yielder(ga, gb, t_a, t_b, None, faces)
            cut = by_id.get(yid, gb)
            cutter = ga if cut is gb else gb
            width = (cutter.thickness or 0.0)

            seg = plate_joint_segment(cut, cutter, faces, eps)
            if seg is None:
                continue
            joints.append(
                PlateJoint(cutter_id=cutter.id, cut_id=cut.id, a=seg[0], b=seg[1],
                           width=width, kind="slot")
            )

    # NOTA: las "marcas de apoyo" (surface) se detectan en resolve_support_joints, pero NO
    # se agregan acá: inundaban la lámina de corte con rectángulos/diagonales rojos y
    # ensuciaban plate_joints (que el visor 3D usa para las intersecciones). Si se quieren
    # exponer, van en un array/entregable propio, no mezcladas en plate_joints.
    return joints


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
