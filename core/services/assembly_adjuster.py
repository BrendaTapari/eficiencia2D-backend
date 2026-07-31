import math
from typing import List, Dict, Optional, Literal, Tuple
from dataclasses import dataclass

# Importamos nuestros tipos y servicios
from core.services.types import Face3D, normalize
from core.services.joint_detector import Joint, MIN_JOINT_ANGLE_DEG
from core.group_classifier import GeometryGroup
from core.services.joint_topology_classifier import (
    JointTopology,
    JointTopologyInfo,
    classify_joint_topology,
    is_critical_joint,
    wall_run_length,
)

# Espesor por defecto cuando el modelo no lo trae (mallas de una sola cara). Al cortar
# en MDF el grosor existe: es el estándar de 3 mm. Permite resolver los encastres
# muro-muro (quién corta a quién) aunque la malla no tenga espesor.
DEFAULT_MDF_THICKNESS_M = 0.003
# Espesor físico de la placa de MDF, en metros de plancha. Fuente única de todo lo que
# depende del MATERIAL. Se importa del mismo lugar que usa la ranura de encastre para que
# tope y ranura no puedan desalinearse.
from core.services.plate_intersect import (  # noqa: E402
    PLATE_THICKNESS_M,
    mid_plane_offset,
    pair_key,
)

# Ángulo mínimo entre dos muros para tratarlos como una unión real. Se importa del
# detector (única definición) para que no quede una franja de uniones detectadas pero
# nunca ajustadas, ni al revés.


def oblique_trim_factor(dihedral_angle_deg: float) -> float:
    """Cuánto hay que acortar el muro que cede, en múltiplos de su espesor.

    A 90° el muro que cede se acorta exactamente el espesor de la placa contra la que
    topa. En una unión OBLICUA de ángulo θ, la misma placa se interpone a lo largo de
    una distancia mayor medida sobre el eje del muro que cede: el recorte correcto es
    espesor / sen(θ). Ejemplos: 90° → ×1.00, 45° → ×1.41, 30° → ×2.00.
    Sin esta corrección, un muro en diagonal queda largo y las piezas chocan igual.
    """
    ang = max(MIN_JOINT_ANGLE_DEG, min(90.0, dihedral_angle_deg))
    return 1.0 / math.sin(math.radians(ang))

# ============================================================================
# Assembly Adjuster
#
# Dadas las uniones detectadas y los grosores de los componentes, calcula los
# ajustes de dimensiones para que las piezas cortadas con láser encajen físicamente.
# ============================================================================


@dataclass
class DimensionAdjustment:
    group_id: int
    delta: float
    # "height"     → recorta la BASE del muro (muro encima de losa de piso)
    # "height_top" → recorta la CIMA del muro (techo/losa encima del muro)
    # "width"      → recorta un lado lateral (muro-muro)
    # "plane"      → recorta la pieza EN SU PROPIO PLANO contra `against_group_id`; el eje
    #                (u o v) y el lado salen de la normal de ese grupo. Es el caso de una
    #                losa que tiene que entrar entre dos muros continuos.
    axis: Literal["height", "height_top", "width", "plane"]
    reason: str
    joint_index: int
    # Grupo CONTRA el que se recorta. Sólo lo usa `axis="plane"`, donde el eje del panel
    # (u o v) y el lado se deducen de la normal de ese grupo.
    against_group_id: Optional[int] = None
    # El ajuste tiene DOS componentes con unidades distintas, y hay que sumarlas:
    #   `delta`       -> metros de EDIFICIO (p. ej. el voladizo que el muro del modelo
    #                    mete más allá del plano medio del vecino). Escala con el modelo.
    #   `delta_plate` -> metros de PLANCHA (la media placa de MDF). NO escala: son 1.5mm
    #                    a cualquier escala, así que se multiplica por scale_denom al
    #                    descomponer para sobrevivir al 1/scale_denom del nesting.
    # Mezclarlas es imprescindible: usar una sola hacía que el error cambiara de signo
    # con la escala (+0.5mm a 1:20, -0.5mm a 1:50 en una esquina en L).
    delta_plate: float = 0.0


@dataclass
class WallWallJoint:
    joint_index: int
    group_a: int
    group_b: int
    yield_group_id: Optional[int] = None
    suggested_yield_group_id: Optional[int] = None
    topology: Optional[JointTopology] = None
    critical: Optional[bool] = None


@dataclass
class AdjustmentsResult:
    adjustments: List[DimensionAdjustment]
    wall_wall_joints: List[WallWallJoint]


def yield_by_pair(result: "AdjustmentsResult") -> Dict[Tuple[int, int], int]:
    """`pair_key(a,b) -> id del grupo que cede`, para que la ranura use exactamente la
    misma decisión que el recorte.

    Se omiten las juntas sin resolver (cruces en medio de ambos muros): ahí no hay tope
    que valga y `resolve_plate_joints` decide sola a quién le abre la ranura.
    """
    out: Dict[Tuple[int, int], int] = {}
    for ww in result.wall_wall_joints:
        if ww.yield_group_id is None:
            continue
        # Una misma pareja puede tener más de una junta detectada; la primera manda para
        # que el resultado no dependa del orden de la lista.
        out.setdefault(pair_key(ww.group_a, ww.group_b), ww.yield_group_id)
    return out


# ---------------------------------------------------------------------------
# API Pública
# ---------------------------------------------------------------------------


def compute_adjustments(
    joints: List[Joint],
    groups: List[GeometryGroup],
    wall_wall_decisions: Optional[Dict[int, int]] = None,
    faces: Optional[List[Face3D]] = None,
) -> AdjustmentsResult:
    """
    Calcula los ajustes automáticos de dimensiones para uniones muro-piso e
    identifica las uniones muro-muro que requieren resolución manual.
    """
    group_by_id: Dict[int, GeometryGroup] = {g.id: g for g in groups}
    wall_wall_decisions_map = wall_wall_decisions or {}

    adjustments: List[DimensionAdjustment] = []
    wall_wall_joints: List[WallWallJoint] = []

    for ji, joint in enumerate(joints):
        # Muro-muro admite uniones OBLICUAS (ver MIN_JOINT_ANGLE_DEG); muro-losa sigue
        # restringido a ~90° más abajo, para no alterar el trato de techos inclinados.
        if joint.dihedral_angle < MIN_JOINT_ANGLE_DEG or joint.dihedral_angle > 95:
            continue

        g_a = group_by_id.get(joint.group_a)
        g_b = group_by_id.get(joint.group_b)

        if not g_a or not g_b:
            continue
        if g_a.category == "discard" or g_b.category == "discard":
            continue

        abs_y_a = abs(g_a.representative_normal.y)
        abs_y_b = abs(g_b.representative_normal.y)

        a_is_floor = g_a.category == "floor" and abs_y_a > 0.5
        b_is_floor = g_b.category == "floor" and abs_y_b > 0.5

        if a_is_floor != b_is_floor:
            # Unión Muro–Losa (piso o techo). Se mantiene acotada a ~90°: la lógica de
            # apoyo (is_wall_on_top / is_roof_above_wall) asume encuentro perpendicular.
            if joint.dihedral_angle < 75:
                continue
            floor = g_a if a_is_floor else g_b
            wall = g_b if a_is_floor else g_a

            # El muro se acorta el grosor del MATERIAL REAL que se va a cortar (MDF de
            # 3 mm), no el espesor de la losa que trae el archivo 3D: esa es una cota del
            # edificio (p. ej. 25 cm) y al reducirla por la escala daba un recorte
            # distinto en cada escala (5 mm a 1:50, 2.5 mm a 1:100) en vez de los 3 mm
            # físicos que mide la placa. Mismo criterio que las juntas muro-muro.
            # El borde del muro debe apoyar sobre la CARA de la placa de piso: a media
            # placa de su plano medio. Se descuenta además el voladizo que el muro del
            # modelo mete más allá de ese plano.
            voladizo = overshoot_past_plane_m(wall, floor, faces) or 0.0
            media_placa = PLATE_THICKNESS_M / 2.0

            label = floor.label if floor.label else f"Grupo {floor.id}"

            if is_wall_on_top(wall, floor, joint):
                # Muro encima de la losa → recortar BASE del muro
                adjustments.append(
                    DimensionAdjustment(
                        group_id=wall.id,
                        delta=-voladizo,
                        delta_plate=-media_placa,
                        axis="height",
                        reason=f"Apoyo en {label} (voladizo {voladizo * 1000:.0f}mm + media placa)",
                        joint_index=ji,
                    )
                )
            elif is_roof_above_wall(floor, wall, joint):
                # Techo/losa encima del muro → recortar CIMA del muro
                adjustments.append(
                    DimensionAdjustment(
                        group_id=wall.id,
                        delta=-voladizo,
                        delta_plate=-media_placa,
                        axis="height_top",
                        reason=f"Techo {label} (voladizo {voladizo * 1000:.0f}mm + media placa)",
                        joint_index=ji,
                    )
                )

        elif not a_is_floor and not b_is_floor:
            # Unión Muro–Muro
            t_a = g_a.thickness or 0.0
            t_b = g_b.thickness or 0.0
            # Si el modelo no trae espesor (mallas de una sola cara), igual habrá grosor
            # al cortar: el MDF estándar (3 mm). Se usa ese fallback para poder SUGERIR y
            # APLICAR quién corta a quién, en vez de "no se detectó espesor".
            eff_t_a = t_a if t_a > 0.001 else DEFAULT_MDF_THICKNESS_M
            eff_t_b = t_b if t_b > 0.001 else DEFAULT_MDF_THICKNESS_M

            topo_info = (
                classify_joint_topology(joint, g_a, g_b, faces) if faces else None
            )
            topology = topo_info.topology if topo_info else "unknown"

            # QUIÉN PUEDE CEDER lo decide la posición del cruce, no el grosor: acortar
            # una pieza le quita material de un EXTREMO. Si el cruce cae en el medio de
            # un muro, acortarlo no toca ese choque y encima lo deja corto donde sí
            # tenía que llegar (era la causa de piezas que no alcanzaban a la vecina).
            #
            # Pero ANTES hay que saber QUÉ dimensión sobra. Un contacto HORIZONTAL —dos
            # muros que se encuentran a lo largo, como un faldón de techo contra la banda
            # de borde— recorre las dos piezas de punta a punta: preguntar si la junta cae
            # en un extremo LATERAL no tiene sentido, y acortar el ancho no resuelve nada.
            # Lo que sobra ahí es ALTURA. Sin esta distinción esas juntas quedaban
            # marcadas como "cruce en medio de ambos", se delegaban a una ranura que
            # tampoco correspondía (no se atraviesan, sólo se tocan) y terminaban sin
            # recorte y sin encastre. Medido: 3 de 38 uniones en un modelo real, las 3 sin
            # resolver y las 3 pisándose.
            if joint.horizontal_frac >= 0.5:
                eje_a = vertical_end_axis(g_a, joint)
                eje_b = vertical_end_axis(g_b, joint)
                a_at_end = eje_a is not None
                b_at_end = eje_b is not None
            else:
                eje_a = eje_b = "width"
                fa = joint_position_frac(g_a, joint, faces)
                fb = joint_position_frac(g_b, joint, faces)
                a_at_end = fa is None or fa <= END_ZONE_FRAC or fa >= 1.0 - END_ZONE_FRAC
                b_at_end = fb is None or fb <= END_ZONE_FRAC or fb >= 1.0 - END_ZONE_FRAC

            if a_at_end and not b_at_end:
                suggested_yield_group_id = g_a.id   # sólo A puede acortarse
            elif b_at_end and not a_at_end:
                suggested_yield_group_id = g_b.id
            elif not a_at_end and not b_at_end:
                # Cruce en el medio de AMBOS (dos muros que se atraviesan): acortar no
                # resuelve nada. El encastre acá es una ranura, que resuelve
                # resolve_plate_joints; no se propone recorte.
                suggested_yield_group_id = None
            else:
                suggested_yield_group_id = choose_wall_wall_yielder(
                    g_a, g_b, eff_t_a, eff_t_b, topo_info, faces
                )
            critical = is_critical_joint(topology, eff_t_a, eff_t_b)

            new_ww_joint = WallWallJoint(
                joint_index=ji,
                group_a=g_a.id,
                group_b=g_b.id,
                suggested_yield_group_id=suggested_yield_group_id,
                topology=topology,
                critical=critical,
            )
            wall_wall_joints.append(new_ww_joint)

            # El sistema SIEMPRE resuelve la junta: si el usuario no eligió, se aplica
            # la sugerencia. Dejarla sin resolver haría que las piezas se solapen al
            # cortar y la maqueta no se pueda armar si el usuario no repasa cada cruce.
            # Su elección manual, cuando existe, tiene prioridad.
            decision = wall_wall_decisions_map.get(ji)
            if decision is None:
                decision = suggested_yield_group_id
            if decision is None and (a_at_end or b_at_end):
                # Red de seguridad: ninguna regla decidió pese a haber un extremo
                # disponible. Se elige de forma determinística, priorizando siempre un
                # muro que PUEDA acortarse.
                if a_at_end and b_at_end:
                    decision = min(g_a.id, g_b.id)
                else:
                    decision = g_a.id if a_at_end else g_b.id
            if decision is not None:
                yield_group = group_by_id.get(decision)
                other_group_id = g_b.id if decision == g_a.id else g_a.id
                other_group = group_by_id.get(other_group_id)

                if yield_group and other_group:
                    # El borde del muro que cede debe apoyar sobre la CARA de la placa
                    # vecina, o sea a MEDIA PLACA de su plano medio. Hay que quitar dos
                    # cosas, y están en unidades distintas:
                    #   1) el voladizo: el contorno del panel se mide hasta la piel
                    #      EXTERIOR del muro, así que en la esquina asoma más allá del
                    #      eje del vecino. Es una cota del EDIFICIO y escala con él.
                    #   2) la media placa: 1.5mm de MDF, iguales en toda escala.
                    # Antes se restaba un valor fijo de 3mm que ignoraba el voladizo, y
                    # el error cambiaba de signo con la escala (+0.5mm a 1:20, -0.5mm a
                    # 1:50 en una esquina en L de 20cm).
                    # En uniones oblicuas ambos términos se estiran por 1/sen(θ), porque
                    # la placa se interpone a lo largo de más recorrido del muro.
                    factor = oblique_trim_factor(joint.dihedral_angle)
                    voladizo = overshoot_past_plane_m(yield_group, other_group, faces) or 0.0
                    media_placa = PLATE_THICKNESS_M / 2.0

                    detalle = f"voladizo {voladizo * 1000:.0f}mm + media placa"
                    if factor >= 1.02:
                        detalle += f", a {joint.dihedral_angle:.0f}° (x{factor:.2f})"
                    # El eje sale de la orientación del contacto: "width" si se encuentran
                    # por un lateral, "height"/"height_top" si se encuentran a lo largo.
                    eje = eje_a if decision == g_a.id else eje_b
                    new_ww_joint.yield_group_id = decision
                    adjustments.append(
                        DimensionAdjustment(
                            group_id=decision,
                            delta=-voladizo * factor,
                            delta_plate=-media_placa * factor,
                            axis=eje or "width",
                            reason=f"Junta con {other_group.label} ({detalle})",
                            joint_index=ji,
                        )
                    )

    # Encuentros muro-losa: se calculan por geometría, no por juntas (ver las funciones).
    # La losa entra entre los muros que la atraviesan; el extremo del muro apoya sobre la
    # cara de la placa de losa. Los duplicados con lo que ya emitió el bucle de juntas los
    # absorbe el dedupe de altura de más abajo.
    adjustments.extend(floor_fit_adjustments(groups, faces))
    adjustments.extend(wall_end_fit_adjustments(groups, faces))

    # Deduplicar ajustes de altura: mantener el delta más negativo por grupo y eje.
    # Los ajustes de ancho (muro-muro) pasan directos.
    seen_height: Dict[int, DimensionAdjustment] = {}
    seen_height_top: Dict[int, DimensionAdjustment] = {}
    kept_width: List[DimensionAdjustment] = []

    for adj in adjustments:
        if adj.axis == "height":
            existing = seen_height.get(adj.group_id)
            if not existing or adj.delta < existing.delta:
                seen_height[adj.group_id] = adj
        elif adj.axis == "height_top":
            existing = seen_height_top.get(adj.group_id)
            if not existing or adj.delta < existing.delta:
                seen_height_top[adj.group_id] = adj
        else:
            kept_width.append(adj)

    final_adjustments = (
        list(seen_height.values()) + list(seen_height_top.values()) + kept_width
    )
    return AdjustmentsResult(
        adjustments=final_adjustments, wall_wall_joints=wall_wall_joints
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


FLOOR_CROSS_TOL_M = 0.05
# Hasta dónde puede estar el borde de la losa del plano medio del muro para considerar
# que se encuentran. Más lejos que esto es un hueco de DISEÑO (un vacío, un patio) y
# achicar la losa abriría una luz en vez de cerrarla.
FLOOR_REACH_TOL_M = 0.30


def floor_fit_adjustments(
    groups: List[GeometryGroup], faces: Optional[List[Face3D]]
) -> List[DimensionAdjustment]:
    """Recortes para que cada losa ENTRE entre los muros que la atraviesan.

    Va por geometría y NO por juntas detectadas, a propósito. `detect_joints` es
    topológico (exige arista compartida) y su complemento geométrico
    `detect_contact_joints` sólo mira pares muro-muro, así que una losa que no esté
    soldada a los muros en la malla no genera ninguna junta con ellos: el entrepiso se
    quedaba sin recorte EN SILENCIO y salía con el ancho que tiene entre las pieles del
    modelo. Al armar la maqueta no entraba. Acá se recorren los pares (losa, muro) —son
    pocos: 2 x 26 en un modelo real— y se mide directamente.

    Sólo se recorta contra muros que ATRAVIESAN el nivel de la losa. Si el muro nace o
    muere en ella, la losa se apoya y no hay nada entre lo que entrar: por eso la losa de
    planta baja conserva su tamaño y el entrepiso no.
    """
    out: List[DimensionAdjustment] = []
    for f, w, s in floor_wall_encounters(groups, faces):
        # La losa llega HASTA EL PLANO MEDIO del muro, no media placa antes: su borde se
        # mete en la ranura pasante que `resolve_plate_joints` abre en el muro a ese
        # nivel. Antes se la recortaba media placa para que topara contra la cara del
        # muro, y así entraba sin chocar pero NADA la sostenía: quedaba encajada a
        # fricción entre dos muros continuos, sin ranura y sin apoyo.
        out.append(
            DimensionAdjustment(
                group_id=f.id,
                delta=-s,
                delta_plate=0.0,
                axis="plane",
                reason=f"Entra en la ranura de {w.label or w.id} (borde {s * 1000:+.0f}mm)",
                joint_index=-1,
                against_group_id=w.id,
            )
        )
    return out


def floor_wall_encounters(
    groups: List[GeometryGroup], faces: Optional[List[Face3D]]
) -> List[Tuple[GeometryGroup, GeometryGroup, float]]:
    """`(losa, muro, s)` de cada encuentro donde un muro CONTINUO atraviesa el nivel de
    una losa. `s` es la distancia con signo del borde de la losa al plano medio del muro.

    FUENTE ÚNICA de ese encuentro: la consumen el recorte de la losa
    (`floor_fit_adjustments`) y la ranura del muro (`plate_intersect.resolve_plate_joints`).
    Tenerlo en dos lados fue exactamente el defecto que costó más caro en este pipeline —
    el recorte y la ranura decidiendo por separado y contradiciéndose en 7 de 15 pares.
    """
    out: List[Tuple[GeometryGroup, GeometryGroup, float]] = []
    if not faces:
        return out

    floors = [
        g for g in groups
        if g.category == "floor" and abs(normalize(g.representative_normal).y) > 0.5
    ]
    walls = [
        g for g in groups
        if g.category == "wall" and abs(normalize(g.representative_normal).y) <= 0.5
    ]
    if not floors or not walls:
        return out

    def caja(g: GeometryGroup):
        xs = []; ys = []; zs = []
        for fi in g.face_indices:
            if 0 <= fi < len(faces):
                for v in faces[fi].vertices:
                    xs.append(v.x); ys.append(v.y); zs.append(v.z)
        if not xs:
            return None
        return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))

    cajas = {g.id: caja(g) for g in floors + walls}

    for f in floors:
        cf = cajas.get(f.id)
        if cf is None:
            continue
        for w in walls:
            if not wall_crosses_level(w, f):
                continue
            cw = cajas.get(w.id)
            if cw is None:
                continue
            # Tienen que coincidir en planta: si no, es un muro de otra parte del edificio.
            tol = FLOOR_CROSS_TOL_M
            if (min(cf[1], cw[1]) - max(cf[0], cw[0]) < -tol
                    or min(cf[5], cw[5]) - max(cf[4], cw[4]) < -tol):
                continue

            n = normalize(w.representative_normal)
            d = _mid_plane_offset(w, faces, n)
            proj = [
                n.x * v.x + n.y * v.y + n.z * v.z
                for fi in f.face_indices
                if 0 <= fi < len(faces)
                for v in faces[fi].vertices
            ]
            if not proj:
                continue
            lo, hi = min(proj), max(proj)
            centro = (lo + hi) / 2.0
            # Distancia CON SIGNO del borde de la losa al plano medio del muro, hacia
            # afuera: positiva = la losa pasa del plano; negativa = se queda corta.
            s = (hi - d) if centro < d else (d - lo)
            if abs(s) > FLOOR_REACH_TOL_M:
                continue  # no se encuentran: es un vacío de diseño, no un encastre
            out.append((f, w, s))
    return out


def wall_end_fit_adjustments(
    groups: List[GeometryGroup], faces: Optional[List[Face3D]]
) -> List[DimensionAdjustment]:
    """Recortes para que el EXTREMO de cada muro apoye sobre la cara de la placa de losa.

    Es la contraparte de `floor_fit_adjustments`: allá la losa entra entre los muros,
    acá el muro topa contra la losa. Las dos ramas que ya existían para esto
    (`is_wall_on_top` y `is_roof_above_wall`) viven dentro del bucle de JUNTAS, así que
    sólo actúan si `detect_joints` encontró la unión — y ese detector es topológico, con
    un complemento geométrico que sólo mira pares muro-muro. Una losa que no comparte
    vértices con el muro no genera ninguna junta con él, y el encuentro se quedaba sin
    resolver en silencio. Medido: en dos modelos distintos, un muro que nace dentro del
    canto de la losa y otro cuya cima queda dentro de la placa del entrepiso.

    Va por geometría, como el encaje de las losas, y por los mismos motivos.
    """
    out: List[DimensionAdjustment] = []
    if not faces:
        return out

    floors = [
        g for g in groups
        if g.category == "floor" and abs(normalize(g.representative_normal).y) > 0.5
    ]
    walls = [
        g for g in groups
        if g.category == "wall" and abs(normalize(g.representative_normal).y) <= 0.5
    ]
    if not floors or not walls:
        return out

    def caja(g: GeometryGroup):
        xs = []; ys = []; zs = []
        for fi in g.face_indices:
            if 0 <= fi < len(faces):
                for v in faces[fi].vertices:
                    xs.append(v.x); ys.append(v.y); zs.append(v.z)
        return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)) if xs else None

    media_placa = PLATE_THICKNESS_M / 2.0
    cajas = {g.id: caja(g) for g in floors + walls}

    for w in walls:
        cw = cajas.get(w.id)
        if cw is None:
            continue
        for f in floors:
            cf = cajas.get(f.id)
            if cf is None:
                continue
            # Tienen que coincidir en planta.
            tol = FLOOR_CROSS_TOL_M
            if (min(cw[1], cf[1]) - max(cw[0], cf[0]) < -tol
                    or min(cw[5], cf[5]) - max(cw[4], cf[4]) < -tol):
                continue

            # Plano medio de la losa, en Y (las losas son horizontales, así que el eje es
            # el vertical y no hace falta arrastrar el signo de su normal).
            d_y = (cf[2] + cf[3]) / 2.0
            w_lo, w_hi = cw[2], cw[3]
            if w_hi - w_lo <= 1e-6:
                continue
            centro = (w_lo + w_hi) / 2.0

            if centro > d_y:
                # El muro está por ENCIMA: apoya sobre la losa, se recorta su BASE.
                s = d_y - w_lo
                eje = "height"
                extremo = w_lo
            else:
                # La losa está por encima: el muro topa con ella, se recorta su CIMA.
                s = w_hi - d_y
                eje = "height_top"
                extremo = w_hi
            if abs(extremo - d_y) > FLOOR_REACH_TOL_M:
                continue  # el extremo del muro no llega a la losa: no se encuentran

            out.append(
                DimensionAdjustment(
                    group_id=w.id,
                    delta=-s,
                    delta_plate=-media_placa,
                    axis=eje,
                    reason=(
                        f"Topa con {f.label or f.id} "
                        f"(extremo {s * 1000:+.0f}mm del plano medio + media placa)"
                    ),
                    joint_index=-1,
                    against_group_id=f.id,
                )
            )
    return out


def wall_crosses_level(wall: GeometryGroup, floor: GeometryGroup) -> bool:
    """El muro atraviesa el nivel de la losa: sigue de largo por ARRIBA y por ABAJO.

    Distingue el entrepiso —que tiene que entrar entre las placas de muro— del piso de
    planta baja, sobre el que los muros simplemente apoyan. Si el muro NACE en la losa
    (min_y ≈ el nivel de la losa) no la cruza: la sostiene.
    """
    if None in (wall.min_y, wall.max_y, floor.min_y, floor.max_y):
        return False
    return (
        wall.min_y < floor.min_y - FLOOR_CROSS_TOL_M
        and wall.max_y > floor.max_y + FLOOR_CROSS_TOL_M
    )


def overshoot_past_plane_m(
    group: GeometryGroup,
    other: GeometryGroup,
    faces: Optional[List[Face3D]],
) -> Optional[float]:
    """Distancia CON SIGNO del borde de `group` al plano medio de `other`, hacia afuera.

    En metros de EDIFICIO. Positiva = el material pasa del plano (hay que quitarlo);
    NEGATIVA = el borde se queda corto y el recorte total es menor que la media placa.
    Antes se clampeaba a 0, con lo cual una pieza que ya llegaba corta igual perdía la
    media placa entera y quedaba con una luz del tamaño de lo que le faltaba (1 cm en el
    entrepiso de un modelo real). El clamp del recorte total sigue existiendo, pero está
    donde corresponde: en `decompose_panels_from_groups`, que es quien conoce la escala y
    puede sumar las dos componentes antes de decidir si hay algo que recortar.

    El contorno del panel se mide hasta la piel EXTERIOR del muro (project_faces_to_2d
    proyecta ambas pieles y toma el bbox de la unión), así que en una esquina el muro
    llega más allá del eje del vecino: ese sobrante hay que quitarlo, además de la media
    placa.
    """
    if not faces:
        return None
    n = normalize(other.representative_normal)
    d_other = _mid_plane_offset(other, faces, n)
    proj = [
        n.x * v.x + n.y * v.y + n.z * v.z
        for fi in group.face_indices
        if 0 <= fi < len(faces)
        for v in faces[fi].vertices
    ]
    if not proj:
        return None
    centro = (min(proj) + max(proj)) / 2.0
    # El cuerpo de la pieza está de un lado del plano; el borde que mira al vecino es el
    # extremo del otro lado. Sin clamp: si se queda corto el valor sale negativo.
    if centro >= d_other:
        return d_other - min(proj)
    return max(proj) - d_other


def _mid_plane_offset(group: GeometryGroup, faces: List[Face3D], n) -> float:
    """Plano medio del grupo a lo largo de `n`. Delega en la fuente única de
    `plate_intersect` para que el tope y la ranura no puedan referirse a planos distintos
    (ver `plate_intersect.mid_plane_offset`)."""
    return mid_plane_offset(group, faces, n)


def vertical_end_axis(group: GeometryGroup, joint: Joint) -> Optional[str]:
    """Qué borde HORIZONTAL de la pieza toca la junta: `"height"` (la base),
    `"height_top"` (la cima), o `None` si el contacto cae en su franja media.

    La contraparte vertical de `joint_position_frac`. Un muro sólo puede ceder ahí donde
    tiene un borde: si el contacto horizontal le cae por el medio, acortarlo no toca el
    choque y encima lo deja corto donde sí llegaba.
    """
    if group.min_y is None or group.max_y is None:
        return None
    span = group.max_y - group.min_y
    if span <= 1e-6:
        return None
    f = (joint.edge_mid.y - group.min_y) / span
    if f <= END_ZONE_FRAC:
        return "height"
    if f >= 1.0 - END_ZONE_FRAC:
        return "height_top"
    return None


def joint_position_frac(
    group: GeometryGroup, joint: Joint, faces: Optional[List[Face3D]]
) -> Optional[float]:
    """Posición del cruce a lo largo del muro, de 0 (un extremo) a 1 (el otro).

    Sirve para distinguir un encuentro en el EXTREMO —que se resuelve acortando la
    pieza— de uno en el MEDIO, donde acortar no toca el choque y además deja la pieza
    corta justo donde tenía que llegar.
    """
    if not faces:
        return None
    n = group.representative_normal
    plan = math.hypot(n.x, n.z)
    if plan < 1e-6:
        return None
    ux, uz = -n.z / plan, n.x / plan  # dirección horizontal a lo largo del muro
    ts = [
        v.x * ux + v.z * uz
        for fi in group.face_indices
        if 0 <= fi < len(faces)
        for v in faces[fi].vertices
    ]
    if not ts:
        return None
    t0, t1 = min(ts), max(ts)
    if t1 - t0 < 1e-6:
        return None
    t = joint.edge_mid.x * ux + joint.edge_mid.z * uz
    return (t - t0) / (t1 - t0)


# Un cruce se considera "en el extremo" si cae dentro de este margen de una punta.
END_ZONE_FRAC = 0.15


def choose_wall_wall_yielder(
    g_a: GeometryGroup,
    g_b: GeometryGroup,
    t_a: float,
    t_b: float,
    topo_info: Optional[JointTopologyInfo] = None,
    faces: Optional[List[Face3D]] = None,
) -> Optional[int]:
    """Decide qué muro cede en una unión muro-muro utilizando reglas físicas determinísticas."""

    # Sin grosores -> nada que restar
    if t_a <= 0.001 and t_b <= 0.001:
        return None
    # Solo un lado medido -> el otro cede
    if t_a > 0.001 and t_b <= 0.001:
        return g_b.id
    if t_b > 0.001 and t_a <= 0.001:
        return g_a.id

    # Ambos medidos. Claramente diferentes (ratio < 0.9) -> el más delgado cede
    lo = min(t_a, t_b)
    hi = max(t_a, t_b)
    if (lo / hi) < 0.9:
        return g_a.id if t_a <= t_b else g_b.id

    # Longitud del muro: el claramente más corto cede
    if faces is not None:
        len_a = wall_run_length(g_a, faces)
        len_b = wall_run_length(g_b, faces)
        lo_len = min(len_a, len_b)
        hi_len = max(len_a, len_b)
        if hi_len > 0.5 and (lo_len / hi_len) < 0.6 and (hi_len - lo_len) > 2.0:
            return g_a.id if len_a <= len_b else g_b.id

    # Grosores casi iguales: desempatar geométricamente
    if topo_info and topo_info.topology == "T":
        # El "tallo" (el muro cuya arista asienta en su propio extremo) cede.
        if topo_info.a_at_end and not topo_info.b_at_end:
            return g_a.id
        if topo_info.b_at_end and not topo_info.a_at_end:
            return g_b.id

    # L / X / empate desconocido -> Muro Norte-Sur gana, Este-Oeste cede.
    a_is_east_west = abs(g_a.representative_normal.z) >= abs(
        g_a.representative_normal.x
    )
    b_is_east_west = abs(g_b.representative_normal.z) >= abs(
        g_b.representative_normal.x
    )

    if a_is_east_west and not b_is_east_west:
        return g_a.id
    if b_is_east_west and not a_is_east_west:
        return g_b.id

    # Misma orientación -> fallback estable por orden de ID
    return g_a.id if g_a.id <= g_b.id else g_b.id


def is_wall_on_top(wall: GeometryGroup, floor: GeometryGroup, joint: Joint) -> bool:
    """Verifica si un muro se asienta sobre una losa de piso."""
    if wall.min_y is None or floor.max_y is None:
        return False

    tol = max(floor.thickness or 0.0, 0.05)
    if wall.min_y < floor.max_y - tol:
        return False

    if joint.horizontal_frac < 0.5:
        return False

    return True


def is_roof_above_wall(roof: GeometryGroup, wall: GeometryGroup, joint: Joint) -> bool:
    """Verifica si una losa/techo se asienta sobre la parte SUPERIOR del muro."""
    if roof.min_y is None or wall.max_y is None:
        return False

    # Tolerancia: al menos 15 cm para cubrir modelos con pequeñas discrepancias
    y_extent = (
        (roof.max_y - roof.min_y)
        if roof.max_y is not None and roof.min_y is not None
        else 0.0
    )
    tol = max(roof.thickness or 0.0, y_extent, 0.15)

    # El borde inferior del techo debe estar cerca de la cima del muro
    if roof.min_y < wall.max_y - tol:
        return False

    # La arista compartida debe ser predominantemente horizontal
    if joint.horizontal_frac < 0.5:
        return False

    return True
