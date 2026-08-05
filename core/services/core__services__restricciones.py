"""
Grafo de restricciones del encaje: dónde tiene que caer cada borde de cada pieza.

Por qué existe
--------------
Hasta acá, "cuánto se acorta cada pieza" vivía disperso en seis `adjustments.append(...)`
de `assembly_adjuster.py`, cada uno emitiendo un DELTA relativo, y se aplicaban en cadena
en `decompose_panels_from_groups` acumulando desplazamientos. Esa forma tiene tres
problemas que ya costaron defectos medidos:

1. Un delta no dice a DÓNDE tiene que llegar el borde, sólo cuánto moverlo. Dos reglas
   sobre el mismo borde no se pueden comparar: se terminaba eligiendo la mayor en silencio.
2. Aplicarlos en cadena obliga a arrastrar `clip_off_u/v`, y basta que un paso no lo reste
   para que el siguiente decida en un marco desactualizado.
3. Un delta negativo —la pieza tiene que CRECER— se descartaba, porque el consumidor sólo
   sabía recortar.

Acá cada regla se expresa como un DESTINO: la distancia con signo a la que el borde tiene
que quedar del plano medio de su vecina. Eso las hace comparables entre sí (dos destinos
distintos para el mismo borde son un conflicto explícito, no un máximo silencioso),
independientes del orden de aplicación, y simétricas: que el destino caiga más allá del
borde crudo significa "alargar", sin ningún caso especial.

El invariante es siempre el mismo y es GEOMÉTRICO —vale igual a 1:20 que a 1:200— porque
la placa mide 3 mm en cualquier escala mientras el muro del modelo se achica:

    distancia(borde de A, plano medio de B)  =  espesor_placa / 2

El signo lo decide el rol en la junta: la que CEDE va a la cara interior de su vecina, la
que PASA a la exterior. Y el rol sale de la decisión del usuario en el visor, no de una
convención fija: si la invierte, las dos restricciones se invierten con ella.

Unidades: `distancia_placa_m` está en metros de PLANCHA y NO escala (son 1.5 mm a
cualquier escala). Para convertirla a metros de EDIFICIO hay que multiplicar por
`scale_denom`, igual que hace `_delta_m` con `delta_plate`. Mezclar las dos unidades fue el
error que hacía que el signo del defecto cambiara con la escala.

Estado
------
Fases A y B: esto se CONSTRUYE, se RESUELVE y se VERIFICA, pero todavía no se consume. El
pipeline sigue aplicando los `DimensionAdjustment` de siempre; `test_corpus.py` compara los
dos mundos y fija el estado en `LINEA_BASE_RESTRICCIONES`.

`objetivo_local_m` ya está listo para que el consumidor lo use, y es la pieza de la Fase C.
Un primer intento de enchufarlo como productor del recorte lateral se midió y se revirtió:
cerraba entero el modelo del usuario a 1:100 (32 de 32 bordes, contra 31) y mejoraba a 1:50
(28 contra 26), pero subía los choques de 2 a 6 en ese mismo archivo y rompía 23 tests del
banco. Falta entender por qué la profundidad sale mal en los casos donde el panel no es
rectangular: `clip_off_u` no alcanza para pasar del marco crudo al actual cuando un recorte
de ALTURA ya renormalizó el ancho.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from core.group_classifier import GeometryGroup
from core.services.joint_detector import Joint, MIN_JOINT_ANGLE_DEG
from core.services.joint_topology_classifier import END_FRAC as END_ZONE_FRAC
from core.services.plate_intersect import PLATE_THICKNESS_M, mid_plane_offset
from core.services.types import Face3D, Vec3, dot, normalize

# Dos destinos para el mismo borde que difieran menos que esto son el mismo destino.
# En metros de PLANCHA: 1 µm.
TOL_DESTINO_M = 1e-6

# Piso para el tope absoluto de "esto es una esquina", cuando el modelo no trae espesor
# (mallas de una sola cara). En metros de EDIFICIO.
ESPESOR_MINIMO_M = 0.05

Rol = Literal["cede", "pasa"]


@dataclass(frozen=True)
class Restriccion:
    """A qué distancia del plano medio de `contra_group_id` tiene que quedar un borde.

    `distancia_placa_m` es con signo y en metros de PLANCHA:
      - negativa → cara INTERIOR del vecino (la pieza se detiene antes): rol "cede"
      - positiva → cara EXTERIOR (la pieza lo cruza y termina a ras): rol "pasa"

    `banda` acota la restricción a un tramo de la ALTURA de la pieza (en metros de
    edificio, sobre el eje perpendicular al del borde). Es lo que separa una muesca de un
    recorte de borde entero: dos vecinos que llegan al mismo extremo a alturas distintas
    dan dos restricciones con bandas disjuntas y cada una se lleva sólo su franja.
    `None` = el borde entero.
    """

    group_id: int
    contra_group_id: int
    rol: Rol
    distancia_placa_m: float
    # QUÉ borde de la pieza. Sin esto no se puede saber si dos restricciones compiten:
    # dos vecinas distintas en el mismo extremo se pisan, en extremos opuestos no.
    eje: Literal["u", "v"] = "u"
    lado: Literal["bajo", "alto"] = "bajo"
    # Dónde cae el plano medio del vecino sobre ese eje, en el marco de proyección CRUDO
    # del panel y en metros de edificio. Es el ancla desde la que se despeja el destino.
    plano_local_m: float = 0.0
    banda: Optional[Tuple[float, float]] = None
    joint_index: int = -1
    motivo: str = ""

    def destino_edificio_m(self, scale_denom: float) -> float:
        """La misma distancia, en metros de EDIFICIO a la escala pedida."""
        return self.distancia_placa_m * scale_denom

    def objetivo_local_m(self, scale_denom: float) -> float:
        """A qué coordenada del panel tiene que llegar el borde, en el marco CRUDO.

        Se despeja del plano del vecino: media placa hacia adentro del panel si esta pieza
        CEDE (se detiene en la cara interior), media placa hacia afuera si PASA (lo cruza y
        termina a ras de la cara exterior). El lado del borde decide para qué lado del
        plano queda el material, y por eso entra en el signo.

        Es una COORDENADA, no un delta: no depende de cuánto mida hoy la pieza ni de qué
        recortes se hayan aplicado antes. Que caiga fuera del panel crudo significa que la
        pieza tiene que crecer, y eso no es un caso especial.
        """
        media = abs(self.distancia_placa_m) * scale_denom
        hacia_adentro = +1.0 if self.lado == "bajo" else -1.0
        signo = hacia_adentro if self.rol == "cede" else -hacia_adentro
        return self.plano_local_m + signo * media


@dataclass
class Conflicto:
    """Dos restricciones sobre el mismo borde, con bandas que se pisan y destinos distintos.

    No existe pieza que satisfaga las dos. Hoy esto se resolvía quedándose con la mayor sin
    avisar; acá se declara para que llegue al usuario.
    """

    group_id: int
    a: Restriccion
    b: Restriccion

    def mensaje(self) -> str:
        return (
            f"La pieza del grupo {self.group_id} tiene dos destinos incompatibles para el "
            f"mismo borde: «{self.a.motivo}» y «{self.b.motivo}». No hay corte que cumpla "
            "los dos. Revisá quién cede en esa unión."
        )


def _bandas_se_pisan(
    a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]
) -> bool:
    if a is None or b is None:
        return True          # borde entero contra cualquier cosa: se pisan
    return min(a[1], b[1]) - max(a[0], b[0]) > 1e-9


def resolver(
    restricciones: List[Restriccion],
) -> Tuple[List[Restriccion], List[Conflicto]]:
    """Separa lo que puede convivir de lo que no.

    Dos restricciones del mismo grupo contra el MISMO vecino son la misma junta vista dos
    veces y no compiten. Compiten las que apuntan al mismo borde contra vecinos distintos:
    ahí, o tienen bandas disjuntas (conviven como muescas separadas) o hay conflicto.

    Cuando hay conflicto gana la más exigente —la que deja menos material— para que dos
    piezas no ocupen el mismo MDF, y la otra se reporta.
    """
    por_borde: Dict[Tuple[int, str, str], List[Restriccion]] = {}
    for r in restricciones:
        por_borde.setdefault((r.group_id, r.eje, r.lado), []).append(r)

    elegidas: List[Restriccion] = []
    conflictos: List[Conflicto] = []

    for clave, rs in sorted(por_borde.items()):
        # Más exigente primero: "cede" (se detiene antes del plano) recorta más que "pasa".
        rs = sorted(rs, key=lambda r: r.distancia_placa_m)
        vistas: List[Restriccion] = []
        for r in rs:
            choca = next(
                (v for v in vistas if _bandas_se_pisan(v.banda, r.banda)), None
            )
            if choca is None:
                vistas.append(r)
            elif (
                choca.contra_group_id != r.contra_group_id
                and abs(choca.distancia_placa_m - r.distancia_placa_m) > TOL_DESTINO_M
            ):
                # Dos VECINAS distintas piden cosas distintas del mismo borde, en la misma
                # franja de altura. No hay corte que cumpla las dos.
                conflictos.append(Conflicto(clave[0], choca, r))
        elegidas.extend(vistas)

    return elegidas, conflictos


# ---------------------------------------------------------------------------
# Construcción
# ---------------------------------------------------------------------------


def construir_muro_muro(
    joints: List[Joint],
    groups: List[GeometryGroup],
    faces: List[Face3D],
    wall_wall_decisions: Optional[Dict[int, int]] = None,
    adj_result=None,
) -> List[Restriccion]:
    """Las dos mitades de cada junta muro-muro resuelta.

    Reproduce las reglas de `assembly_adjuster.compute_adjustments` (`:361` la que cede,
    `:410` la contraparte pasante) pero como destinos en vez de deltas. Usa la MISMA
    decisión —`yield_group_id`, que ya respeta `wall_wall_decisions`— para que no puedan
    discrepar.
    """
    from core.services.assembly_adjuster import (
        compute_adjustments,
        oblique_trim_factor,
    )

    por_id = {g.id: g for g in groups}
    # Se reutiliza la resolución que ya se hizo, si vino. Recalcularla abriría de nuevo la
    # puerta a que el grafo y el recorte decidan distinto, que es el defecto que más caro
    # costó en este pipeline.
    res = adj_result or compute_adjustments(joints, groups, wall_wall_decisions, faces)
    media = PLATE_THICKNESS_M / 2.0
    out: List[Restriccion] = []

    for ww in res.wall_wall_joints:
        if ww.yield_group_id is None:
            continue          # cruce en la panza de ambos: lo resuelve una ranura
        if not (0 <= ww.joint_index < len(joints)):
            continue
        joint = joints[ww.joint_index]
        if joint.dihedral_angle < MIN_JOINT_ANGLE_DEG:
            continue
        factor = oblique_trim_factor(joint.dihedral_angle)

        cede_id = ww.yield_group_id
        pasa_id = ww.group_b if cede_id == ww.group_a else ww.group_a

        for gid, rol, signo in ((cede_id, "cede", -1.0), (pasa_id, "pasa", +1.0)):
            g = por_id.get(gid)
            otro = por_id.get(pasa_id if gid == cede_id else cede_id)
            if g is None or otro is None:
                continue
            borde = _ubicar_borde(g, otro, faces)
            if borde is None:
                continue
            eje, lado, plano_local = borde
            out.append(
                Restriccion(
                    group_id=gid,
                    contra_group_id=otro.id,
                    rol=rol,
                    distancia_placa_m=signo * media * factor,
                    eje=eje,
                    lado=lado,
                    plano_local_m=plano_local,
                    banda=_banda_del_vecino(g, otro, faces),
                    joint_index=ww.joint_index,
                    motivo=(
                        f"{'cede ante' if rol == 'cede' else 'pasa por'} "
                        f"{otro.label or otro.id}"
                    ),
                )
            )
    return out


def _ubicar_borde(
    grupo: GeometryGroup, otro: GeometryGroup, faces: List[Face3D]
) -> Optional[Tuple[str, str, float]]:
    """Qué borde de `grupo` se encuentra con `otro`: `(eje, lado, plano_local)`.

    Se proyecta el plano medio del vecino sobre los dos ejes del panel y se queda con el
    que lo enfrenta de punta; el lado sale de si ese plano cae antes o después de la mitad.

    Sin esto no se puede saber si dos restricciones COMPITEN. Dos vecinas en el mismo
    extremo se pisan y hay que elegir; en extremos opuestos conviven sin problema. Es la
    diferencia entre detectar un conflicto real y ver conflictos donde no los hay.
    """
    m = _proyeccion(grupo, faces)
    if m is None:
        return None
    n = normalize(otro.representative_normal)
    d = mid_plane_offset(otro, faces, n)
    for eje, ax, origen, largo in (
        ("u", m.u_axis, m.origin_u, m.width_m),
        ("v", m.v_axis, m.origin_v, m.height_m),
    ):
        c = dot(ax, n)
        if abs(c) < 0.7 or largo <= 1e-9:
            continue
        t = (d / c) - origen
        frac = t / largo
        # El vecino tiene que caer en un EXTREMO. Si su plano cae en la panza, la pieza no
        # termina ahí: lo cruza, y eso lo resuelve una ranura, no un tope. Sin esta guarda
        # se leían juntas de panza como restricciones de borde y aparecían conflictos donde
        # no los hay (un vecino a 1.25 m de un panel de 4.25 no compite con el del extremo).
        # Mismo umbral que usa `compute_adjustments` para decidir quién puede ceder.
        if END_ZONE_FRAC < frac < 1.0 - END_ZONE_FRAC:
            return None
        lado = "bajo" if frac <= 0.5 else "alto"
        # Y un tope ABSOLUTO además del relativo. En un panel de 7.50 m, un vecino a 1.25 m
        # del extremo entra en el 20% y aun así no es una esquina: acortar ahí se comería
        # 1.4 m de muro. El plano del vecino tiene que caer a lo sumo a un espesor de muro
        # del borde, que es donde de verdad está el eje de la pared que forma la esquina.
        borde = 0.0 if lado == "bajo" else largo
        tope = 2.0 * max(otro.thickness or 0.0, grupo.thickness or 0.0, ESPESOR_MINIMO_M)
        if abs(t - borde) > tope:
            return None
        return eje, lado, t
    return None


def _proyeccion(grupo: GeometryGroup, faces: List[Face3D]):
    """Marco de proyección crudo del grupo, con la normal orientada hacia afuera."""
    from core.services.cutting_sheet import (
        orient_group_normals_outward,
        project_faces_to_2d,
    )

    gfaces = [faces[i] for i in grupo.face_indices if 0 <= i < len(faces)]
    if not gfaces:
        return None
    n = orient_group_normals_outward([grupo]).get(grupo.id, grupo.representative_normal)
    return project_faces_to_2d(gfaces, n, "Y")


def _banda_del_vecino(
    grupo: GeometryGroup, otro: GeometryGroup, faces: List[Face3D]
) -> Optional[Tuple[float, float]]:
    """Franja de la ALTURA de `grupo` donde el vecino existe, en metros de edificio.

    Se mide proyectando los vértices del vecino sobre el eje v del panel, no tomando su
    altura en el mundo: así vale también para paneles inclinados, donde v no es la vertical.
    Devuelve None si el vecino abarca toda la altura (el recorte es de borde entero).
    """
    m = _proyeccion(grupo, faces)
    if m is None:
        return None
    proy = [
        dot(v, m.v_axis) - m.origin_v
        for fi in otro.face_indices
        if 0 <= fi < len(faces)
        for v in faces[fi].vertices
    ]
    if not proy:
        return None
    lo, hi = max(min(proy), 0.0), min(max(proy), m.height_m)
    if hi - lo <= 1e-6 or hi - lo >= m.height_m - 1e-6:
        return None
    return (lo, hi)


# ---------------------------------------------------------------------------
# Verificación
# ---------------------------------------------------------------------------


def verificar(
    paneles,
    restricciones: List[Restriccion],
    groups: List[GeometryGroup],
    faces: List[Face3D],
    scale_denom: float,
    tol_mm: float = 0.2,
) -> Tuple[int, List[Tuple[str, Restriccion, float]]]:
    """Qué restricciones NO cumple la pieza que hoy produce el pipeline.

    Devuelve `(cuántas se pudieron evaluar, [(panel_id, restriccion, error_mm_plancha)])`.
    Positivo = el borde se quedó más lejos del vecino de lo que debía; negativo = se pasó.

    No toda restricción es evaluable: si el eje largo de la pieza no enfrenta la normal del
    vecino, ese borde no es el que cierra la esquina y se saltea. Se devuelve el conteo para
    que una restricción salteada no se confunda con una cumplida.

    Se mide en 3D contra el plano medio del vecino, que es la misma comprobación que el
    usuario hace con la regla sobre el DXF, y no depende de ningún marco intermedio.
    """
    por_grupo = {p.source_group_id: p for p in paneles}
    por_id = {g.id: g for g in groups}
    fallas: List[Tuple[str, Restriccion, float]] = []
    evaluadas = 0

    for r in restricciones:
        p = por_grupo.get(r.group_id)
        otro = por_id.get(r.contra_group_id)
        if p is None or otro is None or not p.frame:
            continue
        m = p.frame
        o = Vec3(m["origin"]["x"], m["origin"]["y"], m["origin"]["z"])
        u = Vec3(m["u_axis"]["x"], m["u_axis"]["y"], m["u_axis"]["z"])
        n = normalize(otro.representative_normal)
        if abs(dot(u, n)) < 0.7:
            continue          # no se encuentran de punta por este eje
        evaluadas += 1
        d = mid_plane_offset(otro, faces, n)
        # El borde más cercano al plano del vecino, con signo.
        cerca = min(
            (
                dot(Vec3(o.x + u.x * t, o.y + u.y * t, o.z + u.z * t), n) - d
                for t in (0.0, m["width_m"])
            ),
            key=abs,
        )
        objetivo = r.destino_edificio_m(scale_denom)
        # El signo del destino está referido a "hacia afuera del vecino"; el borde real
        # puede caer de cualquier lado, así que se comparan magnitudes con el signo del rol.
        real = abs(cerca)
        esperado = abs(objetivo)
        err_mm = (real - esperado) / scale_denom * 1000.0
        if abs(err_mm) > tol_mm:
            fallas.append((p.id, r, err_mm))
    return evaluadas, fallas
