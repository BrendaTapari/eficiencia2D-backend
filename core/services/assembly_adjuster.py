import math
from typing import List, Dict, Optional, Literal, Tuple
from dataclasses import dataclass

# Importamos nuestros tipos y servicios
from core.services.types import Face3D
from core.services.joint_detector import Joint
from core.group_classifier import GeometryGroup
from core.services.joint_topology_classifier import (
    JointTopology,
    JointTopologyInfo,
    classify_joint_topology,
    is_critical_joint,
    wall_run_length,
)

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
    # Qué dimensión recortar: "height" recorta la base (muro-piso), "width" recorta un lado (muro-muro).
    axis: Literal["height", "width"]
    reason: str
    joint_index: int


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
        # Solo manejar uniones cercanas a 90°
        if joint.dihedral_angle < 75 or joint.dihedral_angle > 95:
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
            # Unión Muro–Piso
            floor = g_a if a_is_floor else g_b
            wall = g_b if a_is_floor else g_a

            if not floor.thickness or floor.thickness < 0.001:
                continue

            # Filtro: el muro debe asentarse SOBRE el piso (wall.min_y ≈ floor.max_y)
            # Y la arista compartida debe ser predominantemente horizontal.
            wall_on_top = is_wall_on_top(wall, floor, joint)
            if not wall_on_top:
                continue

            delta = -floor.thickness
            label = floor.label if floor.label else f"Grupo {floor.id}"

            adjustments.append(
                DimensionAdjustment(
                    group_id=wall.id,
                    delta=delta,
                    axis="height",
                    reason=f"Junta con {label} (grosor {floor.thickness * 100:.1f}cm)",
                    joint_index=ji,
                )
            )

        elif not a_is_floor and not b_is_floor:
            # Unión Muro–Muro
            t_a = g_a.thickness or 0.0
            t_b = g_b.thickness or 0.0

            topo_info = (
                classify_joint_topology(joint, g_a, g_b, faces) if faces else None
            )
            topology = topo_info.topology if topo_info else "unknown"

            suggested_yield_group_id = choose_wall_wall_yielder(
                g_a, g_b, t_a, t_b, topo_info, faces
            )
            critical = is_critical_joint(topology, t_a, t_b)

            new_ww_joint = WallWallJoint(
                joint_index=ji,
                group_a=g_a.id,
                group_b=g_b.id,
                suggested_yield_group_id=suggested_yield_group_id,
                topology=topology,
                critical=critical,
            )
            wall_wall_joints.append(new_ww_joint)

            # Aplicar la decisión del usuario si está presente
            decision = wall_wall_decisions_map.get(ji)
            if decision is not None:
                yield_group = group_by_id.get(decision)
                other_group_id = g_b.id if decision == g_a.id else g_a.id
                other_group = group_by_id.get(other_group_id)

                if yield_group and other_group:
                    # Preferir el grosor del muro que no cede; sino el del que cede
                    trim_thickness = 0.0
                    if other_group.thickness and other_group.thickness > 0.001:
                        trim_thickness = other_group.thickness
                    elif yield_group.thickness and yield_group.thickness > 0.001:
                        trim_thickness = yield_group.thickness

                    if trim_thickness > 0.001:
                        new_ww_joint.yield_group_id = decision
                        adjustments.append(
                            DimensionAdjustment(
                                group_id=decision,
                                delta=-trim_thickness,
                                axis="width",
                                reason=f"Junta con {other_group.label} (grosor {trim_thickness * 100:.1f}cm)",
                                joint_index=ji,
                            )
                        )

    # Deduplicar ajustes de altura (muro-piso): mantener el delta más grande (el más negativo) por grupo.
    # Los ajustes de ancho (muro-muro) pasan directos.
    seen_height: Dict[int, DimensionAdjustment] = {}
    kept_width: List[DimensionAdjustment] = []

    for adj in adjustments:
        if adj.axis == "height":
            existing = seen_height.get(adj.group_id)
            if not existing or adj.delta < existing.delta:
                seen_height[adj.group_id] = adj
        else:
            kept_width.append(adj)

    final_adjustments = list(seen_height.values()) + kept_width
    return AdjustmentsResult(
        adjustments=final_adjustments, wall_wall_joints=wall_wall_joints
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
