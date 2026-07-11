import time
import logging
from typing import List, Dict, Optional, Literal, Tuple
from dataclasses import dataclass

from core.profiler import PipelineTimer
from core.services.types import Face3D, PipelineOptions, OutputFile, Vec3, IndexedFace3D
from core.services.facade_extractor import detect_up_axis
from core.group_classifier import (
    GeometryGroup,
    classify_into_groups,
    peel_buried_walls,
    polish_groups,
    split_groups_by_tags,
    split_thin_groups_by_pane,
    DEFAULT_MIN_REAL_AREA,
)
from core.services.mesh_splitter import split_wall_groups_at_floors
from core.services.joint_detector import Joint, detect_joints
from core.services.assembly_adjuster import (
    DimensionAdjustment,
    WallWallJoint,
    compute_adjustments,
)
# La generación (Fase 2) vive en core.review_generate y se importa lazy dentro de
# generate_pipeline; pipeline.py ya no usa cutting_sheet/pdf_writer/sheet_nester/extractors.

logger = logging.getLogger("eficiencia2d.pipeline")


@dataclass
class Phase1Result:
    faces: List[Face3D]
    raw_faces: List[Face3D]  # caras pre-split: el front las usa para reclassifyWithAxis
    applied_axis: Literal["Y", "Z"]
    groups: List[GeometryGroup]
    joints: List[Joint]
    adjustments: List[DimensionAdjustment]
    wall_wall_joints: List[WallWallJoint]
    stem: str
    warnings: List[str]
    pre_split_face_count: int
    suggested_merges: List[List[int]]
    timing: Optional[Dict] = None  # Reporte de timing para debug


def _rotate_z_to_y(face: Face3D) -> Face3D:
    def rot(v: Vec3) -> Vec3:
        return Vec3(v.x, v.z, -v.y)

    new_verts = [rot(v) for v in face.vertices]
    new_normal = rot(face.normal)
    new_inner = [[rot(v) for v in loop] for loop in face.inner_loops]
    if isinstance(face, IndexedFace3D):
        return IndexedFace3D(
            vertices=new_verts,
            normal=new_normal,
            inner_loops=new_inner,
            panel_id=face.panel_id,
            material=getattr(face, "material", None),
            source_object=getattr(face, "source_object", None),
            vertex_indices=list(face.vertex_indices),
        )
    return Face3D(
        vertices=new_verts,
        normal=new_normal,
        inner_loops=new_inner,
        panel_id=face.panel_id,
        material=getattr(face, "material", None),
        source_object=getattr(face, "source_object", None),
    )


def parse_pipeline(
    file_name: str,
    faces: List[Face3D],
    warnings: List[str],
    force_axis: Optional[Literal["Y", "Z"]] = None,
    min_real_area: float = DEFAULT_MIN_REAL_AREA,
) -> Phase1Result:
    timer = PipelineTimer(f"parse_pipeline({file_name})")
    stem = file_name.rsplit(".", 1)[0]

    raw_faces = list(faces)
    # `force_axis` permite al usuario fijar el eje vertical (endpoint /recompute);
    # si no se fuerza, se autodetecta como siempre.
    detected_up = force_axis if force_axis in ("Y", "Z") else detect_up_axis(raw_faces)
    working_faces = raw_faces
    if detected_up == "Z":
        working_faces = [_rotate_z_to_y(f) for f in raw_faces]

    # Soldado canónico de vértices (snap-rounding único): unifica los índices para que
    # la conectividad use igualdad entera exacta y suelda vértices coincidentes que el
    # export pudo duplicar (costuras que sobre-cortan la pared). Ver core.services.topology.
    from core.services.topology import weld_vertices
    weld_vertices(working_faces)

    logger.info(
        f"parse_pipeline: {len(working_faces):,} caras de entrada (eje {detected_up})"
    )

    # 1. Clasificación inicial
    with timer.step("classify_into_groups", face_count=len(working_faces)):
        groups = classify_into_groups(
            working_faces, min_real_area=min_real_area, out_warnings=warnings
        )

    # 1.5 Separar por etiquetas del archivo (objeto/material) si existen: el vidrio,
    # el marco y la puerta no deben quedar fundidos con el muro. No-op si el formato
    # no trae etiquetas (cae a la separación por grosor).
    with timer.step("split_groups_by_tags", group_count=len(groups)):
        groups = split_groups_by_tags(groups, working_faces)

    logger.info(f"  → {len(groups)} grupos: " + 
                ", ".join(f"{sum(1 for g in groups if g.category==c)} {c}" 
                          for c in ["wall","floor","discard"]))

    # 2. Peel buried walls
    with timer.step("peel_buried_walls", group_count=len(groups)):
        groups = peel_buried_walls(working_faces, groups)

    logger.info(f"  → tras peel: {len(groups)} grupos")

    # 3. Split por pisos
    pre_split_face_count = len(working_faces)
    with timer.step("split_wall_groups_at_floors", face_count=pre_split_face_count):
        split_faces, split_groups = split_wall_groups_at_floors(
            working_faces, groups, {}, min_real_area=min_real_area
        )

    logger.info(
        f"  → split: {pre_split_face_count:,} → {len(split_faces):,} caras, "
        f"{len(split_groups)} grupos"
    )

    # 4. Polish groups
    with timer.step("polish_groups", group_count=len(split_groups)):
        polish_groups(split_groups, min_real_area)

    # 4.5 Separar vidrios/elementos finos: cada ventana, cada vidrio, su propio grupo
    # (sólo grupos finos por grosor; los muros gruesos quedan intactos).
    with timer.step("split_thin_groups_by_pane", group_count=len(split_groups)):
        split_groups = split_thin_groups_by_pane(split_groups, split_faces)

    # 5. Detección de joints
    with timer.step("detect_joints", group_count=len(split_groups), face_count=len(split_faces)):
        joints = detect_joints(split_faces, split_groups)

    logger.info(f"  → {len(joints)} joints detectados")

    # 6. Ajustes de dimensiones
    with timer.step("compute_adjustments", joint_count=len(joints)):
        adj_result = compute_adjustments(joints, split_groups, faces=split_faces)

    logger.info(
        f"  → {len(adj_result.adjustments)} ajustes, "
        f"{len(adj_result.wall_wall_joints)} wall-wall joints"
    )

    timing_report = timer.report()

    return Phase1Result(
        faces=split_faces,
        raw_faces=raw_faces,
        applied_axis=detected_up,
        groups=split_groups,
        joints=joints,
        adjustments=adj_result.adjustments,
        wall_wall_joints=adj_result.wall_wall_joints,
        stem=stem,
        warnings=warnings,
        pre_split_face_count=pre_split_face_count,
        suggested_merges=[],
        timing=timing_report,
    )


def generate_pipeline(
    phase1: Phase1Result,
    opts: PipelineOptions,
    overrides: Optional[Dict[int, str]] = None,
    wall_wall_decisions: Optional[Dict[int, int]] = None,
    merges: Optional[List[List[int]]] = None,
    marks: Optional[List[int]] = None,
    user_cuts: Optional[List[dict]] = None,
    flex: Optional[List[dict]] = None,
    mark_lines: Optional[List[dict]] = None,
    ribs: Optional[List[dict]] = None,
) -> List[OutputFile]:
    from core.review_generate import generate_from_review

    return generate_from_review(
        phase1,
        opts,
        overrides=overrides,
        wall_wall_decisions=wall_wall_decisions,
        merges=merges,
        marks=marks,
        user_cuts=user_cuts,
        flex=flex,
        mark_lines=mark_lines,
        ribs=ribs,
    )


def apply_review_edits(
    base: Phase1Result,
    axis: Optional[str] = None,
    min_real_area: Optional[float] = None,
    splits: Optional[List[dict]] = None,
) -> Phase1Result:
    """
    Re-deriva la topología de forma determinística a partir de las caras originales
    (`base.raw_faces`) aplicando, en orden: eje vertical → área mínima → splits.

    No aplica merges (eso lo hace review_generate.apply_merges, compartido por
    /recompute, /nesting-preview y /generate). Reusa parse_pipeline para que la
    cadena sea idéntica a la del upload, sin re-leer el archivo de disco.
    """
    from dataclasses import replace as _replace
    from core.group_classifier import apply_splits

    ax = axis if axis in ("Y", "Z") else base.applied_axis
    mra = DEFAULT_MIN_REAL_AREA if min_real_area is None else float(min_real_area)
    splits = splits or []

    # Fast-path: sin ediciones reales (mismo eje, área por defecto, sin splits) →
    # devolver la base tal cual y evitar re-clasificar (~2 s en modelos grandes).
    if (
        not splits
        and (axis is None or ax == base.applied_axis)
        and abs(mra - DEFAULT_MIN_REAL_AREA) < 1e-9
    ):
        return base

    p1 = parse_pipeline(
        f"{base.stem}.obj",
        base.raw_faces,
        [],
        force_axis=ax,
        min_real_area=mra,
    )

    if splits:
        groups = apply_splits(p1.groups, p1.faces, splits)
        joints = detect_joints(p1.faces, groups)
        adj = compute_adjustments(joints, groups, faces=p1.faces)
        p1 = _replace(
            p1,
            groups=groups,
            joints=joints,
            adjustments=adj.adjustments,
            wall_wall_joints=adj.wall_wall_joints,
        )

    return p1
