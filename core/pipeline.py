import time
import logging
from typing import List, Dict, Optional, Literal, Tuple
from dataclasses import dataclass

from core.profiler import PipelineTimer
from core.services.types import Face3D, Facade, FloorPlan, PipelineOptions, OutputFile
from core.group_classifier import (
    GeometryGroup,
    classify_into_groups,
    peel_buried_walls,
    polish_groups,
    DEFAULT_MIN_REAL_AREA,
)
from core.services.mesh_splitter import split_wall_groups_at_floors
from core.services.joint_detector import Joint, detect_joints
from core.services.assembly_adjuster import (
    DimensionAdjustment,
    WallWallJoint,
    compute_adjustments,
)
from core.services.cutting_sheet import (
    decompose_into_panels,
    generate_cutting_sheets,
    Panel,
    PanelCategory,
)
from core.services.facade_extractor import extract_facades
from core.services.floor_plan_extractor import extract_floor_plans
from core.services.pdf_writer import generate_pdf, generate_nesting_pdf
from core.services.sheet_nester import nest_panels, NestingResult, SheetConfig

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


def parse_pipeline(
    file_name: str, faces: List[Face3D], warnings: List[str]
) -> Phase1Result:
    timer = PipelineTimer(f"parse_pipeline({file_name})")
    stem = file_name.rsplit(".", 1)[0]

    logger.info(f"parse_pipeline: {len(faces):,} caras de entrada")

    # 1. Clasificación inicial
    with timer.step("classify_into_groups", face_count=len(faces)):
        groups = classify_into_groups(faces, out_warnings=warnings)

    logger.info(f"  → {len(groups)} grupos: " + 
                ", ".join(f"{sum(1 for g in groups if g.category==c)} {c}" 
                          for c in ["wall","floor","discard"]))

    # 2. Peel buried walls
    with timer.step("peel_buried_walls", group_count=len(groups)):
        groups = peel_buried_walls(faces, groups)

    logger.info(f"  → tras peel: {len(groups)} grupos")

    # 3. Split por pisos
    pre_split_face_count = len(faces)
    with timer.step("split_wall_groups_at_floors", face_count=pre_split_face_count):
        split_faces, split_groups = split_wall_groups_at_floors(faces, groups, {})

    logger.info(
        f"  → split: {pre_split_face_count:,} → {len(split_faces):,} caras, "
        f"{len(split_groups)} grupos"
    )

    # 4. Polish groups
    with timer.step("polish_groups", group_count=len(split_groups)):
        polish_groups(split_groups, DEFAULT_MIN_REAL_AREA)

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
        raw_faces=faces,
        applied_axis="Y",
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
) -> List[OutputFile]:

    files: List[OutputFile] = []
    return files
