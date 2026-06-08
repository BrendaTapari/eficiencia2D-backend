from typing import List, Dict, Optional, Literal, Tuple
from dataclasses import dataclass

# Importaciones de todos los servicios que traducimos
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


@dataclass
class Phase1Result:
    faces: List[Face3D]
    raw_faces: List[Face3D]
    applied_axis: Literal["Y", "Z"]
    groups: List[GeometryGroup]
    joints: List[Joint]
    adjustments: List[DimensionAdjustment]
    wall_wall_joints: List[WallWallJoint]
    stem: str
    warnings: List[str]
    pre_split_face_count: int
    suggested_merges: List[List[int]]


def parse_pipeline(
    file_name: str, faces: List[Face3D], warnings: List[str]
) -> Phase1Result:
    stem = file_name.rsplit(".", 1)[0]

    # 1. Clasificación inicial
    groups = peel_buried_walls(
        faces, classify_into_groups(faces, out_warnings=warnings)
    )

    # 2. Split por pisos
    pre_split_face_count = len(faces)
    split_faces, split_groups = split_wall_groups_at_floors(faces, groups, {})
    polish_groups(split_groups, DEFAULT_MIN_REAL_AREA)

    # 3. Detección de uniones y ajustes
    joints = detect_joints(split_faces, split_groups)
    adj_result = compute_adjustments(joints, split_groups, faces=split_faces)

    # Aquí iría la lógica de suggest_coplanar_merges si la necesitás

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
    )


def generate_pipeline(
    phase1: Phase1Result,
    opts: PipelineOptions,
    overrides: Optional[Dict[int, str]] = None,
) -> List[OutputFile]:

    # Este proceso invoca a los writers (DXF/PDF)
    # 1. Clasificación final según overrides
    # 2. generate_cutting_sheets -> retorna lista de DXFs
    # 3. generate_pdf -> retorna PDF

    files: List[OutputFile] = []

    # Ejemplo de uso de lo que ya programamos:
    # dxffiles = generate_cutting_sheets(phase1.faces, "Y", opts.scale_denom)

    return files
