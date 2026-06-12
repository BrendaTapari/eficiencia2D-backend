"""
Generación de planos a partir del estado de revisión del frontend.

El upload/clasificación corre en parse_pipeline; acá se aplican overrides,
fusiones y decisiones muro-muro, se descompone por grupo y se exporta DXF/PDF.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from core.group_classifier import GeometryGroup
from core.pipeline import Phase1Result
from core.services.assembly_adjuster import compute_adjustments
from core.services.cutting_sheet import (
    Panel,
    clip_panel_at_u,
    clip_panel_at_v,
    mirror_edges_horizontal,
    nested_sheets_to_dxf,
    project_faces_to_2d,
)
from core.services.facade_extractor import extract_facades
from core.services.floor_plan_extractor import extract_floor_plans
from core.services.joint_detector import detect_joints
from core.services.pdf_writer import generate_nesting_pdf, generate_pdf
from core.services.sheet_nester import (
    NestingPanel,
    NestingResult,
    SheetConfig,
    Vec2,
    Edge,
    nest_panels,
)
from core.services.types import OutputFile, PipelineOptions, Vec3, dot


def apply_merges(phase1: Phase1Result, merges: List[List[int]]) -> Phase1Result:
    if not merges:
        return phase1

    group_by_id: Dict[int, GeometryGroup] = {g.id: g for g in phase1.groups}
    merged_ids: set[int] = set()

    for merge_set in merges:
        members = [
            group_by_id[gid]
            for gid in merge_set
            if gid in group_by_id and group_by_id[gid].category != "discard"
        ]
        if len(members) < 2:
            continue

        survivor = max(members, key=lambda g: g.totalArea)
        combined_faces: List[int] = []
        total_area = 0.0
        cx = cy = cz = 0.0
        min_y = max_y = None

        for m in members:
            combined_faces.extend(m.face_indices)
            total_area += m.totalArea
            cx += m.centroid.x * m.totalArea
            cy += m.centroid.y * m.totalArea
            cz += m.centroid.z * m.totalArea
            if m.min_y is not None:
                min_y = m.min_y if min_y is None else min(min_y, m.min_y)
            if m.max_y is not None:
                max_y = m.max_y if max_y is None else max(max_y, m.max_y)
            if m.id != survivor.id:
                merged_ids.add(m.id)

        centroid = (
            Vec3(x=cx / total_area, y=cy / total_area, z=cz / total_area)
            if total_area > 0
            else survivor.centroid
        )
        group_by_id[survivor.id] = replace(
            survivor,
            face_indices=combined_faces,
            total_area=total_area,
            centroid=centroid,
            min_y=min_y,
            max_y=max_y,
        )

    new_groups = [
        group_by_id[g.id] for g in phase1.groups if g.id not in merged_ids
    ]
    joints = detect_joints(phase1.faces, new_groups)
    adj = compute_adjustments(joints, new_groups, None, phase1.faces)

    return replace(
        phase1,
        groups=new_groups,
        joints=joints,
        adjustments=adj.adjustments,
        wall_wall_joints=adj.wall_wall_joints,
        suggested_merges=[],
    )


def _effective_category(
    group: GeometryGroup, overrides: Dict[int, str]
) -> str:
    return overrides.get(group.id, group.category)


def decompose_panels_from_groups(
    phase1: Phase1Result,
    opts: PipelineOptions,
    overrides: Optional[Dict[int, str]] = None,
    wall_wall_decisions: Optional[Dict[int, int]] = None,
) -> Tuple[List[Panel], List[Panel]]:
    overrides = overrides or {}
    min_area = opts.min_area_m2 if opts.min_area_m2 is not None else 0.01

    effective_decisions: Dict[int, int] = {}
    for ww in phase1.wall_wall_joints:
        if ww.suggested_yield_group_id is not None:
            effective_decisions[ww.joint_index] = ww.suggested_yield_group_id
    if wall_wall_decisions:
        effective_decisions.update(wall_wall_decisions)

    adj_result = compute_adjustments(
        phase1.joints,
        phase1.groups,
        effective_decisions,
        phase1.faces,
    )

    height_adj: Dict[int, float] = {}
    width_adjs: Dict[int, list] = {}
    for adj in adj_result.adjustments:
        if adj.axis == "height":
            height_adj[adj.group_id] = height_adj.get(adj.group_id, 0.0) + adj.delta
        else:
            width_adjs.setdefault(adj.group_id, []).append(adj)

    wall_panels: List[Panel] = []
    floor_panels: List[Panel] = []
    wall_count = floor_count = 0

    for group in phase1.groups:
        cat = _effective_category(group, overrides)
        if cat == "discard":
            continue

        is_floor = cat == "floor"
        faces = [phase1.faces[fi] for fi in group.face_indices if fi < len(phase1.faces)]
        if not faces:
            continue

        result = project_faces_to_2d(faces, group.representative_normal, "Y")
        if not result:
            continue

        width_m, height_m, edges = result.width_m, result.height_m, result.edges
        if width_m * height_m < min_area:
            continue

        height_delta = height_adj.get(group.id, 0.0)
        if height_delta < 0 and not is_floor:
            strip = min(-height_delta, height_m - 0.01)
            if strip > 0.001:
                base_at_min_v = result.v_up >= 0
                clipped = (
                    clip_panel_at_v(edges, strip, True)
                    if base_at_min_v
                    else clip_panel_at_v(edges, height_m - strip, False)
                )
                if clipped:
                    width_m, height_m, edges = (
                        clipped.width_m,
                        clipped.height_m,
                        clipped.edges,
                    )

        for w_adj in width_adjs.get(group.id, []):
            if w_adj.delta >= 0 or is_floor:
                continue
            strip = min(-w_adj.delta, width_m - 0.01)
            if strip <= 0.001:
                continue
            joint = phase1.joints[w_adj.joint_index]
            u = dot(joint.edge_mid, result.u_axis) - result.origin_u
            joint_on_left = u < width_m / 2
            clipped = (
                clip_panel_at_u(edges, width_m - strip, False)
                if joint_on_left
                else clip_panel_at_u(edges, strip, True)
            )
            if clipped:
                width_m, height_m, edges = (
                    clipped.width_m,
                    clipped.height_m,
                    clipped.edges,
                )

        edges = mirror_edges_horizontal(edges, width_m)

        if is_floor:
            floor_count += 1
            floor_panels.append(
                Panel(
                    id=f"B{floor_count}",
                    group_name=f"floor_{floor_count}",
                    category="floor",
                    floor_index=0,
                    width_m=width_m,
                    height_m=height_m,
                    edges=edges,
                    source_group_id=group.id,
                )
            )
        else:
            wall_count += 1
            wall_panels.append(
                Panel(
                    id=f"A{wall_count}",
                    group_name=f"wall_{wall_count}",
                    category="wall",
                    floor_index=0,
                    width_m=width_m,
                    height_m=height_m,
                    edges=edges,
                    source_group_id=group.id,
                )
            )

    return wall_panels, floor_panels


def _panels_to_nesting(panels: List[Panel], scale_denom: float) -> List[NestingPanel]:
    s = 1.0 / scale_denom
    out: List[NestingPanel] = []
    for p in panels:
        out.append(
            NestingPanel(
                id=p.id,
                category=p.category,
                width_m=p.width_m * s,
                height_m=p.height_m * s,
                edges=[
                    Edge(
                        a=Vec2(e.a.x * s, e.a.y * s),
                        b=Vec2(e.b.x * s, e.b.y * s),
                    )
                    for e in p.edges
                ],
            )
        )
    return out


def generate_from_review(
    phase1: Phase1Result,
    opts: PipelineOptions,
    overrides: Optional[Dict[int, str]] = None,
    wall_wall_decisions: Optional[Dict[int, int]] = None,
    merges: Optional[List[List[int]]] = None,
) -> List[OutputFile]:
    work = apply_merges(phase1, merges or [])
    wall_panels, floor_panels = decompose_panels_from_groups(
        work, opts, overrides, wall_wall_decisions
    )

    sc = opts.sheet_config
    sheet_cfg = SheetConfig(
        width_m=sc.width_m if sc else 1.0,
        height_m=sc.height_m if sc else 0.6,
        gap_m=sc.gap_m if sc else 0.003,
    )
    scale = opts.scale_denom
    stem = work.stem

    wall_nesting = nest_panels(_panels_to_nesting(wall_panels, scale), sheet_cfg, scale)
    floor_nesting = nest_panels(_panels_to_nesting(floor_panels, scale), sheet_cfg, scale)

    files: List[OutputFile] = []

    def add_nesting_outputs(nesting: NestingResult, label: str, prefix: str) -> None:
        if not nesting.sheets:
            return
        files.append(
            OutputFile(
                name=f"{stem}_{prefix}_con_referencias.dxf",
                blob=nested_sheets_to_dxf(nesting, True).encode("utf-8"),
            )
        )
        ref_pdf = generate_nesting_pdf(nesting, label, True)
        if ref_pdf:
            files.append(
                OutputFile(name=f"{stem}_{prefix}_con_referencias.pdf", blob=ref_pdf)
            )
        files.append(
            OutputFile(
                name=f"{stem}_{prefix}_corte.dxf",
                blob=nested_sheets_to_dxf(nesting, False).encode("utf-8"),
            )
        )
        cut_pdf = generate_nesting_pdf(nesting, label, False)
        if cut_pdf:
            files.append(OutputFile(name=f"{stem}_{prefix}_corte.pdf", blob=cut_pdf))

    add_nesting_outputs(wall_nesting, "Paredes", "Paredes")
    add_nesting_outputs(floor_nesting, "Pisos", "Pisos")

    facades = extract_facades(work.faces, "Y")
    floor_plans = extract_floor_plans(work.faces, "Y")
    plan_pdf = generate_pdf(facades, floor_plans, scale, opts.paper)
    if plan_pdf:
        files.append(OutputFile(name=f"{stem}_planos.pdf", blob=plan_pdf))

    return files
