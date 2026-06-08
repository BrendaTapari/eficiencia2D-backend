import math
from typing import List, Dict, Tuple, Any
import re

# Importamos nuestros tipos (asumiendo que exportamos Facade y FloorPlan en types.py)
from core.services.types import Facade, FloorPlan

# Importamos el NestingResult y rotate_edges del sheet_nester
from core.services.sheet_nester import NestingResult, rotate_edges

# ============================================================================
# PDF Writer
#
# Genera un archivo PDF multipágina:
#   - Una página por elevación de fachada
#   - Una página por plano de planta
#   - Generación de PDF de planchas de corte láser (Nesting)
#
# Escribe operadores PDF crudos (raw PDF operators) sin dependencias externas.
# ============================================================================

PAPERS = {
    "A4": {"w": 297.0, "h": 210.0},  # horizontal (landscape) en mm
    "A3": {"w": 420.0, "h": 297.0},
    "A1": {"w": 841.0, "h": 594.0},
}
MM_TO_PT = 72.0 / 25.4


def m_to_pts(m: float, scale_denom: float) -> float:
    return (m / scale_denom) * 1000.0 * MM_TO_PT


def build_facade_content(
    facade: Facade, scale_denom: float, paper: Dict[str, float] = PAPERS["A4"]
) -> str:
    page_w = paper["w"] * MM_TO_PT
    page_h = paper["h"] * MM_TO_PT
    margin = 40.0
    font_size = 10.0

    avail_w = page_w - 2 * margin
    avail_h = page_h - 2 * margin - 30.0

    facade_w_pts = m_to_pts(facade.width, scale_denom)
    facade_h_pts = m_to_pts(facade.height, scale_denom)

    fit_scale = 1.0
    if facade_w_pts > avail_w and facade_w_pts > 0:
        fit_scale = min(fit_scale, avail_w / facade_w_pts)
    if facade_h_pts > avail_h and facade_h_pts > 0:
        fit_scale = min(fit_scale, avail_h / facade_h_pts)

    effective_w = facade_w_pts * fit_scale
    effective_h = facade_h_pts * fit_scale

    ox = (page_w - effective_w) / 2.0
    oy = margin + (avail_h - effective_h) / 2.0

    def tx(vx: float) -> float:
        return ox + m_to_pts(vx, scale_denom) * fit_scale

    def ty(vy: float) -> float:
        return oy + m_to_pts(vy, scale_denom) * fit_scale

    cs: List[str] = []

    # Líneas negras para la fachada
    cs.append("0 0 0 RG\n0.4 w")
    for poly in facade.polygons:
        verts = poly.vertices
        n_verts = len(verts)
        if n_verts < 2:
            continue
        if n_verts == 2:
            cs.append(
                f"{tx(verts[0].x):.4f} {ty(verts[0].y):.4f} m\n{tx(verts[1].x):.4f} {ty(verts[1].y):.4f} l\nS"
            )
        else:
            cs.append(f"{tx(verts[0].x):.4f} {ty(verts[0].y):.4f} m")
            for i in range(1, n_verts):
                cs.append(f"{tx(verts[i].x):.4f} {ty(verts[i].y):.4f} l")
            cs.append("s")

    # Título
    cs.append(
        f"BT\n/F1 {font_size + 2} Tf\n{(page_w / 2.0):.2f} {(oy + effective_h + 16.0):.2f} Td\n({pdf_escape(facade.label)}) Tj\nET"
    )

    # Cotas (Ancho y Alto)
    cs.append(
        f"BT\n/F1 {font_size} Tf\n{(page_w / 2.0):.2f} {(oy - 16.0):.2f} Td\n({facade.width:.2f} m) Tj\nET"
    )
    cs.append(
        f"BT\n/F1 {font_size} Tf\n{(ox + effective_w + 8.0):.2f} {(oy + effective_h / 2.0):.2f} Td\n({facade.height:.2f} m) Tj\nET"
    )

    # Anotación de Escala
    cs.append(
        f"BT\n/F1 {font_size - 2} Tf\n{(page_w - margin):.2f} {(margin / 2.0):.2f} Td"
    )
    if fit_scale < 0.999:
        cs.append("(Escala: ajustada para caber en pagina) Tj")
    else:
        cs.append(f"(Escala: 1:{scale_denom}) Tj")
    cs.append("ET")

    return "\n".join(cs) + "\n"


def build_floor_plan_content(
    plan: FloorPlan, scale_denom: float, paper: Dict[str, float] = PAPERS["A4"]
) -> str:
    page_w = paper["w"] * MM_TO_PT
    page_h = paper["h"] * MM_TO_PT
    margin = 40.0
    font_size = 10.0

    avail_w = page_w - 2 * margin
    avail_h = page_h - 2 * margin - 30.0

    plan_w_pts = m_to_pts(plan.width, scale_denom)
    plan_h_pts = m_to_pts(plan.height, scale_denom)

    fit_scale = 1.0
    if plan_w_pts > avail_w and plan_w_pts > 0:
        fit_scale = min(fit_scale, avail_w / plan_w_pts)
    if plan_h_pts > avail_h and plan_h_pts > 0:
        fit_scale = min(fit_scale, avail_h / plan_h_pts)

    effective_w = plan_w_pts * fit_scale
    effective_h = plan_h_pts * fit_scale

    ox = (page_w - effective_w) / 2.0
    oy = margin + (avail_h - effective_h) / 2.0

    def tx(vx: float) -> float:
        return ox + m_to_pts(vx, scale_denom) * fit_scale

    def ty(vy: float) -> float:
        return oy + m_to_pts(vy, scale_denom) * fit_scale

    cs: List[str] = []

    # Trazar muros
    cs.append("0 0 0 RG\n0.6 w")
    for seg in plan.segments:
        cs.append(
            f"{tx(seg.a.x):.4f} {ty(seg.a.y):.4f} m\n{tx(seg.b.x):.4f} {ty(seg.b.y):.4f} l\nS"
        )

    # Trazar puertas (gris, trazo punteado)
    if plan.doors:
        cs.append("0.45 0.45 0.45 RG\n0.3 w")
        for door in plan.doors:
            cs.append("[3 2] 0 d")
            steps = 24
            start_rad = math.radians(door.start_angle)
            end_rad = math.radians(door.end_angle)
            sweep_rad = end_rad - start_rad
            if sweep_rad < 0:
                sweep_rad += 2 * math.pi

            for i in range(steps + 1):
                angle = start_rad + sweep_rad * (i / float(steps))
                px = door.hinge.x + door.width * math.cos(angle)
                py = door.hinge.y + door.width * math.sin(angle)
                if i == 0:
                    cs.append(f"{tx(px):.4f} {ty(py):.4f} m")
                else:
                    cs.append(f"{tx(px):.4f} {ty(py):.4f} l")
            cs.append("S")

            # Hoja de la puerta (sólida)
            cs.append("[] 0 d")
            cs.append(
                f"{tx(door.hinge.x):.4f} {ty(door.hinge.y):.4f} m\n{tx(door.leaf_end.x):.4f} {ty(door.leaf_end.y):.4f} l\nS"
            )

        # Resetear estado de gráficos
        cs.append("0 0 0 RG\n0.6 w\n[] 0 d")

    # Título
    cs.append(
        f"BT\n/F1 {font_size + 2} Tf\n{(page_w / 2.0):.2f} {(oy + effective_h + 16.0):.2f} Td\n({pdf_escape(plan.label)}) Tj\nET"
    )

    # Cotas
    cs.append(
        f"BT\n/F1 {font_size} Tf\n{(page_w / 2.0):.2f} {(oy - 16.0):.2f} Td\n({plan.width:.2f} m) Tj\nET"
    )
    cs.append(
        f"BT\n/F1 {font_size} Tf\n{(ox + effective_w + 8.0):.2f} {(oy + effective_h / 2.0):.2f} Td\n({plan.height:.2f} m) Tj\nET"
    )

    # Escala
    cs.append(
        f"BT\n/F1 {font_size - 2} Tf\n{(page_w - margin):.2f} {(margin / 2.0):.2f} Td"
    )
    if fit_scale < 0.999:
        cs.append("(Escala: ajustada para caber en pagina) Tj")
    else:
        cs.append(f"(Escala: 1:{scale_denom}) Tj")
    cs.append("ET")

    return "\n".join(cs) + "\n"


def assemble_pdf(page_contents: List[str], page_w: float, page_h: float) -> bytes:
    """Shared PDF assembly — takes page content strings and builds a valid PDF raw bytes block."""
    if not page_contents:
        return b""

    pw = f"{page_w:.2f}"
    ph = f"{page_h:.2f}"
    num_pages = len(page_contents)
    parts: List[str] = []
    offsets: List[int] = []
    cursor = 0

    def emit(s: str):
        nonlocal cursor
        parts.append(s)
        # PDF relies on exact byte counts (ASCII), so len() of string works if strictly ASCII
        cursor += len(s)

    emit("%PDF-1.4\n")

    # Object 1: Catalog
    offsets.append(cursor)
    emit("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

    # Object 2: Pages
    page_obj_ids = [4 + i * 2 for i in range(num_pages)]
    kids = " ".join(f"{obj_id} 0 R" for obj_id in page_obj_ids)
    offsets.append(cursor)
    emit(f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>\nendobj\n")

    # Object 3: Font
    offsets.append(cursor)
    emit("3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    # Pages and content streams
    for i in range(num_pages):
        page_id = 4 + i * 2
        stream_id = page_id + 1
        content = page_contents[i]

        offsets.append(cursor)
        emit(
            f"{page_id} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw} {ph}] /Contents {stream_id} 0 R /Resources << /Font << /F1 3 0 R >> >> >>\nendobj\n"
        )

        offsets.append(cursor)
        emit(
            f"{stream_id} 0 obj\n<< /Length {len(content)} >>\nstream\n{content}endstream\nendobj\n"
        )

    # Xref table
    total_objs = len(offsets) + 1
    xref_off = cursor
    emit(f"xref\n0 {total_objs}\n")
    emit("0000000000 65535 f \n")
    for off in offsets:
        emit(f"{str(off).zfill(10)} 00000 n \n")

    emit(f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\n")
    emit(f"startxref\n{xref_off}\n%%EOF\n")

    # In a real backend, we return bytes so it can be served as a file directly via FastAPI Response
    return "".join(parts).encode("ascii", errors="replace")


def generate_pdf(
    facades: List[Facade],
    floor_plans: List[FloorPlan],
    scale_denom: float,
    paper_name: str = "A4",
) -> bytes:
    paper = PAPERS.get(paper_name, PAPERS["A4"])
    page_contents: List[str] = []

    for facade in facades:
        page_contents.append(build_facade_content(facade, scale_denom, paper))

    for plan in floor_plans:
        page_contents.append(build_floor_plan_content(plan, scale_denom, paper))

    if not page_contents:
        return b""

    return assemble_pdf(page_contents, paper["w"] * MM_TO_PT, paper["h"] * MM_TO_PT)


# ---------------------------------------------------------------------------
# Nesting PDF — renders nested cutting sheets to PDF
# ---------------------------------------------------------------------------

CHAR_W_RATIO_PDF = 0.62
LABEL_H_M = 0.008  # 8mm for panel ID
DIM_H_M = 0.005  # 5mm for dimensions


def fit_text_height_pdf(
    text: str, max_w: float, max_h: float, target_h: float
) -> float:
    if not text:
        return 0.0
    by_width = (max_w * 0.88) / (len(text) * CHAR_W_RATIO_PDF)
    return min(target_h, by_width, max_h)


def pdf_escape(s: str) -> str:
    # A diferencia de TS, usamos replace encadenados o regex simple
    s = s.replace("\\", "\\\\")
    s = s.replace("(", "\\(")
    s = s.replace(")", "\\)")
    # Sanitizar caracteres no-ASCII temporalmente (PDF raw R12/1.4 no maneja bien UTF-8 sin fuentes embebidas)
    return s.encode("ascii", "ignore").decode("ascii")


def generate_nesting_pdf(
    nesting: NestingResult, label: str, include_text: bool
) -> bytes:
    sheets = nesting.sheets
    config = nesting.config

    if not sheets:
        return b""

    SHEET_SPACING_M = 0.10
    cols = min(len(sheets), 3)
    rows = math.ceil(len(sheets) / cols)

    total_w = cols * config.width_m + (cols - 1) * SHEET_SPACING_M
    total_h = rows * config.height_m + (rows - 1) * SHEET_SPACING_M
    sheet_label_space_m = 0.04
    total_h_with_labels = total_h + rows * sheet_label_space_m

    paper = PAPERS["A3"]
    page_w = paper["w"] * MM_TO_PT
    page_h = paper["h"] * MM_TO_PT
    margin = 40.0

    avail_w = page_w - 2 * margin
    avail_h = page_h - 2 * margin

    m_to_pt = 1000.0 * MM_TO_PT
    raw_w = total_w * m_to_pt
    raw_h = total_h_with_labels * m_to_pt

    fit_scale = 1.0
    if raw_w > avail_w and raw_w > 0:
        fit_scale = min(fit_scale, avail_w / raw_w)
    if raw_h > avail_h and raw_h > 0:
        fit_scale = min(fit_scale, avail_h / raw_h)

    scale = m_to_pt * fit_scale

    drawn_w = total_w * scale
    drawn_h = total_h_with_labels * scale
    ox_page = margin + (avail_w - drawn_w) / 2.0
    oy_page = margin + (avail_h - drawn_h) / 2.0

    def tx(mx: float) -> float:
        return ox_page + mx * scale

    def ty(my: float) -> float:
        return oy_page + (total_h_with_labels - my) * scale

    cs: List[str] = []

    for si, sheet in enumerate(sheets):
        col = si % cols
        row = si // cols
        sx = col * (config.width_m + SHEET_SPACING_M)
        sy_top = (
            row * (config.height_m + SHEET_SPACING_M + sheet_label_space_m)
            + sheet_label_space_m
        )

        # Borde de la plancha (negro)
        cs.append("0 0 0 RG\n0.4 w")
        x0, y0 = sx, sy_top
        x1, y1 = sx + config.width_m, sy_top + config.height_m
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

        for ci in range(4):
            ax, ay = corners[ci]
            bx, by = corners[(ci + 1) % 4]
            cs.append(f"{tx(ax):.4f} {ty(ay):.4f} m\n{tx(bx):.4f} {ty(by):.4f} l\nS")

        # Etiqueta de la plancha
        sheet_label = f"Plancha {si + 1}"
        label_cx = tx(sx + config.width_m / 2.0)
        label_cy = ty(sy_top - sheet_label_space_m * 0.4)
        cs.append(
            f"BT\n0 0 0 rg\n/F1 10 Tf\n{label_cx:.2f} {label_cy:.2f} Td\n({pdf_escape(sheet_label)}) Tj\nET"
        )

        # Paneles dentro de la plancha
        for placed in sheet.panels:
            panel = placed.panel
            px, py = placed.x, placed.y
            pw, ph = placed.effective_w, placed.effective_h

            edges = (
                rotate_edges(panel.edges, panel.width_m)
                if placed.rotated
                else panel.edges
            )

            # Dibujar bordes del panel
            cs.append("0 0 0 RG\n0.3 w")
            for edge in edges:
                eax, eay = sx + px + edge.a.x, sy_top + py + edge.a.y
                ebx, eby = sx + px + edge.b.x, sy_top + py + edge.b.y
                cs.append(
                    f"{tx(eax):.4f} {ty(eay):.4f} m\n{tx(ebx):.4f} {ty(eby):.4f} l\nS"
                )

            if include_text:
                real_w = pw * nesting.scale_denom
                real_h = ph * nesting.scale_denom
                dim_text = f"{real_w:.2f} x {real_h:.2f} m"

                label_h = fit_text_height_pdf(panel.id, pw, ph * 0.45, LABEL_H_M)
                dim_h = fit_text_height_pdf(dim_text, pw, ph * 0.30, DIM_H_M)
                MIN_H = 0.002

                if label_h >= MIN_H:
                    label_pt_size = label_h * scale
                    lcx = tx(sx + px + pw / 2.0)
                    lcy = ty(sy_top + py + ph - label_h * 1.5)
                    cs.append(
                        f"BT\n0 0 1 rg\n/F1 {label_pt_size:.2f} Tf\n{lcx:.2f} {lcy:.2f} Td\n({pdf_escape(panel.id)}) Tj\nET"
                    )

                if dim_h >= MIN_H and (label_h + dim_h * 3.0) < ph:
                    dim_pt_size = dim_h * scale
                    dcx = tx(sx + px + pw / 2.0)
                    dcy = ty(sy_top + py + dim_h * 0.6)
                    cs.append(
                        f"BT\n0.5 0.5 0.5 rg\n/F1 {dim_pt_size:.2f} Tf\n{dcx:.2f} {dcy:.2f} Td\n({pdf_escape(dim_text)}) Tj\nET"
                    )

    # Título General
    cs.append(
        f"BT\n0 0 0 rg\n/F1 14 Tf\n{(page_w / 2.0):.2f} {(page_h - margin / 2.0 - 4.0):.2f} Td\n({pdf_escape(label)}) Tj\nET"
    )

    content_str = "\n".join(cs) + "\n"
    return assemble_pdf([content_str], page_w, page_h)
