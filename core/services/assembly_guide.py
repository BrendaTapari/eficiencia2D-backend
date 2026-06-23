"""
Guía de ensamble — PDF multipágina.

Genera un PDF complementario con:
  1. Vista isométrica del modelo 3D con el código de cada panel anotado
  2. Cuatro elevaciones ortogonales (Frente / Atrás / Derecha / Izquierda)
  3. Directorio de paneles (tabla: ID | Tipo | Ancho | Alto)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from core.group_classifier import GeometryGroup
from core.services.cutting_sheet import Panel, project_faces_to_2d
from core.services.pdf_writer import PAPERS, MM_TO_PT, assemble_pdf, pdf_escape
from core.services.types import Face3D, Vec3, dot

# ---------------------------------------------------------------------------
# Constantes de proyección
# ---------------------------------------------------------------------------

_ISO_C = math.cos(math.radians(30))  # ≈ 0.866
_ISO_S = 0.5                          # sin(30°)
_WORLD_UP = Vec3(0.0, 1.0, 0.0)

# Elevaciones estándar: (nombre, u_axis, v_axis)
# u_axis = dirección horizontal en la vista; v_axis = dirección vertical
_ELEVATIONS: List[Tuple[str, Vec3, Vec3, Vec3]] = [
    # (nombre, view_dir, u_axis, v_axis)
    ("Frente",    Vec3( 0, 0,  1), Vec3( 1, 0,  0), Vec3(0, 1, 0)),
    ("Atrás",     Vec3( 0, 0, -1), Vec3(-1, 0,  0), Vec3(0, 1, 0)),
    ("Derecha",   Vec3( 1, 0,  0), Vec3( 0, 0, -1), Vec3(0, 1, 0)),
    ("Izquierda", Vec3(-1, 0,  0), Vec3( 0, 0,  1), Vec3(0, 1, 0)),
]

_MARGIN = 48.0    # pt — margen de página
_TITLE_H = 24.0   # pt — altura reservada para el título


# ---------------------------------------------------------------------------
# Proyección
# ---------------------------------------------------------------------------

def _iso(v: Vec3) -> Tuple[float, float]:
    """Proyección isométrica estándar desde arriba-derecha-frente."""
    return (v.x - v.z) * _ISO_C, v.y + (v.x + v.z) * _ISO_S


def _proj(v: Vec3, u: Vec3, up: Vec3) -> Tuple[float, float]:
    """Proyección ortogonal sobre el plano (u, up)."""
    return dot(v, u), dot(v, up)


# ---------------------------------------------------------------------------
# Utilidades de escala y layout
# ---------------------------------------------------------------------------

def _fit(
    pts: List[Tuple[float, float]],
    avail_w: float,
    avail_h: float,
    padding: float = 0.90,
) -> Tuple[float, float, float]:
    """Calcula (scale, off_x, off_y) para centrar pts en el área disponible."""
    if not pts:
        return 1.0, avail_w / 2, avail_h / 2
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    dw = (max(xs) - min(xs)) or 1.0
    dh = (max(ys) - min(ys)) or 1.0
    s = min(avail_w / dw, avail_h / dh) * padding
    ox = avail_w / 2 - ((min(xs) + max(xs)) / 2) * s
    oy = avail_h / 2 - ((min(ys) + max(ys)) / 2) * s
    return s, ox, oy


def _px(x: float, s: float, ox: float, base: float) -> float:
    return base + x * s + ox


def _py(y: float, s: float, oy: float, base: float) -> float:
    return base + y * s + oy


# ---------------------------------------------------------------------------
# Colores por categoría
# ---------------------------------------------------------------------------

_CAT_FILL = {
    "wall":  ("0.18 0.48 0.78", "0.12 0.36 0.62"),  # (rg fill, RG stroke)
    "floor": ("0.45 0.22 0.68", "0.34 0.15 0.52"),
}
_LEADER_COLOR = "0 0 0"  # negro para líneas líder
_WIRE_COLOR   = "0.80 0.80 0.80"  # gris claro wireframe


def _effective_category(
    group: GeometryGroup, overrides: Dict[int, str]
) -> str:
    return overrides.get(group.id, group.category)


def _panel_bbox_centroid(panel: Panel) -> Tuple[float, float]:
    xs = [e.a.x for e in panel.edges] + [e.b.x for e in panel.edges]
    ys = [e.a.y for e in panel.edges] + [e.b.y for e in panel.edges]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def _panel_centroid_3d(
    panel: Panel,
    group: GeometryGroup,
    faces: List[Face3D],
) -> Vec3:
    """
    Ubica la etiqueta en el centro de la pieza 2D final, mapeado al 3D del grupo.
    Si un corte manual partió el panel, cada pieza queda en su propia posición.
    """
    group_faces = [faces[fi] for fi in group.face_indices if fi < len(faces)]
    proj = project_faces_to_2d(group_faces, group.representative_normal, "Y")
    if not proj:
        return group.centroid

    cx_m, cy_m = _panel_bbox_centroid(panel)
    u_local = panel.width_m - cx_m  # des-espejar (mirror_edges_horizontal)
    v_local = cy_m
    target_u = proj.origin_u + u_local
    target_v = proj.origin_v + v_local

    best: Optional[Vec3] = None
    best_d = float("inf")
    for face in group_faces:
        for vertex in face.vertices:
            pu = dot(vertex, proj.u_axis)
            pv = dot(vertex, proj.v_axis)
            d = (pu - target_u) ** 2 + (pv - target_v) ** 2
            if d < best_d:
                best_d = d
                best = vertex
    return best if best is not None else group.centroid


def _active_group_ids(
    panels: List[Panel],
    groups: List[GeometryGroup],
    overrides: Dict[int, str],
) -> Set[int]:
    """Grupos que produjeron paneles finales (respeta overrides/descartes)."""
    panel_groups = {p.source_group_id for p in panels}
    return {
        g.id
        for g in groups
        if g.id in panel_groups and _effective_category(g, overrides) != "discard"
    }


# ---------------------------------------------------------------------------
# Página 1: Vista isométrica
# ---------------------------------------------------------------------------

def _build_iso_page(
    panels: List[Panel],
    faces: List[Face3D],
    group_by_id: Dict[int, GeometryGroup],
    paper: Dict[str, float],
    overrides: Dict[int, str],
) -> str:
    pw = paper["w"] * MM_TO_PT
    ph = paper["h"] * MM_TO_PT
    ox0 = _MARGIN
    oy0 = _MARGIN
    avail_w = pw - 2 * _MARGIN
    avail_h = ph - 2 * _MARGIN - _TITLE_H

    active_ids = _active_group_ids(panels, list(group_by_id.values()), overrides)

    # Recopilar puntos isométricos de los grupos activos para calcular escala
    all_face_indices: Set[int] = set()
    for gid in active_ids:
        g = group_by_id.get(gid)
        if g:
            for fi in g.face_indices:
                all_face_indices.add(fi)

    all_pts: List[Tuple[float, float]] = []
    for fi in all_face_indices:
        if fi < len(faces):
            for v in faces[fi].vertices:
                all_pts.append(_iso(v))

    s, off_x, off_y = _fit(all_pts, avail_w, avail_h)

    def tx(x: float) -> float:
        return _px(x, s, off_x, ox0)

    def ty(y: float) -> float:
        return _py(y, s, off_y, oy0)

    cs: List[str] = []

    # ── Título ──────────────────────────────────────────────────────────────
    cs.append(
        f"BT\n/F1 13 Tf\n{pw/2:.1f} {ph - _MARGIN * 0.6:.1f} Td\n"
        f"(Guia de Ensamble - Vista Isometrica) Tj\nET"
    )

    # ── Wireframe de la geometría (grupos no descartados) ───────────────────
    cs.append(f"{_WIRE_COLOR} RG\n0.25 w")
    for fi in all_face_indices:
        if fi >= len(faces):
            continue
        verts = faces[fi].vertices
        n = len(verts)
        for i in range(n):
            ax, ay = _iso(verts[i])
            bx, by = _iso(verts[(i + 1) % n])
            cs.append(
                f"{tx(ax):.2f} {ty(ay):.2f} m\n"
                f"{tx(bx):.2f} {ty(by):.2f} l S"
            )

    # ── Marcadores y etiquetas de paneles ───────────────────────────────────
    for panel in sorted(panels, key=lambda p: p.id):
        group = group_by_id.get(panel.source_group_id)
        if not group or panel.source_group_id not in active_ids:
            continue

        label_pt = _panel_centroid_3d(panel, group, faces)
        cx_w, cy_w = _iso(label_pt)
        px_pt = tx(cx_w)
        py_pt = ty(cy_w)

        fill, stroke = _CAT_FILL.get(panel.category, ("0.3 0.3 0.3", "0 0 0"))

        # Cruz marcadora
        r = 5.0
        cs.append(f"{stroke} RG\n1.8 w")
        cs.append(f"{px_pt - r:.2f} {py_pt:.2f} m {px_pt + r:.2f} {py_pt:.2f} l S")
        cs.append(f"{px_pt:.2f} {py_pt - r:.2f} m {px_pt:.2f} {py_pt + r:.2f} l S")

        # Círculo relleno
        _append_circle(cs, px_pt, py_pt, r * 0.7, fill, stroke)

        # Etiqueta con fondo blanco
        lx = px_pt + r + 2.0
        ly = py_pt - 3.5
        label = panel.id
        # Fondo
        lw = len(label) * 5.5 + 3
        lh = 9.0
        cs.append(
            f"1 1 1 rg {_LEADER_COLOR} RG 0.3 w\n"
            f"{lx - 1:.2f} {ly - 1:.2f} {lw:.2f} {lh:.2f} re f\n"
            f"{lx - 1:.2f} {ly - 1:.2f} {lw:.2f} {lh:.2f} re S"
        )
        cs.append(
            f"BT\n/F1 7 Tf\n{fill} rg\n"
            f"{lx:.1f} {ly:.1f} Td\n({pdf_escape(label)}) Tj\nET"
        )

    # ── Leyenda ──────────────────────────────────────────────────────────────
    lx, ly = ox0 + 8.0, oy0 + 6.0
    _append_circle(cs, lx + 4, ly + 4, 4, _CAT_FILL["wall"][0], _CAT_FILL["wall"][1])
    cs.append(
        f"BT\n/F1 7 Tf\n0 0 0 rg\n{lx + 11:.1f} {ly + 1:.1f} Td\n"
        f"(Pared) Tj\nET"
    )
    ly += 14
    _append_circle(cs, lx + 4, ly + 4, 4, _CAT_FILL["floor"][0], _CAT_FILL["floor"][1])
    cs.append(
        f"BT\n/F1 7 Tf\n0 0 0 rg\n{lx + 11:.1f} {ly + 1:.1f} Td\n"
        f"(Piso / Techo) Tj\nET"
    )

    cs.append("0 0 0 RG\n0 0 0 rg")
    return "\n".join(cs) + "\n"


# ---------------------------------------------------------------------------
# Páginas 2-5: Elevaciones ortogonales
# ---------------------------------------------------------------------------

def _build_elevation_page(
    name: str,
    view_dir: Vec3,
    u_axis: Vec3,
    v_axis: Vec3,
    panels: List[Panel],
    faces: List[Face3D],
    group_by_id: Dict[int, GeometryGroup],
    paper: Dict[str, float],
    overrides: Dict[int, str],
) -> Optional[str]:
    """Genera una página de elevación para la dirección dada."""
    pw = paper["w"] * MM_TO_PT
    ph = paper["h"] * MM_TO_PT
    ox0 = _MARGIN
    oy0 = _MARGIN
    avail_w = pw - 2 * _MARGIN
    avail_h = ph - 2 * _MARGIN - _TITLE_H

    active_ids = _active_group_ids(panels, list(group_by_id.values()), overrides)

    # ── Grupos visibles desde esta dirección ──────────────────────────────
    visible_groups = [
        g for g in group_by_id.values()
        if g.id in active_ids and dot(g.representative_normal, view_dir) > 0.25
    ]
    if not visible_groups:
        return None

    vis_face_idx: Set[int] = set()
    for g in visible_groups:
        for fi in g.face_indices:
            vis_face_idx.add(fi)

    # Puntos proyectados para calcular escala
    all_pts = []
    for fi in vis_face_idx:
        if fi < len(faces):
            for v in faces[fi].vertices:
                all_pts.append(_proj(v, u_axis, v_axis))

    if not all_pts:
        return None

    s, off_x, off_y = _fit(all_pts, avail_w, avail_h)

    def tx(x: float) -> float:
        return _px(x, s, off_x, ox0)

    def ty(y: float) -> float:
        return _py(y, s, off_y, oy0)

    cs: List[str] = []

    # ── Título ──────────────────────────────────────────────────────────────
    cs.append(
        f"BT\n/F1 13 Tf\n{pw/2:.1f} {ph - _MARGIN * 0.6:.1f} Td\n"
        f"(Elevacion: {pdf_escape(name)}) Tj\nET"
    )

    # ── Ordenar grupos por profundidad (los más alejados primero) ───────────
    def _depth(g: GeometryGroup) -> float:
        return dot(g.centroid, view_dir)

    ordered_groups = sorted(visible_groups, key=_depth)

    # ── Dibujar wireframe de todos los grupos visibles ───────────────────────
    cs.append(f"{_WIRE_COLOR} RG\n0.25 w")
    for g in ordered_groups:
        for fi in g.face_indices:
            if fi >= len(faces):
                continue
            verts = faces[fi].vertices
            n = len(verts)
            for i in range(n):
                ax, ay = _proj(verts[i], u_axis, v_axis)
                bx, by = _proj(verts[(i + 1) % n], u_axis, v_axis)
                cs.append(
                    f"{tx(ax):.2f} {ty(ay):.2f} m\n"
                    f"{tx(bx):.2f} {ty(by):.2f} l S"
                )

    # ── Paneles con etiquetas ────────────────────────────────────────────────
    vis_group_ids = {g.id for g in visible_groups}
    vis_panels = [p for p in panels if p.source_group_id in vis_group_ids]

    for panel in sorted(vis_panels, key=lambda p: p.id):
        group = group_by_id.get(panel.source_group_id)
        if not group:
            continue

        label_pt = _panel_centroid_3d(panel, group, faces)
        cx2, cy2 = _proj(label_pt, u_axis, v_axis)
        px_pt = tx(cx2)
        py_pt = ty(cy2)

        fill, stroke = _CAT_FILL.get(panel.category, ("0.3 0.3 0.3", "0 0 0"))

        # Contorno del grupo proyectado (color según categoría, bordes más visibles)
        cs.append(f"{stroke} RG\n0.7 w")
        for fi in group.face_indices:
            if fi >= len(faces):
                continue
            verts = faces[fi].vertices
            n = len(verts)
            for i in range(n):
                ax, ay = _proj(verts[i], u_axis, v_axis)
                bx, by = _proj(verts[(i + 1) % n], u_axis, v_axis)
                cs.append(
                    f"{tx(ax):.2f} {ty(ay):.2f} m\n"
                    f"{tx(bx):.2f} {ty(by):.2f} l S"
                )

        # Marcador en el centroide
        r = 4.5
        _append_circle(cs, px_pt, py_pt, r, fill, stroke)

        # Etiqueta
        lx = px_pt + r + 2.0
        ly = py_pt - 3.5
        label = panel.id
        lw = len(label) * 5.5 + 3
        lh = 9.0
        cs.append(
            f"1 1 1 rg {_LEADER_COLOR} RG 0.3 w\n"
            f"{lx - 1:.2f} {ly - 1:.2f} {lw:.2f} {lh:.2f} re f\n"
            f"{lx - 1:.2f} {ly - 1:.2f} {lw:.2f} {lh:.2f} re S"
        )
        cs.append(
            f"BT\n/F1 7 Tf\n{fill} rg\n"
            f"{lx:.1f} {ly:.1f} Td\n({pdf_escape(label)}) Tj\nET"
        )

    cs.append("0 0 0 RG\n0 0 0 rg")
    return "\n".join(cs) + "\n"


# ---------------------------------------------------------------------------
# Última página: Directorio de paneles
# ---------------------------------------------------------------------------

def _build_directory_page(
    panels: List[Panel],
    paper: Dict[str, float],
) -> str:
    pw = paper["w"] * MM_TO_PT
    ph = paper["h"] * MM_TO_PT

    all_panels = sorted(panels, key=lambda p: (0 if p.category == "wall" else 1, p.id))

    cs: List[str] = []

    # Título
    cs.append(
        f"BT\n/F1 13 Tf\n{pw/2:.1f} {ph - _MARGIN * 0.6:.1f} Td\n"
        f"(Directorio de Paneles) Tj\nET"
    )

    # Encabezados de columna
    cols = [
        ("ID",      _MARGIN + 4,   48),
        ("Tipo",    _MARGIN + 54,  48),
        ("Ancho",   _MARGIN + 120, 52),
        ("Alto",    _MARGIN + 180, 52),
        ("Area m2", _MARGIN + 240, 60),
    ]
    row_h = 14.0
    hdr_y = ph - _MARGIN - _TITLE_H - row_h

    cs.append("0.92 0.92 0.95 rg")
    cs.append(
        f"{_MARGIN:.1f} {hdr_y:.1f} "
        f"{pw - 2*_MARGIN:.1f} {row_h:.1f} re f"
    )
    cs.append("0 0 0 rg\n0.55 0.55 0.55 RG\n0.4 w")
    cs.append(
        f"{_MARGIN:.1f} {hdr_y:.1f} "
        f"{pw - 2*_MARGIN:.1f} {row_h:.1f} re S"
    )
    for col_name, col_x, _ in cols:
        cs.append(
            f"BT\n/F1 8 Tf\n0 0 0 rg\n"
            f"{col_x:.1f} {hdr_y + 4:.1f} Td\n"
            f"({pdf_escape(col_name)}) Tj\nET"
        )

    y = hdr_y - row_h
    for i, panel in enumerate(all_panels):
        if y < _MARGIN:
            break  # sin paginación por ahora

        # Fila alternada
        if i % 2 == 0:
            cs.append(f"0.97 0.97 0.97 rg")
            cs.append(
                f"{_MARGIN:.1f} {y:.1f} "
                f"{pw - 2*_MARGIN:.1f} {row_h:.1f} re f"
            )

        tipo = "Pared" if panel.category == "wall" else "Piso/Techo"
        area = panel.width_m * panel.height_m
        row_data = [
            panel.id,
            tipo,
            f"{panel.width_m:.3f} m",
            f"{panel.height_m:.3f} m",
            f"{area:.4f}",
        ]
        fill, _ = _CAT_FILL.get(panel.category, ("0 0 0", "0 0 0"))
        for j, (_, col_x, _) in enumerate(cols):
            text_color = fill if j == 0 else "0 0 0"
            font_size = 8 if j > 0 else 9
            cs.append(
                f"BT\n/F1 {font_size} Tf\n{text_color} rg\n"
                f"{col_x:.1f} {y + 4:.1f} Td\n"
                f"({pdf_escape(row_data[j])}) Tj\nET"
            )

        # Línea divisoria
        cs.append(f"0.80 0.80 0.80 RG\n0.25 w")
        cs.append(
            f"{_MARGIN:.1f} {y:.1f} m "
            f"{pw - _MARGIN:.1f} {y:.1f} l S"
        )
        y -= row_h

    # Totales al pie
    if y >= _MARGIN + row_h:
        total_area = sum(p.width_m * p.height_m for p in all_panels)
        n_wall = sum(1 for p in all_panels if p.category == "wall")
        n_floor = sum(1 for p in all_panels if p.category != "wall")
        summary = (
            f"Total: {len(all_panels)} paneles  "
            f"({n_wall} paredes, {n_floor} pisos/techos)  |  "
            f"Area total: {total_area:.2f} m2"
        )
        cs.append(
            f"BT\n/F1 8 Tf\n0.3 0.3 0.3 rg\n"
            f"{_MARGIN + 4:.1f} {y + 4:.1f} Td\n"
            f"({pdf_escape(summary)}) Tj\nET"
        )

    cs.append("0 0 0 RG\n0 0 0 rg")
    return "\n".join(cs) + "\n"


# ---------------------------------------------------------------------------
# Helper: círculo en PDF (aproximación con bezier cúbico)
# ---------------------------------------------------------------------------

_K = 0.5523  # factor para aproximar círculo con 4 beziers


def _append_circle(
    cs: List[str], cx: float, cy: float, r: float, fill: str, stroke: str
) -> None:
    """Agrega un círculo relleno al stream de contenido PDF."""
    k = _K * r
    cs.append(
        f"{fill} rg\n{stroke} RG\n0.5 w\n"
        f"{cx + r:.2f} {cy:.2f} m\n"
        f"{cx + r:.2f} {cy + k:.2f} {cx + k:.2f} {cy + r:.2f} {cx:.2f} {cy + r:.2f} c\n"
        f"{cx - k:.2f} {cy + r:.2f} {cx - r:.2f} {cy + k:.2f} {cx - r:.2f} {cy:.2f} c\n"
        f"{cx - r:.2f} {cy - k:.2f} {cx - k:.2f} {cy - r:.2f} {cx:.2f} {cy - r:.2f} c\n"
        f"{cx + k:.2f} {cy - r:.2f} {cx + r:.2f} {cy - k:.2f} {cx + r:.2f} {cy:.2f} c\n"
        f"b"
    )


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def generate_assembly_guide_pdf(
    wall_panels: List[Panel],
    floor_panels: List[Panel],
    faces: List[Face3D],
    groups: List[GeometryGroup],
    overrides: Optional[Dict[int, str]] = None,
    scale_denom: float = 100.0,
    paper_name: str = "A3",
) -> bytes:
    """
    Genera el PDF de guía de ensamble.

    Devuelve bytes del PDF, o b"" si no hay paneles.
    """
    all_panels = list(wall_panels) + list(floor_panels)
    if not all_panels:
        return b""

    overrides = overrides or {}
    paper = PAPERS.get(paper_name, PAPERS["A3"])

    group_by_id: Dict[int, GeometryGroup] = {g.id: g for g in groups}

    pages: List[str] = []

    # Página 1 — isométrica
    pages.append(_build_iso_page(all_panels, faces, group_by_id, paper, overrides))

    # Páginas 2-5 — elevaciones
    for name, view_dir, u_axis, v_axis in _ELEVATIONS:
        page = _build_elevation_page(
            name, view_dir, u_axis, v_axis,
            all_panels, faces, group_by_id, paper, overrides,
        )
        if page:
            pages.append(page)

    # Última página — directorio de paneles
    pages.append(_build_directory_page(all_panels, paper))

    if not pages:
        return b""

    return assemble_pdf(pages, paper["w"] * MM_TO_PT, paper["h"] * MM_TO_PT)
