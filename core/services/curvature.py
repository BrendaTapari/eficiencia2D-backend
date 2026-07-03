"""
Detección de curvatura y desarrollo (unroll / flatten) de superficies curvas.

El pipeline base asume caras coplanares: una superficie curva se fragmenta en el
agrupado. Este módulo permite (a) medir la curvatura de un conjunto de caras y
(b) desarrollarlo a un panel plano para cortar el patrón de flexión encima.

- **Curvatura simple** (desarrollable, p. ej. cilindro) → `unroll`, preservando la
  longitud de arco. El patrón *kerf* se corta sobre ese plano.
- **Doble curvatura** (no desarrollable, p. ej. cúpula) → aplanado aproximado
  (proyección al plano PCA + escala para preservar área). El patrón *auxético*
  absorbe la diferencia al expandirse.

Todo en METROS, marco Y-up (igual que el resto del pipeline). Usa numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from core.services.cutting_sheet import Edge2D
from core.services.types import Face3D, Vec2, Vec3

# Umbrales (grados / adimensionales)
FLAT_SPREAD_DEG = 12.0     # dispersión de normales por debajo de la cual es "plano"
FIT_TOL = 0.06             # residuo RMS máximo relativo al TAMAÑO del panel (no al radio)
MIN_FACES = 8              # menos caras que esto no se considera superficie curva
MIN_SWEEP_DEG = 20.0       # barrido angular mínimo para considerar curvatura real
MAX_RADIUS_FACTOR = 4.0    # radio/extent máximo: por encima la superficie es casi plana
MIN_RADIAL_ALIGN = 0.6     # |dot(normal, radial)| medio mínimo (normales realmente radiales)
MIN_SIDEDNESS = 0.75       # todas las normales al mismo lado (rechaza muros de doble piel)


@dataclass
class CurvatureInfo:
    curved: bool
    kind: str                       # "flat" | "single" | "double"
    bend_radius_m: Optional[float]
    principal_dir: Vec3             # dirección de máxima curvatura (para orientar kerf)
    normal_spread_deg: float


@dataclass
class UnrolledPanel:
    width_m: float
    height_m: float
    edges: List[Edge2D]
    kind: str
    bend_radius_m: Optional[float]


def build_curvature_map(groups, faces) -> dict:
    """Metadata de curvatura por grupo NO descartado (contrato §2.1).

    Sólo incluye los grupos detectados como curvos. El front la usa para marcar
    componentes curvos y sugerir método (kerf si simple, auxético si doble) + spacing.
    """
    out: dict = {}
    for g in groups:
        if getattr(g, "category", None) == "discard":
            continue
        idxs = getattr(g, "face_indices", None) or []
        gfaces = [faces[i] for i in idxs if 0 <= i < len(faces)]
        if len(gfaces) < MIN_FACES:
            continue
        try:
            info = detect_curvature(gfaces)
        except Exception:
            continue
        if not info.curved:
            continue
        out[str(g.id)] = {
            "curved": True,
            "kind": info.kind,
            "bend_radius_m": info.bend_radius_m,
            "principal_dir": {
                "x": info.principal_dir.x,
                "y": info.principal_dir.y,
                "z": info.principal_dir.z,
            },
            "normal_spread_deg": round(info.normal_spread_deg, 1),
        }
    return out


# ---------------------------------------------------------------------------
# Utilidades numpy
# ---------------------------------------------------------------------------


def _verts_array(faces: List[Face3D]) -> np.ndarray:
    pts = []
    for f in faces:
        for v in f.vertices:
            pts.append((v.x, v.y, v.z))
    return np.asarray(pts, dtype=float)


def _normals_array(faces: List[Face3D]) -> np.ndarray:
    ns = []
    for f in faces:
        n = np.array([f.normal.x, f.normal.y, f.normal.z], dtype=float)
        ln = np.linalg.norm(n)
        if ln > 1e-9:
            ns.append(n / ln)
    return np.asarray(ns, dtype=float) if ns else np.zeros((0, 3))


def _tris(face: Face3D) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fan-triangulación de una cara (soporta polígonos de >3 vértices)."""
    vs = [np.array([v.x, v.y, v.z], dtype=float) for v in face.vertices]
    out = []
    for i in range(1, len(vs) - 1):
        out.append((vs[0], vs[i], vs[i + 1]))
    return out


def _tri_area3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


# ---------------------------------------------------------------------------
# Ajuste de cilindro (curvatura simple)
# ---------------------------------------------------------------------------


def _fit_cylinder(verts: np.ndarray, normals: np.ndarray):
    """Estima el eje del cilindro y el círculo de la sección.

    Devuelve (axis, e_u, e_v, center3d, radius). El eje es la dirección en la que la
    superficie NO curva: las normales son perpendiculares al eje, así que el eje es el
    vector singular más chico de la matriz de normales. `e_u`/`e_v` son la base del plano
    perpendicular al eje donde se ajusta el círculo (Kåsa).
    """
    # Eje = dirección menos representada en las normales.
    _, _, vh = np.linalg.svd(normals - normals.mean(axis=0), full_matrices=False)
    axis = vh[-1]
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    # Base ortonormal del plano perpendicular al eje.
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e_u = np.cross(axis, ref)
    e_u /= np.linalg.norm(e_u) + 1e-12
    e_v = np.cross(axis, e_u)
    e_v /= np.linalg.norm(e_v) + 1e-12

    center = verts.mean(axis=0)
    rel = verts - center
    xu = rel @ e_u
    xv = rel @ e_v

    # Ajuste algebraico de círculo (Kåsa): x^2+y^2 + D x + E y + F = 0.
    A = np.column_stack([xu, xv, np.ones_like(xu)])
    b = -(xu ** 2 + xv ** 2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = sol
    cu, cv = -D / 2.0, -E / 2.0
    radius = math.sqrt(max(cu * cu + cv * cv - F, 1e-9))
    circle_center3d = center + cu * e_u + cv * e_v
    return axis, e_u, e_v, circle_center3d, radius


def _cylinder_residual(verts: np.ndarray, axis, e_u, e_v, circle_center3d, radius) -> float:
    """RMS ABSOLUTO (metros) del ajuste cilíndrico: |dist_al_eje − radio|."""
    rel = verts - circle_center3d
    pu = rel @ e_u
    pv = rel @ e_v
    dist = np.sqrt(pu ** 2 + pv ** 2)
    return float(np.sqrt(np.mean((dist - radius) ** 2)))


def _cylinder_sweep_deg(verts: np.ndarray, axis, e_u, e_v, circle_center3d) -> float:
    """Barrido angular (grados) de los vértices alrededor del eje."""
    rel = verts - circle_center3d
    theta = np.arctan2(rel @ e_v, rel @ e_u)
    m = np.arctan2(np.sin(theta).sum(), np.cos(theta).sum())
    d = np.arctan2(np.sin(theta - m), np.cos(theta - m))
    return float(np.degrees(d.max() - d.min()))


def _face_centroids_normals(faces: List[Face3D]):
    cs, ns = [], []
    for f in faces:
        n = np.array([f.normal.x, f.normal.y, f.normal.z], dtype=float)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        c = np.mean([[v.x, v.y, v.z] for v in f.vertices], axis=0)
        cs.append(c)
        ns.append(n / ln)
    return np.asarray(cs), np.asarray(ns)


def _radial_consistency(centroids, normals, axis, center) -> Tuple[float, float]:
    """¿Las normales apuntan radialmente y todas hacia el mismo lado?

    Devuelve (align, sidedness): align = |dot(normal, radial)| medio (1 = radial puro);
    sidedness = |media del signo| (1 = todas afuera o todas adentro; 0 = doble piel).
    """
    rel = centroids - center
    if axis is not None:
        rel = rel - np.outer(rel @ axis, axis)   # componente perpendicular al eje
    norms = np.linalg.norm(rel, axis=1)
    ok = norms > 1e-9
    if ok.sum() < 3:
        return 0.0, 0.0
    radial = rel[ok] / norms[ok, None]
    d = np.sum(normals[ok] * radial, axis=1)
    align = float(np.mean(np.abs(d)))
    strong = d[np.abs(d) > 0.3]
    sidedness = float(abs(np.mean(np.sign(strong)))) if len(strong) else 0.0
    return align, sidedness


def _fit_sphere(verts: np.ndarray):
    """Ajuste algebraico de esfera. Devuelve (center, radius, residuo_rms_relativo)."""
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    A = np.column_stack([x, y, z, np.ones_like(x)])
    b = -(x ** 2 + y ** 2 + z ** 2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F, G = sol
    center = np.array([-D / 2.0, -E / 2.0, -F / 2.0])
    radius = math.sqrt(max(center @ center - G, 1e-9))
    dist = np.linalg.norm(verts - center, axis=1)
    res = float(np.sqrt(np.mean((dist - radius) ** 2)))   # RMS ABSOLUTO (metros)
    return center, radius, res


# ---------------------------------------------------------------------------
# Detección
# ---------------------------------------------------------------------------


def detect_curvature(faces: List[Face3D]) -> CurvatureInfo:
    """Mide la curvatura de un conjunto de caras (una superficie candidata)."""
    normals = _normals_array(faces)
    if len(faces) < MIN_FACES or normals.shape[0] < MIN_FACES:
        return CurvatureInfo(False, "flat", None, Vec3(1, 0, 0), 0.0)

    # Normal media ~ representativa; dispersión angular respecto a ella.
    mean_n = normals.mean(axis=0)
    mean_n /= np.linalg.norm(mean_n) + 1e-12
    dots = np.clip(normals @ mean_n, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    spread = float(np.percentile(angles, 95))

    if spread < FLAT_SPREAD_DEG:
        return CurvatureInfo(False, "flat", None, Vec3(1, 0, 0), spread)

    verts = _verts_array(faces)
    # Tamaño del panel (diagonal del bbox de vértices): escala para normalizar residuos.
    extent = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    if extent < 1e-6:
        return CurvatureInfo(False, "flat", None, Vec3(1, 0, 0), spread)
    tol = FIT_TOL * extent

    # Discriminador principal: ¿los vértices encajan en un cilindro o una esfera con
    # residuo bajo RELATIVO AL TAMAÑO DEL PANEL, y con radio razonable? Un grupo jagged
    # (normales opuestas, esquina, L) NO encaja y se descarta como "no curvo suave".
    centroids, fnormals = _face_centroids_normals(faces)

    res_cyl = float("inf")
    radius_cyl: Optional[float] = None
    principal = Vec3(1, 0, 0)
    sweep = 0.0
    cyl_align = cyl_side = 0.0
    try:
        axis, e_u, e_v, cc3d, r = _fit_cylinder(verts, normals)
        res_cyl = _cylinder_residual(verts, axis, e_u, e_v, cc3d, r)
        sweep = _cylinder_sweep_deg(verts, axis, e_u, e_v, cc3d)
        radius_cyl = float(r)
        principal = Vec3(float(e_u[0]), float(e_u[1]), float(e_u[2]))
        cyl_align, cyl_side = _radial_consistency(centroids, fnormals, axis, cc3d)
    except Exception:
        pass

    res_sph = float("inf")
    radius_sph: Optional[float] = None
    sph_align = sph_side = 0.0
    try:
        c_sph, rs, res_sph = _fit_sphere(verts)
        radius_sph = float(rs)
        sph_align, sph_side = _radial_consistency(centroids, fnormals, None, c_sph)
    except Exception:
        pass

    max_radius = MAX_RADIUS_FACTOR * extent
    # Una superficie curva REAL: buen ajuste, radio razonable, barrido angular real, y
    # normales radiales apuntando todas al mismo lado (rechaza muros de doble piel y
    # grupos jagged que encajan flojamente en un cilindro/esfera grande).
    single_ok = (
        res_cyl < tol
        and sweep >= MIN_SWEEP_DEG
        and radius_cyl is not None
        and radius_cyl < max_radius
        and cyl_align >= MIN_RADIAL_ALIGN
        and cyl_side >= MIN_SIDEDNESS
    )
    double_ok = (
        res_sph < tol
        and radius_sph is not None
        and radius_sph < max_radius
        and sph_align >= MIN_RADIAL_ALIGN
        and sph_side >= MIN_SIDEDNESS
    )

    if single_ok and res_cyl <= res_sph:
        return CurvatureInfo(True, "single", radius_cyl, principal, spread)
    if double_ok:
        return CurvatureInfo(True, "double", radius_sph, principal, spread)
    if single_ok:
        return CurvatureInfo(True, "single", radius_cyl, principal, spread)

    # No encaja en ninguna primitiva suave → no es una superficie curva desarrollable.
    return CurvatureInfo(False, "flat", None, Vec3(1, 0, 0), spread)


# ---------------------------------------------------------------------------
# Desarrollo (unroll / flatten) → panel plano
# ---------------------------------------------------------------------------


def _polygon_to_edges(geom) -> List[Edge2D]:
    edges: List[Edge2D] = []
    if geom.is_empty:
        return edges
    if geom.geom_type == "Polygon":
        polys = [geom]
    elif geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
    else:
        return edges
    for p in polys:
        ext = list(p.exterior.coords)
        for i in range(len(ext) - 1):
            edges.append(Edge2D(a=Vec2(*ext[i]), b=Vec2(*ext[i + 1])))
        for interior in p.interiors:
            ic = list(interior.coords)
            for i in range(len(ic) - 1):
                edges.append(Edge2D(a=Vec2(*ic[i]), b=Vec2(*ic[i + 1]), hole=True))
    return edges


def _panel_from_uv_triangles(tris_uv: List[Tuple[Tuple[float, float], ...]]) -> Optional[UnrolledPanel]:
    polys = []
    for t in tris_uv:
        try:
            p = Polygon(t)
            if not p.is_valid:
                p = make_valid(p)
            if p.area > 1e-9:
                polys.append(p)
        except Exception:
            continue
    if not polys:
        return None
    outline = unary_union(polys)
    if outline.is_empty:
        return None
    if isinstance(outline, MultiPolygon):
        outline = max(outline.geoms, key=lambda g: g.area)
    minx, miny, maxx, maxy = outline.bounds
    outline = _shift(outline, -minx, -miny)
    edges = _polygon_to_edges(outline)
    w, h = maxx - minx, maxy - miny
    if w < 0.01 or h < 0.01:
        return None
    return UnrolledPanel(width_m=w, height_m=h, edges=edges, kind="", bend_radius_m=None)


def _shift(geom, dx: float, dy: float):
    from shapely import affinity

    return affinity.translate(geom, xoff=dx, yoff=dy)


def unroll(faces: List[Face3D], info: Optional[CurvatureInfo] = None) -> Optional[UnrolledPanel]:
    """Desarrolla la superficie curva a un panel plano (marco local, metros)."""
    if info is None:
        info = detect_curvature(faces)
    if not info.curved:
        return None

    verts = _verts_array(faces)
    normals = _normals_array(faces)
    if verts.shape[0] < 3 or normals.shape[0] < 1:
        return None

    if info.kind == "single":
        panel = _unroll_developable(faces, verts, normals)
    else:
        panel = _flatten_double(faces, verts)

    if panel is not None:
        panel.kind = info.kind
        panel.bend_radius_m = info.bend_radius_m
    return panel


def _unroll_developable(
    faces: List[Face3D], verts: np.ndarray, normals: np.ndarray
) -> Optional[UnrolledPanel]:
    """Cilindro → tira plana. u = R·θ (arco), v = coordenada a lo largo del eje."""
    try:
        axis, e_u, e_v, center3d, radius = _fit_cylinder(verts, normals)
    except Exception:
        return None
    if radius <= 1e-6 or radius > 1e4:
        return None

    # Ángulo medio del parche: fija el corte de rama 180° opuesto para que ningún
    # triángulo cruce el salto ±π (válido para un parche simplemente conexo de <2π).
    rel_all = verts - center3d
    mean_theta = math.atan2(float((rel_all @ e_v).sum()), float((rel_all @ e_u).sum()))

    def uv(p: np.ndarray) -> Tuple[float, float]:
        rel = p - center3d
        t = float(rel @ axis)                       # posición a lo largo del eje
        pu = float(rel @ e_u)
        pv = float(rel @ e_v)
        theta = math.atan2(pv, pu) - mean_theta      # centrado en el parche
        theta = math.atan2(math.sin(theta), math.cos(theta))  # normaliza a (-π,π]
        return (radius * theta, t)                   # (arco, eje)

    tris_uv: List[Tuple[Tuple[float, float], ...]] = []
    for f in faces:
        for (a, b, c) in _tris(f):
            tris_uv.append((uv(a), uv(b), uv(c)))
    return _panel_from_uv_triangles(tris_uv)


def _flatten_double(faces: List[Face3D], verts: np.ndarray) -> Optional[UnrolledPanel]:
    """Doble curvatura → proyección al plano PCA, escalada para preservar área total.

    Aproximado (no isométrico): el patrón auxético absorbe la diferencia al expandirse.
    """
    center = verts.mean(axis=0)
    _, _, vh = np.linalg.svd(verts - center, full_matrices=False)
    e0, e1 = vh[0], vh[1]

    area3d = 0.0
    tris_uv: List[Tuple[Tuple[float, float], ...]] = []
    for f in faces:
        for (a, b, c) in _tris(f):
            area3d += _tri_area3d(a, b, c)
            tri = tuple(
                (float((p - center) @ e0), float((p - center) @ e1)) for p in (a, b, c)
            )
            tris_uv.append(tri)

    panel = _panel_from_uv_triangles(tris_uv)
    if panel is None:
        return None

    # Escala uniforme para que el área 2D iguale el área 3D (preserva superficie total).
    area2d = 0.0
    for t in tris_uv:
        (x0, y0), (x1, y1), (x2, y2) = t
        area2d += abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2.0
    if area2d > 1e-9 and area3d > 1e-9:
        scale = math.sqrt(area3d / area2d)
        if 0.5 < scale < 3.0:
            panel.edges = [
                Edge2D(
                    a=Vec2(e.a.x * scale, e.a.y * scale),
                    b=Vec2(e.b.x * scale, e.b.y * scale),
                    hole=e.hole,
                )
                for e in panel.edges
            ]
            panel.width_m *= scale
            panel.height_m *= scale
    return panel
