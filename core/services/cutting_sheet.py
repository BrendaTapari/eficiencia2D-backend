import math
from typing import List, Dict, Set, Optional, Tuple, Literal
from dataclasses import dataclass, field
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# Importamos nuestros tipos y servicios
from core.services.types import (
    Face3D,
    Vec2,
    Vec3,
    cross,
    dot,
    get_vertex_indices,
    normalize,
    sub,
    vlength,
)
from core.services.floor_plan_extractor import detect_floor_levels
from core.services.wall_pairing import are_thin_twins, TwinCandidate
from core.services.sheet_nester import NestingResult, rotate_edges

# ============================================================================
# Cutting Sheet — Plancha de Corte
#
# Descompone el modelo 3D en componentes estructurales individuales usando
# agrupamiento por coplanaridad geométrica.
# ============================================================================

GAP_M = 0.003  # 3mm de brecha para corte láser
SHEET_SPACING_M = 0.10  # Brecha visual entre planchas en el DXF
NORMAL_CLUSTER_DOT = 0.85  # Caras con dot > esto son de la "misma dirección"
NEAR_PARALLEL_EPS = 0.01  # Tolerancia cercana a cero para producto cruzado
THIN_TWIN_THRESHOLD = 0.40  # Fusionar grupos coplanares gemelos más cercanos que esto

UpAxis = Literal["Y", "Z"]
PanelCategory = Literal["wall", "floor"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_up(v: Vec3, up: UpAxis) -> float:
    return v.y if up == "Y" else v.z


def get_up_vec(up: UpAxis) -> Vec3:
    return Vec3(0, 1, 0) if up == "Y" else Vec3(0, 0, 1)


def face_area(face: Face3D) -> float:
    verts = face.vertices
    if len(verts) < 3:
        return 0.0
    sx, sy, sz = 0.0, 0.0, 0.0
    for i in range(1, len(verts) - 1):
        e1 = sub(verts[i], verts[0])
        e2 = sub(verts[i + 1], verts[0])
        c = cross(e1, e2)
        sx += c.x
        sy += c.y
        sz += c.z
    return 0.5 * math.sqrt(sx**2 + sy**2 + sz**2)


def snap(v: float) -> float:
    return round(v * 100) / 100.0


def vert_key(x: float, y: float) -> str:
    return f"{snap(x)},{snap(y)}"


def edge_key(ax: float, ay: float, bx: float, by: float) -> str:
    a = vert_key(ax, ay)
    b = vert_key(bx, by)
    return f"{a}|{b}" if a < b else f"{b}|{a}"


def snap3(v: float) -> float:
    return round(v * 100) / 100.0


def r_str(n: float) -> str:
    # Evitar notación científica y limitar a 4 decimales
    return f"{n:.4f}".rstrip("0").rstrip(".") if "." in f"{n:.4f}" else f"{n:.4f}"


MIN_HOLE_AREA = 0.0025


def ring_2d_area(ring: List[Tuple[float, float]]) -> float:
    """Área con signo (shoelace) de un anillo, ABIERTO o CERRADO.

    Antes se sumaba sólo hasta len(ring)-1, omitiendo el término que cierra el
    polígono. Con los anillos de shapely no se notaba (traen el primer vértice
    repetido al final), pero los triángulos crudos del OBJ vienen abiertos y su área
    salía mal: en algunos casos los términos se cancelaban y daba exactamente 0, así
    que union_outline los descartaba por "degenerados" y esa porción de pared quedaba
    como un hueco con forma de cuña en la plancha.
    """
    n = len(ring)
    if n < 3:
        return 0.0
    # Si el anillo ya viene cerrado (shapely), ignorar el vértice duplicado final.
    if abs(ring[0][0] - ring[-1][0]) < 1e-12 and abs(ring[0][1] - ring[-1][1]) < 1e-12:
        n -= 1
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += ring[i][0] * ring[j][1] - ring[j][0] * ring[i][1]
    return a / 2.0


def sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


# ---------------------------------------------------------------------------
# Tipos de Paneles
# ---------------------------------------------------------------------------


@dataclass
class Edge2D:
    a: Vec2
    b: Vec2
    hole: bool = False  # True si la arista pertenece a un anillo interior (abertura)
    joint: bool = False  # True si es una línea de encastre (junta transversal 3D)
    score: bool = False  # True si es una línea de pliegue/score (corte manual tipo "line")
    flex: bool = False  # True si es un corte del patrón de flexión (kerf / auxético)


@dataclass
class Panel:
    id: str
    group_name: str
    category: PanelCategory
    floor_index: int
    width_m: float
    height_m: float
    edges: List[Edge2D]
    source_group_id: int
    is_mark: bool = False  # True si las aberturas de este panel se graban (no se cortan)
    # Marco de proyección 3D de la pieza YA RECORTADA (mismas claves que
    # `compute_group_placement`). El instructivo dibujaba `build_placements`, que es la
    # proyección CRUDA del modelo: 20 de 28 piezas de un modelo real se mostraban más
    # grandes de lo que se cortan (dos muros, 55 cm de más), así que en el instructivo se
    # pisaban unas con otras aunque en la plancha encastraran bien.
    frame: Optional[Dict] = None
    # Ids de los grupos vecinos cuya ranura quedó REALMENTE cortada en esta pieza. No es
    # lo mismo que `plate_joints`: ahí hay ranuras que después se descartan porque su
    # segmento cae fuera del panel ya recortado (el tope resolvió la junta y la ranura
    # sobra). Quien verifique el ensamble tiene que mirar esto y no la lista de juntas, o
    # va a excusar un choque por una ranura que no existe.
    slots_against: List[int] = field(default_factory=list)


@dataclass
class PlacedPanel:
    panel: Panel
    x: float
    y: float


@dataclass
class CoplanarGroup:
    normal: Vec3
    d: float
    faces: List[Face3D]
    total_area: float
    category: PanelCategory


@dataclass
class RawEdge:
    ax: float
    ay: float
    bx: float
    by: float
    via: Optional[int] = None
    vib: Optional[int] = None
    hole: bool = False  # True si proviene de un anillo interior (abertura)


# ---------------------------------------------------------------------------
# Agrupamiento por Coplanaridad
# ---------------------------------------------------------------------------


def cluster_by_coplanarity(faces: List[Face3D], up: UpAxis) -> List[CoplanarGroup]:
    groups: List[CoplanarGroup] = []
    D_TOLERANCE = 0.15

    for face in faces:
        n = face.normal
        if vlength(n) < 0.01:
            continue
        area = face_area(face)
        if area < 1e-6:
            continue

        d = dot(n, face.vertices[0])
        up_comp = abs(get_up(n, up))
        category: PanelCategory = "floor" if up_comp > 0.75 else "wall"

        placed = False
        for group in groups:
            if (
                group.category == category
                and abs(dot(n, group.normal)) > NORMAL_CLUSTER_DOT
                and abs(d - group.d) < D_TOLERANCE
            ):
                group.faces.append(face)
                group.total_area += area
                placed = True
                break

        if not placed:
            groups.append(
                CoplanarGroup(
                    normal=normalize(n),
                    d=d,
                    faces=[face],
                    total_area=area,
                    category=category,
                )
            )

    return groups


def split_connected_components(faces: List[Face3D]) -> List[List[Face3D]]:
    if len(faces) <= 1:
        return [faces]

    vert_to_faces: Dict[str, List[int]] = {}
    for fi, face in enumerate(faces):
        indices = get_vertex_indices(face)
        if indices:
            for vi in indices:
                key = str(vi)
                if key not in vert_to_faces:
                    vert_to_faces[key] = []
                vert_to_faces[key].append(fi)
        else:
            for v in face.vertices:
                key = f"{snap3(v.x)},{snap3(v.y)},{snap3(v.z)}"
                if key not in vert_to_faces:
                    vert_to_faces[key] = []
                vert_to_faces[key].append(fi)

    parent = list(range(len(faces)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for face_indices in vert_to_faces.values():
        for i in range(1, len(face_indices)):
            union(face_indices[0], face_indices[i])

    comp_map: Dict[int, List[Face3D]] = {}
    for fi, face in enumerate(faces):
        root = find(fi)
        if root not in comp_map:
            comp_map[root] = []
        comp_map[root].append(face)

    return list(comp_map.values())


# ---------------------------------------------------------------------------
# Rastreo de Contornos (Half-Edge Dart Traversal)
# ---------------------------------------------------------------------------


def trace_contours(boundary_edges: List[RawEdge]) -> List[RawEdge]:
    if len(boundary_edges) <= 2:
        return boundary_edges

    def vert_id(e: RawEdge, side: str) -> str:
        if side == "a":
            return f"i{e.via}" if e.via is not None else vert_key(e.ax, e.ay)
        return f"i{e.vib}" if e.vib is not None else vert_key(e.bx, e.by)

    adj: Dict[str, List[int]] = {}
    for i, e in enumerate(boundary_edges):
        for side in ("a", "b"):
            vk = vert_id(e, side)
            if vk not in adj:
                adj[vk] = []
            adj[vk].append(i)

    removed: Set[int] = set()
    changed = True
    while changed:
        changed = False
        for vk, indices in list(adj.items()):
            live = [i for i in indices if i not in removed]
            if len(live) == 1:
                removed.add(live[0])
                changed = True
            adj[vk] = [] if len(live) <= 1 else live

    vert_coord: Dict[str, Vec2] = {}
    for e in boundary_edges:
        ak, bk = vert_id(e, "a"), vert_id(e, "b")
        if ak not in vert_coord:
            vert_coord[ak] = Vec2(e.ax, e.ay)
        if bk not in vert_coord:
            vert_coord[bk] = Vec2(e.bx, e.by)

    live_edges = [i for i in range(len(boundary_edges)) if i not in removed]
    if not live_edges:
        return []

    def dart_from(dart: int) -> str:
        ei = dart >> 1
        return vert_id(boundary_edges[ei], "a" if (dart & 1) == 0 else "b")

    def dart_to(dart: int) -> str:
        ei = dart >> 1
        return vert_id(boundary_edges[ei], "b" if (dart & 1) == 0 else "a")

    def dart_angle(dart: int) -> float:
        frm, to = vert_coord[dart_from(dart)], vert_coord[dart_to(dart)]
        return math.atan2(to.y - frm.y, to.x - frm.x)

    outgoing: Dict[str, List[int]] = {}
    for ei in live_edges:
        for dir_bit in (0, 1):
            dart = ei * 2 + dir_bit
            frm = dart_from(dart)
            if frm not in outgoing:
                outgoing[frm] = []
            outgoing[frm].append(dart)

    for arr in outgoing.values():
        arr.sort(key=dart_angle)

    def next_dart(dart: int) -> int:
        w = dart_to(dart)
        arr = outgoing[w]
        twin = (dart >> 1) * 2 + 1 if (dart & 1) == 0 else (dart >> 1) * 2
        idx = arr.index(twin)
        prev = (idx - 1 + len(arr)) % len(arr)
        return arr[prev]

    dart_face: Dict[int, int] = {}
    face_sign: List[int] = []
    face_id = 0

    for ei in live_edges:
        for dir_bit in (0, 1):
            start = ei * 2 + dir_bit
            if start in dart_face:
                continue

            loop_darts: List[int] = []
            d = start
            guard = 0
            limit = len(live_edges) * 2 + 4
            while guard < limit:
                if d in dart_face:
                    break
                dart_face[d] = face_id
                loop_darts.append(d)
                d = next_dart(d)
                if d == start:
                    break
                guard += 1

            area2 = 0.0
            for dd in loop_darts:
                frm, to = vert_coord[dart_from(dd)], vert_coord[dart_to(dd)]
                area2 += frm.x * to.y - to.x * frm.y
            face_sign.append(sign(area2))
            face_id += 1

    kept: Set[int] = set()
    for ei in live_edges:
        f0 = dart_face.get(ei * 2)
        f1 = dart_face.get(ei * 2 + 1)
        if f0 is None or f1 is None:
            continue
        s0, s1 = face_sign[f0], face_sign[f1]
        if s0 != 0 and s1 != 0 and s0 != s1:
            kept.add(ei)

    return [e for i, e in enumerate(boundary_edges) if i in kept]


# ---------------------------------------------------------------------------
# Boolean Union & Proyecciones (Reemplazo de polyclip-ts por Shapely)
# ---------------------------------------------------------------------------

UNION_SNAP = 1e4


def snap_union(v: float) -> float:
    return round(v * UNION_SNAP) / UNION_SNAP


def union_outline(
    faces: List[Face3D], u_axis: Vec3, v_axis: Vec3
) -> Optional[List[RawEdge]]:
    polys: List[Polygon] = []
    for face in faces:
        ring = [
            (snap_union(dot(v, u_axis)), snap_union(dot(v, v_axis)))
            for v in face.vertices
        ]
        if len(ring) < 3:
            continue
        if abs(ring_2d_area(ring)) < 1e-7:
            continue

        # Shapely requiere anillos cerrados, pero asume cierre si se lo damos en orden
        polys.append(Polygon(ring))

    if not polys:
        return None

    try:
        merged = unary_union(polys)
    except Exception:
        return None

    out: List[RawEdge] = []

    def process_poly(poly: Polygon):
        # Boundary Exterior
        ext = list(poly.exterior.coords)
        for i in range(len(ext) - 1):
            out.append(
                RawEdge(ax=ext[i][0], ay=ext[i][1], bx=ext[i + 1][0], by=ext[i + 1][1])
            )

        # Huecos Internos (Windows/Doors) -> hole=True (aberturas)
        for interior in poly.interiors:
            int_coords = list(interior.coords)
            if abs(ring_2d_area(int_coords)) < MIN_HOLE_AREA:
                continue
            for i in range(len(int_coords) - 1):
                out.append(
                    RawEdge(
                        ax=int_coords[i][0],
                        ay=int_coords[i][1],
                        bx=int_coords[i + 1][0],
                        by=int_coords[i + 1][1],
                        hole=True,
                    )
                )

    if isinstance(merged, MultiPolygon):
        for p in merged.geoms:
            process_poly(p)
    elif isinstance(merged, Polygon):
        process_poly(merged)
    else:
        return None

    return out if out else None


def legacy_boundary(faces: List[Face3D], u_axis: Vec3, v_axis: Vec3) -> List[RawEdge]:
    edge_face_count: Dict[str, int] = {}
    edge_coords: Dict[str, RawEdge] = {}

    for face in faces:
        pts = [Vec2(dot(v, u_axis), dot(v, v_axis)) for v in face.vertices]
        vi = get_vertex_indices(face)
        n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            key = (
                f"{vi[i]}|{vi[j]}"
                if (vi and vi[i] < vi[j])
                else (
                    f"{vi[j]}|{vi[i]}"
                    if vi
                    else edge_key(pts[i].x, pts[i].y, pts[j].x, pts[j].y)
                )
            )

            edge_face_count[key] = edge_face_count.get(key, 0) + 1
            if key not in edge_coords:
                edge_coords[key] = RawEdge(
                    ax=pts[i].x,
                    ay=pts[i].y,
                    bx=pts[j].x,
                    by=pts[j].y,
                    via=vi[i] if vi else None,
                    vib=vi[j] if vi else None,
                )

    boundary_edges = [
        edge_coords[key] for key, count in edge_face_count.items() if count == 1
    ]
    if not boundary_edges:
        return []
    return trace_contours(boundary_edges)


@dataclass
class ProjectedFace:
    width_m: float
    height_m: float
    edges: List[Edge2D]
    v_up: float
    u_axis: Vec3
    v_axis: Vec3
    origin_u: float
    origin_v: float


def orient_group_normals_outward(groups) -> dict:
    """Normal canónica hacia AFUERA del edificio por grupo: {group_id: Vec3}.

    El espejado de la plancha (quemado del lado interior) y el instructivo asumen que
    representative_normal apunta hacia afuera, pero el clasificador la toma tal cual
    viene del archivo: un modelo con normales invertidas dejaría esa pieza espejada al
    revés (quemado exterior). Regla determinística:
      - techos/pisos (|n.y| ≥ 0.5)  → n.y > 0 (hacia el cielo; el espejo dibuja la cara
        de abajo/interior, que es la que se quema)
      - muros                        → alejándose del centroide horizontal del edificio
        (tabiques interiores equidistantes: se deja como viene, cualquier cara es interior)
    No muta los grupos; sólo devuelve la normal a usar al proyectar.
    """
    out: dict = {}
    considered = [g for g in groups if getattr(g, "category", None) != "discard"]
    if not considered:
        considered = list(groups)

    total = sum(g.total_area for g in considered) or 1.0
    bx = sum(g.centroid.x * g.total_area for g in considered) / total
    bz = sum(g.centroid.z * g.total_area for g in considered) / total

    for g in groups:
        n = g.representative_normal
        if abs(n.y) >= 0.5:
            out[g.id] = n if n.y > 0 else Vec3(-n.x, -n.y, -n.z)
            continue
        dx, dz = g.centroid.x - bx, g.centroid.z - bz
        d = dx * n.x + dz * n.z
        # d≈0: muro que pasa por el centro (tabique) → orientación indistinta, se respeta
        out[g.id] = Vec3(-n.x, -n.y, -n.z) if d < -1e-9 else n
    return out


def project_faces_to_2d(
    faces: List[Face3D], group_normal: Vec3, up: UpAxis
) -> Optional[ProjectedFace]:
    if not faces:
        return None

    world_up = get_up_vec(up)
    u_axis = normalize(cross(world_up, group_normal))

    if vlength(u_axis) < NEAR_PARALLEL_EPS:
        u_axis = Vec3(1, 0, 0)
        v_axis = normalize(cross(group_normal, u_axis))
        if vlength(v_axis) < NEAR_PARALLEL_EPS:
            v_axis = Vec3(0, 0, 1) if up == "Y" else Vec3(0, 1, 0)
    else:
        v_axis = normalize(cross(group_normal, u_axis))

    contoured = union_outline(faces, u_axis, v_axis) or legacy_boundary(
        faces, u_axis, v_axis
    )
    if not contoured:
        return None

    min_u, max_u = float("inf"), float("-inf")
    min_v, max_v = float("inf"), float("-inf")

    for e in contoured:
        min_u = min(min_u, e.ax, e.bx)
        max_u = max(max_u, e.ax, e.bx)
        min_v = min(min_v, e.ay, e.by)
        max_v = max(max_v, e.ay, e.by)

    w, h = max_u - min_u, max_v - min_v
    if w < 0.01 or h < 0.01:
        return None

    edges = [
        Edge2D(
            a=Vec2(e.ax - min_u, e.ay - min_v),
            b=Vec2(e.bx - min_u, e.by - min_v),
            hole=e.hole,
        )
        for e in contoured
    ]

    return ProjectedFace(
        width_m=w,
        height_m=h,
        edges=edges,
        v_up=dot(v_axis, world_up),
        u_axis=u_axis,
        v_axis=v_axis,
        origin_u=min_u,
        origin_v=min_v,
    )


# ---------------------------------------------------------------------------
# Marco de proyección 3D por pieza (instructivo de armado)
#
# Permite al front mapear un punto del panel local (u, v) en metros de vuelta a 3D:
#     world = origin + u·u_axis + v·v_axis
# `origin` es el punto 3D que proyecta a panel (0,0). {u_axis, v_axis, normal} es una
# base ortonormal (la misma de project_faces_to_2d). `mirrored=True` avisa que el
# contorno de corte viene espejado horizontalmente (u_local = width_m − u).
# ---------------------------------------------------------------------------


def compute_group_placement(
    faces: List[Face3D], group_normal: Vec3, up: UpAxis = "Y"
) -> Optional[Dict]:
    res = project_faces_to_2d(faces, group_normal, up)
    if not res:
        return None

    n = normalize(group_normal)
    u, v = res.u_axis, res.v_axis

    # d = desplazamiento del plano (n·X = d). Es el PUNTO MEDIO entre las dos pieles, la
    # misma definición que usan el recorte y la ranura (plate_intersect.mid_plane_offset).
    # Antes promediaba los vértices, que se sesga hacia la piel más teselada: el
    # instructivo dibujaba la placa desplazada respecto del plano donde realmente va.
    from core.services.plate_intersect import mid_plane_offset_faces
    d = mid_plane_offset_faces(faces, n)

    ou, ov = res.origin_u, res.origin_v
    origin = Vec3(
        ou * u.x + ov * v.x + d * n.x,
        ou * u.y + ov * v.y + d * n.y,
        ou * u.z + ov * v.z + d * n.z,
    )

    def vec(w: Vec3) -> Dict:
        return {"x": w.x, "y": w.y, "z": w.z}

    return {
        "origin": vec(origin),
        "u_axis": vec(u),
        "v_axis": vec(v),
        "normal": vec(n),
        "width_m": res.width_m,
        "height_m": res.height_m,
        "mirrored": True,
    }


def shift_placement(placement: Dict, off_u: float, off_v: float,
                    width_m: float, height_m: float, area_m2: float) -> Dict:
    """Mismo marco, corrido a la pieza YA RECORTADA.

    Los recortes de ensamble re-normalizan el panel a un origen nuevo (`offset_u/v`
    acumulados). El marco crudo sigue apuntando al origen viejo, así que hay que
    trasladarlo por esos offsets y reemplazar las medidas por las finales. Sin esto el
    instructivo dibuja la pieza sin recortar y en su posición original.
    """
    o = placement["origin"]
    u = placement["u_axis"]
    v = placement["v_axis"]
    out = dict(placement)
    # 1) Al marco crudo se le aplican los offsets de los recortes, que están en el marco
    #    de la PROYECCIÓN (antes del espejado).
    ox = o["x"] + off_u * u["x"] + off_v * v["x"]
    oy = o["y"] + off_u * u["y"] + off_v * v["y"]
    oz = o["z"] + off_u * u["z"] + off_v * v["z"]
    # 2) El contorno del panel se ESPEJA antes de cortarse (mirror_edges_horizontal), para
    #    que el quemado del láser quede del lado interior de la maqueta. O sea que las
    #    coordenadas `u` de las aristas son `ancho - u_proyección`, y mapearlas al mundo
    #    con `origin + u·u_axis` deja cada pieza asimétrica dada vuelta sobre su propio
    #    eje: medido en un modelo real, hasta 2.49 m de desvío contra los 0.37 m de la
    #    fórmula correcta. El espejado se hornea acá, en el marco: se corre el origen al
    #    otro extremo y se invierte `u_axis`. Así `world = origin + u·u_axis + v·v_axis`
    #    vale TAL CUAL para las coordenadas del panel, sin que el consumidor compense
    #    nada — y por eso `mirrored` queda en False.
    out["origin"] = {
        "x": ox + width_m * u["x"],
        "y": oy + width_m * u["y"],
        "z": oz + width_m * u["z"],
    }
    out["u_axis"] = {"x": -u["x"], "y": -u["y"], "z": -u["z"]}
    out["mirrored"] = False
    out["width_m"] = width_m
    out["height_m"] = height_m
    # Área de la PIEZA que se corta (contorno menos aberturas). El front venía mostrando
    # `GeometryGroup.total_area`, que es la suma de las caras de la malla: para un sólido
    # cuenta las dos pieles y para una chapa de una sola cara, una. Dos piezas casi
    # iguales se veían con áreas muy distintas (55.91 m² contra 46.65 m²).
    out["area_m2"] = area_m2
    return out


def panel_cut_area_m2(edges: List[Edge2D], width_m: float, height_m: float) -> float:
    """Área de MATERIAL de la pieza (contorno menos aberturas), en m² de edificio.

    Se polygoniza la sopa de aristas y se aplica la regla PAR-IMPAR: una región contenida
    en un número par de otras es material, en un número impar es hueco. No sirve quedarse
    con el polígono más grande y restarle lo de adentro (que es lo que hace
    `_edges_to_polygon`, pensado para un contorno único): una pieza puede tener VARIAS
    regiones sueltas, y así se perdían todas menos la mayor. Medido en un muro real de
    7.74 m² de malla: se reportaban 1.87.

    Si el polígono no se puede reconstruir se cae al bounding box, que es una cota
    superior honesta."""
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union

    segs = [
        LineString([(e.a.x, e.a.y), (e.b.x, e.b.y)])
        for e in edges
        if not getattr(e, "flex", False) and not getattr(e, "score", False)
    ]
    if not segs:
        return width_m * height_m
    caras = sorted(polygonize(unary_union(segs)), key=lambda p: p.area, reverse=True)
    if not caras:
        return width_m * height_m
    area = 0.0
    for i, c in enumerate(caras):
        p = c.representative_point()
        contenida_en = sum(1 for j, o in enumerate(caras) if j != i and o.area > c.area and o.contains(p))
        if contenida_en % 2 == 0:
            area += c.area
    return area if area > 1e-9 else width_m * height_m


def build_placements(groups, faces: List[Face3D], up: UpAxis = "Y") -> Dict[str, Dict]:
    """Marco de proyección 3D por grupo (no-discard). group_id (str) -> placement."""
    out: Dict[str, Dict] = {}
    # Misma normal canónica (hacia afuera) que usa la plancha → el instructivo y el
    # corte comparten marco aunque el archivo traiga normales invertidas.
    oriented = orient_group_normals_outward(groups)
    for g in groups:
        if getattr(g, "category", None) == "discard":
            continue
        gfaces = [faces[fi] for fi in g.face_indices if 0 <= fi < len(faces)]
        if not gfaces:
            continue
        pl = compute_group_placement(
            gfaces, oriented.get(g.id, g.representative_normal), up
        )
        if pl:
            out[str(g.id)] = pl
    return out


# ---------------------------------------------------------------------------
# Recortes y Simetría (Assembly Compensation)
# ---------------------------------------------------------------------------


def clip_panel_at_v(
    edges: List[Edge2D], cut: float, keep_above: bool
) -> Optional[Dict]:
    def in_side(y: float) -> bool:
        return y >= cut - 1e-9 if keep_above else y <= cut + 1e-9

    out: List[Edge2D] = []
    crossings: List[float] = []

    for e in edges:
        a_in, b_in = in_side(e.a.y), in_side(e.b.y)
        if a_in and b_in:
            out.append(e)
        elif not a_in and not b_in:
            continue
        else:
            t = (cut - e.a.y) / (e.b.y - e.a.y)
            ix = e.a.x + t * (e.b.x - e.a.x)
            cut_pt = Vec2(ix, cut)
            keep = e.a if a_in else e.b
            out.append(
                Edge2D(a=keep, b=cut_pt, hole=e.hole)
                if a_in
                else Edge2D(a=cut_pt, b=keep, hole=e.hole)
            )
            crossings.append(ix)

    crossings.sort()
    for i in range(0, len(crossings) - 1, 2):
        out.append(Edge2D(a=Vec2(crossings[i], cut), b=Vec2(crossings[i + 1], cut)))

    if len(out) < 3:
        return None

    min_u, min_v = float("inf"), float("inf")
    max_u, max_v = float("-inf"), float("-inf")
    for e in out:
        min_u = min(min_u, e.a.x, e.b.x)
        max_u = max(max_u, e.a.x, e.b.x)
        min_v = min(min_v, e.a.y, e.b.y)
        max_v = max(max_v, e.a.y, e.b.y)

    w, h = max_u - min_u, max_v - min_v
    if w < 0.01 or h < 0.01:
        return None

    normalized = [
        Edge2D(
            a=Vec2(e.a.x - min_u, e.a.y - min_v),
            b=Vec2(e.b.x - min_u, e.b.y - min_v),
            hole=e.hole,
        )
        for e in out
    ]
    # offset_u/offset_v: el recorte re-normaliza el panel a un origen NUEVO. Quien
    # proyecte algo más tarde en el marco original (p. ej. las ranuras de encastre)
    # debe restar este desplazamiento acumulado, o dibujará fuera de la pieza.
    return {
        "width_m": w,
        "height_m": h,
        "edges": normalized,
        "offset_u": min_u,
        "offset_v": min_v,
    }


def clip_panel_at_u(
    edges: List[Edge2D], cut: float, keep_right: bool
) -> Optional[Dict]:
    def in_side(x: float) -> bool:
        return x >= cut - 1e-9 if keep_right else x <= cut + 1e-9

    out: List[Edge2D] = []
    crossings: List[float] = []

    for e in edges:
        a_in, b_in = in_side(e.a.x), in_side(e.b.x)
        if a_in and b_in:
            out.append(e)
        elif not a_in and not b_in:
            continue
        else:
            t = (cut - e.a.x) / (e.b.x - e.a.x)
            iy = e.a.y + t * (e.b.y - e.a.y)
            cut_pt = Vec2(cut, iy)
            keep = e.a if a_in else e.b
            out.append(
                Edge2D(a=keep, b=cut_pt, hole=e.hole)
                if a_in
                else Edge2D(a=cut_pt, b=keep, hole=e.hole)
            )
            crossings.append(iy)

    crossings.sort()
    for i in range(0, len(crossings) - 1, 2):
        out.append(Edge2D(a=Vec2(cut, crossings[i]), b=Vec2(cut, crossings[i + 1])))

    if len(out) < 3:
        return None

    min_u, min_v = float("inf"), float("inf")
    max_u, max_v = float("-inf"), float("-inf")
    for e in out:
        min_u = min(min_u, e.a.x, e.b.x)
        max_u = max(max_u, e.a.x, e.b.x)
        min_v = min(min_v, e.a.y, e.b.y)
        max_v = max(max_v, e.a.y, e.b.y)

    w, h = max_u - min_u, max_v - min_v
    if w < 0.01 or h < 0.01:
        return None

    normalized = [
        Edge2D(
            a=Vec2(e.a.x - min_u, e.a.y - min_v),
            b=Vec2(e.b.x - min_u, e.b.y - min_v),
            hole=e.hole,
        )
        for e in out
    ]
    # offset_u/offset_v: el recorte re-normaliza el panel a un origen NUEVO. Quien
    # proyecte algo más tarde en el marco original (p. ej. las ranuras de encastre)
    # debe restar este desplazamiento acumulado, o dibujará fuera de la pieza.
    return {
        "width_m": w,
        "height_m": h,
        "edges": normalized,
        "offset_u": min_u,
        "offset_v": min_v,
    }


def mirror_edges_horizontal(edges: List[Edge2D], width_m: float) -> List[Edge2D]:
    return [
        Edge2D(
            a=Vec2(width_m - e.a.x, e.a.y),
            b=Vec2(width_m - e.b.x, e.b.y),
            hole=e.hole,
            joint=getattr(e, "joint", False),
            score=getattr(e, "score", False),
        )
        for e in edges
    ]


# ---------------------------------------------------------------------------
# Cortes manuales del usuario (user_cuts) — ver CONTRATO_user_cuts_backend.
#
# Los cortes llegan en el marco local del panel (post project_faces_to_2d,
# normalizado a (0,0), metros), el mismo del front. rect/square/circle se RESTAN
# (boolean difference, pueden partir el panel en varias piezas y dejar huecos);
# line es marca de pliegue/score (no parte el panel, se emite como score).
# ---------------------------------------------------------------------------


def _edges_to_polygon(edges: List[Edge2D]):
    """Reconstruye el polígono del panel (exterior + huecos) desde sus aristas."""
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union

    segs = [
        LineString([(e.a.x, e.a.y), (e.b.x, e.b.y)])
        for e in edges
        if not getattr(e, "score", False)
    ]
    if not segs:
        return None
    faces = list(polygonize(unary_union(segs)))
    if not faces:
        return None
    faces.sort(key=lambda p: p.area, reverse=True)
    outer = faces[0]
    holes = [f for f in faces[1:] if outer.contains(f.representative_point())]
    poly = outer.difference(unary_union(holes)) if holes else outer
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if not poly.is_empty else None


def _polygon_to_edges(poly) -> List[Edge2D]:
    out: List[Edge2D] = []

    def ring(coords, hole: bool):
        for i in range(len(coords) - 1):
            out.append(
                Edge2D(
                    a=Vec2(coords[i][0], coords[i][1]),
                    b=Vec2(coords[i + 1][0], coords[i + 1][1]),
                    hole=hole,
                )
            )

    ring(list(poly.exterior.coords), False)
    for interior in poly.interiors:
        ring(list(interior.coords), True)
    return out


def _cut_polygon(cut: Dict, width_m: float, height_m: float):
    """Polígono sustractivo de un corte rect/square/circle (None para line / degenerado)."""
    from shapely.geometry import Polygon

    kind = cut.get("kind")
    if kind == "line":
        return None
    u0, v0 = float(cut["u0"]), float(cut["v0"])
    u1, v1 = float(cut["u1"]), float(cut["v1"])
    min_u, max_u = min(u0, u1), max(u0, u1)
    min_v, max_v = min(v0, v1), max(v0, v1)

    if kind == "square":
        side = max(max_u - min_u, max_v - min_v)
        max_u, max_v = min_u + side, min_v + side

    if kind == "circle":
        cx, cy = (min_u + max_u) / 2, (min_v + max_v) / 2
        rx, ry = (max_u - min_u) / 2, (max_v - min_v) / 2
        if rx < 0.02 or ry < 0.02:
            return None
        pts = [
            (cx + math.cos(2 * math.pi * i / 32) * rx, cy + math.sin(2 * math.pi * i / 32) * ry)
            for i in range(33)
        ]
        return Polygon(pts)

    # rect / square
    if (max_u - min_u) < 0.02 or (max_v - min_v) < 0.02:
        return None
    min_u, min_v = max(0.0, min_u), max(0.0, min_v)
    max_u, max_v = min(width_m, max_u), min(height_m, max_v)
    if (max_u - min_u) <= 0 or (max_v - min_v) <= 0:
        return None
    return Polygon([(min_u, min_v), (max_u, min_v), (max_u, max_v), (min_u, max_v)])


def apply_user_cuts(
    width_m: float, height_m: float, edges: List[Edge2D], cuts: List[Dict]
) -> Tuple[List[List[Edge2D]], List[Tuple[Vec2, Vec2]]]:
    """
    Aplica los cortes a un panel (marco local, sin espejar). Devuelve:
      - piezas: lista de listas de aristas (contorno+huecos) en el MISMO marco, una
        por pieza resultante (un corte que cruza el panel lo parte en varias).
      - score_lines: segmentos (a,b) de los cortes tipo "line" (marcas de pliegue).
    Si no hay cortes sustractivos efectivos, devuelve el panel original como única pieza.
    """
    from shapely.geometry import MultiPolygon

    score_lines: List[Tuple[Vec2, Vec2]] = [
        (Vec2(float(c["u0"]), float(c["v0"])), Vec2(float(c["u1"]), float(c["v1"])))
        for c in cuts
        if c.get("kind") == "line"
    ]

    subtractive = [c for c in cuts if c.get("kind") != "line"]
    if not subtractive:
        return [edges], score_lines

    panel = _edges_to_polygon(edges)
    if panel is None:
        return [edges], score_lines

    from shapely.ops import unary_union

    geom = panel
    cut_polys = []
    for cut in subtractive:
        cp = _cut_polygon(cut, width_m, height_m)
        if cp is None or not cp.is_valid:
            continue
        cut_polys.append(cp)
        try:
            geom = geom.difference(cp)
        except Exception:
            continue

    def _polys_of(g):
        if g is None or g.is_empty:
            return []
        return list(g.geoms) if isinstance(g, MultiPolygon) else (
            [g] if g.geom_type == "Polygon" else [])

    pieces: List[List[Edge2D]] = []
    # RESTO: el panel con el/los hueco(s) o partido en varias piezas.
    for p in _polys_of(geom):
        if p.area < 1e-4:
            continue
        pe = _polygon_to_edges(p)
        if len(pe) >= 3:
            pieces.append(pe)

    # RECORTE (B1): el material removido (panel ∩ cortes) como pieza(s) independiente(s).
    if cut_polys:
        try:
            recorte = panel.intersection(unary_union(cut_polys))
        except Exception:
            recorte = None
        for p in _polys_of(recorte):
            if p.area < 1e-4:
                continue
            pe = _polygon_to_edges(p)
            if len(pe) >= 3:
                pieces.append(pe)

    if not pieces:
        return [], score_lines
    return pieces, score_lines


# ---------------------------------------------------------------------------
# Filtrado Simple y Twin Merging
# ---------------------------------------------------------------------------


def compute_group_geom(group: CoplanarGroup) -> Dict:
    sx, sy, sz, count = 0.0, 0.0, 0.0, 0
    min_x, min_y, min_z = float("inf"), float("inf"), float("inf")
    max_x, max_y, max_z = float("-inf"), float("-inf"), float("-inf")

    for face in group.faces:
        for v in face.vertices:
            sx += v.x
            sy += v.y
            sz += v.z
            count += 1
            if v.x < min_x:
                min_x = v.x
            if v.y < min_y:
                min_y = v.y
            if v.z < min_z:
                min_z = v.z
            if v.x > max_x:
                max_x = v.x
            if v.y > max_y:
                max_y = v.y
            if v.z > max_z:
                max_z = v.z

    return {
        "centroid": Vec3(sx / count, sy / count, sz / count),
        "extent": max(max_x - min_x, max_y - min_y, max_z - min_z),
    }


def merge_thin_twin_groups(groups: List[CoplanarGroup]) -> List[CoplanarGroup]:
    geom = [compute_group_geom(g) for g in groups]
    drop: Set[int] = set()

    for i in range(len(groups)):
        if i in drop:
            continue
        for j in range(i + 1, len(groups)):
            if j in drop:
                continue
            if groups[i].category != groups[j].category:
                continue

            a = TwinCandidate(
                normal=groups[i].normal,
                d=groups[i].d,
                centroid=geom[i]["centroid"],
                extent=geom[i]["extent"],
            )
            b = TwinCandidate(
                normal=groups[j].normal,
                d=groups[j].d,
                centroid=geom[j]["centroid"],
                extent=geom[j]["extent"],
            )

            if not are_thin_twins(a, b, THIN_TWIN_THRESHOLD):
                continue

            if groups[i].total_area >= groups[j].total_area:
                drop.add(j)
            else:
                drop.add(i)
                break

    return [g for i, g in enumerate(groups) if i not in drop]


def filter_groups_for_simple_mode(groups: List[CoplanarGroup]) -> List[CoplanarGroup]:
    walls = [g for g in groups if g.category == "wall"]
    floors = [g for g in groups if g.category == "floor"]

    used: Set[int] = set()
    kept: List[CoplanarGroup] = []

    for i in range(len(walls)):
        if i in used:
            continue
        used.add(i)
        best = walls[i]

        for j in range(i + 1, len(walls)):
            if j in used:
                continue
            if dot(walls[i].normal, walls[j].normal) > -0.85:
                continue
            if abs(walls[i].d + walls[j].d) > 0.5:
                continue

            used.add(j)
            if walls[j].total_area > best.total_area:
                best = walls[j]

        kept.append(best)

    max_area = max([g.total_area for g in kept] + [0])
    filtered = [g for g in kept if g.total_area >= max_area * 0.10]

    return filtered + floors


# ---------------------------------------------------------------------------
# API Pública - Decomposición y DXF
# ---------------------------------------------------------------------------


def decompose_into_panels(
    faces: List[Face3D], up: UpAxis, simple_mode: bool, min_area_m2: float = 0.01
) -> List[Panel]:
    coplanar_groups = cluster_by_coplanarity(faces, up)
    coplanar_groups = merge_thin_twin_groups(coplanar_groups)
    if simple_mode:
        coplanar_groups = filter_groups_for_simple_mode(coplanar_groups)

    levels = detect_floor_levels(faces, up)
    raw_panels = []

    for group in coplanar_groups:
        components = split_connected_components(group.faces)
        for comp_faces in components:
            res = project_faces_to_2d(comp_faces, group.normal, up)
            if not res or (res.width_m * res.height_m < min_area_m2):
                continue

            all_elevs = [get_up(v, up) for f in comp_faces for v in f.vertices]
            mid = (min(all_elevs) + max(all_elevs)) / 2.0

            floor_idx = 0
            for i in range(len(levels) - 1, -1, -1):
                if mid >= levels[i] - 0.5:
                    floor_idx = i
                    break

            raw_panels.append(
                {
                    "category": group.category,
                    "floor_index": floor_idx,
                    "width_m": res.width_m,
                    "height_m": res.height_m,
                    "edges": res.edges,
                }
            )

    walls = sorted(
        [p for p in raw_panels if p["category"] == "wall"],
        key=lambda x: (x["floor_index"], -(x["width_m"] * x["height_m"])),
    )
    floors = sorted(
        [p for p in raw_panels if p["category"] == "floor"],
        key=lambda x: (x["floor_index"], -(x["width_m"] * x["height_m"])),
    )

    panels: List[Panel] = []

    for i, rp in enumerate(walls, 1):
        panels.append(
            Panel(
                id=f"A{i}",
                group_name=f"wall_{i}",
                category="wall",
                floor_index=rp["floor_index"],
                width_m=rp["width_m"],
                height_m=rp["height_m"],
                edges=rp["edges"],
                source_group_id=-1,
            )
        )

    for i, rp in enumerate(floors, 1):
        panels.append(
            Panel(
                id=f"B{i}",
                group_name=f"floor_{i}",
                category="floor",
                floor_index=rp["floor_index"],
                width_m=rp["width_m"],
                height_m=rp["height_m"],
                edges=rp["edges"],
                source_group_id=-1,
            )
        )

    return panels


def layout_panels(panels: List[Panel]) -> List[PlacedPanel]:
    if not panels:
        return []
    sorted_panels = sorted(panels, key=lambda p: p.height_m, reverse=True)
    max_row_w = max([p.width_m for p in sorted_panels] + [2.0]) * 4

    placed: List[PlacedPanel] = []
    row_x, row_y, row_max_h = 0.0, 0.0, 0.0

    for panel in sorted_panels:
        pw, ph = panel.width_m, panel.height_m
        if row_x > 0 and row_x + pw > max_row_w:
            row_y += row_max_h + GAP_M
            row_x, row_max_h = 0.0, 0.0

        placed.append(PlacedPanel(panel=panel, x=row_x, y=row_y))
        row_x += pw + GAP_M
        row_max_h = max(row_max_h, ph)

    return placed


CS_LAYERS = [
    {"name": "CUT_EXTERIOR", "aci": "7"},
    {"name": "ENGRAVE_VECTOR", "aci": "5"},
    {"name": "ENGRAVE_RASTER", "aci": "8"},
    {"name": "CUT_INTERIOR", "aci": "3"},
    {"name": "MARK_VECTOR", "aci": "1"},  # rojo: aberturas a grabar (no cortar)
    {"name": "FLEX_CUT", "aci": "7"},  # negro: patrón de flexión (kerf/auxético) — SE CORTA
]


def emit_dxf_header(lines: List[str], layer_count: int):
    lines.extend(
        [
            "0",
            "SECTION",
            "2",
            "HEADER",
            "9",
            "$ACADVER",
            "1",
            "AC1009",
            "9",
            "$INSUNITS",
            "70",
            "6",
            "0",
            "ENDSEC",
            "0",
            "SECTION",
            "2",
            "TABLES",
            "0",
            "TABLE",
            "2",
            "LTYPE",
            "70",
            "1",
            "0",
            "LTYPE",
            "2",
            "CONTINUOUS",
            "70",
            "0",
            "3",
            "Solid line",
            "72",
            "65",
            "73",
            "0",
            "40",
            "0.0",
            "0",
            "ENDTAB",
            "0",
            "TABLE",
            "2",
            "LAYER",
            "70",
            str(layer_count),
        ]
    )
    for l in CS_LAYERS:
        lines.extend(
            ["0", "LAYER", "2", l["name"], "70", "0", "62", l["aci"], "6", "CONTINUOUS"]
        )
    lines.extend(["0", "ENDTAB", "0", "ENDSEC"])


def fit_text_height(text: str, max_w: float, max_h: float, target_h: float) -> float:
    if not text:
        return 0.0
    by_width = (max_w * 0.88) / (len(text) * 0.62)
    return min(target_h, by_width, max_h)


def emit_panel_entities(
    lines: List[str],
    edges: List[Edge2D],
    pw: float,
    ph: float,
    panel_id: str,
    ox: float,
    oy: float,
    scale_denom: float = 1.0,
    include_text: bool = True,
    is_mark: bool = False,
    label_dims: Optional[Tuple[float, float]] = None,
):
    for edge in edges:
        # Los encastres (joint) NO se dibujan en la lámina de corte: son geometría de
        # ensamble (van en el array plate_joints para el visor 3D), no en el corte del
        # taller. Evita el "verde" espurio en la plancha.
        if getattr(edge, "joint", False):
            continue
        if getattr(edge, "flex", False):
            # Patrón de flexión (kerf/auxético): SE CORTA. Capa propia FLEX_CUT en negro
            # (color de corte) para que el láser lo corte y el material pueda plegarse.
            layer, aci = "FLEX_CUT", "7"
        elif getattr(edge, "score", False):
            # Corte manual tipo "line" -> marca de pliegue/score (no corta): MARK_VECTOR.
            layer, aci = "MARK_VECTOR", "1"
        elif getattr(edge, "joint", False):
            # Encastre = CORTE (negro), no una capa verde aparte (B4).
            layer, aci = "CUT_EXTERIOR", "7"
        elif is_mark and getattr(edge, "hole", False):
            layer, aci = "MARK_VECTOR", "1"
        else:
            layer, aci = "CUT_EXTERIOR", "7"
        lines.extend(
            [
                "0",
                "LINE",
                "8",
                layer,
                "62",
                aci,
                "10",
                r_str(ox + edge.a.x),
                "20",
                r_str(oy + edge.a.y),
                "11",
                r_str(ox + edge.b.x),
                "21",
                r_str(oy + edge.b.y),
            ]
        )

    if not include_text:
        return

    # Las medidas se rotulan SIEMPRE en el marco propio de la pieza, no en el que quedó
    # tras rotarla para que entre en la plancha. Con las rotadas al revés, dos piezas que
    # se comparan (un entrepiso contra el piso de abajo, por ejemplo) imprimían sus ejes
    # cruzados: una decía "6.96 x 7.18" y la otra "7.51 x 7.01", y comparar los primeros
    # números era comparar la Z de una contra la X de la otra. La etiqueta llevaba a
    # conclusiones falsas sobre piezas que estaban bien.
    lw, lh = label_dims if label_dims else (pw, ph)
    real_w, real_h = lw * scale_denom, lh * scale_denom
    dim_text = f"{real_w:.2f} x {real_h:.2f} m"
    label_h = fit_text_height(panel_id, pw, ph * 0.45, 0.008)
    dim_h = fit_text_height(dim_text, pw, ph * 0.30, 0.005)

    if label_h >= 0.002:
        lx, ly = r_str(ox + pw / 2), r_str(oy + ph - label_h * 1.5)
        lines.extend(
            [
                "0",
                "TEXT",
                "8",
                "ENGRAVE_VECTOR",
                "62",
                "5",
                "10",
                lx,
                "20",
                ly,
                "40",
                r_str(label_h),
                "1",
                panel_id,
                "72",
                "1",
                "11",
                lx,
                "21",
                ly,
            ]
        )

    if dim_h >= 0.002 and label_h + dim_h * 3 < ph:
        dx, dy = r_str(ox + pw / 2), r_str(oy + dim_h * 0.6)
        lines.extend(
            [
                "0",
                "TEXT",
                "8",
                "ENGRAVE_RASTER",
                "62",
                "8",
                "10",
                dx,
                "20",
                dy,
                "40",
                r_str(dim_h),
                "1",
                dim_text,
                "72",
                "1",
                "11",
                dx,
                "21",
                dy,
            ]
        )


def panels_to_dxf(placed: List[PlacedPanel]) -> str:
    lines: List[str] = []
    emit_dxf_header(lines, len(CS_LAYERS))
    lines.extend(["0", "SECTION", "2", "ENTITIES"])
    for p in placed:
        emit_panel_entities(
            lines,
            p.panel.edges,
            p.panel.width_m,
            p.panel.height_m,
            p.panel.id,
            p.x,
            p.y,
            is_mark=getattr(p.panel, "is_mark", False),
        )
    lines.extend(["0", "ENDSEC", "0", "EOF"])
    return "\n".join(lines) + "\n"


def generate_cutting_sheets(
    faces: List[Face3D], up_axis: UpAxis, scale_denom: float, mode: str = "simple"
) -> List[Dict[str, str]]:
    panels = decompose_into_panels(faces, up_axis, mode == "simple")
    if not panels:
        return []

    results = []
    wall_panels = [p for p in panels if p.category == "wall"]
    floor_panels = [p for p in panels if p.category == "floor"]

    if wall_panels:
        results.append(
            {
                "name": "Descomposicion_Paredes.dxf",
                "content": panels_to_dxf(layout_panels(wall_panels)),
            }
        )
    if floor_panels:
        results.append(
            {
                "name": "Descomposicion_Pisos.dxf",
                "content": panels_to_dxf(layout_panels(floor_panels)),
            }
        )

    return results


def nested_sheets_to_dxf(nesting: NestingResult, include_text: bool = True) -> str:
    sheets, config = nesting.sheets, nesting.config
    if not sheets:
        return ""

    lines: List[str] = []
    emit_dxf_header(lines, len(CS_LAYERS))
    lines.extend(["0", "SECTION", "2", "ENTITIES"])

    cols = min(len(sheets), 3)

    for si, sheet in enumerate(sheets):
        col, row = si % cols, si // cols
        sx = col * (config.width_m + SHEET_SPACING_M)
        sy = -(row * (config.height_m + SHEET_SPACING_M))

        x0, y0, x1, y1 = sx, sy, sx + config.width_m, sy + config.height_m
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for ci in range(4):
            ax, ay = corners[ci]
            bx, by = corners[(ci + 1) % 4]
            lines.extend(
                [
                    "0",
                    "LINE",
                    "8",
                    "ENGRAVE_RASTER",
                    "62",
                    "7",
                    "10",
                    r_str(ax),
                    "20",
                    r_str(ay),
                    "11",
                    r_str(bx),
                    "21",
                    r_str(by),
                ]
            )

        if include_text:
            cx, cy = r_str(sx + config.width_m / 2), r_str(sy + config.height_m + 0.02)
            lines.extend(
                [
                    "0",
                    "TEXT",
                    "8",
                    "ENGRAVE_RASTER",
                    "62",
                    "8",
                    "10",
                    cx,
                    "20",
                    cy,
                    "40",
                    r_str(0.03),
                    "1",
                    f"Plancha {si + 1}",
                    "72",
                    "1",
                    "11",
                    cx,
                    "21",
                    cy,
                ]
            )

        for placed in sheet.panels:
            edges = (
                rotate_edges(placed.panel.edges, placed.panel.width_m)
                if placed.rotated
                else placed.panel.edges
            )
            emit_panel_entities(
                lines,
                edges,
                placed.effective_w,
                placed.effective_h,
                placed.panel.id,
                sx + placed.x,
                sy + placed.y,
                nesting.scale_denom,
                include_text,
                is_mark=getattr(placed.panel, "is_mark", False),
                label_dims=(placed.panel.width_m, placed.panel.height_m),
            )

    lines.extend(["0", "ENDSEC", "0", "EOF"])
    return "\n".join(lines) + "\n"
