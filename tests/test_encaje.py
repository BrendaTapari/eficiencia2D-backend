"""
Banco de pruebas del ENCAJE: que las piezas cortadas armen la maqueta sin luces ni choques.

Por qué existe
--------------
Es el criterio de aceptación de todo el pipeline de encastres. Se construyen modelos
sintéticos cuyo resultado correcto se calcula a mano, y se afirma sobre la maqueta ARMADA,
no sobre números intermedios. Un intento anterior de verificar el ensamble se validó de
forma superficial (se comprobó que el centro de cada pieza cayera en su plano, lo que no
valida la orientación dentro del plano) y se entregó un chequeo que daba "todo bien" con
piezas visiblemente superpuestas.

El invariante que se afirma
---------------------------
Cada placa es un rectángulo de `PLATE_THICKNESS` centrado en el plano medio de su grupo.
Dos placas que se topan encajan si el borde de una cae sobre la CARA de la otra:

    distancia(borde de A, plano medio de B) == espesor_placa / 2

Es un invariante GEOMÉTRICO, y por eso vale en todas las escalas. El largo en milímetros,
en cambio, NO es proporcional entre escalas: la placa mide 3 mm siempre mientras el muro
del modelo se achica. Con un muro de 20 cm y 8 m de largo:

    1:20  -> bruto 400.00mm, recorte 6.50mm, final 393.50mm
    1:100 -> bruto  80.00mm, recorte 2.50mm, final  77.50mm

Afirmar "mismo largo en mm" sería incorrecto; se afirma la distancia al plano vecino.
"""

import math

import pytest

from core.pipeline import parse_pipeline
from core.review_generate import _decompose
from core.services.cutting_sheet import (
    orient_group_normals_outward,
    project_faces_to_2d,
)
from core.services.obj_parser import parse_obj
from core.services.types import PipelineOptions, dot

# Espesor físico de la placa de MDF, en metros de plancha.
PLATE_THICKNESS_M = 0.003
# Tolerancia de las afirmaciones, en milímetros de plancha. Del orden del kerf del láser y
# de la variación del propio MDF: es lo que se puede sostener en producción. Perseguir
# centésimas marcaría en rojo desvíos que en la pieza real no son medibles.
TOL_MM = 0.2

ESCALAS = (20.0, 50.0, 100.0, 200.0)


# ---------------------------------------------------------------------------
# Construcción de modelos sintéticos
# ---------------------------------------------------------------------------


class _ObjBuilder:
    """Arma un OBJ en memoria a partir de cajas, soldando vértices repetidos."""

    def __init__(self):
        self._vmap = {}
        self._verts = []
        self._faces = []

    def _vid(self, p):
        key = tuple(round(c, 6) for c in p)
        if key not in self._vmap:
            self._vmap[key] = len(self._verts) + 1
            self._verts.append(key)
        return self._vmap[key]

    def quad(self, a, b, c, d):
        self._faces.append(tuple(self._vid(p) for p in (a, b, c, d)))

    def box(self, x0, x1, y0, y1, z0, z1):
        """Caja sólida: seis caras. Representa un muro o losa con espesor real."""
        self.quad((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0))
        self.quad((x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1))
        self.quad((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1))
        self.quad((x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0))
        self.quad((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1))
        self.quad((x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0))

    def text(self) -> str:
        lines = [f"v {x} {y} {z}" for (x, y, z) in self._verts]
        lines += [f"f {' '.join(map(str, f))}" for f in self._faces]
        return "\n".join(lines)


def _procesar(obj_text: str, scale_denom: float):
    """Corre el pipeline completo y devuelve (work, paneles, grupos por id)."""
    parsed = parse_obj(obj_text)
    # Los modelos de este banco están escritos con Y hacia ARRIBA. Sin forzar el eje,
    # `detect_up_axis` los leía como Z-up y el pipeline intercambiaba Y y Z: la esquina
    # en L pasaba a ser "losa horizontal + muro vertical", el grupo que cedía era la
    # tapa en L de los extremos (bbox 8x6) y la medición no significaba nada. El eje se
    # fija acá para que el modelo sintético sea lo que dice ser; en producción se sigue
    # autodetectando.
    p1 = parse_pipeline("sintetico.obj", parsed["faces"], parsed["warnings"], force_axis="Y")
    opts = PipelineOptions(
        scale_denom=scale_denom, paper="A4", min_area_m2=1.0, sheet_config=None
    )
    work, walls, floors, plate_joints = _decompose(p1, opts, None, None, None, None)
    grupos = {g.id: g for g in work.groups}
    return work, walls + floors, grupos, plate_joints


# ---------------------------------------------------------------------------
# Medición sobre la maqueta ARMADA
# ---------------------------------------------------------------------------


def _plano_medio(grupo, faces):
    """Plano medio del grupo: (normal unitaria, offset). Es donde va la placa."""
    n = grupo.representative_normal
    ln = math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z)
    n = type(n)(n.x / ln, n.y / ln, n.z / ln)
    vals = [
        n.x * v.x + n.y * v.y + n.z * v.z
        for fi in grupo.face_indices
        if 0 <= fi < len(faces)
        for v in faces[fi].vertices
    ]
    # Punto medio entre las dos pieles, NO el promedio de vértices: el promedio se sesga
    # hacia la piel más teselada y no cae en el plano medio (0.0667 en vez de 0.100 en un
    # muro de 20cm). La placa va en el plano medio geométrico.
    return n, ((min(vals) + max(vals)) / 2.0 if vals else 0.0)


def _extremos_a_lo_largo(panel, grupo, faces, direccion):
    """Posiciones (min, max) del panel FINAL proyectadas sobre `direccion`, en metros
    de edificio. Reconstruye el largo ya recortado a partir del ancho del panel."""
    oriented = orient_group_normals_outward([grupo])
    res = project_faces_to_2d(
        [faces[i] for i in grupo.face_indices if i < len(faces)],
        oriented.get(grupo.id, grupo.representative_normal),
        "Y",
    )
    if res is None:
        return None
    # El eje u del panel, expresado sobre `direccion`.
    sgn = dot(res.u_axis, direccion)
    if abs(sgn) < 0.5:
        return None
    # Extremos del panel BRUTO, expresados sobre `direccion`. El eje u del panel puede
    # quedar ANTIPARALELO a esa dirección (la normal canónica se decide por el centroide
    # y puede invertirse), y en ese caso hay que dar vuelta el intervalo: sin esto se
    # comparaba contra el extremo equivocado y la medición daba el largo entero de la
    # pieza (78.9mm donde se esperaban 1.5mm).
    lo_u = res.origin_u
    hi_u = res.origin_u + res.width_m
    if sgn >= 0:
        bruto_lo, bruto_hi = lo_u, hi_u
    else:
        bruto_lo, bruto_hi = -hi_u, -lo_u
    # El panel final es más corto: el recorte salió de uno de los extremos.
    recorte = res.width_m - panel.width_m
    return bruto_lo, bruto_hi, recorte, sgn


# ---------------------------------------------------------------------------
# Casos
# ---------------------------------------------------------------------------


def _modelo_esquina_L(t=0.20, largo=8.0, alto=3.0, fondo=6.0):
    """Dos muros sólidos que forman una esquina en L.

    Muro A corre en X, ocupa z ∈ [0, t]  -> su plano medio es z = t/2
    Muro B corre en Z, ocupa x ∈ [0, t]  -> su plano medio es x = t/2

    El extremo de A llega a x = 0, o sea que SOBRESALE t/2 del plano medio de B.
    Para encajar, el borde de A debe quedar a espesor_placa/2 de ese plano.
    """
    b = _ObjBuilder()
    b.box(0.0, largo, 0.0, alto, 0.0, t)      # muro A
    b.box(0.0, t, 0.0, alto, 0.0, fondo)      # muro B
    return b.text()


@pytest.mark.parametrize("scale", ESCALAS)
def test_esquina_L_borde_cae_sobre_la_cara_vecina(scale):
    """El borde del muro que cede debe apoyar sobre la CARA de la placa vecina.

    Es el invariante central: se cumple en toda escala aunque el largo en mm cambie.
    """
    t = 0.20
    work, paneles, grupos, _ = _procesar(_modelo_esquina_L(t=t), scale)

    placa_m = PLATE_THICKNESS_M * scale  # espesor de placa en metros de EDIFICIO
    objetivo_m = placa_m / 2.0           # el borde va a media placa del plano medio

    # Se busca la junta muro-muro y se mide el muro que cede.
    assert work.wall_wall_joints, "no se detectó la unión de la esquina"
    ww = work.wall_wall_joints[0]
    assert ww.yield_group_id is not None, "la junta quedó sin resolver"

    cede = grupos[ww.yield_group_id]
    otro = grupos[ww.group_b if ww.yield_group_id == ww.group_a else ww.group_a]
    panel = next(p for p in paneles if p.source_group_id == cede.id)

    n_otro, d_otro = _plano_medio(otro, work.faces)

    # Distancia del borde recortado de `cede` al plano medio de `otro`.
    ext = _extremos_a_lo_largo(panel, cede, work.faces, n_otro)
    assert ext is not None, "el muro que cede no corre a lo largo de la normal del otro"
    bruto_lo, bruto_hi, recorte, sgn = ext

    # El extremo que mira al otro muro es el más cercano a su plano medio.
    borde_bruto = bruto_lo if abs(bruto_lo - d_otro) < abs(bruto_hi - d_otro) else bruto_hi
    # Tras el recorte, ese borde se aleja del plano medio.
    borde_final = borde_bruto + recorte if borde_bruto < d_otro else borde_bruto - recorte
    distancia_m = abs(borde_final - d_otro)

    distancia_mm = distancia_m / scale * 1000.0
    objetivo_mm = objetivo_m / scale * 1000.0  # == PLATE_THICKNESS_M/2 * 1000 == 1.5

    assert objetivo_mm == pytest.approx(1.5, abs=1e-9)
    assert distancia_mm == pytest.approx(objetivo_mm, abs=TOL_MM), (
        f"1:{scale:.0f} el borde quedó a {distancia_mm:.3f}mm del plano medio vecino, "
        f"y debía quedar a {objetivo_mm:.3f}mm "
        f"(diferencia {distancia_mm - objetivo_mm:+.3f}mm en la pieza real)"
    )


@pytest.mark.parametrize("scale", ESCALAS)
def test_esquina_L_solo_cede_uno(scale):
    """Sólo una de las dos placas se acorta: si se acortan las dos, queda luz."""
    work, paneles, grupos, _ = _procesar(_modelo_esquina_L(), scale)
    ww = work.wall_wall_joints[0]
    recortados = []
    for gid in (ww.group_a, ww.group_b):
        g = grupos[gid]
        panel = next((p for p in paneles if p.source_group_id == gid), None)
        if panel is None:
            continue
        oriented = orient_group_normals_outward([g])
        res = project_faces_to_2d(
            [work.faces[i] for i in g.face_indices if i < len(work.faces)],
            oriented.get(gid, g.representative_normal),
            "Y",
        )
        if res and (res.width_m - panel.width_m) > 1e-6:
            recortados.append(gid)
    assert len(recortados) == 1, f"se acortaron {len(recortados)} placas, debía ser 1"


@pytest.mark.parametrize("scale", ESCALAS)
def test_ranura_tiene_el_ancho_de_la_placa(scale):
    """La ranura debe medir lo que la placa que la atraviesa (3mm), más el kerf.

    Hoy sale del espesor del MODELO sin escalar: a 1:100 da 2mm para una placa de 3mm
    (no entra) y con malla sin espesor da 0 (no hay ranura).
    """
    # Cruce en X: un muro atraviesa a otro por el medio de ambos.
    b = _ObjBuilder()
    t = 0.20
    b.box(-4.0, 4.0, 0.0, 3.0, 0.0, t)       # muro que corre en X
    b.box(-0.1, 0.1, 0.0, 3.0, -3.0, 3.0)    # muro que corre en Z, lo cruza al medio
    work, paneles, grupos, plate_joints = _procesar(b.text(), scale)

    assert plate_joints, "no se generó ninguna ranura para el cruce en X"
    # PlateJoint.width está en metros de PLANCHA: es el material real, no una cota
    # del edificio, así que no depende de la escala.
    for pj in plate_joints:
        ancho_mm = pj.width * 1000.0
        assert ancho_mm == pytest.approx(3.0, abs=0.2), (
            f"1:{scale:.0f} ranura de {ancho_mm:.2f}mm para una placa de 3.00mm"
        )


@pytest.mark.parametrize("scale", ESCALAS)
def test_muro_sobre_losa_se_acorta_el_espesor_de_placa(scale):
    """El muro apoyado sobre una losa se acorta el espesor de la PLACA, no el de la losa."""
    # El muro va INSETADO respecto del borde de la losa. Si comparten plano lateral, el
    # clasificador fusiona la cara de la losa con la del muro en un solo grupo coplanar
    # que abarca toda la altura: ese grupo ATRAVIESA el entrepiso en vez de apoyarse en
    # él, y el test terminaba midiendo un caso distinto del que nombra.
    b = _ObjBuilder()
    b.box(0.0, 8.0, 0.0, 0.25, 0.0, 6.0)          # losa de 25cm de canto
    b.box(1.0, 7.0, 0.25, 3.25, 2.0, 2.20)        # muro encima, sin tocar los bordes
    work, paneles, grupos, _ = _procesar(b.text(), scale)

    for panel in paneles:
        g = grupos[panel.source_group_id]
        if g.category != "wall":
            continue
        # Sólo el muro que este test construye (arranca en la cara superior de la losa).
        # El canto de la losa también se clasifica como muro y no es lo que se mide acá.
        if g.min_y is None or abs(g.min_y - 0.25) > 0.02:
            continue
        oriented = orient_group_normals_outward([g])
        res = project_faces_to_2d(
            [work.faces[i] for i in g.face_indices if i < len(work.faces)],
            oriented.get(g.id, g.representative_normal),
            "Y",
        )
        if res is None:
            continue
        recorte_m = res.height_m - panel.height_m
        if recorte_m <= 1e-9:
            continue
        recorte_mm = recorte_m / scale * 1000.0
        # El muro apoya sobre la CARA de la placa de piso: se acorta media placa. No hay
        # voladizo, porque su base ya coincide con la cara superior de la losa.
        # (La aserción anterior usaba `recorte_mm % 1.5`, que sobre un float no afirma
        # nada útil; se reemplaza por el valor exacto esperado.)
        assert recorte_mm == pytest.approx(1.5, abs=TOL_MM), (
            f"1:{scale:.0f} el muro se acortó {recorte_mm:.2f}mm y debía acortarse "
            f"1.50mm (media placa)"
        )


@pytest.mark.parametrize("scale", ESCALAS)
def test_un_borde_no_se_recorta_dos_veces(scale):
    """Dos juntas del mismo lado no deben acumular dos recortes sobre el mismo borde."""
    b = _ObjBuilder()
    t = 0.20
    b.box(0.0, 8.0, 0.0, 3.0, 0.0, t)        # muro central
    b.box(0.0, t, 0.0, 3.0, 0.0, 4.0)        # topa en el extremo x=0
    b.box(0.0, t, 0.0, 3.0, -4.0, 0.0)       # otro que topa en el MISMO extremo
    work, paneles, grupos, _ = _procesar(b.text(), scale)

    for panel in paneles:
        g = grupos[panel.source_group_id]
        oriented = orient_group_normals_outward([g])
        res = project_faces_to_2d(
            [work.faces[i] for i in g.face_indices if i < len(work.faces)],
            oriented.get(g.id, g.representative_normal),
            "Y",
        )
        if res is None:
            continue
        recorte_mm = (res.width_m - panel.width_m) / scale * 1000.0
        # Con dos juntas en el mismo borde, el recorte no puede duplicarse.
        assert recorte_mm < 8.0, (
            f"1:{scale:.0f} la pieza {panel.id} perdió {recorte_mm:.2f}mm de largo: "
            "hay recortes acumulados sobre el mismo borde"
        )
