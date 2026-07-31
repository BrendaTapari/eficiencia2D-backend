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
from core.services.assembly_adjuster import compute_adjustments, yield_by_pair
from core.services.assembly_verify import verificar_ensamble
from core.services.cutting_sheet import (
    orient_group_normals_outward,
    project_faces_to_2d,
)
from core.services.obj_parser import parse_obj
from core.services.plate_intersect import group_plane, mid_plane_offset, pair_key
from core.services.types import PipelineOptions, dot, normalize

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
        self.poligono([a, b, c, d])

    def poligono(self, pts):
        self._faces.append(tuple(self._vid(p) for p in pts))

    def box(self, x0, x1, y0, y1, z0, z1):
        """Caja sólida: seis caras. Representa un muro o losa con espesor real."""
        self.quad((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0))
        self.quad((x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1))
        self.quad((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1))
        self.quad((x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0))
        self.quad((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1))
        self.quad((x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0))

    def muro(self, x0, z0, x1, z1, t, y0, y1):
        """Muro sólido a lo largo del eje (x0,z0)→(x1,z1), espesor `t`, altura y0..y1.

        `box` sólo construye cajas paralelas a los ejes, así que no puede representar una
        unión OBLICUA. Esto sí: el muro se centra sobre su eje, de modo que su plano medio
        es exactamente ese segmento.
        """
        dx, dz = x1 - x0, z1 - z0
        largo = math.hypot(dx, dz)
        ux, uz = dx / largo, dz / largo
        nx, nz = -uz, ux          # normal en planta
        h = t / 2.0
        planta = [
            (x0 + nx * h, z0 + nz * h),
            (x1 + nx * h, z1 + nz * h),
            (x1 - nx * h, z1 - nz * h),
            (x0 - nx * h, z0 - nz * h),
        ]
        self.prisma(planta, y0, y1)

    def losa_inclinada(self, x0, x1, z0, y0, z1, y1, t):
        """Placa inclinada: sección en el plano (z,y) de (z0,y0) a (z1,y1), espesor `t`,
        extruida en X de x0 a x1. `prisma` sólo extruye en vertical, así que no puede
        representar un faldón de techo."""
        dz, dy = z1 - z0, y1 - y0
        L = math.hypot(dz, dy)
        uz, uy = dz / L, dy / L
        mz, my = -uy, uz          # normal de la sección
        h = t / 2.0
        sec = [
            (z0 + mz * h, y0 + my * h), (z1 + mz * h, y1 + my * h),
            (z1 - mz * h, y1 - my * h), (z0 - mz * h, y0 - my * h),
        ]
        a = [(x0, y, z) for (z, y) in sec]
        b = [(x1, y, z) for (z, y) in sec]
        for i in range(4):
            j = (i + 1) % 4
            self.quad(a[i], a[j], b[j], b[i])
        self.quad(*reversed(a))
        self.quad(*b)

    def prisma(self, planta, y0, y1):
        """Extruye un polígono en planta [(x,z), ...] entre las alturas y0 e y1."""
        n = len(planta)
        bajo = [(x, y0, z) for (x, z) in planta]
        alto = [(x, y1, z) for (x, z) in planta]
        for i in range(n):
            j = (i + 1) % n
            self.quad(bajo[i], bajo[j], alto[j], alto[i])
        self.poligono(list(reversed(bajo)))
        self.poligono(alto)

    def text(self) -> str:
        lines = [f"v {x} {y} {z}" for (x, y, z) in self._verts]
        lines += [f"f {' '.join(map(str, f))}" for f in self._faces]
        return "\n".join(lines)


def _procesar(obj_text: str, scale_denom: float, wall_wall_decisions=None):
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
    work, walls, floors, plate_joints = _decompose(
        p1, opts, None, wall_wall_decisions, None, None
    )
    grupos = {g.id: g for g in work.groups}
    return work, walls + floors, grupos, plate_joints


def _decisiones(work, forzar=None):
    """`pair_key(a,b) -> grupo que cede`, tal como lo ve el recorte."""
    return yield_by_pair(
        compute_adjustments(work.joints, work.groups, forzar, work.faces)
    )


def _forzar_lo_contrario(work):
    """`joint_index -> el muro CONTRARIO al que el sistema sugiere`.

    Simula al usuario cambiando la decisión en el selector del visor 3D.
    """
    res = compute_adjustments(work.joints, work.groups, None, work.faces)
    forz = {}
    for ww in res.wall_wall_joints:
        if ww.suggested_yield_group_id is None:
            continue
        forz[ww.joint_index] = (
            ww.group_b if ww.suggested_yield_group_id == ww.group_a else ww.group_a
        )
    return forz


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


def _ocupa(panel, direccion):
    """Intervalo que ocupa la pieza FINAL sobre `direccion`, LEÍDO de su marco 3D.

    Es la diferencia entre verificar el TAMAÑO y verificar la POSICIÓN. Reconstruir el
    borde a partir de la proyección y del recorte —que es lo que hacía este banco— da por
    sentado de qué extremo salió el material, así que no puede detectar que el código lo
    saque del otro. Y eso pasó: las dos ramas del clip lateral estaban invertidas, la
    pieza salía con el largo correcto y corrida el recorte entero, y los 41 casos pasaban
    en verde. Acá se lee dónde quedó la pieza y se compara contra el mundo.
    """
    m = panel.frame
    assert m is not None, f"la pieza {panel.id} no trae marco 3D"
    o, u, v = m["origin"], m["u_axis"], m["v_axis"]
    proy = [
        direccion.x * (o["x"] + a * u["x"] + b * v["x"])
        + direccion.y * (o["y"] + a * u["y"] + b * v["y"])
        + direccion.z * (o["z"] + a * u["z"] + b * v["z"])
        for a, b in ((0.0, 0.0), (m["width_m"], 0.0),
                     (m["width_m"], m["height_m"]), (0.0, m["height_m"]))
    ]
    return min(proy), max(proy)


def _borde_contra(panel, otro, faces):
    """Distancia del borde de `panel` al plano medio de `otro`, sobre la posición REAL."""
    n = normalize(otro.representative_normal)
    d = mid_plane_offset(otro, faces, n)
    lo, hi = _ocupa(panel, n)
    borde = lo if abs(lo - d) < abs(hi - d) else hi
    return abs(borde - d)


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

    # Distancia del borde recortado al plano medio del vecino, medida sobre la POSICIÓN
    # real de la pieza. Antes se reconstruía a partir de la proyección y del recorte,
    # asumiendo de qué extremo había salido el material: por eso este mismo caso pasaba
    # en verde con las ramas del clip invertidas.
    distancia_mm = _borde_contra(panel, otro, work.faces) / scale * 1000.0
    objetivo_mm = objetivo_m / scale * 1000.0  # == PLATE_THICKNESS_M/2 * 1000 == 1.5

    assert objetivo_mm == pytest.approx(1.5, abs=1e-9)
    assert distancia_mm == pytest.approx(objetivo_mm, abs=TOL_MM), (
        f"1:{scale:.0f} el borde quedó a {distancia_mm:.3f}mm del plano medio vecino, "
        f"y debía quedar a {objetivo_mm:.3f}mm "
        f"(diferencia {distancia_mm - objetivo_mm:+.3f}mm en la pieza real)"
    )

    # Y que el material se haya sacado del extremo CORRECTO: la pieza tiene que seguir
    # llegando a su extremo libre, no haberse corrido entera.
    n_otro = normalize(otro.representative_normal)
    lo_f, hi_f = _ocupa(panel, n_otro)
    proy_bruto = [
        dot(n_otro, v)
        for fi in cede.face_indices
        if 0 <= fi < len(work.faces)
        for v in work.faces[fi].vertices
    ]
    d_otro = mid_plane_offset(otro, work.faces, n_otro)
    lejano_bruto = max(proy_bruto) if abs(max(proy_bruto) - d_otro) > abs(min(proy_bruto) - d_otro) else min(proy_bruto)
    lejano_final = hi_f if abs(hi_f - d_otro) > abs(lo_f - d_otro) else lo_f
    assert lejano_final == pytest.approx(lejano_bruto, abs=1e-6), (
        f"1:{scale:.0f} el extremo LIBRE de la pieza se movió de {lejano_bruto:.4f} a "
        f"{lejano_final:.4f}: el recorte salió del extremo equivocado"
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
def test_muro_sobre_losa_apoya_en_la_cara_de_la_placa(scale):
    """La base del muro debe apoyar sobre la CARA de la placa de piso.

    O sea a media placa del PLANO MEDIO de la losa, que es el mismo invariante
    geométrico de la esquina en L. La versión anterior afirmaba "se acorta 1.5 mm en
    toda escala", y eso era falso por la misma razón que ya se había corregido para los
    muros: la losa del modelo tiene 25 cm de canto y su placa se centra en el plano
    medio, así que la cara superior de la placa cae en 0.125 + media_placa, que se mueve
    con la escala. Números reales de este modelo:

        1:20  -> el muro llega 4.75 mm CORTO (habría que alargarlo: no se puede)
        1:50  -> llega 1.00 mm corto
        1:100 -> sobra 0.25 mm -> se recorta
        1:200 -> sobra 0.87 mm -> se recorta

    Cuando la pieza ya llega corta no se la extiende (puede haber aberturas junto al
    borde): se deja como está, y eso es lo que se afirma en esa rama.
    """
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
        media_placa_m = PLATE_THICKNESS_M * scale / 2.0
        plano_medio_losa = 0.125          # losa y=0..0.25
        base_bruta = 0.25                 # base del muro en el modelo
        necesario = (plano_medio_losa + media_placa_m) - base_bruta

        if necesario <= 0.0:
            # El muro ya llega corto: no se lo puede alargar, así que no se toca.
            assert recorte_m == pytest.approx(0.0, abs=1e-6), (
                f"1:{scale:.0f} el muro llegaba {(-necesario)/scale*1000:.2f}mm corto y "
                f"aun así se le recortaron {recorte_m/scale*1000:.2f}mm"
            )
            continue

        base_final = base_bruta + recorte_m
        distancia_mm = (base_final - plano_medio_losa) / scale * 1000.0
        assert distancia_mm == pytest.approx(1.5, abs=TOL_MM), (
            f"1:{scale:.0f} la base del muro quedó a {distancia_mm:.2f}mm del plano medio "
            f"de la losa y debía quedar a 1.50mm (media placa)"
        )


# ---------------------------------------------------------------------------
# Coherencia entre el RECORTE y la RANURA
#
# Eran dos decisiones independientes: `compute_adjustments` elegía quién se acorta y
# `resolve_plate_joints` volvía a elegir, por su cuenta, quién recibe la ranura — con
# `topo_info=None`, con los espesores crudos y sin ver la decisión del usuario. En un
# modelo real se contradecían en 7 de 15 pares: se quitaba material de una placa y se
# abría la ranura en la otra. Ése era el fallo del par 255/261 en la maqueta física.
# ---------------------------------------------------------------------------


def _modelo_cruce_en_extremo(t=0.20, largo=8.0, alto=3.0, fondo=3.0):
    """Un muro pasa DE LARGO por el extremo de otro: hay tope Y hay ranura.

    Es la geometría del par 255/261. El muro B atraviesa el plano de A (por eso se genera
    ranura) pero lo hace en el EXTREMO de A (por eso A tiene decisión de recorte). Un
    simple tope en T no sirve para este caso: sin penetración no hay ranura que comparar.
    """
    b = _ObjBuilder()
    b.box(0.0, largo, 0.0, alto, 0.0, t)          # A corre en X
    b.box(0.0, t, 0.0, alto, -fondo, fondo)       # B corre en Z y cruza el extremo de A
    return b.text()


@pytest.mark.parametrize("scale", ESCALAS)
def test_recorte_y_ranura_eligen_la_misma_placa(scale):
    """La placa que se acorta y la que recibe la ranura son la misma.

    GUARDA, no reproducción: sobre dos muros sintéticos este caso también pasaba ANTES
    del arreglo, porque la regla de posición (quién tiene un extremo en la junta) decide
    sola y las reglas de `choose_wall_wall_yielder` donde las dos rutas discrepaban ni
    siquiera llegan a correr. El desacuerdo real (7 de 15 pares) aparece con topologías de
    3+ muros. Lo que sí queda cerrado acá es que un cambio futuro no pueda volver a
    separar las dos decisiones sin que el banco se entere.
    """
    work, _, _, plate_joints = _procesar(_modelo_cruce_en_extremo(), scale)
    ceden = _decisiones(work)

    comparables = 0
    for pj in plate_joints:
        k = pair_key(pj.cutter_id, pj.cut_id)
        if k not in ceden:
            # Cruce en medio de AMBOS muros: no hay tope, sólo ranura. Sin decisión que
            # respetar, y así debe ser.
            continue
        comparables += 1
        assert pj.cut_id == ceden[k], (
            f"1:{scale:.0f} par {k}: se acorta g{ceden[k]} pero la ranura va a "
            f"g{pj.cut_id}. Se saca material de una placa y se abre la ranura en la otra."
        )
    assert comparables, "el modelo de cruce en extremo no generó ninguna ranura comparable"


@pytest.mark.parametrize("scale", ESCALAS)
def test_la_decision_del_usuario_llega_a_la_ranura(scale):
    """Si el usuario cambia quién cede en el visor, la ranura tiene que seguirlo.

    `resolve_plate_joints` ni siquiera recibía `wall_wall_decisions`: la elección movía
    el recorte y dejaba la ranura donde estaba. Es el síntoma de "la intersección entre
    A10 y A3 no está resuelta pese a que en el selector hay una decisión tomada".
    """
    obj = _modelo_cruce_en_extremo()
    work_auto, _, _, _ = _procesar(obj, scale)
    forzado = _forzar_lo_contrario(work_auto)
    assert forzado, "el modelo de cruce en extremo no ofreció ninguna junta para elegir"

    work, _, _, plate_joints = _procesar(obj, scale, wall_wall_decisions=forzado)
    ceden = _decisiones(work, forzado)

    comparables = 0
    for pj in plate_joints:
        k = pair_key(pj.cutter_id, pj.cut_id)
        if k not in ceden:
            continue
        comparables += 1
        assert pj.cut_id == ceden[k], (
            f"1:{scale:.0f} par {k}: el usuario eligió acortar g{ceden[k]}, pero la "
            f"ranura se abrió en g{pj.cut_id}"
        )
    assert comparables, "ninguna ranura quedó ligada a la decisión del usuario"


def test_el_plano_de_la_ranura_es_el_mismo_que_el_del_recorte():
    """`group_plane` (ranura) y `_mid_plane_offset` (recorte) deben dar el MISMO plano.

    Uno promediaba los vértices y el otro tomaba el punto medio entre pieles. El promedio
    se sesga hacia la piel más teselada, así que la ranura se ubicaba en un plano y el
    recorte se medía contra otro. Acá una piel está subdividida en 4 y la otra no, que es
    lo que pasa en un modelo real con aberturas.
    """
    b = _ObjBuilder()
    largo, alto, t = 8.0, 3.0, 0.20
    # Piel z=t subdividida en 4 tramos; piel z=0 de una sola pieza.
    for i in range(4):
        xa = largo * i / 4.0
        xb = largo * (i + 1) / 4.0
        b.quad((xa, 0.0, t), (xa, alto, t), (xb, alto, t), (xb, 0.0, t))
    b.quad((0.0, 0.0, 0.0), (largo, 0.0, 0.0), (largo, alto, 0.0), (0.0, alto, 0.0))
    # Cantos, para que el grupo tenga volumen cerrado.
    b.quad((0.0, 0.0, 0.0), (0.0, alto, 0.0), (0.0, alto, t), (0.0, 0.0, t))
    b.quad((largo, 0.0, 0.0), (largo, 0.0, t), (largo, alto, t), (largo, alto, 0.0))

    work, _, grupos, _ = _procesar(b.text(), 50.0)

    revisados = 0
    for g in grupos.values():
        if g.category == "discard":
            continue
        n, d_ranura = group_plane(g, work.faces)
        d_recorte = mid_plane_offset(g, work.faces, normalize(g.representative_normal))
        assert d_ranura == pytest.approx(d_recorte, abs=1e-9), (
            f"grupo {g.id}: la ranura se ubica en {d_ranura:.4f} y el recorte se mide "
            f"contra {d_recorte:.4f}"
        )
        # Y que el test muerda: sobre esta malla el promedio SÍ difiere del punto medio.
        proy = [
            dot(n, v)
            for fi in g.face_indices
            if 0 <= fi < len(work.faces)
            for v in work.faces[fi].vertices
        ]
        if proy and (max(proy) - min(proy)) > 0.01:
            promedio = sum(proy) / len(proy)
            assert abs(promedio - d_recorte) > 1e-6, (
                "la malla de prueba quedó simétrica: el caso no distingue promedio de "
                "punto medio y no prueba nada"
            )
            revisados += 1
    assert revisados, "ningún grupo con espesor: el caso no ejercita el sesgo"


# ---------------------------------------------------------------------------
# Uniones oblicuas
# ---------------------------------------------------------------------------


def _distancia_borde_a_plano_vecino(panel, grupo, otro, faces):
    """Distancia del borde recortado de `panel` al plano medio de `otro`, en metros.

    Sirve para cualquier ángulo: el recorte se mide a lo largo del eje `u` del panel, y
    acá se lo proyecta sobre la normal del vecino. `_extremos_a_lo_largo` da por sentado
    que `u` es paralelo a esa normal, cosa que sólo vale a 90°.
    """
    oriented = orient_group_normals_outward([grupo])
    res = project_faces_to_2d(
        [faces[i] for i in grupo.face_indices if i < len(faces)],
        oriented.get(grupo.id, grupo.representative_normal),
        "Y",
    )
    if res is None:
        return None
    n = normalize(otro.representative_normal)
    su = dot(res.u_axis, n)
    if abs(su) < 1e-6:
        return None
    d_otro = mid_plane_offset(otro, faces, n)
    proy = [
        dot(n, v)
        for fi in grupo.face_indices
        if 0 <= fi < len(faces)
        for v in faces[fi].vertices
    ]
    if not proy:
        return None
    lo, hi = min(proy), max(proy)
    borde = lo if abs(lo - d_otro) < abs(hi - d_otro) else hi
    # El recorte se hace sobre `u`; lo que se aleja del plano vecino es su proyección.
    desplazamiento = (res.width_m - panel.width_m) * abs(su)
    borde_final = borde + desplazamiento if borde < d_otro else borde - desplazamiento
    return abs(borde_final - d_otro)


def _inter_rectas(p0, u0, p1, u1):
    """Intersección de dos rectas dadas por punto + dirección, en planta."""
    det = u0[0] * (-u1[1]) - u0[1] * (-u1[0])
    if abs(det) < 1e-12:
        return None
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    s = (dx * (-u1[1]) - dz * (-u1[0])) / det
    return (p0[0] + s * u0[0], p0[1] + s * u0[1])


def _planta_mitrada(eje, t):
    """Contorno en planta de una polilínea de muros de espesor `t`, a inglete.

    Dos muros oblicuos modelados como cajas sueltas NO comparten vértices, y entonces
    ninguno de los dos detectores los une: el topológico exige arista compartida, y
    `detect_contact_joints` compara VÉRTICES contra triángulos con una tolerancia de 2 cm
    — en una esquina de muros de 20 cm a 45° los vértices quedan a ~7 cm. Un modelo real
    trae la esquina resuelta a inglete, que es lo que se construye acá.
    """
    h = t / 2.0
    dirs = []
    for i in range(len(eje) - 1):
        dx, dz = eje[i + 1][0] - eje[i][0], eje[i + 1][1] - eje[i][1]
        L = math.hypot(dx, dz)
        dirs.append((dx / L, dz / L))

    def lado(signo):
        pts = []
        offs = [
            ((eje[i][0] - dirs[i][1] * h * signo, eje[i][1] + dirs[i][0] * h * signo), dirs[i])
            for i in range(len(dirs))
        ]
        pts.append(offs[0][0])
        for i in range(len(offs) - 1):
            p = _inter_rectas(offs[i][0], offs[i][1], offs[i + 1][0], offs[i + 1][1])
            pts.append(p if p else offs[i][0])
        p_ult, u_ult = offs[-1]
        pts.append((p_ult[0] + u_ult[0] * _largo(eje, -1), p_ult[1] + u_ult[1] * _largo(eje, -1)))
        return pts

    return lado(+1) + list(reversed(lado(-1)))


def _largo(eje, i):
    a, b = eje[i - 1], eje[i]
    return math.hypot(b[0] - a[0], b[1] - a[1])


@pytest.mark.parametrize("scale", ESCALAS)
def test_esquina_oblicua_a_45_grados(scale):
    """En una unión a 45° el borde también debe apoyar sobre la cara del vecino.

    `oblique_trim_factor` estira el recorte por 1/sen(θ) porque la placa vecina se
    interpone a lo largo de más recorrido del muro que cede. No había ni un caso oblicuo
    en el banco: todos los modelos eran ortogonales. COBERTURA NUEVA, no regresión: la
    fórmula ya era correcta y este caso también pasa con el código anterior.
    """
    t = 0.20
    a = 6.0 / math.sqrt(2.0)
    eje = [(0.0, 0.0), (8.0, 0.0), (8.0 + a, a)]   # tramo en X + tramo a 45°
    b = _ObjBuilder()
    b.prisma(_planta_mitrada(eje, t), 0.0, 3.0)
    work, paneles, grupos, _ = _procesar(b.text(), scale)

    assert work.wall_wall_joints, "no se detectó la unión oblicua"
    ww = next((j for j in work.wall_wall_joints if j.yield_group_id is not None), None)
    assert ww is not None, "la unión oblicua quedó sin resolver"

    cede = grupos[ww.yield_group_id]
    otro = grupos[ww.group_b if ww.yield_group_id == ww.group_a else ww.group_a]
    panel = next(p for p in paneles if p.source_group_id == cede.id)

    d = _distancia_borde_a_plano_vecino(panel, cede, otro, work.faces)
    assert d is not None, "no se pudo medir el borde contra el plano del vecino"
    distancia_mm = d / scale * 1000.0
    assert distancia_mm == pytest.approx(1.5, abs=TOL_MM), (
        f"1:{scale:.0f} en la esquina a 45° el borde quedó a {distancia_mm:.3f}mm del "
        f"plano medio vecino, y debía quedar a 1.500mm"
    )


# ---------------------------------------------------------------------------
# Losas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", ESCALAS)
def test_entrepiso_entra_entre_muros_continuos(scale):
    """Un entrepiso entre dos muros que siguen de largo debe ENTRAR entre las placas.

    Los pisos no recibían NINGÚN recorte: salían con el ancho que tienen entre las pieles
    del modelo y al armar la maqueta no entraban. En un modelo real el entrepiso sobraba
    6.5 cm por lado (2.6 mm de plancha a 1:50) y se trababa. El muro acá no se puede
    acortar: el encuentro cae en su mitad, sigue de largo por arriba y por abajo.
    """
    # Tabique de 5 cm. El espesor importa para poder construir el caso: la losa tiene que
    # terminar contra la CARA interior del muro (si entrara en su cuerpo,
    # `split_wall_groups_at_floors` partiría el muro en dos tramos y el caso pasaría a ser
    # "la losa va entre dos tramos", que se resuelve recortando los muros). Con la losa
    # apoyada en la cara, su borde queda a t/2 del plano medio, y para que haya recorte a
    # TODA escala hace falta t/2 < media placa en la escala más gruesa: 1:20 da 3 cm, así
    # que un muro de 20 cm no sirve para este caso y uno de 5 cm sí.
    t = 0.05
    b = _ObjBuilder()
    b.box(0.0, t, 0.0, 6.0, 0.0, 5.0)            # muro oeste, entero de y=0 a y=6
    b.box(8.0 - t, 8.0, 0.0, 6.0, 0.0, 5.0)      # muro este, idem
    b.box(t, 8.0 - t, 3.0, 3.20, 0.5, 4.5)       # entrepiso a media altura
    work, paneles, grupos, _ = _procesar(b.text(), scale)

    losas = [p for p in paneles if grupos[p.source_group_id].category == "floor"
             and grupos[p.source_group_id].min_y is not None
             and abs(grupos[p.source_group_id].min_y - 3.0) < 0.05]
    assert losas, "no se detectó el entrepiso"

    media_placa_m = PLATE_THICKNESS_M * scale / 2.0
    revisados = 0
    for panel in losas:
        g = grupos[panel.source_group_id]
        marco = panel.frame
        assert marco is not None, "el panel del entrepiso no trae marco 3D"
        o, u, v = marco["origin"], marco["u_axis"], marco["v_axis"]

        for w in work.groups:
            if w.category != "wall" or w.min_y is None or w.max_y is None:
                continue
            # Sólo los muros que ATRAVIESAN el nivel de la losa.
            if not (w.min_y < g.min_y - 0.05 and w.max_y > g.max_y + 0.05):
                continue
            n = normalize(w.representative_normal)
            if abs(n.y) > 0.5:
                continue
            d_muro = mid_plane_offset(w, work.faces, n)
            # Sólo los muros que la losa TOCA. Las tapas laterales de los cajones de muro
            # también cruzan el nivel, pero la losa se queda a medio metro de ellas: ahí
            # no hay encastre y recortar abriría una luz. Mismo criterio que el código.
            bruto = [
                dot(n, v)
                for fi in g.face_indices
                if 0 <= fi < len(work.faces)
                for v in work.faces[fi].vertices
            ]
            if min(abs(min(bruto) - d_muro), abs(max(bruto) - d_muro)) > 0.30:
                continue
            # Borde de la pieza FINAL más cercano al plano de ese muro.
            proy = [
                n.x * (o["x"] + a * u["x"] + c * v["x"])
                + n.y * (o["y"] + a * u["y"] + c * v["y"])
                + n.z * (o["z"] + a * u["z"] + c * v["z"])
                for a, c in ((0.0, 0.0), (marco["width_m"], 0.0),
                             (marco["width_m"], marco["height_m"]), (0.0, marco["height_m"]))
            ]
            borde = min(proy, key=lambda p: abs(p - d_muro))
            distancia_mm = abs(d_muro - borde) / scale * 1000.0
            revisados += 1
            assert distancia_mm == pytest.approx(1.5, abs=TOL_MM), (
                f"1:{scale:.0f} el borde del entrepiso quedó a {distancia_mm:.2f}mm del "
                f"plano medio del muro g{w.id} y debía quedar a 1.50mm: no entra"
            )
    assert revisados >= 2, f"sólo se midieron {revisados} muros continuos, debían ser 2"


@pytest.mark.parametrize("scale", ESCALAS)
def test_extremo_del_muro_apoya_en_la_cara_de_la_losa_sin_junta_detectada(scale):
    """El extremo del muro topa contra la placa de losa AUNQUE no haya junta detectada.

    Las dos ramas que resolvían esto (`is_wall_on_top` / `is_roof_above_wall`) viven
    dentro del bucle de juntas, y `detect_joints` es topológico: si la losa no comparte
    vértices con el muro, no hay junta y el encuentro se quedaba sin resolver EN SILENCIO.
    Acá el muro va insetado en planta, así que no comparte ningún vértice con la losa.
    """
    b = _ObjBuilder()
    b.box(0.0, 8.0, 2.0, 2.40, 0.0, 6.0)          # losa de 40cm, plano medio y=2.20
    b.box(1.0, 7.0, 2.20, 5.0, 2.0, 2.20)         # muro que NACE en ese plano
    work, paneles, grupos, _ = _procesar(b.text(), scale)

    objetivo_mm = 1.5
    revisados = 0
    for panel in paneles:
        g = grupos[panel.source_group_id]
        if g.category != "wall" or g.min_y is None:
            continue
        if abs(g.min_y - 2.20) > 0.02:      # sólo el muro que este caso construye
            continue
        base_final = g.min_y + (_altura_bruta(g, work.faces) - panel.height_m)
        distancia_mm = abs(base_final - 2.20) / scale * 1000.0
        revisados += 1
        assert distancia_mm == pytest.approx(objetivo_mm, abs=TOL_MM), (
            f"1:{scale:.0f} la base del muro quedó a {distancia_mm:.2f}mm del plano medio "
            f"de la losa y debía quedar a {objetivo_mm:.2f}mm (media placa)"
        )
    assert revisados, "no se encontró el muro del caso"


def _altura_bruta(grupo, faces):
    """Alto del panel SIN recortar, para deducir cuánto se le quitó."""
    oriented = orient_group_normals_outward([grupo])
    res = project_faces_to_2d(
        [faces[i] for i in grupo.face_indices if i < len(faces)],
        oriented.get(grupo.id, grupo.representative_normal),
        "Y",
    )
    return res.height_m if res else 0.0


@pytest.mark.parametrize("scale", ESCALAS)
def test_losa_de_base_no_se_recorta(scale):
    """Sobre la losa de planta baja APOYAN los muros: no hay que achicarla.

    Es la contraparte del caso anterior y lo que hace que la distinción importe: si se
    recortara toda losa, la de base quedaría más chica que la huella del edificio.
    """
    t = 0.20
    b = _ObjBuilder()
    b.box(0.0, 8.0, 0.0, 0.20, 0.0, 5.0)         # losa de base
    b.box(0.0, t, 0.20, 6.0, 0.0, 5.0)           # muros que NACEN sobre ella
    b.box(8.0 - t, 8.0, 0.20, 6.0, 0.0, 5.0)
    work, paneles, grupos, _ = _procesar(b.text(), scale)

    base = [p for p in paneles if grupos[p.source_group_id].category == "floor"
            and grupos[p.source_group_id].min_y is not None
            and grupos[p.source_group_id].min_y < 0.05]
    assert base, "no se detectó la losa de base"
    for panel in base:
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
        assert recorte_mm == pytest.approx(0.0, abs=1e-6), (
            f"1:{scale:.0f} la losa de base se achicó {recorte_mm:.2f}mm, y los muros "
            "apoyan sobre ella: no hay nada entre lo que entrar"
        )


# ---------------------------------------------------------------------------
# Fase D — la verificación del ensamble tiene que valer para algo
#
# Un chequeo que puede dar un falso OK es peor que no tener ninguno. Estos casos son la
# validación del verificador: tiene que dar 0 donde el banco afirma que encastra, >0 donde
# se reintroduce un defecto conocido, y NO condenar los encastres por ranura, que se
# atraviesan a propósito.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", ESCALAS)
def test_verificacion_no_marca_un_ensamble_correcto(scale):
    """La esquina en L encastra —lo afirman los otros casos— así que no debe marcar nada."""
    work, paneles, grupos, pjs = _procesar(_modelo_esquina_L(t=0.20), scale)
    fallos = verificar_ensamble(paneles, pjs, scale)
    assert fallos == [], (
        f"1:{scale:.0f} el verificador marca {len(fallos)} choques en un ensamble que el "
        f"resto del banco da por bueno: {[(f.pieza_a, f.pieza_b, round(f.penetracion_mm,2)) for f in fallos]}"
    )


def _placa(pid, gid, origen, u_dir, v_dir, ancho, alto):
    """Placa rectangular puesta a mano, para probar el verificador contra casos cuya
    respuesta se calcula sin correr el pipeline."""
    from core.services.cutting_sheet import Edge2D, Panel
    from core.services.types import Vec2

    esq = [(0.0, 0.0), (ancho, 0.0), (ancho, alto), (0.0, alto)]
    edges = [
        Edge2D(a=Vec2(*esq[i]), b=Vec2(*esq[(i + 1) % 4])) for i in range(4)
    ]
    return Panel(
        id=pid, group_name=pid, category="wall", floor_index=0,
        width_m=ancho, height_m=alto, edges=edges, source_group_id=gid,
        frame={
            "origin": {"x": origen[0], "y": origen[1], "z": origen[2]},
            "u_axis": {"x": u_dir[0], "y": u_dir[1], "z": u_dir[2]},
            "v_axis": {"x": v_dir[0], "y": v_dir[1], "z": v_dir[2]},
            "normal": {"x": 0.0, "y": 0.0, "z": 0.0},
            "width_m": ancho, "height_m": alto, "mirrored": False,
        },
    )


@pytest.mark.parametrize("scale", ESCALAS)
def test_verificacion_detecta_dos_placas_que_se_pisan(scale):
    """Dos placas perpendiculares donde una ATRAVIESA a la otra: tiene que marcarlo.

    Es la otra mitad de la validación: sin este caso, un verificador que devolviera
    siempre la lista vacía pasaría el caso anterior. Se arma a mano en vez de romper el
    pipeline, porque quitarle los recortes al modelo en L hace que la ranura resuelva la
    junta legítimamente — o sea que no queda roto.
    """
    media = PLATE_THICKNESS_M * scale / 2.0
    # A en el plano z=0, de x 0 a 4. B en el plano x=2, de z -1 a 1: lo cruza al medio.
    a = _placa("A1", 1, (0.0, 0.0, 0.0), (1, 0, 0), (0, 1, 0), 4.0, 3.0)
    b = _placa("A2", 2, (2.0, 0.0, -1.0), (0, 0, 1), (0, 1, 0), 2.0, 3.0)
    fallos = verificar_ensamble([a, b], [], scale)
    assert fallos, f"1:{scale:.0f} dos placas que se atraviesan y no se detectó nada"
    # La profundidad es una COTA INFERIOR (sale del muestreo): tiene que ser positiva y no
    # puede pasar de media placa, que es lo máximo que una placa puede meterse en otra
    # antes de salir del otro lado.
    p = fallos[0].penetracion_mm
    assert 0.0 < p <= 1.5 + TOL_MM, (
        f"1:{scale:.0f} penetración informada fuera de rango: {p:.3f}mm (media placa = 1.5)"
    )


@pytest.mark.parametrize("scale", ESCALAS)
def test_verificacion_no_marca_un_tope_correcto(scale):
    """Un tope bien resuelto es TANGENTE: el borde de A apoya sobre la cara de B.

    Si el verificador marcara esto, condenaría todos los encastres correctos.
    """
    media = PLATE_THICKNESS_M * scale / 2.0
    # B en el plano x=2. A termina justo a media placa de ese plano: apoya, no penetra.
    a = _placa("A1", 1, (0.0, 0.0, 0.0), (1, 0, 0), (0, 1, 0), 2.0 - media, 3.0)
    b = _placa("A2", 2, (2.0, 0.0, -1.0), (0, 0, 1), (0, 1, 0), 2.0, 3.0)
    fallos = verificar_ensamble([a, b], [], scale)
    assert fallos == [], (
        f"1:{scale:.0f} un tope tangente se marcó como choque: "
        f"{[(f.pieza_a, f.pieza_b, round(f.penetracion_mm,3)) for f in fallos]}"
    )


@pytest.mark.parametrize("scale", ESCALAS)
def test_verificacion_no_condena_un_encastre_por_ranura(scale):
    """Dos placas encastradas SE ATRAVIESAN: la ranura es el hueco que lo permite.

    Una primera versión de este chequeo las marcaba como choque. Es el falso positivo que
    haría inservible toda la Fase D.
    """
    # Cruce en X: un muro atraviesa a otro por el medio de ambos → se resuelve con ranura.
    b = _ObjBuilder()
    t = 0.20
    b.box(-4.0, 4.0, 0.0, 3.0, 0.0, t)
    b.box(-0.1, 0.1, 0.0, 3.0, -3.0, 3.0)
    work, paneles, grupos, pjs = _procesar(b.text(), scale)
    assert pjs, "el cruce en X no generó ninguna ranura: el caso no prueba lo que dice"

    con_ranura = {pair_key(pj.cutter_id, pj.cut_id) for pj in pjs}
    gid = {p.id: p.source_group_id for p in paneles}
    for f in verificar_ensamble(paneles, pjs, scale):
        k = pair_key(gid[f.pieza_a], gid[f.pieza_b])
        assert k not in con_ranura, (
            f"1:{scale:.0f} {f.pieza_a}/{f.pieza_b} tienen ranura entre sí y el "
            "verificador las marcó como choque"
        )


@pytest.mark.parametrize("scale", ESCALAS)
def test_contacto_horizontal_se_resuelve_recortando_la_altura(scale):
    """Dos muros que se encuentran A LO LARGO ceden en ALTURA, no en ancho.

    Un faldón de techo apoyado sobre el borde de un muro: el contacto es horizontal y
    recorre las dos piezas de punta a punta. Preguntar si la junta cae en un extremo
    LATERAL no tiene sentido ahí, y acortar el ancho no resuelve nada. Antes esas juntas
    se marcaban como "cruce en medio de ambos", se delegaban a una ranura que tampoco
    correspondía —no se atraviesan, sólo se tocan— y quedaban sin recorte y sin encastre.
    """
    b = _ObjBuilder()
    b.box(0.0, 8.0, 0.0, 2.0, 0.0, 0.20)                    # muro vertical
    b.losa_inclinada(0.0, 8.0, 0.10, 2.0, 3.10, 5.0, 0.20)  # faldón que apoya en su cima
    work, paneles, grupos, pjs = _procesar(b.text(), scale)

    horizontales = [
        j for j in work.wall_wall_joints
        if work.joints[j.joint_index].horizontal_frac >= 0.5
    ]
    assert horizontales, "el modelo no produjo ninguna unión de contacto horizontal"
    for j in horizontales:
        assert j.yield_group_id is not None, (
            f"1:{scale:.0f} la unión horizontal g{j.group_a}-g{j.group_b} quedó sin "
            "resolver: no se recortó ninguna de las dos y no hay ranura"
        )

    # Y que esas dos piezas efectivamente dejen de pisarse. La Fase D como oráculo del
    # arreglo. Se mira SÓLO el par de cada unión horizontal: el modelo genera además
    # tapas horizontales que se clasifican como piso, y sus encuentros son otro caso
    # (muro-losa) que este test no cubre.
    gid = {p.id: p.source_group_id for p in paneles}
    pares = {pair_key(j.group_a, j.group_b) for j in horizontales}
    choques = [
        f for f in verificar_ensamble(paneles, pjs, scale)
        if pair_key(gid[f.pieza_a], gid[f.pieza_b]) in pares
    ]
    assert choques == [], (
        f"1:{scale:.0f} las piezas de la unión horizontal se siguen pisando: "
        f"{[(f.pieza_a, f.pieza_b, round(f.penetracion_mm, 2)) for f in choques]}"
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
