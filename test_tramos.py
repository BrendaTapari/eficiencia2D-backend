"""
Los tramos de un mismo muro comparten el plano de su placa.

Origen
------
De los 43 problemas que el instructivo atribuía "al modelo" en `Prueba app.obj` a 1:100,
42 desaparecían con una placa más fina: los generaba el sistema, no el modelo. El grupo
más numeroso era siempre el mismo: un muro que el modelo trae en dos tramos de distinto
espesor (25 cm de un lado de un encuentro, 20 del otro, cara exterior a ras). Cada tramo
ponía su placa en SU plano medio y las dos placas del mismo muro quedaban corridas 2.5 cm:
la pieza que llega al encuentro no puede terminar contra las dos.

Con un solo plano por muro, a 1:100 en ese modelo: piezas verificadas 204 -> 227, lugares
para revisar 52 -> 37, atribuidos al modelo 43 -> 28, sin empeorar nada (demo.obj no
cambia: no tiene tramos así).
"""

import pytest

import modelo_edificio_prueba as mp
from core.pipeline import parse_pipeline
from core.review_generate import _decompose
from core.services.obj_parser import parse_obj
from core.services.types import PipelineOptions

ESCALAS = (50.0, 100.0)


def _armar(texto, escala):
    parsed = parse_obj(texto)
    p1 = parse_pipeline("m.obj", parsed["faces"], parsed["warnings"], force_axis="Y")
    opts = PipelineOptions(scale_denom=escala, paper="A4", min_area_m2=1.0, sheet_config=None)
    return _decompose(p1, opts, None, None, None, None)


def _plano(pieza):
    """Posición del plano de la pieza a lo largo de su normal (signo hacia +eje)."""
    f = pieza.frame
    n = (f["normal"]["x"], f["normal"]["y"], f["normal"]["z"])
    o = (f["origin"]["x"], f["origin"]["y"], f["origin"]["z"])
    k = max(range(3), key=lambda i: abs(n[i]))
    return k, o[k]


def _dos_tramos():
    """El muro oeste del modelo de prueba partido en sus dos tramos, como grupos
    separados (así llegan en un modelo real; acá el clasificador los junta porque
    comparten la cara exterior)."""
    from dataclasses import replace

    parsed = parse_obj(mp.texto_muro_en_tramos())
    p1 = parse_pipeline("m.obj", parsed["faces"], parsed["warnings"], force_axis="Y")
    oeste = next(
        g for g in p1.groups
        if g.category == "wall" and abs(g.representative_normal.x) > 0.9 and g.centroid.x < 1
    )

    def z_medio(i):
        vs = p1.faces[i].vertices
        return sum(v.z for v in vs) / len(vs)

    # Las pieles interiores llegan como grupos aparte; el armado las junta con la
    # exterior (`placas_superpuestas`, espalda con espalda). Acá se hace lo mismo.
    interiores = [
        g for g in p1.groups
        if g.category == "wall" and abs(g.representative_normal.x) > 0.9
        and g.id != oeste.id and g.centroid.x < 1
    ]
    norte = [i for i in oeste.face_indices if z_medio(i) < 4.0] + [
        i for g in interiores if g.centroid.z < 4.0 for i in g.face_indices]
    sur = [i for i in oeste.face_indices if z_medio(i) > 4.0] + [
        i for g in interiores if g.centroid.z > 4.0 for i in g.face_indices]
    usados = {oeste.id} | {g.id for g in interiores}
    otros = [g for g in p1.groups if g.id not in usados]
    tramos = [
        replace(oeste, id=9001, face_indices=norte),
        replace(oeste, id=9002, face_indices=sur),
    ]
    return p1, otros + tramos


@pytest.mark.parametrize("escala", ESCALAS)
def test_los_dos_tramos_del_muro_comparten_plano(escala):
    from core.services import plate_intersect as pi
    from core.services.types import Vec3

    p1, grupos = _dos_tramos()
    movidos = pi.alinear_tramos(grupos, p1.faces, lambda g: g.category, 0.003 * escala)
    assert set(movidos) == {9001, 9002}, movidos
    d1, d2 = (abs(movidos[g]) for g in (9001, 9002))
    assert d1 == pytest.approx(d2, abs=1e-9)
    # Entre los dos planos medios (12.5 y 10 cm): no se va afuera del muro.
    assert 0.10 - 1e-6 <= d1 <= 0.125 + 1e-6
    # Y es el plano que ve todo el sistema (placa, recorte, ranura, instructivo).
    for g in grupos:
        if g.id in (9001, 9002):
            caras = [p1.faces[i] for i in g.face_indices]
            n = g.representative_normal
            assert abs(pi.mid_plane_offset_faces(caras, Vec3(n.x, n.y, n.z))) == pytest.approx(d1)
    pi.olvidar_alineacion()


@pytest.mark.parametrize("escala", ESCALAS)
def test_no_se_alinea_lo_que_no_son_tramos_lado_a_lado(escala):
    """Muros de plantas distintas (con una losa en el medio) y caras sueltas sin espesor
    conservan su plano: alinearlos movía piezas que estaban bien (demo.obj)."""
    from core.services import plate_intersect as pi

    for texto in (mp.texto_casa(None), mp.texto(None)):
        parsed = parse_obj(texto)
        p1 = parse_pipeline("m.obj", parsed["faces"], parsed["warnings"], force_axis="Y")
        movidos = pi.alinear_tramos(
            p1.groups, p1.faces, lambda g: g.category, 0.003 * escala,
        )
        assert movidos == {}, movidos
        pi.olvidar_alineacion()


def test_el_registro_no_confunde_grupos_de_otro_armado():
    """El plano común se guarda con las caras mismas: un grupo de otro modelo con la
    misma cantidad de caras no lo hereda."""
    from core.services import plate_intersect as pi
    from core.services.types import Vec3

    p1, grupos = _dos_tramos()
    movidos = pi.alinear_tramos(grupos, p1.faces, lambda g: g.category, 0.3)
    assert len(movidos) == 2
    otro = parse_obj(mp.texto_muro_en_tramos())
    p2 = parse_pipeline("m.obj", otro["faces"], otro["warnings"], force_axis="Y")
    for g in p2.groups:
        caras = [p2.faces[i] for i in g.face_indices]
        n = g.representative_normal
        assert pi._PLANOS_ALINEADOS.buscar(caras, Vec3(n.x, n.y, n.z)) is None
    pi.olvidar_alineacion()


# Línea base del modelo real donde aparece el problema. Si cambia, que sea con una
# explicación en este mismo lugar.
#   (piezas verificadas, lugares para revisar, de ellos atribuidos al modelo)
#   1:100  (204, 52, 43) -> (227, 37, 28) con un solo plano por muro
#   1:50   (189, 71, 29) -> (207, 59, 29)
#
# Losas por volumen, muros bajo viga, cruces que no se recortan y pieles dentro del espesor
# del muro. Juzgado primero por FIDELIDAD (test_fidelidad.py, contra el sólido del .obj) y
# después por el verificador:
#   1:50   caras del modelo representadas 95.67% -> 97.43%; no arman 59 -> 43
#   1:100  caras del modelo representadas 96.79% -> 97.38%; no arman 25 -> 29
# A 1:100 bajan 3 piezas verificadas (227 -> 224) y suben 4 uniones que no arman: muros que
# ahora llegan a la viga o conservan los metros que se les recortaban, y se cruzan con
# piezas cuyo encastre no los contemplaba; y una carpintería de 3 cm en un vano que llega
# al borde (pendiente: elementos más finos que la placa).
#   1:100  (227, 37, 28) -> (224, 34, 23)
#   1:50   (207, 59, 29) -> (211, 58, 24)
#
# Plano medio por niveles de cara y cierre de ranuras finas: 1:50 mejora en todo
# (212 -> 229 piezas verificadas en el reporte global, no arman 42 -> 38); a 1:100 suben 2 piezas
# verificadas y 2 nudos (esquinas contra fachadas partidas en franjas, pendiente).
#   1:100  (224, 34, 23) -> (226, 36, 21)
LINEA_BASE_PRUEBA_APP = {50.0: (211, 58, 24), 100.0: (226, 36, 21)}


@pytest.mark.parametrize("escala", ESCALAS)
def test_linea_base_del_modelo_real(escala):
    from pathlib import Path

    from core.review_generate import compute_verificacion

    ruta = Path(__file__).parent / "Prueba app.obj"
    if not ruta.exists():
        pytest.skip("falta el modelo grande")
    parsed = parse_obj(ruta.read_text())
    p1 = parse_pipeline(ruta.name, parsed["faces"], parsed["warnings"])
    opts = PipelineOptions(scale_denom=escala, paper="A4", min_area_m2=1.0, sheet_config=None)
    r = compute_verificacion(p1, opts)[0].verificacion["resumen"]
    actual = (r["piezas_ok"], r["nudos"], r["nudos_por_quien"]["modelo"])
    ok, nudos, modelo = LINEA_BASE_PRUEBA_APP[escala]
    assert actual[0] >= ok and actual[1] <= nudos and actual[2] <= modelo, (
        f"1:{escala:.0f}: {actual} contra {LINEA_BASE_PRUEBA_APP[escala]}. Si es una mejora, "
        "actualizá LINEA_BASE_PRUEBA_APP."
    )
