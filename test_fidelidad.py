"""
Fidelidad al modelo: el control que no depende de ninguna decisión del sistema.

Origen
------
Tres arreglos para las losas mejoraban una escala y empeoraban la otra, y no había forma
de saber cuál era el correcto: todo se medía contra la línea base del propio motor (cuántas
uniones encajan), que dice si las piezas encajan ENTRE SÍ, no si son el edificio del .obj.
Este control mide contra la malla misma.

Con él aparecieron dos defectos que el verificador no podía ver:
* las losas que el modelo trae con una sola cara (la de arriba) tenían la placa centrada
  en esa cara, media placa por encima de la losa real; y las que traían caras de vigas
  colgadas, centrada en el medio de la viga;
* el nivel de la losa contra el que se cortan los muros salía de OTRA cuenta (el centro de
  la caja de sus caras), así que cuando la placa de la losa se ubicaba bien, los muros
  quedaban cortados contra la posición vieja.
"""

import numpy as np
import pytest

import modelo_edificio_prueba as mp
from core.pipeline import parse_pipeline
from core.review_generate import _decompose
from core.services.fidelidad import Solido, _distancia_punto_triangulos, plano_en_el_material
from core.services.obj_parser import parse_obj
from core.services.planos_por_volumen import capa_de_losa
from core.services.types import PipelineOptions

ESCALAS = (50.0, 100.0)


def _p1(texto):
    parsed = parse_obj(texto)
    return parse_pipeline("m.obj", parsed["faces"], parsed["warnings"], force_axis="Y")


# ---------------------------------------------------------------------------
# ¿Un punto está dentro del material?
# ---------------------------------------------------------------------------

CASA = [
    ((2.0, 1.0, 0.10), True), ((2.0, 1.0, 0.30), False), ((2.0, 2.9, 3.0), True),
    ((2.0, 2.75, 3.0), False), ((0.0, 1.0, 0.0), True), ((4.0, -0.1, 3.0), True),
    ((10.0, 1.0, 3.0), False), ((4.0, 1.0, 3.0), True), ((4.2, 1.0, 3.0), False),
]
EDIFICIO = [
    ((2.0, 1.5, 0.075), False),   # vano de una ventana
    ((2.0, 0.5, 0.075), True),    # antepecho
    ((8.5, 2.775, 6.0), False),   # hueco de la escalera
    ((5.0, 2.775, 6.0), True),    # entrepiso
    ((4.9, 1.0, 0.075), False),   # vano de la puerta
]


@pytest.mark.parametrize("texto,puntos", [(mp.texto_casa(None), CASA), (mp.texto(None), EDIFICIO)],
                         ids=["casa", "edificio"])
def test_el_solido_sabe_que_es_material_y_que_es_aire(texto, puntos):
    s = Solido.desde_caras(parse_obj(texto)["faces"])
    for p, esperado in puntos:
        assert s.contiene(p) == esperado, p


def test_distancia_exacta_a_un_triangulo():
    T = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
    for p, d in [((0.2, 0.2, 0.5), 0.5), ((2, 0, 0), 1.0), ((-1, -1, 0), 2 ** 0.5),
                 ((0.5, 0.5, 0), 0.0), ((1, 1, 0), 0.5 ** 0.5), ((0.5, -1, 0), 1.0)]:
        assert _distancia_punto_triangulos(np.array(p, float), T)[0] == pytest.approx(d)


# ---------------------------------------------------------------------------
# La placa de cada losa va en su capa de material
# ---------------------------------------------------------------------------

def test_la_capa_de_cada_losa_es_la_del_modelo():
    p1 = _p1(mp.texto(None))
    s = Solido.desde_caras(p1.raw_faces)
    losas = sorted((g for g in p1.groups if g.category == "floor"), key=lambda g: g.centroid.y)
    esperadas = [(-mp.T, 0.0), mp.Y_EP, mp.Y_TE]
    assert len(losas) == 3
    for g, (y0, y1) in zip(losas, esperadas):
        c = capa_de_losa(g, p1.faces, s)
        assert c is not None
        assert c[0] == pytest.approx(y0, abs=0.006) and c[1] == pytest.approx(y1, abs=0.006)


@pytest.mark.parametrize("escala", ESCALAS)
@pytest.mark.parametrize("texto", [mp.texto_casa(None), mp.texto(None)], ids=["casa", "edificio"])
def test_en_un_modelo_correcto_todas_las_placas_pasan_por_material(texto, escala):
    p1 = _p1(texto)
    s = Solido.desde_caras(p1.raw_faces)
    opts = PipelineOptions(scale_denom=escala, paper="A4", min_area_m2=1.0, sheet_config=None)
    _w, muros, pisos, _ = _decompose(p1, opts, None, None, None, None)
    r = plano_en_el_material(muros + pisos, s, escala, buscar_corrimiento=False)
    mal = [(x.pieza, round(x.fraccion, 2)) for x in r if x.fraccion < 0.9]
    assert not mal, mal


def test_el_solido_se_arma_con_las_caras_originales():
    """Las caras de trabajo están partidas a la altura de las losas y esa partición rompe
    el conteo de cruces. El sólido del sistema tiene que salir de las originales."""
    from core.services.planos_por_volumen import solido_del_modelo

    p1 = _p1(mp.texto(None))
    s = solido_del_modelo(p1.raw_faces, p1.applied_axis)
    for p, esperado in EDIFICIO:
        assert s.contiene(p) == esperado, p


def test_un_tabique_de_dos_laminas_sueltas_cuenta_como_material():
    """Un tabique de 5 cm que el modelo trae como dos caras sin cerrar no encierra volumen:
    la paridad de rayos lo ve como aire. Entre las dos láminas igual es material."""
    texto = "\n".join([
        "v 0 0 0", "v 0 3 0", "v 0 3 4", "v 0 0 4",
        "v 0.05 0 0", "v 0.05 3 0", "v 0.05 3 4", "v 0.05 0 4",
        "f 1 2 3 4", "f 5 6 7 8",
    ])
    sol = Solido.desde_caras(_p1(texto).raw_faces)
    n = np.array([1.0, 0.0, 0.0])
    adentro = np.array([[0.025, 1.5, 2.0]])
    assert not sol.contiene_varios(adentro)[0]
    assert sol.en_material_o_superficie(adentro, normal=n)[0]
    afuera = np.array([[0.25, 1.5, 2.0]])
    assert not sol.en_material_o_superficie(afuera, normal=n)[0]
    # una sola lámina cerca no alcanza: tiene que haber material de los dos lados
    al_lado = np.array([[-0.04, 1.5, 2.0]])
    assert not sol.en_material_o_superficie(al_lado, tol=0.02, normal=n)[0]


def _girar_y_a_z(texto):
    """El mismo .obj con Z hacia arriba: (x, y, z) -> (x, -z, y)."""
    out = []
    for linea in texto.splitlines():
        if linea.startswith("v "):
            x, y, z = (float(t) for t in linea.split()[1:4])
            linea = f"v {x} {-z} {y}"
        out.append(linea)
    return "\n".join(out)


@pytest.mark.parametrize("texto", [mp.texto_casa(None), mp.texto(None)], ids=["casa", "edificio"])
def test_el_eje_vertical_de_una_casa_casi_cubica(texto):
    """Una casa casi cúbica tiene losas y fachadas de área parecida: la heurística de áreas
    no decide y la de la caja tampoco. Las losas cubren la planta; las fachadas no."""
    from core.services.facade_extractor import detect_up_axis

    assert detect_up_axis(parse_obj(texto)["faces"]) == "Y"
    assert detect_up_axis(parse_obj(_girar_y_a_z(texto))["faces"]) == "Z"
