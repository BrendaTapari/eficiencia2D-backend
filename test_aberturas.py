"""
Las aberturas del modelo tienen que salir cortadas.

Origen
------
Se armó un edificio de prueba con ventanas para ver si el verificador detectaba errores de
modelado, y lo que apareció fue otra cosa: en el edificio CORRECTO las piezas salían sin
ventanas. Un muro de 10 x 2.7 m con una ventana se cortaba como un rectángulo macizo.

No era del modelo de prueba. Medido sobre los modelos reales: `demo.obj` a 1:100 cortaba un
muro de 16 ventanas sin ninguna, y `Prueba app.obj` cortaba macizas dos losas con huecos de
unos 21 m² cada una. Las puertas, que tocan el borde de la pieza, no se perdían.

La causa era una sola función (`_material_de_la_pieza`) que rellenaba toda abertura cerrada
de cualquier pieza que pasara por el recorte de encaje. Nadie lo vio porque los modelos del
corpus traen vidrios y carpinterías modelados dentro de los vanos: ahí "ventana tapada"
parecía una decisión y no un error.

Qué se fija acá
---------------
* Una abertura VACÍA se corta abierta, con su medida.
* Una abertura con algo modelado adentro (vidrio, carpintería) se corta cerrada, se AVISA,
  y se abre descartando ese elemento.
* En ningún modelo del corpus una pieza tiene más material que el que tiene en el modelo.
"""

import pytest
from shapely.geometry import Polygon

import modelo_edificio_prueba as edificio
import test_corpus as tc
from core.pipeline import parse_pipeline
from core.review_generate import _decompose, _material_de_la_pieza
from core.services.cutting_sheet import (
    Edge2D,
    _edges_to_polygon,
    orient_group_normals_outward,
    project_faces_to_2d,
)
from core.services.obj_parser import parse_obj
from core.services.plate_intersect import PLATE_THICKNESS_M
from core.services.types import PipelineOptions, Vec2

ESCALAS = (50.0, 100.0)


def _anillo(pts, hole=False):
    return [
        Edge2D(a=Vec2(*pts[i]), b=Vec2(*pts[(i + 1) % len(pts)]), hole=hole)
        for i in range(len(pts))
    ]


def _huecos(poly):
    """[área de cada abertura cerrada] de una pieza."""
    return [
        Polygon(h).area for g in getattr(poly, "geoms", [poly]) for h in g.interiors
    ]


def _procesar(texto, escala, overrides=None):
    parsed = parse_obj(texto)
    p1 = parse_pipeline("edificio.obj", parsed["faces"], parsed["warnings"], force_axis="Y")
    opts = PipelineOptions(scale_denom=escala, paper="A4", min_area_m2=1.0, sheet_config=None)
    work, muros, pisos, _ = _decompose(p1, opts, overrides, None, None, None)
    return p1, work, muros + pisos


# ---------------------------------------------------------------------------
# La función que estaba mal, con casos que se calculan a mano
# ---------------------------------------------------------------------------

def test_el_material_de_un_muro_con_ventana_tiene_la_ventana():
    bordes = _anillo([(0, 0), (10, 0), (10, 2.7), (0, 2.7)]) + _anillo(
        [(1.2, 0.9), (2.7, 0.9), (2.7, 2.1), (1.2, 2.1)], True
    )
    material = _material_de_la_pieza(bordes)
    assert material.area == pytest.approx(27.0 - 1.8)
    assert len(material.interiors) == 1


def test_el_material_respeta_islas_dentro_de_una_abertura():
    """Par-impar de verdad: una región dentro de un hueco vuelve a ser material."""
    bordes = (
        _anillo([(0, 0), (10, 0), (10, 10), (0, 10)])
        + _anillo([(2, 2), (8, 2), (8, 8), (2, 8)], True)
        + _anillo([(4, 4), (6, 4), (6, 6), (4, 6)])
    )
    material = _material_de_la_pieza(bordes)
    assert material.area == pytest.approx(100 - 36 + 4)


def test_el_material_de_dos_piezas_sueltas_con_ventana_cada_una():
    bordes = (
        _anillo([(0, 0), (4, 0), (4, 3), (0, 3)])
        + _anillo([(1, 1), (2, 1), (2, 2), (1, 2)], True)
        + _anillo([(6, 0), (10, 0), (10, 3), (6, 3)])
        + _anillo([(7, 1), (9, 1), (9, 2), (7, 2)], True)
    )
    material = _material_de_la_pieza(bordes)
    assert material.area == pytest.approx(12 - 1 + 12 - 2)


# ---------------------------------------------------------------------------
# El edificio de prueba: respuesta conocida
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("escala", ESCALAS)
def test_cada_ventana_del_edificio_sale_cortada_con_su_medida(escala):
    _, work, piezas = _procesar(edificio.texto(None), escala)
    cortadas = sorted(
        a for p in piezas if p.category == "wall" for a in _huecos(_edges_to_polygon(p.edges))
    )
    esperadas = sorted(
        w * h for planta in edificio.VENTANAS.values() for (w, h) in planta
    )
    assert len(cortadas) == len(esperadas), (
        f"1:{escala:.0f}: el modelo tiene {len(esperadas)} ventanas y se cortan {len(cortadas)}"
    )
    for a, b in zip(cortadas, esperadas):
        assert a == pytest.approx(b, abs=1e-3)
    # Y el hueco de la escalera, en el entrepiso.
    en_losas = [a for p in piezas if p.category == "floor" for a in _huecos(_edges_to_polygon(p.edges))]
    w, h = edificio.HUECO_DE_ESCALERA
    assert en_losas == [pytest.approx(w * h, abs=1e-3)]
    # Ninguna abertura vacía tiene por qué generar un aviso.
    assert not [a for a in (work.warnings or []) if "CERRADAS" in a]


@pytest.mark.parametrize("escala", ESCALAS)
def test_el_edificio_correcto_sigue_dando_listo(escala):
    """Abrir las ventanas no puede inventar problemas de encaje."""
    from core.review_generate import reporte_de_verificacion

    _, work, piezas = _procesar(edificio.texto(None), escala)
    muros = [p for p in piezas if p.category == "wall"]
    pisos = [p for p in piezas if p.category == "floor"]
    v = reporte_de_verificacion(work, muros, pisos, escala)
    assert v["veredicto"] == "listo", [n["titulo"] for n in v["nudos"]]
    assert len(piezas) == 13


@pytest.mark.parametrize("escala", ESCALAS)
def test_una_ventana_con_vidrio_sale_cerrada_y_se_avisa(escala):
    """Lo que el modelo trae dentro del vano ocupa la misma placa que el muro: se corta
    con él y el vano sale cerrado. No puede pasar en silencio."""
    p1, work, piezas = _procesar(edificio.texto("con_vidrios"), escala)
    assert len(piezas) == 13, "los vidrios no son piezas aparte"
    ventanas = [a for p in piezas if p.category == "wall" for a in _huecos(_edges_to_polygon(p.edges))]
    total = sum(len(v) for v in edificio.VENTANAS.values())
    assert len(ventanas) == total - edificio.VENTANAS_CON_VIDRIO
    avisos = [a for a in (work.warnings or []) if "CERRADAS" in a]
    assert len(avisos) == 1, work.warnings
    assert avisos[0].startswith(f"{edificio.VENTANAS_CON_VIDRIO} abertura(s)"), avisos[0]
    assert "descartá" in avisos[0]


# ---------------------------------------------------------------------------
# En cualquier modelo: una pieza no puede tener más material que el modelo
# ---------------------------------------------------------------------------

def _material_de_mas(work, piezas, escala):
    """[(etiqueta, m² de más)] de las piezas con más material que su grupo en el modelo.

    El recorte de encaje puede ALARGAR una pieza para tapar un canto: a lo sumo una placa
    por cada lado. Lo que pase de eso es material inventado; una ventana rellena lo supera
    de lejos.
    """
    normales = orient_group_normals_outward(work.groups)
    por_id = {g.id: g for g in work.groups}
    placa = PLATE_THICKNESS_M * escala
    de_mas = []
    for p in piezas:
        g = por_id.get(p.source_group_id)
        if g is None:
            continue
        r = project_faces_to_2d(
            [work.faces[i] for i in g.face_indices],
            normales.get(g.id, g.representative_normal), "Y",
        )
        if not r:
            continue
        crudo = _edges_to_polygon(r.edges)
        final = _edges_to_polygon([e for e in p.edges if not getattr(e, "flex", False)])
        margen = 2.0 * (r.width_m + r.height_m) * placa + 0.05
        sobra = final.area - crudo.area - margen
        if sobra > 0:
            de_mas.append((p.id, round(sobra, 2)))
    return de_mas


# PENDIENTE, anterior a esta corrección y sin relación con las aberturas: en demo.obj la
# losa B5 pasa de 14 m² en el modelo a unos 30 m² al recortarla (da idéntico con y sin el
# arreglo de las aberturas). No se investigó. Se deja a la vista para que no se confunda
# con una abertura rellena y para que, si se arregla, esta lista haya que achicarla.
MATERIAL_DE_MAS_CONOCIDO = {("demo.obj", "B5")}


@pytest.mark.skipif(not tc.MODELOS, reason="no hay modelos disponibles")
@pytest.mark.parametrize("path", tc.MODELOS, ids=lambda p: p.name)
@pytest.mark.parametrize("escala", ESCALAS)
def test_ninguna_pieza_tiene_mas_material_que_el_modelo(path, escala):
    work, muros, pisos, _ = tc._procesar(path, escala)
    de_mas = [
        x for x in _material_de_mas(work, muros + pisos, escala)
        if (path.name, x[0]) not in MATERIAL_DE_MAS_CONOCIDO
    ]
    assert not de_mas, (
        f"{path.name} 1:{escala:.0f}: piezas con material que el modelo no tiene "
        f"(¿una abertura rellena?): {de_mas[:8]}"
    )


@pytest.mark.parametrize("escala", ESCALAS)
def test_ninguna_pieza_del_edificio_tiene_mas_material_que_el_modelo(escala):
    for variante in (None, "con_vidrios"):
        _, work, piezas = _procesar(edificio.texto(variante), escala)
        assert not _material_de_mas(work, piezas, escala), variante
