"""
Huecos que el modelo deja en la maqueta (`core/services/sospechas_modelo.py`).

Origen
------
Se armaron dos edificios con 16 errores de modelado a propósito y el verificador dijo
"LISTO" en los 16. Midiendo las piezas cortadas: unos los corrige el motor solo, otros
los reproduce tal cual (y sólo quien diseñó sabe si eran la idea), y otros dejan un HUECO
en la maqueta. Éstos son los que se tienen que avisar.

Lo que fija esta prueba
-----------------------
* Ningún aviso en los modelos correctos. Un aviso falso en un modelo bien hecho es lo que
  hace que la gente deje de leer los avisos.
* Cada error que deja un hueco en las piezas se avisa, con las dos piezas y la medida.
* El aviso no cambia el veredicto ni las piezas.
"""

import pytest

import modelo_edificio_prueba as mp
from core.pipeline import parse_pipeline
from core.review_generate import _decompose, reporte_de_verificacion
from core.services.obj_parser import parse_obj
from core.services.sospechas_modelo import detectar_huecos
from core.services.types import PipelineOptions

ESCALAS = (50.0, 100.0)


def _armar(texto, escala):
    parsed = parse_obj(texto)
    p1 = parse_pipeline("m.obj", parsed["faces"], parsed["warnings"], force_axis="Y")
    opts = PipelineOptions(scale_denom=escala, paper="A4", min_area_m2=1.0, sheet_config=None)
    work, muros, pisos, _ = _decompose(p1, opts, None, None, None, None)
    return work, muros, pisos


@pytest.mark.parametrize("escala", ESCALAS)
@pytest.mark.parametrize("texto", [mp.texto_casa(None), mp.texto(None), mp.texto("con_vidrios")],
                         ids=["casa", "edificio", "edificio_con_vidrios"])
def test_ningun_aviso_en_un_modelo_correcto(texto, escala):
    work, muros, pisos = _armar(texto, escala)
    huecos = detectar_huecos(muros + pisos, escala)
    assert huecos == [], [h["texto"] for h in huecos]


# (modelo, escala, pieza que no llega o par esperado por categoría, separación mínima en cm)
CON_HUECO = [
    ("casa", "corto25", 50.0, 20), ("casa", "corto25", 100.0, 15),
    ("casa", "corto5", 50.0, 5),
    ("casa", "flotante", 50.0, 15), ("casa", "flotante", 100.0, 10),
    ("casa", "corrido", 50.0, 10), ("casa", "corrido", 100.0, 10),
    ("edificio", "esquina_abierta", 50.0, 8), ("edificio", "esquina_abierta", 100.0, 1),
    ("edificio", "tabique_no_llega", 50.0, 6),
]


@pytest.mark.parametrize("familia,error,escala,minimo_cm", CON_HUECO)
def test_un_error_que_deja_hueco_se_avisa(familia, error, escala, minimo_cm):
    texto = mp.texto_casa(error) if familia == "casa" else mp.texto(error)
    work, muros, pisos = _armar(texto, escala)
    huecos = detectar_huecos(muros + pisos, escala)
    assert huecos, f"{familia}/{error} 1:{escala:.0f}: el hueco no se avisó"
    assert max(h["medida_cm_edificio"] for h in huecos) >= minimo_cm
    for h in huecos:
        assert h["largo_cm_edificio"] >= 50
        assert len(h["piezas"]) == 2 and h["texto"] and h["pregunta"]


@pytest.mark.parametrize("escala", ESCALAS)
def test_el_aviso_va_en_el_reporte_sin_cambiar_el_veredicto(escala):
    work, muros, pisos = _armar(mp.texto("esquina_abierta"), escala)
    reporte = reporte_de_verificacion(work, muros, pisos, escala)
    assert reporte["veredicto"] == "listo"
    assert reporte["resumen"]["huecos"] == len(reporte["huecos"]) >= 1
    ok = reporte_de_verificacion(*_armar(mp.texto(None), escala), escala)
    assert ok["huecos"] == [] and ok["resumen"]["huecos"] == 0


def test_la_memoria_entre_pedidos_da_lo_mismo():
    for error in (None, "esquina_abierta", "tabique_no_llega"):
        work, muros, pisos = _armar(mp.texto(error), 50.0)
        cache = {}
        primero = detectar_huecos(muros + pisos, 50.0, cache)
        assert detectar_huecos(muros + pisos, 50.0, cache) == primero
        assert detectar_huecos(muros + pisos, 50.0) == primero
