"""
Banco de pruebas de la VERIFICACIÓN: el chequeo que le dice a una persona si puede cortar.

Por qué existe
--------------
Un verificador que puede dar un falso "todo bien" es peor que no tener ninguno: ya pasó una
vez (un "Ensamble verificado" con las piezas chocando a la vista) y dos más durante el
desarrollo de éste —piezas con el contorno abierto que el chequeo salteaba en silencio, y
astillas de 0.02 mm que hacían pasar por correcto un borde que no llegaba—.

Por eso acá no se prueba que el verificador "corra": se prueba que diga lo que tiene que
decir en casos cuya respuesta se calcula A MANO, incluidos los tres que el verificador
anterior no podía ver (la luz, la esquina sin tapar y las placas montadas entre media y una
placa), y que diga "listo" en los modelos que el banco de encaje ya afirma correctos.
"""

import json

import pytest

import test_encaje as te
from core.pipeline import parse_pipeline
from core.review_generate import (
    EnsambleNoArma,
    _decompose,
    generate_from_review,
    reporte_de_verificacion,
)
from core.services.obj_parser import parse_obj
from core.services.plate_intersect import PLATE_THICKNESS_M, pair_key
from core.services.reporte_verificacion import construir_reporte
from core.services.sheet_nester import SheetConfig
from core.services.types import PipelineOptions
from core.services.verificacion_uniones import TOL_OK_MM, verificar_uniones

ESCALAS = te.ESCALAS
P = te._placa


def _una(piezas, scale, prevista=True):
    us = verificar_uniones(piezas, scale, {pair_key(1, 2)} if prevista else set())
    assert len(us) <= 1, f"se esperaba a lo sumo una unión, hay {len(us)}"
    return us[0] if us else None


# ---------------------------------------------------------------------------
# Casos armados a mano: dos placas puestas donde se quiere, respuesta conocida
#
# A está en el plano z = 0 y corre en X. B está en un plano x = cte y corre en Z.
# `h` es media placa en metros de edificio.
# ---------------------------------------------------------------------------


def _caso(nombre, scale):
    t = PLATE_THICKNESS_M * scale
    h = t / 2.0
    mm = scale / 1000.0          # 1 mm de plancha, en metros de edificio
    A = lambda x0=0.0: P("A1", 1, (x0, 0.0, 0.0), (1, 0, 0), (0, 1, 0), 4.0, 3.0)
    B = lambda x, z0: P("A2", 2, (x, 0.0, z0), (0, 0, 1), (0, 1, 0), 2.0, 3.0)
    return {
        # B cruza a A por el medio y ninguna tiene ranura.
        "X sin ranura": ([A(), B(2.0, -1.0)], "choque", 3.0),
        # El canto de B apoya en la cara de A.
        "T a tope": ([A(), B(2.0, h)], "encaja", 0.0),
        # B queda 1 mm corta.
        "T con luz": ([A(), B(2.0, h + 1.0 * mm)], "luz", 1.0),
        # B entra 1 mm en A.
        "T que entra": ([A(), B(2.0, h - 1.0 * mm)], "choque", 1.0),
        # B llega hasta el plano medio de A: nadie la recortó.
        "T sin recortar": ([A(), B(2.0, 0.0)], "choque", 1.5),
        # Esquina bien cerrada: A tapa el canto de B, B hace tope en la cara de A.
        "L cerrada": ([A(-h), B(0.0, h)], "encaja", 0.0),
        # A llega sólo hasta el eje de B: medio canto de B queda a la vista.
        "L a medio tapar": ([A(0.0), B(0.0, h)], "sin_cubrir", 1.5),
        # Las dos "pasan": se comen la esquina entera.
        "L las dos pasan": ([A(-h), B(0.0, -h)], "choque", 3.0),
    }[nombre]


@pytest.mark.parametrize("scale", ESCALAS)
@pytest.mark.parametrize("nombre", [
    "X sin ranura", "T a tope", "T con luz", "T que entra", "T sin recortar",
    "L cerrada", "L a medio tapar", "L las dos pasan",
])
def test_cada_union_se_mide_exacta(nombre, scale):
    """Tipo y milímetros de cada encuentro, contra el valor calculado a mano.

    Las medidas son en mm de PLANCHA y por eso no dependen de la escala: una placa que
    entra 1 mm en otra entra 1 mm a 1:20 y a 1:200.
    """
    piezas, tipo, mm = _caso(nombre, scale)
    u = _una(piezas, scale)
    assert u is not None, f"{nombre} 1:{scale:.0f}: no se informó ninguna unión"
    assert u.tipo == tipo, f"{nombre} 1:{scale:.0f}: se informó {u.tipo!r}"
    assert u.medida_mm == pytest.approx(mm, abs=0.01), (
        f"{nombre} 1:{scale:.0f}: {u.medida_mm:.3f} mm, se esperaban {mm}"
    )
    assert (u.estado == "ok") == (mm <= TOL_OK_MM)
    # El contacto corre a lo largo de los 3 m de alto que comparten.
    assert u.largo_m == pytest.approx(3.0, abs=1e-6)


@pytest.mark.parametrize("scale", ESCALAS)
def test_placas_paralelas_montadas_se_ven_tambien_entre_media_y_una_placa(scale):
    """El punto ciego del verificador anterior: separación entre t/2 y t.

    Con el plano medio de una FUERA del volumen de la otra, el chequeo por muestreo no
    veía nada, y las dos placas se pisan igual (un cuarto de placa, en este caso).
    """
    t = PLATE_THICKNESS_M * scale
    a = P("A1", 1, (0.0, 0.0, 0.0), (1, 0, 0), (0, 1, 0), 4.0, 3.0)
    b = P("A2", 2, (1.0, 0.0, 0.75 * t), (1, 0, 0), (0, 1, 0), 4.0, 3.0)
    u = _una([a, b], scale, prevista=False)
    assert u is not None and u.tipo == "superpuestas"
    assert u.medida_mm == pytest.approx(0.75, abs=0.01)

    # Dos tramos coplanares que sólo se tocan de canto NO están montados.
    c = P("A2", 2, (4.0, 0.0, 0.0), (1, 0, 0), (0, 1, 0), 4.0, 3.0)
    assert _una([a, c], scale, prevista=False) is None


@pytest.mark.parametrize("scale", ESCALAS)
def test_dos_piezas_que_el_modelo_no_une_solo_se_informan_si_chocan(scale):
    """Que dos piezas cercanas no se toquen no es un defecto si el modelo no las une."""
    t = PLATE_THICKNESS_M * scale
    a = P("A1", 1, (0.0, 0.0, 0.0), (1, 0, 0), (0, 1, 0), 4.0, 3.0)
    cerca = P("A2", 2, (2.0, 0.0, t), (0, 0, 1), (0, 1, 0), 2.0, 3.0)   # a media placa de luz
    assert _una([a, cerca], scale, prevista=False) is None
    assert _una([a, cerca], scale, prevista=True).tipo == "luz"


# ---------------------------------------------------------------------------
# Sobre el pipeline completo
# ---------------------------------------------------------------------------


def _reporte(obj_text, scale, **kw):
    work, paneles, _grupos, _pjs = te._procesar(obj_text, scale, **kw)
    muros = [p for p in paneles if p.category == "wall"]
    pisos = [p for p in paneles if p.category == "floor"]
    return work, reporte_de_verificacion(work, muros, pisos, scale)


@pytest.mark.parametrize("scale", ESCALAS)
@pytest.mark.parametrize("modelo", ["esquina_L", "cruce_en_extremo"])
def test_un_modelo_que_arma_da_listo(modelo, scale):
    """Donde el banco de encaje afirma que las piezas arman, el veredicto es "listo".

    Es la mitad que evita el falso negativo: un verificador que marcara todo en rojo
    también pasaría los casos de arriba.
    """
    obj = {
        "esquina_L": te._modelo_esquina_L(),
        "cruce_en_extremo": te._modelo_cruce_en_extremo(),
    }[modelo]
    _work, r = _reporte(obj, scale)
    assert r is not None, "el reporte no se pudo calcular"
    assert r["veredicto"] == "listo", (
        f"{modelo} 1:{scale:.0f}: {[(p['titulo'], p['estado']) for p in r['problemas']]}"
    )
    assert r["resumen"]["uniones"] > 0, "no midió ninguna unión: 'listo' no significaría nada"
    assert r["resumen"]["uniones_ok"] == r["resumen"]["uniones"]
    assert r["resumen"]["piezas_ok"] == r["resumen"]["piezas"]
    json.dumps(r)        # tiene que poder viajar tal cual


def test_el_reporte_trae_la_accion_y_aplicarla_lo_resuelve():
    """Cada problema trae qué hacer, ya como el cambio que se le manda al backend.

    Muro con una chapa paralela a 11 cm. A 1:50 la placa son 15 cm de edificio: las dos
    superficies quedan como piezas separadas (están a más de media placa) pero sus
    volúmenes se pisan 4 cm, o sea 0.8 mm de plancha. El reporte lo tiene que decir,
    ofrecer fusionarlas, y al mandar ESE cambio el veredicto tiene que pasar a "listo".
    """
    obj = te._modelo_muro_con_revestimiento(0.11)
    _work, r = _reporte(obj, 50.0)
    assert r["veredicto"] == "no_arma"
    montadas = [p for p in r["problemas"] if p["tipo"] == "superpuestas"]
    assert len(montadas) == 1
    pr = montadas[0]
    assert pr["peor_mm"] == pytest.approx(0.8, abs=0.01)
    assert pr["peor_cm_edificio"] == pytest.approx(4.0, abs=0.1)
    fusionar = [a for a in pr["acciones"] if a["tipo"] == "fusionar"]
    assert fusionar, f"no se ofreció fusionar: {[a['tipo'] for a in pr['acciones']]}"
    merges = fusionar[0]["aplicar"]["merges"]

    parsed = parse_obj(obj)
    p1 = parse_pipeline("sintetico.obj", parsed["faces"], parsed["warnings"], force_axis="Y")
    opts = PipelineOptions(scale_denom=50.0, paper="A4", min_area_m2=1.0, sheet_config=None)
    work, muros, pisos, _ = _decompose(p1, opts, None, None, merges, None)
    r2 = reporte_de_verificacion(work, muros, pisos, 50.0)
    assert r2["veredicto"] == "listo", [p["titulo"] for p in r2["problemas"]]


def test_treinta_uniones_contra_la_misma_pieza_son_un_problema():
    """Lo que hace legible el reporte: se agrupa por causa, no se lista por unión."""
    scale = 100.0
    largo = P("B1", 1, (0.0, 0.0, 0.0), (1, 0, 0), (0, 1, 0), 40.0, 3.0)
    cruzan = [
        P(f"A{i}", 10 + i, (1.0 + i, 0.0, -1.0), (0, 0, 1), (0, 1, 0), 2.0, 3.0)
        for i in range(1, 31)
    ]
    uniones = verificar_uniones([largo] + cruzan, scale)
    assert len(uniones) == 30
    r = construir_reporte([largo] + cruzan, uniones, scale)

    assert r["resumen"]["uniones_no_arma"] == 30
    assert len(r["problemas"]) == 1, [p["titulo"] for p in r["problemas"]]
    pr = r["problemas"][0]
    assert pr["cantidad"] == 30 and pr["pieza_comun"] == "B1"
    assert pr["estado"] == "no_arma" and len(pr["uniones"]) == 30
    # La acción de grupo es sobre la pieza común: una sola resuelve las treinta.
    assert pr["acciones"][0]["aplicar"] == {"overrides": {"1": "discard"}}
    # Y el estado llega a cada pieza, que es lo que el visor pinta.
    assert r["piezas"]["B1"]["estado"] == "no_arma" and r["piezas"]["B1"]["problemas"] == 30
    assert all(r["piezas"][f"A{i}"]["estado"] == "no_arma" for i in range(1, 31))
    assert r["veredicto"] == "no_arma"


def test_la_decision_por_par_vale_lo_mismo_que_por_indice():
    """`wall_wall_decisions_by_pair` es la forma estable de decir quién cede.

    El índice de junta es posicional y se pierde con cada fusión, división o cambio de
    escala. El par de grupos no. Las dos formas tienen que producir las mismas piezas.
    """
    obj = te._modelo_esquina_L()
    parsed = parse_obj(obj)
    p1 = parse_pipeline("sintetico.obj", parsed["faces"], parsed["warnings"], force_axis="Y")
    por_indice = te._forzar_lo_contrario(p1)
    assert por_indice
    por_par = [
        {
            "group_a": p1.joints[ji].group_a,
            "group_b": p1.joints[ji].group_b,
            "yield_group_id": cede,
        }
        for ji, cede in por_indice.items()
    ]
    opts = PipelineOptions(scale_denom=50.0, paper="A4", min_area_m2=1.0, sheet_config=None)

    def medidas(**kw):
        _w, muros, pisos, _ = _decompose(p1, opts, None, kw.get("idx"), None, None,
                                         decisiones_por_par=kw.get("par"))
        return sorted((p.source_group_id, round(p.width_m, 6), round(p.height_m, 6))
                      for p in muros + pisos)

    sin_decidir = medidas()
    assert medidas(idx=por_indice) != sin_decidir, "forzar lo contrario no cambió nada"
    assert medidas(par=por_par) == medidas(idx=por_indice)


def test_se_puede_bloquear_la_generacion_si_no_arma():
    """Pedido original: si una unión no tiene solución, frenar antes de entregar el corte.

    Es opcional —lo pide quien genera— para no dejar sin poder cortar a un front que
    todavía no muestra el reporte. Sin pedirlo, o forzando, los archivos salen igual.
    """
    obj = te._modelo_muro_con_revestimiento(0.11)       # a 1:50 tiene dos piezas montadas
    parsed = parse_obj(obj)
    p1 = parse_pipeline("sintetico.obj", parsed["faces"], parsed["warnings"], force_axis="Y")
    opts = PipelineOptions(
        scale_denom=50.0, paper="A4", min_area_m2=1.0,
        sheet_config=SheetConfig(width_m=1.0, height_m=0.6, gap_m=0.003),
    )
    with pytest.raises(EnsambleNoArma) as exc:
        generate_from_review(p1, opts, bloquear_si_no_arma=True)
    assert exc.value.reporte["veredicto"] == "no_arma"
    assert generate_from_review(p1, opts), "sin pedir el bloqueo tiene que generar"

    # Un modelo que arma no se bloquea aunque se lo pida.
    parsed = parse_obj(te._modelo_esquina_L())
    ok = parse_pipeline("sintetico.obj", parsed["faces"], parsed["warnings"], force_axis="Y")
    assert generate_from_review(ok, opts, bloquear_si_no_arma=True)


# ---------------------------------------------------------------------------
# El contorno que llega a la plancha
# ---------------------------------------------------------------------------


def test_las_astillas_se_van_y_la_pieza_queda_igual():
    """Una tira más fina que el kerf no es material, ni pegada al cuerpo ni suelta.

    Salían cuando una pieza crecía y un obstáculo le comía casi todo lo agregado. Iban al
    DXF como un contorno aparte y hacían creer a las mediciones que el borde llegaba a
    destino. Medido: 101 de 296 piezas de `Prueba app.obj` a 1:50 tenían al menos una.
    """
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    from core.services.recorte_csg import _sin_astillas

    espesor = PLATE_THICKNESS_M * 50.0                # 15 cm de edificio
    cuerpo = Polygon([(0, 0), (4, 0), (4, 1), (2, 1), (2, 3), (0, 3)])   # una L
    # 1 mm de edificio = 0.02 mm de plancha. Una lengüeta que SALE del cuerpo y una tira
    # separada. (Una franja de 1 mm a todo lo largo de un lado no es una astilla: es la
    # pieza un milímetro más ancha, y no se toca.)
    pegada = box(4.0, 0.2, 4.6, 0.201)
    suelta = box(5.0, 0.0, 5.001, 3.0)

    limpio = _sin_astillas(unary_union([cuerpo, pegada, suelta]), espesor)
    assert limpio.geom_type == "Polygon", "quedó más de una isla"
    assert limpio.area == pytest.approx(cuerpo.area, abs=1e-6)
    assert limpio.bounds == pytest.approx(cuerpo.bounds, abs=1e-6)

    # Y una pieza sana —con una muesca del ancho de una placa— no se toca.
    con_muesca = cuerpo.difference(box(1.0, 0.0, 1.0 + espesor, 0.5))
    assert _sin_astillas(con_muesca, espesor).area == pytest.approx(con_muesca.area, abs=1e-6)


# ---------------------------------------------------------------------------
# Probeta de calibración
# ---------------------------------------------------------------------------


def test_la_probeta_trae_las_muescas_que_dice():
    """Cada muesca y cada ranura mide lo que tiene grabado, y todo entra en una plancha
    de 12 × 8 cm."""
    from core.services.cutting_sheet import _edges_to_polygon
    from core.services.probeta import anchos_mm, construir_probeta, generar_probeta

    esperados = anchos_mm(3.0)
    assert esperados == [2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4]
    piezas = {p.id: p for p in construir_probeta()}
    assert set(piezas) == {"PEINE", "RANURAS", "TESTIGO"}

    # RANURAS: siete huecos cerrados, de 20.2 mm de largo y el ancho grabado.
    ran = _edges_to_polygon([e for e in piezas["RANURAS"].edges])
    huecos = sorted(ran.interiors, key=lambda h: h.bounds[0])
    assert len(huecos) == 7
    for h, w in zip(huecos, esperados):
        x0, y0, x1, y1 = h.bounds
        assert (x1 - x0) * 1000 == pytest.approx(w, abs=1e-6)
        assert (y1 - y0) * 1000 == pytest.approx(20.2, abs=1e-6)

    # PEINE: siete muescas abiertas de 10 mm de fondo. El material que falta respecto del
    # rectángulo lleno es exactamente la suma de las siete.
    peine = _edges_to_polygon([e for e in piezas["PEINE"].edges])
    lleno = piezas["PEINE"].width_m * piezas["PEINE"].height_m
    assert (lleno - peine.area) * 1e6 == pytest.approx(sum(esperados) * 10.0, abs=1e-3)

    # TESTIGO: 40 × 20 con una lengüeta de 20 × 8.
    tes = _edges_to_polygon([e for e in piezas["TESTIGO"].edges])
    assert tes.area * 1e6 == pytest.approx(40 * 20 + 20 * 8, abs=1e-3)

    archivos, nesting = generar_probeta()
    assert not nesting.unplaced and len(nesting.sheets) == 1
    assert archivos[0].name.endswith(".dxf") and len(archivos[0].blob) > 1000
