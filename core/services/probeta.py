"""
Probeta de calibración: tres piezas chicas que dicen si el encastre va a entrar ANTES de
cortar la maqueta.

Por qué existe
--------------
Todo el encaje se calcula con dos números que nadie midió en el material de verdad:

    PLATE_THICKNESS_M = 0.003     el MDF "de 3 mm" puede medir 2.7 o 3.2 según la partida
    KERF_CLEARANCE_M  = 0.0001    lo que el láser se come depende de la máquina y la potencia

La verificación por unión puede afirmar que las piezas arman con esos valores. No puede
saber si esos valores son los de la plancha que está sobre la mesa. Una maqueta de 300
piezas con las ranuras 0.2 mm cerradas no arma, y el cálculo estaba bien.

La única forma de saberlo es cortar algo. Esto arma el "algo" mínimo: una plancha de unos
10 × 7 cm que se corta en un minuto y responde tres preguntas.

Qué trae
--------
PEINE      Siete muescas abiertas en el borde, de ancho creciente alrededor del nominal
           (de −0.2 a +0.4 mm, de a una décima), cada una con su ancho grabado. Se
           presenta el canto de CUALQUIER retazo de la misma placa en cada muesca: la que
           entra justa, sin forzar ni bailar, dice cuánto mide en realidad "espesor +
           kerf". Es el dato que hay que cargar como holgura.

RANURAS    Lo mismo pero con ranuras CERRADAS, que es como salen los encastres de la
           maqueta. Una ranura cerrada se comporta distinto que una muesca: el material no
           cede hacia el borde. Se prueba con la lengüeta del testigo.

TESTIGO    Un rectángulo de 40 × 20 mm con una lengüeta de 20 mm. Se mide el lado de 40
           con un calibre: lo que le falte para 40.00 es el kerf TOTAL (medio de cada
           lado). La lengüeta es lo que se mete en las ranuras cerradas.

Unidades
--------
Todo en metros de PLANCHA: son piezas físicas, no partes de un edificio, así que no llevan
escala. Se anidan con `scale_denom = 1`.

Lo que todavía NO hace el sistema con el resultado
--------------------------------------------------
El espesor y la holgura siguen siendo constantes del código, no parámetros del pedido. La
probeta dice qué valores poner; cargarlos por proyecto queda pendiente.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from core.services.plate_intersect import PLATE_THICKNESS_M
from core.services.sheet_nester import Edge, NestingPanel, Vec2

# Desvíos respecto del espesor nominal que se prueban, en milímetros.
DESVIOS_MM = (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4)

_MM = 0.001

# Trazos de un dígito de siete segmentos, en una caja de 1 (ancho) × 2 (alto).
_SEG = {
    "a": ((0, 2), (1, 2)), "b": ((1, 2), (1, 1)), "c": ((1, 1), (1, 0)),
    "d": ((0, 0), (1, 0)), "e": ((0, 0), (0, 1)), "f": ((0, 1), (0, 2)),
    "g": ((0, 1), (1, 1)),
}
_DIGITOS = {
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
    "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abfgcd",
}


def _texto(s: str, x: float, y: float, alto: float) -> List[Edge]:
    """Trazos para GRABAR un número (dígitos y punto), con la base izquierda en (x, y)."""
    ancho = alto / 2.0
    paso = ancho * 1.5
    out: List[Edge] = []
    cx = x
    for ch in s:
        if ch == ".":
            r = ancho * 0.12
            out.append(Edge(a=Vec2(cx, y), b=Vec2(cx + r, y), score=True))
            out.append(Edge(a=Vec2(cx + r, y), b=Vec2(cx + r, y + r), score=True))
            cx += ancho * 0.6
            continue
        for seg in _DIGITOS.get(ch, ""):
            (ax, ay), (bx, by) = _SEG[seg]
            out.append(Edge(
                a=Vec2(cx + ax * ancho, y + ay * ancho),
                b=Vec2(cx + bx * ancho, y + by * ancho),
                score=True,
            ))
        cx += paso
    return out


def _anillo(pts: List[Tuple[float, float]], hueco: bool = False) -> List[Edge]:
    n = len(pts)
    return [
        Edge(
            a=Vec2(*pts[i]), b=Vec2(*pts[(i + 1) % n]),
            hole=hueco, joint=hueco,
        )
        for i in range(n)
    ]


def anchos_mm(espesor_mm: float) -> List[float]:
    """Los anchos de muesca/ranura que trae la probeta, en mm."""
    return [round(espesor_mm + d, 2) for d in DESVIOS_MM]


def construir_probeta(espesor_m: float = PLATE_THICKNESS_M) -> List[NestingPanel]:
    """Las tres piezas de la probeta, listas para anidar con `scale_denom = 1`."""
    t_mm = espesor_m * 1000.0
    anchos = anchos_mm(t_mm)
    n = len(anchos)

    paso = 12.0 * _MM            # de centro a centro
    margen = 8.0 * _MM
    ancho_total = 2 * margen + (n - 1) * paso
    alto_txt = 3.0 * _MM

    # --- PEINE: muescas abiertas en el borde de arriba --------------------------------
    alto_peine = 26.0 * _MM
    prof = 10.0 * _MM
    pts: List[Tuple[float, float]] = [(0.0, 0.0), (ancho_total, 0.0), (ancho_total, alto_peine)]
    marcas: List[Edge] = []
    for i in reversed(range(n)):
        c = margen + i * paso
        w = anchos[i] * _MM
        pts += [
            (c + w / 2, alto_peine), (c + w / 2, alto_peine - prof),
            (c - w / 2, alto_peine - prof), (c - w / 2, alto_peine),
        ]
    pts.append((0.0, alto_peine))
    for i in range(n):
        c = margen + i * paso
        marcas += _texto(f"{anchos[i]:.1f}", c - 3.4 * _MM, 6.0 * _MM, alto_txt)
    peine = NestingPanel(
        id="PEINE", category="wall", width_m=ancho_total, height_m=alto_peine,
        edges=_anillo(pts) + marcas,
    )

    # --- RANURAS: las mismas, cerradas ------------------------------------------------
    alto_ran = 34.0 * _MM
    largo = 20.2 * _MM           # la lengüeta mide 20.0: dos décimas para que entre de largo
    edges = _anillo([(0.0, 0.0), (ancho_total, 0.0), (ancho_total, alto_ran), (0.0, alto_ran)])
    for i in range(n):
        c = margen + i * paso
        w = anchos[i] * _MM
        y0 = alto_ran - 5.0 * _MM - largo
        edges += _anillo(
            [(c - w / 2, y0), (c + w / 2, y0), (c + w / 2, y0 + largo), (c - w / 2, y0 + largo)],
            hueco=True,
        )
        edges += _texto(f"{anchos[i]:.1f}", c - 3.4 * _MM, 2.5 * _MM, alto_txt)
    ranuras = NestingPanel(
        id="RANURAS", category="wall", width_m=ancho_total, height_m=alto_ran, edges=edges,
    )

    # --- TESTIGO: patrón de 40 mm con lengüeta de 20 ----------------------------------
    cuerpo_w, cuerpo_h = 40.0 * _MM, 20.0 * _MM
    leng_w, leng_h = 20.0 * _MM, 8.0 * _MM
    x0 = (cuerpo_w - leng_w) / 2.0
    testigo = NestingPanel(
        id="TESTIGO", category="wall", width_m=cuerpo_w, height_m=cuerpo_h + leng_h,
        edges=_anillo([
            (0.0, 0.0), (cuerpo_w, 0.0), (cuerpo_w, cuerpo_h),
            (x0 + leng_w, cuerpo_h), (x0 + leng_w, cuerpo_h + leng_h),
            (x0, cuerpo_h + leng_h), (x0, cuerpo_h), (0.0, cuerpo_h),
        ]) + _texto("40", cuerpo_w / 2 - 3.0 * _MM, 6.0 * _MM, 5.0 * _MM),
    )
    return [peine, ranuras, testigo]


def instrucciones(espesor_m: float = PLATE_THICKNESS_M) -> Dict:
    """Qué hacer con la probeta y cómo leerla. Texto para mostrar, no para interpretar."""
    t_mm = espesor_m * 1000.0
    return {
        "espesor_nominal_mm": round(t_mm, 2),
        "anchos_mm": anchos_mm(t_mm),
        "pasos": [
            "Cortá la probeta en la MISMA placa y con la misma potencia y velocidad con "
            "las que vas a cortar la maqueta.",
            "PEINE: presentá el canto de un retazo de esa placa en cada muesca. Anotá el "
            "número grabado de la que entra justa: ni forzando ni con juego.",
            "RANURAS: metiendo la lengüeta del TESTIGO, repetí la prueba. Si el número "
            "es distinto al del peine, valé el de las ranuras: así salen los encastres.",
            "TESTIGO: medí con calibre el lado marcado 40. La diferencia con 40.00 mm es "
            "lo que el láser le quita a cada pieza (el kerf).",
        ],
        "lectura": (
            f"El sistema calcula hoy con placa de {t_mm:.1f} mm y ranuras de "
            f"{t_mm + 0.1:.1f} mm. Si la muesca que entra justa no es la de "
            f"{t_mm + 0.1:.1f}, los encastres de la maqueta van a quedar duros o flojos "
            "en esa misma diferencia."
        ),
    }


def generar_probeta(config=None, paper: str = "A4"):
    """Anida la probeta y devuelve `(archivos, nesting)`.

    `archivos` es una lista de `OutputFile` (DXF de corte y PDF) con el mismo formato que
    los de la maqueta, para que entren al mismo zip y a la misma máquina sin pasos nuevos.
    """
    from core.services.cutting_sheet import nested_sheets_to_dxf
    from core.services.pdf_writer import generate_nesting_pdf
    from core.services.sheet_nester import SheetConfig, nest_panels
    from core.services.types import OutputFile

    cfg = config or SheetConfig(width_m=0.12, height_m=0.08, gap_m=0.003)
    nesting = nest_panels(construir_probeta(), cfg, 1.0)
    archivos = [
        OutputFile(
            name="probeta_calibracion.dxf",
            blob=nested_sheets_to_dxf(nesting, True).encode("utf-8"),
        )
    ]
    try:
        pdf = generate_nesting_pdf(nesting, "Probeta de calibración", True, paper, "single_page")
        if pdf:
            archivos.append(OutputFile(name="probeta_calibracion.pdf", blob=pdf))
    except Exception:
        import logging
        logging.getLogger("eficiencia2d.pipeline").exception("PDF de la probeta falló")
    return archivos, nesting
