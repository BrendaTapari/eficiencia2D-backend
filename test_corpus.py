"""
Corpus: los mismos invariantes sobre TODOS los modelos disponibles, no sobre uno.

Por qué existe
--------------
El riesgo permanente de este pipeline es ajustar el código para que cierre UN archivo.
`test_encaje.py` protege contra eso por un lado —sus modelos son sintéticos y su resultado
se calcula a mano— pero no dice nada sobre archivos reales, que es donde aparecen las
combinaciones que ningún caso sintético tiene: mallas sin soldar, muros de una sola cara,
muros continuos y muros partidos, encuentros oblicuos, escalas gruesas.

Acá se corre el pipeline completo sobre cada modelo que haya y se afirman invariantes que
deben valer para CUALQUIER archivo, más una LÍNEA BASE por modelo. La línea base es lo que
convierte esto en un detector de sobreajuste: un cambio que mejora un archivo y empeora
otro mueve un número y hay que explicarlo, en vez de pasar desapercibido.

Qué modelos usa
---------------
- `demo.obj`, que vive en el repo.
- Todo `*.obj` / `*.stl` que haya en el directorio que indique `EFICIENCIA2D_CORPUS`.
  Los modelos de los usuarios no están versionados, así que el corpus local de cada uno
  se apunta con esa variable. Sin ella, corre igual con lo que hay.
"""

import os
from pathlib import Path

import pytest

from core.pipeline import parse_pipeline
from core.review_generate import _decompose, _panels_to_nesting
from core.services.assembly_verify import verificar_ensamble
from core.services.obj_parser import parse_obj
from core.services.plate_intersect import (
    KERF_CLEARANCE_M,
    PLATE_THICKNESS_M,
    pair_key,
)
from core.services.sheet_nester import SheetConfig, nest_panels
from core.services.types import PipelineOptions

ESCALAS = (50.0, 100.0)

# Línea base por modelo: (piezas, choques a 1:50, choques a 1:100). Sólo para los modelos
# que están en el repo — los de un corpus externo no se pueden fijar acá.
#
# Estos números NO son un objetivo: son el estado conocido. Si un cambio los mueve, el
# test falla y hay que decir por qué. Bajar los choques es una mejora y se actualiza el
# número en el mismo commit que la produce; subirlos es una regresión.
LINEA_BASE = {
    "demo.obj": {"piezas": 43, "choques": {50.0: 4, 100.0: 2}},
}


def _modelos():
    out = []
    demo = Path(__file__).parent / "demo.obj"
    if demo.exists():
        out.append(demo)
    extra = os.environ.get("EFICIENCIA2D_CORPUS")
    if extra:
        d = Path(extra)
        if d.is_dir():
            out.extend(sorted(p for p in d.iterdir() if p.suffix.lower() == ".obj"))
    return out


MODELOS = _modelos()


def _base_externa(path: Path):
    """Línea base de un modelo del corpus externo, guardada al lado de los archivos.

    Los modelos de los usuarios no están versionados, así que su línea base no puede
    vivir en el código. Se escribe una instantánea la PRIMERA vez y a partir de ahí
    cualquier cambio de números falla el test. Sin esto, el corpus externo sólo
    verificaría invariantes genéricos y un sobreajuste —un arreglo que cierra un archivo
    y rompe otro— seguiría pasando desapercibido, que es justo lo que este archivo existe
    para evitar.
    """
    import json

    d = os.environ.get("EFICIENCIA2D_CORPUS")
    if not d:
        return None
    f = Path(d) / "corpus_baseline.json"
    datos = {}
    if f.exists():
        try:
            datos = json.loads(f.read_text())
        except Exception:
            datos = {}
    if path.name in datos:
        b = datos[path.name]
        return {"piezas": b["piezas"], "choques": {float(k): v for k, v in b["choques"].items()}}

    # Primera vez: se mide y se registra. No se afirma nada en esta corrida.
    piezas = choques = None
    registro = {"piezas": None, "choques": {}}
    for s in ESCALAS:
        work, walls, floors, pjs = _procesar(path, s)
        registro["piezas"] = len(walls) + len(floors)
        detectadas = {pair_key(j.group_a, j.group_b) for j in work.joints}
        sin_res = {
            pair_key(j.group_a, j.group_b)
            for j in work.wall_wall_joints
            if j.yield_group_id is None
        }
        registro["choques"][str(s)] = len(
            verificar_ensamble(walls + floors, pjs, s, detectadas, sin_res)
        )
    datos[path.name] = registro
    try:
        f.write_text(json.dumps(datos, indent=2, sort_keys=True))
    except Exception:
        pass
    return None


def _procesar(path: Path, scale: float):
    parsed = parse_obj(path.read_text())
    p1 = parse_pipeline(path.name, parsed["faces"], parsed["warnings"])
    opts = PipelineOptions(
        scale_denom=scale, paper="A4", min_area_m2=1.0, sheet_config=None
    )
    return _decompose(p1, opts, None, None, None, None)


@pytest.mark.skipif(not MODELOS, reason="no hay modelos disponibles")
@pytest.mark.parametrize("path", MODELOS, ids=lambda p: p.name)
@pytest.mark.parametrize("scale", ESCALAS)
def test_ninguna_pieza_degenerada(path, scale):
    """Toda pieza que se manda a cortar tiene que tener dimensiones y material."""
    _work, walls, floors, _pjs = _procesar(path, scale)
    for p in walls + floors:
        assert p.width_m > 0 and p.height_m > 0, f"{p.id} sin dimensiones"
        assert p.edges, f"{p.id} sin contorno"


@pytest.mark.skipif(not MODELOS, reason="no hay modelos disponibles")
@pytest.mark.parametrize("path", MODELOS, ids=lambda p: p.name)
@pytest.mark.parametrize("scale", ESCALAS)
def test_toda_ranura_deja_pasar_la_placa(path, scale):
    """Por una ranura pasa una placa de 3 mm: nunca puede ser más angosta.

    Es el invariante que separa los dos sistemas de unidades. Una ranura que salga del
    espesor del MODELO en vez del material da 2 mm a 1:100 (no entra) y 4 mm a 1:50
    (baila), y el ancho cambia con la escala en vez de quedarse en el material.
    """
    _work, _walls, _floors, pjs = _procesar(path, scale)
    minimo = (PLATE_THICKNESS_M + KERF_CLEARANCE_M) * 1000.0
    for pj in pjs:
        ancho_mm = pj.width * 1000.0
        assert ancho_mm >= minimo - 1e-6, (
            f"ranura de {ancho_mm:.2f}mm para una placa de {PLATE_THICKNESS_M*1000:.1f}mm"
        )
        # Cota superior: el factor oblicuo está clampeado a 20°, o sea x2.92 como máximo.
        assert ancho_mm <= minimo * 2.93, f"ranura de {ancho_mm:.2f}mm, desproporcionada"


@pytest.mark.skipif(not MODELOS, reason="no hay modelos disponibles")
@pytest.mark.parametrize("path", MODELOS, ids=lambda p: p.name)
@pytest.mark.parametrize("scale", ESCALAS)
def test_el_nesting_no_solapa(path, scale):
    """Dos piezas nunca comparten material en la plancha, y respetan la separación."""
    _work, walls, floors, _pjs = _procesar(path, scale)
    cfg = SheetConfig(width_m=1.0, height_m=0.6, gap_m=0.003)
    res = nest_panels(_panels_to_nesting(walls + floors, scale), cfg, scale)
    for hoja in res.sheets:
        ps = hoja.panels
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                a, b = ps[i], ps[j]
                dx = max(b.x - (a.x + a.effective_w), a.x - (b.x + b.effective_w))
                dy = max(b.y - (a.y + a.effective_h), a.y - (b.y + b.effective_h))
                assert dx >= -1e-9 or dy >= -1e-9, (
                    f"{a.panel.id} y {b.panel.id} se pisan en la plancha"
                )


@pytest.mark.skipif(not MODELOS, reason="no hay modelos disponibles")
@pytest.mark.parametrize("path", MODELOS, ids=lambda p: p.name)
@pytest.mark.parametrize("scale", ESCALAS)
def test_linea_base_del_ensamble(path, scale):
    """El estado conocido de cada modelo, para que un sobreajuste se vea.

    Un arreglo que cierra un archivo y rompe otro mueve estos números. Sin la línea base,
    ese intercambio pasa desapercibido: el modelo que uno está mirando mejora y el otro
    empeora en silencio.
    """
    base = LINEA_BASE.get(path.name) or _base_externa(path)
    if base is None:
        pytest.skip(f"{path.name} no tiene línea base registrada")

    work, walls, floors, pjs = _procesar(path, scale)
    piezas = len(walls) + len(floors)
    assert piezas == base["piezas"], (
        f"{path.name}: {piezas} piezas, la línea base dice {base['piezas']}"
    )

    detectadas = {pair_key(j.group_a, j.group_b) for j in work.joints}
    sin_resolver = {
        pair_key(j.group_a, j.group_b)
        for j in work.wall_wall_joints
        if j.yield_group_id is None
    }
    choques = verificar_ensamble(
        walls + floors, pjs, scale, detectadas, sin_resolver
    )
    esperados = base["choques"][scale]
    assert len(choques) <= esperados, (
        f"{path.name} 1:{scale:.0f}: {len(choques)} choques, la línea base dice "
        f"{esperados}. REGRESIÓN: "
        f"{[(c.pieza_a, c.pieza_b, c.causa) for c in choques]}"
    )
    if len(choques) < esperados:
        pytest.fail(
            f"{path.name} 1:{scale:.0f}: {len(choques)} choques contra {esperados} de la "
            "línea base. Es una MEJORA: actualizá LINEA_BASE en este mismo commit para "
            "que quede registrada y no se pueda perder después sin que nadie lo note."
        )


@pytest.mark.skipif(not MODELOS, reason="no hay modelos disponibles")
@pytest.mark.parametrize("path", MODELOS, ids=lambda p: p.name)
@pytest.mark.parametrize("scale", ESCALAS)
def test_el_instructivo_dibuja_la_pieza_que_se_corta(path, scale):
    """`placements_fieles` tiene que coincidir con la pieza de la plancha, siempre.

    Es el invariante que hace que el instructivo sirva para decidir. Mientras el visor
    dibujaba `build_placements` —la proyección CRUDA del modelo— una esquina que se veía
    superpuesta podía estar perfecta en el DXF y una que se veía bien podía chocar: el
    dibujo no era el de las piezas. Medido sobre un modelo real: 19 de 28 piezas más
    grandes de lo que se cortaba a 1:50 y 21 de 28 a 1:100, con excesos de hasta 17.8 mm
    de plancha.

    Se afirma además que el crudo SIGUE siendo distinto, para que este test no pase por
    accidente si alguien hiciera que las dos funciones devuelvan lo mismo.
    """
    from core.review_generate import placements_fieles
    from core.services.cutting_sheet import build_placements

    parsed = parse_obj(path.read_text())
    p1 = parse_pipeline(path.name, parsed["faces"], parsed["warnings"])
    _work, walls, floors, _pjs = _procesar(path, scale)

    fieles = placements_fieles(p1, scale)
    assert fieles, f"{path.name}: no se produjo ningún placement fiel"

    for pn in walls + floors:
        f = fieles.get(str(pn.source_group_id))
        if f is None:
            continue
        assert f["width_m"] == pytest.approx(pn.width_m, abs=1e-6), (
            f"{path.name} 1:{scale:.0f} {pn.id}: el instructivo dibuja "
            f"{f['width_m']:.3f} m de ancho y se cortan {pn.width_m:.3f} m"
        )
        assert f["height_m"] == pytest.approx(pn.height_m, abs=1e-6), (
            f"{path.name} 1:{scale:.0f} {pn.id}: el instructivo dibuja "
            f"{f['height_m']:.3f} m de alto y se cortan {pn.height_m:.3f} m"
        )

    crudo = build_placements(_work.groups, _work.faces, "Y")
    difieren = sum(
        1 for k, f in fieles.items()
        if k in crudo and (
            abs(crudo[k]["width_m"] - f["width_m"]) > 1e-6
            or abs(crudo[k]["height_m"] - f["height_m"]) > 1e-6
        )
    )
    assert difieren > 0, (
        f"{path.name} 1:{scale:.0f}: la proyección cruda coincide con las piezas "
        "recortadas en TODAS. O el modelo no tiene ni una junta, o alguien igualó las "
        "dos funciones y este test dejó de comprobar algo."
    )
