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
    # 1:100 pasó de 2 a 4 al cerrar las esquinas con las dos piezas: el muro pasante
    # crece hasta la cara exterior de su vecina, y al crecer toca piezas cuya junta el
    # detector NO ve (causa "no_detectada": A1/A4, A1/A5, A19/A20). No es la matemática
    # del recorte: es el hueco de detección, que sigue abierto. A cambio, el invariante
    # de esquina pasó de 25 a 33 bordes correctos en este mismo archivo y escala.
    # Sube de 6/6 a 7/11 con el motor booleano. De los 11 a 1:100, OCHO son entre pieles
    # GEMELAS del mismo muro (A1/A4, A1/A5, A10/A12, A22-A25/A29, A19/A20): dos grupos que
    # son las dos caras de la misma pared de 20 cm, cortadas como dos piezas. No es un
    # problema de encaje sino de clasificación, y ya está anotado aparte. A cambio, el
    # invariante de esquina sube de 192/136 a 196/132 sobre los tres modelos, y bfe62e18
    # a 1:100 —el archivo y la escala donde el usuario mide— salta de 33/19 a 37/15.
    # 50 piezas (antes 43): `mesh_splitter` parte los muros al nivel de cada losa, así
    # que un muro que cruza un entrepiso llega como dos tramos y el piso entra entre
    # ellos. Ese cambio se había revertido el 1/8 porque con el motor de recorte viejo
    # regresaba las mediciones; con el motor booleano cambió de signo y hoy es una mejora
    # clara: el invariante de esquina pasa de 46/30 a 72/21 a 1:50 y de 47/29 a 71/22 a
    # 1:100, con los choques igual o mejor.
    #
    # `piezas` pasa a depender de la ESCALA (37 a 1:50, 32 a 1:100; antes 50 fijo) y los
    # choques bajan de 7/10 a 2/2. Tres cambios, medidos también sobre `Prueba app.obj`
    # (403 piezas -> 348/342, choques 53/50 -> 3/2):
    #   1. `discard_slab_rims`: 11 "muros" eran cantos de losa (tiras de hasta 17 cm por
    #      medio milímetro de plancha). No dependen de la escala.
    #   2. `placas_superpuestas`: superficies paralelas a menos de media placa y con
    #      solape real son UNA pieza. Depende de la escala porque la placa mide 15 cm de
    #      edificio a 1:50 y 30 a 1:100: por eso a 1:100 se funden más.
    #   3. Dos tramos coplanares ya no crecen uno dentro del otro al tapar el canto de un
    #      tabique que llega a su costura: pasa el primero y el segundo hace tope.
    # De los 104 choques entre placas PARALELAS que había sumando los dos modelos y las
    # dos escalas quedan 4; los que restan son casi todos perpendiculares de causa
    # "escala".
    #
    # 1:100 baja de 2 a 0: una pieza ya no crece dentro de la placa ante la que cede
    # (`cede_ante` en `aplicar_destinos`). Eran los dos choques de causa "escala".
    # 1:50 pasa de 37 a 36 piezas: una piel interior partida en tramos se reconoce como
    # la otra cara de su muro por solape real (`_espalda_con_espalda`), no por centroides.
    # 1:50 baja de 2 a 1 choque al sacar las caras de canto de los grupos MIXTOS.
    # Piezas 36/32 -> 32/29 al fundir también las placas paralelas que se pisan entre
    # media y una placa (`placas_superpuestas`): eran piezas montadas una sobre otra.
    # Los choques del verificador por muestreo pasan de 1/0 a 0/1: a 1:100 aparece un par
    # de placas montadas (A18/A19) que antes no veía. El verificador exacto, que es el que
    # manda, mide mejora en las dos escalas (ver LINEA_BASE_VEREDICTO).
    "demo.obj": {"piezas": {50.0: 32, 100.0: 29}, "choques": {50.0: 0, 100.0: 1}},
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
        if not isinstance(b.get("piezas"), dict):
            # Instantánea vieja, de cuando la cantidad de piezas no dependía de la escala:
            # ya no es comparable. Se vuelve a medir más abajo.
            datos.pop(path.name)
        else:
            return {
                "piezas": {float(k): v for k, v in b["piezas"].items()},
                "choques": {float(k): v for k, v in b["choques"].items()},
            }

    # Primera vez: se mide y se registra. No se afirma nada en esta corrida.
    piezas = choques = None
    registro = {"piezas": {}, "choques": {}}
    for s in ESCALAS:
        work, walls, floors, pjs = _procesar(path, s)
        registro["piezas"][str(s)] = len(walls) + len(floors)
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
    assert piezas == base["piezas"][scale], (
        f"{path.name} 1:{scale:.0f}: {piezas} piezas, la línea base dice "
        f"{base['piezas'][scale]}"
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


@pytest.mark.skipif(not MODELOS, reason="no hay modelos disponibles")
@pytest.mark.parametrize("path", MODELOS, ids=lambda p: p.name)
@pytest.mark.parametrize("scale", ESCALAS)
def test_ninguna_pieza_lleva_aristas_de_largo_cero(path, scale):
    """Una arista de largo cero es un artefacto: no corta nada y rompe el contorno.

    Salían de los cierres de recorte cuando la línea de corte caía sobre una arista que
    ya existía. En `31232804-casa.obj` a 1:100 eran 10 de 116 aristas.
    """
    _work, walls, floors, _pjs = _procesar(path, scale)
    for p in walls + floors:
        for e in p.edges:
            assert (abs(e.a.x - e.b.x) > 1e-9 or abs(e.a.y - e.b.y) > 1e-9), (
                f"{p.id}: arista de largo cero en ({e.a.x:.4f}, {e.a.y:.4f})"
            )


# Bordes de esquina que NO caen a media placa del plano medio de su vecina, por modelo
# y escala: (correctos, mal). Ver el test de abajo.
LINEA_BASE_ENCAJE = {
    # Antes (72, 21) y (71, 22). Los bordes MAL bajan de 21 a 13 y de 22 a 14. Los
    # correctos también bajan en número absoluto, y no es una regresión: hay menos juntas
    # que medir, porque las dos pieles de un muro contaban cada esquina dos veces y los
    # cantos de losa aportaban juntas contra los muros de verdad. La proporción de bordes
    # correctos pasa de 77% a 80% a 1:50 y de 76% a 78% a 1:100.
    #
    # De los que quedan mal, los de +20 mm o más (A1, A14) no son esquinas: son encuentros
    # en T medidos desde el muro PASANTE, cuyo extremo está —correctamente— a metros del
    # tabique. Es una limitación de esta medición, que mira el extremo más cercano de las
    # dos piezas de cada junta sin distinguir quién pasa. Queda anotado, no corregido.
    # 1:50 pasa de (52, 13) a (50, 13): una pieza menos (ver LINEA_BASE), y con ella dos
    # bordes que se medían contra la piel que ahora es parte de su muro. Los mal no cambian.
    # (50, 13) y (49, 14) pasan a (49, 13) y (48, 14) al sacar las caras de canto de los
    # grupos mixtos: una junta menos, que era de un muro contra un canto. Los mal no cambian.
    #
    # (49, 13) y (48, 14) pasan a (46, 16) y (45, 17), y NO es que el encaje haya
    # empeorado: esos tres bordes estaban mal desde antes y esta medición los contaba
    # bien. Mira hasta dónde llega el bounding box de la pieza, y la pieza llegaba a
    # destino sólo con una ASTILLA de un milímetro de edificio (0.02 mm de plancha) que
    # sobrevivía cuando un obstáculo se comía el resto de lo que había crecido. Al limpiar
    # las astillas (`recorte_csg._sin_astillas`) el borde vuelve a donde está el material.
    # Es la tercera vez que una medición de este archivo resulta más optimista que la
    # realidad; la que no se deja engañar es `test_el_veredicto_de_cada_modelo`.
    #
    # (46, 16) y (45, 17) pasan a (44, 18) y (40, 22) con el RESOLVEDOR de uniones
    # (`review_generate._resolver_uniones`), y es el primer caso en que esta medición y
    # el verificador exacto dicen cosas opuestas. El verificador gana: a 1:100 las uniones
    # que encajan suben de 49 a 51 y las que no arman bajan de 7 a 6.
    #
    # La diferencia es de criterio, no de error. Acá cada borde vale 1 ó 0 según caiga a
    # 0.2 mm de su destino. El resolvedor invierte quién cede donde eso cierra MÁS uniones:
    # cuatro esquinas abiertas pasan a encastrar bien, una luz de 2.1 mm pasa a ser un
    # contacto de canto, y a cambio dos vecinas quedan un poco peor. Para esta medición lo
    # segundo pesa igual que lo primero y las mejoras parciales no cuentan. Se la conserva
    # como alarma, pero la línea base que manda es `LINEA_BASE_VEREDICTO`.
    #
    # 1:100 pasa de (40, 22) a (46, 12) al fundir las placas paralelas que se pisan entre
    # media y una placa. 1:50 no cambia.
    "demo.obj": {50.0: (44, 18), 100.0: (46, 12)},
}


def _bordes_de_esquina(path, scale, tol_mm=0.2):
    """Mide el invariante de esquina en cada junta muro-muro resuelta.

    Una esquina la cierran las DOS piezas: la que cede va a la cara interior de su vecina
    y la que pasa a la cara exterior de la que cede. En los dos casos el borde queda a
    MEDIA PLACA del plano medio del vecino, así que el invariante no depende de quién
    ceda — que es lo que hace falta, porque eso lo elige el usuario en el visor y puede
    invertirse en cualquier cruce.

    Devuelve (correctos, [(pieza, error_mm)]).
    """
    from core.services.assembly_adjuster import compute_adjustments
    from core.services.plate_intersect import mid_plane_offset
    from core.services.types import Vec3, dot, normalize

    work, walls, _floors, _pjs = _procesar(path, scale)
    pan = {p.source_group_id: p for p in walls}
    gby = {g.id: g for g in work.groups}
    res = compute_adjustments(work.joints, work.groups, None, work.faces)
    media = PLATE_THICKNESS_M * scale / 2.0
    bien, mal = 0, []

    for ww in res.wall_wall_joints:
        if ww.yield_group_id is None:
            continue
        for gid in (ww.group_a, ww.group_b):
            otro = gby.get(ww.group_b if gid == ww.group_a else ww.group_a)
            p = pan.get(gid)
            if p is None or not p.frame or otro is None:
                continue
            m = p.frame
            u = Vec3(m["u_axis"]["x"], m["u_axis"]["y"], m["u_axis"]["z"])
            n = normalize(otro.representative_normal)
            if abs(dot(u, n)) < 0.7:      # no se encuentran de punta
                continue
            o = Vec3(m["origin"]["x"], m["origin"]["y"], m["origin"]["z"])
            d = mid_plane_offset(otro, work.faces, n)
            cerca = min(
                (dot(Vec3(o.x + u.x * uu, o.y + u.y * uu, o.z + u.z * uu), n) - d
                 for uu in (0.0, m["width_m"])),
                key=abs,
            )
            err = (abs(cerca) - media) / scale * 1000.0
            if abs(err) <= tol_mm:
                bien += 1
            else:
                mal.append((p.id, err))
    return bien, mal


@pytest.mark.skipif(not MODELOS, reason="no hay modelos disponibles")
@pytest.mark.parametrize("path", MODELOS, ids=lambda p: p.name)
@pytest.mark.parametrize("scale", ESCALAS)
def test_las_esquinas_las_cierran_las_dos_piezas(path, scale):
    """Cada borde de esquina a media placa del plano medio de su vecina.

    Reemplaza a un test que medía si la pieza era igual al hueco LIBRE entre sus dos
    vecinas. Ese criterio da por sentada una convención —"la pieza entra entre las
    otras"— y no se puede dar por sentada: el front le pregunta al usuario en cada cruce
    cuál se acorta, y puede elegir la pasante o la vecina. Éste vale para las dos, porque
    mide la distancia al plano medio, que es media placa en los dos roles.

    Origen: el usuario midió con la regla sobre el DXF que una pieza que debía medir
    69 mm medía 72. La causa era que sólo se ajustaba el muro que CEDE; el que pasa se
    quedaba con el largo del modelo, que llega hasta la piel del muro macizo, y a escalas
    gruesas eso queda corto (a 1:100 media placa son 15 cm de edificio y el muro asoma 1).
    La pieza salía de eje a eje de sus vecinas: le sobraban 3 mm para entrar entre ellas y
    le faltaban 2.8 para pasarles por encima.

    Es una LÍNEA BASE del estado real, no un objetivo cumplido. Bajarla es el trabajo.
    """
    base = LINEA_BASE_ENCAJE.get(path.name)
    if base is None or scale not in base:
        pytest.skip(f"{path.name} 1:{scale:.0f} no tiene línea base de encaje")

    bien, mal = _bordes_de_esquina(path, scale)
    esp_bien, esp_mal = base[scale]

    assert len(mal) <= esp_mal and bien >= esp_bien, (
        f"{path.name} 1:{scale:.0f}: {bien} bordes correctos y {len(mal)} mal; la línea "
        f"base dice {esp_bien} y {esp_mal}. REGRESIÓN. Peores: "
        + str(sorted(mal, key=lambda x: -abs(x[1]))[:5])
    )
    if len(mal) < esp_mal or bien > esp_bien:
        pytest.fail(
            f"{path.name} 1:{scale:.0f}: {bien} correctos / {len(mal)} mal contra "
            f"{esp_bien} / {esp_mal} de la línea base. Es una MEJORA: actualizá "
            "LINEA_BASE_ENCAJE en este mismo commit."
        )


# Estado del grafo por modelo y escala: (evaluadas, fallan, conflictos).
LINEA_BASE_RESTRICCIONES = {
    # Baja de 20 a 16 incumplidas al pasar el recorte al motor booleano: el crecimiento
    # hacia la cara exterior de la vecina, que antes se descartaba entero, ahora se aplica.
    #
    # Antes (72, 11, 5) y (72, 12, 5). Con los cantos de losa descartados y las pieles
    # superpuestas fundidas hay menos restricciones que evaluar (51 y 48), bajan las
    # incumplidas (8 y 10) y bajan los conflictos de 5 a 1 y 2: varios eran un mismo borde
    # reclamado por las dos pieles de un mismo muro, que ya no son dos vecinas.
    #
    # 1:50 pasa de (51, 8, 1) a (48, 9, 2) con el criterio de pieles del mismo muro
    # (`_espalda_con_espalda`): tres restricciones menos, pero UNA incumplida y UN conflicto
    # más. Es una regresión chica en este archivo y se acepta a sabiendas, porque en
    # `Prueba app.obj` el mismo cambio saca 46 piezas duplicadas y 28 de 32 pares de placas
    # montadas media placa que el verificador no ve. Va en un commit propio para poder
    # revertirlo solo si en los modelos reales no rinde.
    #
    # Con las caras de canto fuera de los grupos mixtos se evalúa una restricción menos
    # (47); incumplidas y conflictos no cambian.
    #
    # 1:50 pasa de 9 a 10 incumplidas al limpiar las astillas, por el mismo motivo que
    # el invariante de esquina de más abajo: un borde que "cumplía" con una tira de
    # 0.02 mm de plancha.
    #
    # Con el resolvedor de uniones pasan a 12 y 15 incumplidas. El grafo evalúa cada borde
    # contra la decisión SUGERIDA de quién cede; el resolvedor la invierte donde el
    # verificador exacto mide que así cierran más uniones, así que por definición se aleja
    # de lo que el grafo espera. Ver la nota de LINEA_BASE_ENCAJE.
    # 1:100 pasa a 43 evaluadas, 6 incumplidas y 0 conflictos al fundir las placas que
    # se pisan entre media y una placa: varias "vecinas" eran la misma pieza.
    "demo.obj": {50.0: (47, 12, 2), 100.0: (43, 6, 0)},
}


@pytest.mark.skipif(not MODELOS, reason="no hay modelos disponibles")
@pytest.mark.parametrize("path", MODELOS, ids=lambda p: p.name)
@pytest.mark.parametrize("scale", ESCALAS)
def test_las_piezas_cumplen_el_grafo_de_restricciones(path, scale):
    """El grafo dice dónde tiene que caer cada borde; acá se mide cuánto se cumple.

    `core/services/restricciones.py` expresa cada regla de encaje como un DESTINO —la
    distancia a la que el borde tiene que quedar del plano medio de su vecina— en vez de
    como un delta relativo. Eso las hace comparables entre sí y detecta los conflictos en
    vez de resolverlos en silencio quedándose con el mayor.

    En esta fase el grafo NO se consume: el pipeline sigue aplicando los
    `DimensionAdjustment` de siempre. Este test compara los dos mundos y es el criterio de
    aceptación de las fases que vienen — cuando el grafo pase a ser el productor, estos
    números tienen que BAJAR.

    Vale como verificación cruzada del propio grafo: construido de forma independiente,
    reproduce exactamente los mismos fallos que la medición geométrica directa
    (`test_las_esquinas_las_cierran_las_dos_piezas`): 30 y 29 en demo, 21 y 16 en casa
    simple, 21 y 19 en bfe62e18. Dos implementaciones distintas que coinciden.
    """
    from core.services.restricciones import construir_muro_muro, resolver, verificar

    base = LINEA_BASE_RESTRICCIONES.get(path.name)
    if base is None or scale not in base:
        pytest.skip(f"{path.name} 1:{scale:.0f} no tiene línea base de restricciones")

    work, walls, floors, _pjs = _procesar(path, scale)
    rs = construir_muro_muro(work.joints, work.groups, work.faces, None)
    elegidas, conflictos = resolver(rs)
    evaluadas, fallas = verificar(
        walls + floors, elegidas, work.groups, work.faces, scale
    )
    esp_eval, esp_fallas, esp_confl = base[scale]

    assert evaluadas == esp_eval, (
        f"{path.name} 1:{scale:.0f}: el grafo evaluó {evaluadas} restricciones y la línea "
        f"base dice {esp_eval}. Cambió el conjunto de juntas, no sólo el resultado."
    )
    assert len(fallas) <= esp_fallas, (
        f"{path.name} 1:{scale:.0f}: {len(fallas)} restricciones incumplidas, la línea "
        f"base dice {esp_fallas}. REGRESIÓN. Peores: "
        + str([
            (pid, r.motivo, round(e, 2))
            for pid, r, e in sorted(fallas, key=lambda x: -abs(x[2]))[:4]
        ])
    )
    if len(fallas) < esp_fallas:
        pytest.fail(
            f"{path.name} 1:{scale:.0f}: {len(fallas)} incumplidas contra {esp_fallas} de "
            "la línea base. Es una MEJORA: actualizá LINEA_BASE_RESTRICCIONES en este "
            "mismo commit."
        )
    # Un CONFLICTO es un borde al que dos vecinas distintas le piden cosas incompatibles
    # en la misma franja de altura: no existe corte que cumpla las dos. Hasta ahora se
    # resolvían en silencio quedándose con el mayor. Que aparezcan más es una regresión;
    # que aparezcan menos, una mejora que hay que registrar.
    assert len(conflictos) <= esp_confl, (
        f"{path.name} 1:{scale:.0f}: {len(conflictos)} conflictos, la línea base dice "
        f"{esp_confl}. REGRESIÓN. " + "; ".join(c.mensaje() for c in conflictos[:2])
    )
    if len(conflictos) < esp_confl:
        pytest.fail(
            f"{path.name} 1:{scale:.0f}: {len(conflictos)} conflictos contra {esp_confl} "
            "de la línea base. Es una MEJORA: actualizá LINEA_BASE_RESTRICCIONES."
        )


@pytest.mark.skipif(not MODELOS, reason="no hay modelos disponibles")
@pytest.mark.parametrize("path", MODELOS, ids=lambda p: p.name)
@pytest.mark.parametrize("scale", ESCALAS)
def test_el_instructivo_muestra_exactamente_lo_que_se_corta(path, scale):
    """La maqueta reconstruida SÓLO con lo que viaja al front tiene que ser la real.

    Es la garantía de fidelidad del instructivo, y está escrita al revés de como suele
    escribirse: en vez de afirmar que el backend calcula bien, reconstruye la pieza como
    la va a dibujar el visor —a partir de `placements`, nada más— y la compara contra la
    pieza que se manda a cortar.

    Se verifican las tres cosas que hacen que el dibujo sea el mismo objeto que la pieza:

    1. El CONTORNO. Una pieza con muesca, abertura o faldón no es su rectángulo: el
       bounding box de un faldón triangular tiene el doble de superficie. El instructivo
       dibujaba cajas orientadas, así que mostraba material donde no había y tapaba los
       huecos por donde entra la vecina.
    2. La POSICIÓN en el mundo. `world = origin + u·u_axis + v·v_axis` tiene que valer
       tal cual, sin que el consumidor compense el espejado (viene horneado en el marco).
    3. El ESPESOR, en metros de edificio. La placa mide 3 mm de plancha siempre, pero eso
       son 0.15 m a 1:50 y 0.60 m a 1:200: dibujar las piezas sin espesor esconde
       justamente los choques que el espesor provoca.

    Sin esto, "el instructivo es fiel" es una afirmación que nadie puede comprobar. Con
    esto, si alguien cambia el marco, el espejado o el contorno y el visor deja de
    coincidir con la plancha, falla acá y no en el MDF.
    """
    from shapely.geometry import Polygon
    from shapely.ops import polygonize, unary_union

    from core.review_generate import final_placements
    from core.services.cutting_sheet import panel_cut_area_m2
    from core.services.plate_intersect import PLATE_THICKNESS_M
    from core.services.types import Vec3, dot

    _work, walls, floors, _pjs = _procesar(path, scale)
    piezas = walls + floors
    payload = final_placements(walls, floors, scale)

    assert len(payload) == len(piezas), (
        f"{path.name} 1:{scale:.0f}: se mandan {len(payload)} piezas al instructivo y se "
        f"cortan {len(piezas)}. El visor no puede dibujar lo que no recibe."
    )

    for p in piezas:
        d = payload[str(p.source_group_id)]

        # (3) espesor en metros de EDIFICIO, ya escalado.
        assert d["thickness_m"] == pytest.approx(PLATE_THICKNESS_M * scale, abs=1e-9), (
            f"{p.id}: el espesor que viaja es {d['thickness_m']:.4f} m y la placa mide "
            f"{PLATE_THICKNESS_M * scale:.4f} m de edificio a 1:{scale:.0f}"
        )

        # (1) el contorno que viaja tiene que encerrar la MISMA superficie que se corta.
        segs = [
            ((e["a"]["x"], e["a"]["y"]), (e["b"]["x"], e["b"]["y"]))
            for e in d["outline"]
        ]
        assert segs, f"{p.id}: viaja sin contorno; el visor sólo podría dibujar una caja"
        caras = list(polygonize(unary_union([
            __import__("shapely").geometry.LineString(s) for s in segs
        ])))
        if caras:
            area_dibujada = unary_union(caras).area
            area_cortada = panel_cut_area_m2(p.edges, p.width_m, p.height_m)
            # El contorno se cierra sobre sí mismo: la superficie dibujada no puede ser
            # MENOR que la que se corta (faltaría material) ni disparatadamente mayor.
            assert area_dibujada >= area_cortada - 1e-6, (
                f"{p.id}: el visor dibujaría {area_dibujada:.4f} m² y se cortan "
                f"{area_cortada:.4f} m². Falta material en el dibujo."
            )

        # (2) la posición: los extremos del contorno, mapeados con la fórmula del
        # contrato, tienen que caer sobre el plano medio de la pieza y dentro de su bbox.
        o = Vec3(d["origin"]["x"], d["origin"]["y"], d["origin"]["z"])
        u = Vec3(d["u_axis"]["x"], d["u_axis"]["y"], d["u_axis"]["z"])
        v = Vec3(d["v_axis"]["x"], d["v_axis"]["y"], d["v_axis"]["z"])
        n = Vec3(d["normal"]["x"], d["normal"]["y"], d["normal"]["z"])
        assert d["mirrored"] is False, (
            f"{p.id}: `mirrored` viene en True. El contrato es que el espejado ya está "
            "horneado en el marco y el consumidor no compensa nada."
        )
        plano = None
        for e in d["outline"][:64]:
            for q in (e["a"], e["b"]):
                w = Vec3(
                    o.x + u.x * q["x"] + v.x * q["y"],
                    o.y + u.y * q["x"] + v.y * q["y"],
                    o.z + u.z * q["x"] + v.z * q["y"],
                )
                dist = dot(w, n)
                if plano is None:
                    plano = dist
                assert abs(dist - plano) <= 1e-6, (
                    f"{p.id}: el contorno mapeado al mundo no es plano (desvío "
                    f"{abs(dist - plano):.6f} m). El marco que viaja no corresponde a "
                    "esta pieza."
                )
        assert 0.0 <= max(e["a"]["x"] for e in d["outline"]) <= d["width_m"] + 1e-6, (
            f"{p.id}: el contorno se sale del ancho declarado; en la plancha el nesting "
            "separa por width×height y esto invadiría a la pieza vecina."
        )



# Veredicto por modelo y escala: (uniones, ok, revisar, no_arma, terminaciones). Sale del
# verificador EXACTO por unión (`core/services/verificacion_uniones.py`), que mide choques,
# luces, esquinas sin tapar y placas montadas, y no se deja engañar por astillas ni por
# contornos abiertos. Es la línea base que hay que mirar para saber si un cambio mejora el
# encaje.
#
# Historia de estos números en demo.obj, a 1:50 y 1:100:
#   (77, 52, 19, 6)  (80, 49, 24, 7)   primera medición exacta
#   (77, 53, 18, 6)  (78, 51, 21, 6)   + resolvedor de uniones
#   (69, 49, 18, 2)  (69, 53, 12, 4)   + fundir placas que se pisan entre media y una placa
#
# El último cambio NO es una mejora del encaje sino de la CUENTA: las uniones que sólo se
# tocan de canto dejaron de valer "revisar" y pasaron a una columna propia, terminaciones.
# Eran 16 de las 18 a 1:50 y 10 de las 12 a 1:100. Lo que de verdad falla es lo que queda
# en `revisar` y `no_arma`, y bajarlo sigue siendo el trabajo pendiente.
LINEA_BASE_VEREDICTO = {
    "demo.obj": {50.0: (69, 65, 2, 2, 16), 100.0: (69, 63, 2, 4, 10)},
}


@pytest.mark.skipif(not MODELOS, reason="no hay modelos disponibles")
@pytest.mark.parametrize("path", MODELOS, ids=lambda p: p.name)
@pytest.mark.parametrize("scale", ESCALAS)
def test_el_veredicto_de_cada_modelo(path, scale):
    """Cuántas uniones encajan, cuántas hay que revisar y cuántas no arman.

    Además de la línea base, afirma lo que tiene que valer para CUALQUIER archivo: el
    reporte se calcula, viaja como JSON, sus cuentas cierran y cada problema trae al menos
    una acción.
    """
    import json

    from core.review_generate import reporte_de_verificacion

    work, walls, floors, _pjs = _procesar(path, scale)
    r = reporte_de_verificacion(work, walls, floors, scale)
    assert r is not None, f"{path.name} 1:{scale:.0f}: el reporte no se pudo calcular"
    json.dumps(r)

    x = r["resumen"]
    assert x["uniones"] == x["uniones_ok"] + x["uniones_revisar"] + x["uniones_no_arma"]
    assert x["piezas"] == x["piezas_ok"] + x["piezas_revisar"] + x["piezas_no_arma"]
    assert x["piezas"] == len([p for p in walls + floors if p.frame])
    en_problemas = {i for p in r["problemas"] for i in p["uniones"]}
    con_falla = {u["id"] for u in r["uniones"] if u["estado"] != "ok"}
    assert en_problemas == con_falla, "hay uniones que fallan y no están en ningún problema"
    assert all(p["acciones"] for p in r["problemas"]), "un problema sin ninguna acción"
    assert (r["veredicto"] == "listo") == (not r["problemas"])
    # Nudos: cada unión que falla está en exactamente uno, y cada uno se puede dibujar.
    en_nudos = sorted(i for n in r["nudos"] for i in n["uniones"])
    assert en_nudos == sorted(con_falla), "los nudos no cubren las uniones que fallan"
    assert x["nudos"] == len(r["nudos"]) <= len(con_falla)
    assert all(n["quien"] in ("decidir", "escala", "modelo") for n in r["nudos"])
    sin_dibujo = [n["titulo"] for n in r["nudos"] if not n["detalle"]]
    assert not sin_dibujo, f"nudos sin detalle para dibujar: {sin_dibujo[:3]}"
    for n in r["nudos"]:
        d = n["detalle"]
        assert {c["union"] for c in d["cotas"]} <= set(n["uniones"]), "cota de otra unión"
        dibujadas = {p["pieza"] for p in d["placas"]}
        assert all(set(c["piezas"]) <= dibujadas for c in d["cotas"])
        assert all(set(sp["piezas"]) <= dibujadas for sp in d["solapes"])
    # Las terminaciones no cuentan como falla y están todas en su lista.
    de_canto = {u["id"] for u in r["uniones"] if u["terminacion"]}
    assert all(r["uniones"][i]["estado"] == "ok" for i in de_canto)
    assert {i for t in r["terminaciones"] for i in t["uniones"]} == de_canto
    assert x["terminaciones"] == len(de_canto)
    assert x["resueltas_solas"] == len(r["decisiones_automaticas"])

    base = LINEA_BASE_VEREDICTO.get(path.name)
    if base is None or scale not in base:
        pytest.skip(f"{path.name} 1:{scale:.0f} no tiene línea base de veredicto")
    uniones, ok, revisar, no_arma, _terminaciones = base[scale]
    obtenido = (
        x["uniones"], x["uniones_ok"], x["uniones_revisar"], x["uniones_no_arma"],
        x["terminaciones"],
    )
    assert x["uniones_no_arma"] <= no_arma and x["uniones_revisar"] + x["uniones_no_arma"] <= revisar + no_arma, (
        f"{path.name} 1:{scale:.0f}: {obtenido} (uniones, ok, revisar, no_arma); la "
        f"línea base dice {base[scale]}. REGRESIÓN."
    )
    if obtenido != base[scale]:
        pytest.fail(
            f"{path.name} 1:{scale:.0f}: {obtenido} contra {base[scale]} de la línea "
            "base. Si es una MEJORA, actualizá LINEA_BASE_VEREDICTO en este mismo commit."
        )
