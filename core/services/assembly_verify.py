"""
Fase D — Verificación del ensamble: ¿las piezas que se van a cortar arman el edificio?

Por qué existe
--------------
El pipeline calcula recortes y ranuras junta por junta, y hasta acá nadie comprobaba que
el CONJUNTO cerrara. Cada defecto se descubría pegando el MDF.

Hubo un intento anterior (`assembly_fit`) que informaba "Ensamble verificado" con las
piezas chocando a la vista, y se revirtió. La lección: **un chequeo que puede dar un falso
OK es peor que no tener ninguno**, porque traslada la confianza al lugar equivocado. De ahí
las tres reglas de este módulo:

1. **Conoce las ranuras.** Dos placas encastradas TIENEN que atravesarse: la ranura es el
   hueco que lo permite. Una primera versión de este chequeo condenaba 6 encastres
   correctos por no saberlo.
2. **Se valida contra casos de respuesta conocida** antes de creerle (ver
   `test_encaje.py::test_verificacion_*`): da 0 en un modelo que el banco afirma correcto y
   >0 en el mismo modelo con el defecto reintroducido.
3. **Señala el par y el milímetro**, y CLASIFICA la causa. Un semáforo verde no sirve para
   decidir nada, y decirle al usuario "revisá el archivo" cuando el problema es nuestro
   quema la confianza más rápido que el error mismo.

Cómo mide
---------
Cada pieza es un rectángulo de `PLATE_THICKNESS_M` centrado en su plano medio. Se muestrean
puntos del plano medio de A dentro de su contorno real y se pregunta si caen ESTRICTAMENTE
dentro del volumen de la placa B. Un tope correcto es tangente —el borde de A queda a media
placa del plano de B— así que no cuenta: sólo cuenta la penetración real.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.services.cutting_sheet import Panel, _edges_to_polygon
from core.services.plate_intersect import PLATE_THICKNESS_M, pair_key

# Margen por debajo del cual una penetración es ruido de proyección y no un choque. Del
# orden del kerf, igual que la tolerancia del banco de pruebas.
TOL_MM = 0.2
# Lado de la grilla de muestreo, en metros de PLANCHA. 3 mm es el espesor de la placa: una
# penetración más chica que eso ya está por debajo de lo que se puede detectar sin que el
# muestreo sea el que manda.
PASO_PLANCHA_M = 0.003
# Tope de muestras por pieza, para que el costo no explote en modelos grandes.
MAX_MUESTRAS = 400


@dataclass
class Interferencia:
    """Dos piezas que ocupan el mismo material."""
    pieza_a: str
    pieza_b: str
    # Cuánto se meten una en otra, en mm de PLANCHA. Es una COTA INFERIOR: se mide sobre
    # los puntos muestreados, así que la penetración real es esa o mayor. Sirve para
    # ordenar y para el mensaje ("se pisan al menos X mm"), no para afirmar el valor
    # exacto. Prometer exactitud acá sería la clase de precisión falsa que ya nos costó
    # un chequeo entero.
    penetracion_mm: float
    # Por qué pasa. Define el mensaje y qué puede hacer el usuario:
    #   "sin_recorte"   -> la junta se detectó pero nadie la resolvió (defecto NUESTRO)
    #   "no_detectada"  -> las piezas se tocan y el sistema no vio la unión (defecto NUESTRO)
    #   "escala"        -> a esta escala la placa no entra; a una más fina sí (cambiar escala)
    #   "modelo"        -> el modelo tiene un solape que no es un encuentro (revisar archivo)
    causa: str


def _piezas(panels: List[Panel]):
    """Contorno + marco 3D de cada pieza que tenga posición en el edificio."""
    import numpy as np

    out = []
    for p in panels:
        if not p.frame:
            continue  # refuerzos sueltos: se pegan a mano, no tienen lugar fijo
        poly = _edges_to_polygon([e for e in p.edges if not getattr(e, "flex", False)])
        if poly is None or poly.area <= 1e-9:
            continue
        m = p.frame
        o = np.array([m["origin"]["x"], m["origin"]["y"], m["origin"]["z"]])
        u = np.array([m["u_axis"]["x"], m["u_axis"]["y"], m["u_axis"]["z"]])
        v = np.array([m["v_axis"]["x"], m["v_axis"]["y"], m["v_axis"]["z"]])
        n = np.cross(u, v)
        ln = float(np.linalg.norm(n))
        if ln < 1e-9:
            continue
        n = n / ln
        out.append({
            "id": p.id, "gid": p.source_group_id, "poly": poly,
            "o": o, "u": u, "v": v, "n": n, "d": float(n @ o),
            "slots": set(getattr(p, "slots_against", ()) or ()),
        })
    return out


def _muestrear(pz, paso: float):
    import numpy as np
    from shapely.geometry import Point
    from shapely.prepared import prep

    minx, miny, maxx, maxy = pz["poly"].bounds
    ancho = max(maxx - minx, 1e-9)
    alto = max(maxy - miny, 1e-9)
    # Ajustar el paso si la pieza es grande, para no pasarse de MAX_MUESTRAS.
    n_est = (ancho / paso) * (alto / paso)
    if n_est > MAX_MUESTRAS:
        paso = paso * (n_est / MAX_MUESTRAS) ** 0.5
    listo = prep(pz["poly"])
    pts = []
    y = miny + paso / 2
    while y < maxy:
        x = minx + paso / 2
        while x < maxx:
            if listo.contains(Point(x, y)):
                pts.append(pz["o"] + pz["u"] * x + pz["v"] * y)
            x += paso
        y += paso
    return np.array(pts) if pts else np.zeros((0, 3))


def verificar_ensamble(
    panels: List[Panel],
    plate_joints: Optional[List] = None,
    scale_denom: float = 50.0,
    joints_detectadas: Optional[set] = None,
    juntas_sin_resolver: Optional[set] = None,
) -> List[Interferencia]:
    """Pares de piezas que se interpenetran sin que ninguna ranura lo justifique.

    `joints_detectadas` y `juntas_sin_resolver` son conjuntos de `pair_key(gid_a, gid_b)`
    y sólo sirven para CLASIFICAR la causa; el chequeo geométrico no depende de ellos.
    """
    import numpy as np
    from shapely.geometry import Point
    from shapely.prepared import prep

    placa = PLATE_THICKNESS_M * scale_denom      # espesor en metros de EDIFICIO
    media = placa / 2.0
    eps = TOL_MM / 1000.0 * scale_denom
    paso = PASO_PLANCHA_M * scale_denom

    pz = _piezas(panels)
    if len(pz) < 2:
        return []
    for p in pz:
        p["pts"] = _muestrear(p, paso)
        p["listo"] = prep(p["poly"])
        # AABB de la pieza en el mundo, engordada media placa: dos piezas cuyas cajas no
        # se tocan no pueden interpenetrarse. Sin esta poda el costo es O(n² · muestras)
        # y en un modelo grande no cierra.
        if len(p["pts"]):
            p["lo"] = p["pts"].min(axis=0) - media
            p["hi"] = p["pts"].max(axis=0) + media
        else:
            p["lo"] = p["hi"] = None

    # Pares excusados: SÓLO aquellos cuya ranura quedó realmente cortada en la pieza.
    # `plate_joints` no sirve para esto: contiene juntas cuyo segmento después cae fuera
    # del panel ya recortado y nunca se dibuja. Excusar por esas es inventar un encastre
    # que no existe, que es justo el falso negativo que hizo inservible al chequeo
    # anterior.
    con_ranura = set()
    for p in pz:
        for otro_gid in p["slots"]:
            con_ranura.add(pair_key(p["gid"], otro_gid))
    joints_detectadas = joints_detectadas or set()
    juntas_sin_resolver = juntas_sin_resolver or set()

    peor: Dict[Tuple[str, str], float] = {}
    grupos: Dict[Tuple[str, str], Tuple[int, int]] = {}

    for i in range(len(pz)):
        for j in range(len(pz)):
            if i == j:
                continue
            A, B = pz[i], pz[j]
            if len(A["pts"]) == 0 or B["lo"] is None:
                continue
            if (A["hi"] < B["lo"]).any() or (B["hi"] < A["lo"]).any():
                continue  # cajas separadas: no pueden pisarse
            k_grupo = pair_key(A["gid"], B["gid"])
            if k_grupo in con_ranura:
                continue  # encastre: TIENEN que atravesarse, es lo correcto
            dist = A["pts"] @ B["n"] - B["d"]
            dentro = np.abs(dist) < media - eps
            if not dentro.any():
                continue
            rel = A["pts"][dentro] - B["o"]
            uu = rel @ B["u"]
            vv = rel @ B["v"]
            prof = 0.0
            for a, b, dd in zip(uu, vv, np.abs(dist[dentro])):
                if B["listo"].contains(Point(a, b)):
                    prof = max(prof, media - float(dd))
            if prof <= 0.0:
                continue
            clave = tuple(sorted((A["id"], B["id"])))
            if prof > peor.get(clave, 0.0):
                peor[clave] = prof
                grupos[clave] = k_grupo

    salida: List[Interferencia] = []
    for (a, b), prof in sorted(peor.items(), key=lambda kv: -kv[1]):
        k = grupos[(a, b)]
        if k in juntas_sin_resolver:
            causa = "sin_recorte"
        elif k not in joints_detectadas:
            causa = "no_detectada"
        elif prof < media:
            # La junta se detectó y se resolvió, pero la placa sigue sin entrar: a esta
            # escala ocupa más edificio del que hay disponible.
            causa = "escala"
        else:
            causa = "modelo"
        salida.append(
            Interferencia(
                pieza_a=a, pieza_b=b,
                penetracion_mm=prof / scale_denom * 1000.0,
                causa=causa,
            )
        )
    return salida


MENSAJES = {
    "sin_recorte": (
        "El sistema detectó esta unión pero no la resolvió: ninguna de las dos piezas se "
        "acortó y no se generó ranura. Es un problema del procesamiento, no del archivo."
    ),
    "no_detectada": (
        "Estas dos piezas se tocan en el modelo y el sistema no reconoció la unión. Es un "
        "problema del procesamiento, no del archivo."
    ),
    "escala": (
        "A esta escala la placa de 3 mm representa demasiado edificio y las piezas no "
        "entran. Probá una escala más fina: el mismo modelo suele cerrar a 1:50."
    ),
    "modelo": (
        "Las piezas se superponen más de lo que un encuentro explica. Conviene revisar el "
        "archivo 3D en esa zona."
    ),
}
