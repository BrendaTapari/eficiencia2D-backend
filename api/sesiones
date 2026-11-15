"""Lo que el servidor recuerda entre un pedido y el siguiente.

Por qué existe
--------------
Cada botón del instructivo (fusionar, no cortar, invertir quién cede) vuelve a pedirle al
servidor el estado del modelo. Sin memoria, cada pedido releía el modelo del disco (5 s en
uno de 300 piezas), rearmaba TODAS las piezas (otros 5 a 15 s) y devolvía 120 MB. El
usuario veía "el backend no respondió a tiempo" y el cambio nunca aparecía.

Acá viven tres cosas, todas memoria del proceso (se pierden al reiniciar y entonces el
primer pedido vuelve a tardar lo que tardaba; nunca cambian QUÉ se calcula):

* los últimos modelos leídos (`recordar_modelo` / `modelo_en_memoria`);
* una SESIÓN por modelo + escala + cortes manuales, con lo ya calculado para ese modelo:
  las piezas que no cambian entre una acción y la siguiente, lo ya medido, las etiquetas
  de referencia y las respuestas ya armadas;
* el EFECTO PREVISTO de cada acción que el reporte le ofrece al usuario, calculado en
  segundo plano apenas se responde. Cuando el usuario toca el botón la respuesta ya está.

Un solo cálculo pesado por vez
------------------------------
`CALCULO` es un candado global. Los cálculos son CPU pura en Python: dos a la vez no van
más rápido, sólo se pisan. El trabajo en segundo plano lo toma de a UNA acción; si llega
un pedido del usuario, abandona lo que estaba haciendo y le cede el paso.

Tamaños (variables de entorno)
------------------------------
MODELOS_EN_MEMORIA   (1)  modelos leídos que se conservan. Uno grande ocupa cientos de MB.
SESIONES_EN_MEMORIA  (2)  sesiones que se conservan.
RESPUESTAS_POR_SESION (40) respuestas ya armadas por sesión (unos 2 MB cada una).
EFECTOS_PREVISTOS    (1)  0 = no calcular efectos en segundo plano.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("eficiencia2d.pipeline")


def _entero(nombre: str, por_defecto: int) -> int:
    try:
        return max(0, int(os.environ.get(nombre, por_defecto)))
    except (TypeError, ValueError):
        return por_defecto


MAX_MODELOS = _entero("MODELOS_EN_MEMORIA", 1)
MAX_SESIONES = _entero("SESIONES_EN_MEMORIA", 2)
MAX_RESPUESTAS = _entero("RESPUESTAS_POR_SESION", 40)
EFECTOS_PREVISTOS = bool(_entero("EFECTOS_PREVISTOS", 1))

# ---------------------------------------------------------------------------
# Un cálculo por vez, con prioridad para los pedidos del usuario
# ---------------------------------------------------------------------------

CALCULO = threading.RLock()
_contador = threading.Lock()
_esperando = 0


@contextmanager
def calculo_en_primer_plano():
    """Toma el candado de cálculo avisando que hay un usuario esperando."""
    global _esperando
    with _contador:
        _esperando += 1
    try:
        with CALCULO:
            yield
    finally:
        with _contador:
            _esperando -= 1


def hay_pedidos_esperando() -> bool:
    return _esperando > 0


# ---------------------------------------------------------------------------
# Modelos en memoria
# ---------------------------------------------------------------------------

_tablas = threading.Lock()
_MODELOS: "OrderedDict[str, object]" = OrderedDict()
_SESIONES: "OrderedDict[tuple, Dict]" = OrderedDict()


def modelo_en_memoria(file_id: str):
    with _tablas:
        modelo = _MODELOS.get(file_id)
        if modelo is not None:
            _MODELOS.move_to_end(file_id)
        return modelo


def recordar_modelo(file_id: str, modelo) -> None:
    if MAX_MODELOS <= 0 or modelo is None:
        return
    with _tablas:
        _MODELOS[file_id] = modelo
        _MODELOS.move_to_end(file_id)
        while len(_MODELOS) > MAX_MODELOS:
            _MODELOS.popitem(last=False)


def olvidar_modelo(file_id: str) -> None:
    """El modelo de `file_id` cambió: ni él ni nada calculado sobre él sirve más."""
    with _tablas:
        _MODELOS.pop(file_id, None)
        for clave in [c for c in _SESIONES if c and c[0] == file_id]:
            _SESIONES.pop(clave)["generacion"] = -1


def olvidar_todo() -> None:
    with _tablas:
        _MODELOS.clear()
        for s in _SESIONES.values():
            s["generacion"] = -1
        _SESIONES.clear()


# ---------------------------------------------------------------------------
# Sesiones
# ---------------------------------------------------------------------------

def _canonico(valor) -> str:
    return json.dumps(valor, sort_keys=True, separators=(",", ":"), default=str)


def clave_de_sesion(file_id: str, scale_denom, **lo_que_fija_las_piezas) -> tuple:
    """Todo lo que, si cambia, hace que las piezas sean otras aunque el usuario no haya
    tocado ningún botón del instructivo: modelo, eje, área mínima, divisiones, escala,
    cortes manuales, refuerzos…"""
    return (
        str(file_id),
        round(float(scale_denom or 0.0), 6),
        _canonico({k: v for k, v in lo_que_fija_las_piezas.items() if v}),
    )


def sesion_para(clave: tuple) -> Dict:
    if MAX_SESIONES <= 0:
        return _nueva()
    with _tablas:
        s = _SESIONES.get(clave)
        if s is None:
            s = _SESIONES[clave] = _nueva()
        _SESIONES.move_to_end(clave)
        while len(_SESIONES) > MAX_SESIONES:
            _, vieja = _SESIONES.popitem(last=False)
            vieja["generacion"] = -1          # frena su trabajo en segundo plano
        return s


def sesion_si_existe(clave: tuple) -> Optional[Dict]:
    with _tablas:
        return _SESIONES.get(clave)


def _nueva() -> Dict:
    return {
        "calculo": {},                 # lo que conserva `_decompose` (ver `nueva_sesion`)
        "modelo": None,                # el modelo ya con eje, área mínima y divisiones
        "respuestas": OrderedDict(),   # clave de estado -> (bytes JSON, resumen)
        "efectos": {},                 # clave de estado -> {id de acción: efecto}
        "pendientes": {},              # clave de estado -> acciones que faltan calcular
        "generacion": 0,               # -1 = sesión descartada
        "actual": None,                # (clave, estado, calcular) que el usuario está viendo
        "hilo": None,
        "cerrojo": threading.Lock(),   # protege actual / pendientes / efectos / hilo
    }


# ---------------------------------------------------------------------------
# Estado de revisión: lo que el usuario decidió con los botones
# ---------------------------------------------------------------------------

def estado_de(overrides, merges, wall_wall_decisions, por_par) -> Dict:
    return {
        "overrides": {str(k): v for k, v in (overrides or {}).items()},
        "merges": [list(m) for m in (merges or [])],
        "wall_wall_decisions": {str(k): v for k, v in (wall_wall_decisions or {}).items()},
        "wall_wall_decisions_by_pair": [dict(d) for d in (por_par or [])],
    }


def clave_de_estado(estado: Dict) -> str:
    por_par = sorted(
        (min(d["group_a"], d["group_b"]), max(d["group_a"], d["group_b"]), d["yield_group_id"])
        for d in estado["wall_wall_decisions_by_pair"]
    )
    return _canonico({
        "o": estado["overrides"],
        "m": sorted(sorted(m) for m in estado["merges"]),
        "w": estado["wall_wall_decisions"],
        "p": por_par,
    })


def estado_con(estado: Dict, aplicar: Dict) -> Dict:
    """El estado después de aplicar una acción del reporte. Es lo MISMO que tiene que
    hacer el front con `accion.aplicar` (ver CONTRATO_verificacion.md): los descartes se
    agregan, las fusiones se agregan, y una decisión de quién cede reemplaza a la que
    hubiera para ese par."""
    nuevo = {
        "overrides": dict(estado["overrides"]),
        "merges": [list(m) for m in estado["merges"]],
        "wall_wall_decisions": dict(estado["wall_wall_decisions"]),
        "wall_wall_decisions_by_pair": [dict(d) for d in estado["wall_wall_decisions_by_pair"]],
    }
    for gid, cat in (aplicar.get("overrides") or {}).items():
        nuevo["overrides"][str(gid)] = cat
    for m in aplicar.get("merges") or []:
        if sorted(m) not in [sorted(x) for x in nuevo["merges"]]:
            nuevo["merges"].append(list(m))
    for d in aplicar.get("wall_wall_decisions_by_pair") or []:
        par = {d["group_a"], d["group_b"]}
        nuevo["wall_wall_decisions_by_pair"] = [
            x for x in nuevo["wall_wall_decisions_by_pair"]
            if {x["group_a"], x["group_b"]} != par
        ] + [dict(d)]
    return nuevo


# ---------------------------------------------------------------------------
# Respuestas ya armadas
# ---------------------------------------------------------------------------

def respuesta_guardada(sesion: Dict, clave: str):
    r = sesion["respuestas"].get(clave)
    if r is not None:
        sesion["respuestas"].move_to_end(clave)
    return r


def guardar_respuesta(sesion: Dict, clave: str, cuerpo: bytes, resumen: Dict) -> None:
    sesion["respuestas"][clave] = (cuerpo, resumen)
    sesion["respuestas"].move_to_end(clave)
    while len(sesion["respuestas"]) > MAX_RESPUESTAS:
        vieja, _ = sesion["respuestas"].popitem(last=False)
        sesion["efectos"].pop(vieja, None)
        sesion["pendientes"].pop(vieja, None)


# ---------------------------------------------------------------------------
# Efecto previsto de cada acción
# ---------------------------------------------------------------------------

def acciones_a_prever(reporte: Optional[Dict]) -> List[Dict]:
    """Las acciones que el usuario puede tocar: las de los nudos "para decidir" que
    cambian decisiones (no las de cambiar de escala, que son otro modelo de piezas)."""
    vistas, out = set(), []
    for nudo in (reporte or {}).get("nudos", []):
        if nudo.get("quien") != "decidir":
            continue
        for a in nudo.get("acciones", []):
            aplicar = a.get("aplicar")
            if not aplicar or "scale_denom" in aplicar or not a.get("id"):
                continue
            if a["id"] in vistas:
                continue
            vistas.add(a["id"])
            out.append({"id": a["id"], "aplicar": aplicar, "punto": nudo.get("punto"),
                        "grupos": list(nudo.get("grupos") or []),
                        "radio": 3.0 * (reporte.get("espesor_placa_mm", 3.0) / 1000.0)
                                 * (reporte.get("escala") or 1.0)})
    return out


def resumen_de(reporte: Optional[Dict]) -> Dict:
    r = (reporte or {}).get("resumen", {})
    return {
        "veredicto": (reporte or {}).get("veredicto"),
        "no_arma": r.get("uniones_no_arma", 0),
        "revisar": r.get("uniones_revisar", 0),
        "nudos": r.get("nudos", 0),
        "piezas_ok": r.get("piezas_ok", 0),
        "piezas": r.get("piezas", 0),
        "lugares": [
            ((n["punto"]["x"], n["punto"]["y"], n["punto"]["z"]), set(n.get("grupos") or []))
            for n in (reporte or {}).get("nudos", []) if n.get("punto")
        ],
    }


def efecto(antes: Dict, despues: Dict, accion: Dict) -> Dict:
    """Qué pasa si el usuario toca este botón, en palabras y en números."""
    d_no_arma = despues["no_arma"] - antes["no_arma"]
    d_revisar = despues["revisar"] - antes["revisar"]
    d_nudos = despues["nudos"] - antes["nudos"]
    # ¿Sigue habiendo un nudo en ESTE lugar? Es el mismo lugar si está cerca o si
    # involucra a dos de las mismas piezas (al fusionar, el centro del nudo se corre).
    p, radio = accion.get("punto"), accion.get("radio") or 0.0
    grupos = set(accion.get("grupos") or [])
    lugar = None
    if p:
        sigue = any(
            (x - p["x"]) ** 2 + (y - p["y"]) ** 2 + (z - p["z"]) ** 2 <= radio * radio
            or len(grupos & gs) >= 2
            for (x, y, z), gs in despues["lugares"]
        )
        lugar = "sigue" if sigue else "resuelto"
    peor = d_no_arma > 0 or (d_no_arma == 0 and d_nudos > 0)
    mejor = d_no_arma < 0 or (d_no_arma == 0 and (d_nudos < 0 or (d_nudos == 0 and d_revisar < 0)))

    def cuantos(n: int, uno: str, varios: str) -> str:
        return f"{n} {uno if n == 1 else varios}"

    if peor:
        resultado = "empeora"
        if d_no_arma > 0:
            texto = "Empeora: " + cuantos(d_no_arma, "unión más no arma", "uniones más no arman")
        else:
            texto = "Empeora: " + cuantos(d_nudos, "lugar nuevo para revisar", "lugares nuevos para revisar")
    elif lugar == "resuelto" and d_nudos >= 0:
        # Este lugar queda bien pero la cuenta no baja: el problema se mudó.
        resultado = "igual"
        texto = "Resuelve este lugar, pero aparece otro para revisar"
    elif lugar == "resuelto":
        resultado = "mejora"
        otros = -d_nudos - 1
        texto = "Resuelve este lugar" + (f" y {cuantos(otros, 'otro', 'otros')} más" if otros > 0 else "")
    elif mejor:
        resultado = "mejora"
        texto = "Mejora, pero este lugar sigue para revisar"
    else:
        resultado = "igual"
        texto = "No cambia el resultado: este lugar sigue igual"
    return {
        "resultado": resultado,
        "este_lugar": lugar,
        "texto": texto,
        "veredicto": despues["veredicto"],
        "uniones_no_arma": d_no_arma,
        "uniones_revisar": d_revisar,
        "nudos": d_nudos,
        "piezas_ok": despues["piezas_ok"] - antes["piezas_ok"],
    }


class CalculoCancelado(BaseException):
    """Un cálculo en segundo plano se interrumpió porque llegó un pedido del usuario.

    Hereda de BaseException A PROPÓSITO. El cálculo tiene varios `except Exception` que
    siguen adelante con un resultado peor ("si el resolvedor falla, se usan las decisiones
    sugeridas"). Si esta interrupción cayera en uno de ellos, el cálculo terminaría
    "bien" con un resultado a medias y se guardaría como la respuesta de ese botón. Así
    ninguno la atrapa: sube hasta `_trabajar`, que descarta todo y reintenta después.
    """


def _cancelar_si_hay_pedidos() -> None:
    if hay_pedidos_esperando():
        raise CalculoCancelado()


def prever_efectos(
    sesion: Dict, estado: Dict, acciones: List[Dict],
    calcular: Callable[[Dict], tuple],
) -> None:
    """Anota las acciones del estado que el usuario está viendo y arranca (si no está
    corriendo) el hilo que calcula el efecto de cada una.

    `calcular(estado) -> (bytes, resumen)` es el mismo cálculo del pedido.
    """
    if not EFECTOS_PREVISTOS or sesion["generacion"] < 0:
        return
    clave = clave_de_estado(estado)
    with sesion["cerrojo"]:
        sesion["generacion"] += 1
        hechas = sesion["efectos"].setdefault(clave, {})
        # Sólo interesan las acciones del estado que el usuario tiene delante.
        sesion["pendientes"] = {
            clave: [a for a in acciones if a["id"] not in hechas]
        }
        sesion["actual"] = (clave, estado, calcular)
        if sesion["pendientes"][clave] and sesion.get("hilo") is None:
            hilo = threading.Thread(
                target=_trabajar, args=(sesion,), daemon=True, name="efectos-previstos",
            )
            sesion["hilo"] = hilo
            hilo.start()


def _trabajar(sesion: Dict) -> None:
    """Calcula de a una las acciones pendientes del estado actual.

    El usuario siempre va primero: no empieza una acción si hay un pedido esperando, y si
    llega uno a mitad de un cálculo lo abandona (`CalculoCancelado`) y lo reintenta
    después. Lo ya calculado para piezas sueltas queda guardado, así que el reintento no
    empieza de cero.
    """
    armado = sesion["calculo"].setdefault("armado", {})
    while True:
        if hay_pedidos_esperando():
            time.sleep(0.05)
            continue
        with sesion["cerrojo"]:
            clave, estado, calcular = sesion.get("actual") or (None, None, None)
            cola = sesion["pendientes"].get(clave) or []
            if sesion["generacion"] < 0 or not cola:
                sesion["hilo"] = None
                return
            accion = cola[0]
            base = respuesta_guardada(sesion, clave)
        if base is None:
            with sesion["cerrojo"]:
                cola.clear()
            continue
        resultado = None
        with CALCULO:
            try:
                siguiente = estado_con(estado, accion["aplicar"])
                k = clave_de_estado(siguiente)
                hecha = respuesta_guardada(sesion, k)
                if hecha is None:
                    armado["cancelar"] = _cancelar_si_hay_pedidos
                    try:
                        hecha = calcular(siguiente)
                    finally:
                        armado.pop("cancelar", None)
                    guardar_respuesta(sesion, k, *hecha)
                resultado = efecto(base[1], hecha[1], accion)
            except CalculoCancelado:
                continue                 # queda en la cola; primero el usuario
            except Exception:
                logger.exception("[efectos] no se pudo prever la acción %s", accion["id"])
                resultado = {
                    "resultado": "desconocido", "este_lugar": None,
                    "texto": "No se pudo calcular el efecto",
                }
        with sesion["cerrojo"]:
            sesion["efectos"].setdefault(clave, {})[accion["id"]] = resultado
            if cola and cola[0] is accion:
                cola.pop(0)


def efectos_de(sesion: Dict, estado: Dict) -> Dict:
    clave = clave_de_estado(estado)
    hechos = dict(sesion["efectos"].get(clave, {}))
    faltan = len(sesion["pendientes"].get(clave) or [])
    return {
        "listo": faltan == 0,
        "calculadas": len(hechos),
        "total": len(hechos) + faltan,
        "efectos": hechos,
    }
