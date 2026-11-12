"""
Reporte de verificación: lo que una persona necesita para decidir si manda a cortar.

Por qué existe
--------------
`verificacion_uniones` dice qué pasa en cada encuentro entre dos piezas. Es la verdad, pero
no es legible: una maqueta de 300 piezas tiene más de 500 uniones. Quien la va a armar no
puede revisar 500 renglones, y menos con una entrega encima.

Este módulo convierte esa lista en tres cosas:

1. **Un veredicto.** "listo", "revisar" o "no_arma", con cuántas piezas y uniones hay en
   cada estado. Es lo primero que se muestra: 287 de 295 verificadas, 8 para mirar.
2. **Problemas, no renglones.** Las uniones que fallan se AGRUPAN por causa. Treinta muros
   que entran 0.5 mm en la misma losa no son treinta problemas: es uno, con un arreglo.
3. **Una acción por problema.** Cada uno trae lo que se puede hacer, ya armado como el
   cambio que el front le manda de vuelta al backend (fusionar, descartar, invertir quién
   cede). No hay que traducir el diagnóstico a un clic.

Todas las medidas van en milímetros de PLANCHA —lo que se va a ver en la mano— y además en
centímetros de EDIFICIO, que es la unidad en la que piensa quien tiene que corregir el
modelo 3D.

El contrato de la respuesta está en `CONTRATO_verificacion.md`. Si se cambia una clave de
`construir_reporte`, se cambia ese archivo en el mismo commit.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from core.services.cutting_sheet import Panel
from core.services.plate_intersect import PLATE_THICKNESS_M, pair_key
from core.services.verificacion_uniones import (
    TOL_OK_MM,
    TOL_REVISAR_MM,
    Union,
)

VERSION = 1

_ORDEN_ESTADO = {"no_arma": 0, "revisar": 1, "ok": 2}
# Con cuántas uniones del mismo tipo contra una misma pieza se arma un grupo.
MINIMO_PARA_AGRUPAR = 3


def _peor(a: str, b: str) -> str:
    return a if _ORDEN_ESTADO[a] <= _ORDEN_ESTADO[b] else b


def _mm(x: float) -> float:
    return round(x, 2)


def _punto(p) -> Dict[str, float]:
    return {"x": round(p[0], 4), "y": round(p[1], 4), "z": round(p[2], 4)}


_VERBO = {
    "choque": "se pisan",
    "luz": "no llegan a tocarse",
    "sin_cubrir": "se tocan de canto",
    "superpuestas": "están montadas una sobre otra",
}

_VERBO_GRUPO = {
    "choque": "se pisan con ella",
    "luz": "no llegan a tocarla",
    "sin_cubrir": "la tocan sólo de canto",
    "superpuestas": "están montadas sobre ella",
}

_EXPLICACION = {
    "choque": (
        "Las dos piezas ocupan el mismo lugar: una de las dos tiene que acortarse o "
        "llevar una ranura, y hoy no pasa ninguna de las dos cosas."
    ),
    "luz": (
        "En el modelo estas piezas se tocan, pero cortadas en placa de {t:g} mm queda una "
        "separación entre ellas."
    ),
    "sin_cubrir": (
        "Las piezas se tocan sólo por el canto: ninguna de las dos tapa el espesor de la "
        "otra, así que queda el canto a la vista o medio apoyo en el aire."
    ),
    "superpuestas": (
        "Son dos piezas paralelas que en el modelo están más cerca que el espesor de la "
        "placa: físicamente no hay lugar para las dos."
    ),
    "pieza_fina": (
        "La pieza mide menos que el espesor de la placa: no se puede armar parada."
    ),
}


def _acciones(
    u: Union,
    area_por_pieza: Dict[str, float],
    cede_por_par: Dict[Tuple[int, int], int],
    categoria: Dict[int, str],
) -> List[Dict]:
    """Qué se puede hacer con esta unión, ya como el cambio a mandar al backend.

    Cada acción trae `aplicar`, un objeto con las MISMAS claves que aceptan `/recompute`,
    `/nesting-preview` y `/generate`. El front lo fusiona con su estado y vuelve a pedir;
    no tiene que interpretar nada.
    """
    out: List[Dict] = []
    chica, grande = (
        (u.pieza_a, u.pieza_b)
        if area_por_pieza.get(u.pieza_a, 0.0) <= area_por_pieza.get(u.pieza_b, 0.0)
        else (u.pieza_b, u.pieza_a)
    )
    gid_chica = u.grupo_a if chica == u.pieza_a else u.grupo_b
    ambas_muro = categoria.get(u.grupo_a) == "wall" and categoria.get(u.grupo_b) == "wall"

    if u.tipo == "superpuestas":
        if categoria.get(u.grupo_a) == categoria.get(u.grupo_b):
            out.append({
                "tipo": "fusionar",
                "etiqueta": f"Cortarlas como una sola pieza ({u.pieza_a} + {u.pieza_b})",
                "aplicar": {"merges": [[u.grupo_a, u.grupo_b]]},
            })
        out.append({
            "tipo": "descartar",
            "etiqueta": f"No cortar {chica}",
            "aplicar": {"overrides": {str(gid_chica): "discard"}},
        })
    elif u.tipo in ("choque", "luz", "sin_cubrir"):
        cede = cede_por_par.get(pair_key(u.grupo_a, u.grupo_b))
        if ambas_muro and cede is not None:
            otro = u.grupo_b if cede == u.grupo_a else u.grupo_a
            pieza_otro = u.pieza_b if otro == u.grupo_b else u.pieza_a
            out.append({
                "tipo": "invertir_cede",
                "etiqueta": f"Que la que se acorte sea {pieza_otro}",
                "aplicar": {"wall_wall_decisions_by_pair": [{
                    "group_a": u.grupo_a, "group_b": u.grupo_b, "yield_group_id": otro,
                }]},
            })
        if u.tipo == "choque":
            out.append({
                "tipo": "descartar",
                "etiqueta": f"No cortar {chica}",
                "aplicar": {"overrides": {str(gid_chica): "discard"}},
            })
    out.append({
        "tipo": "escala_mas_fina",
        "etiqueta": "Probar una escala más fina (la placa ocupa menos edificio)",
        "aplicar": None,
    })
    out.append({
        "tipo": "revisar_modelo",
        "etiqueta": "Corregir el modelo 3D en esta zona",
        "aplicar": None,
    })
    return out


def _titulo(tipo: str, piezas: List[str], medida_mm: float) -> str:
    if len(piezas) == 2:
        return f"{piezas[0]} y {piezas[1]} {_VERBO[tipo]} ({medida_mm:.1f} mm)"
    return f"{len(piezas)} piezas: {_VERBO[tipo]}"


def construir_reporte(
    panels: List[Panel],
    uniones: List[Union],
    scale_denom: float,
    cede_por_par: Optional[Dict[Tuple[int, int], int]] = None,
    categoria_por_grupo: Optional[Dict[int, str]] = None,
) -> Dict:
    """Reporte serializable (JSON) de la verificación. Ver `CONTRATO_verificacion.md`."""
    escala = scale_denom or 1.0
    cede_por_par = cede_por_par or {}
    categoria = dict(categoria_por_grupo or {})
    espesor_mm = PLATE_THICKNESS_M * 1000.0
    a_cm_edificio = escala / 10.0          # mm de plancha -> cm de edificio

    piezas: Dict[str, Dict] = {}
    area_por_pieza: Dict[str, float] = {}
    for p in panels:
        if not p.frame or p.source_group_id is None or p.source_group_id < 0:
            continue
        piezas[p.id] = {
            "grupo": p.source_group_id,
            "categoria": p.category,
            "estado": "ok",
            "uniones": 0,
            "problemas": 0,
        }
        area_por_pieza[p.id] = float((p.frame or {}).get("area_m2") or p.width_m * p.height_m)
        categoria.setdefault(p.source_group_id, p.category)

    # --- uniones -----------------------------------------------------------------
    filas: List[Dict] = []
    for i, u in enumerate(uniones):
        filas.append({
            "id": i,
            "pieza_a": u.pieza_a,
            "pieza_b": u.pieza_b,
            "grupo_a": u.grupo_a,
            "grupo_b": u.grupo_b,
            "tipo": u.tipo,
            "estado": u.estado,
            "medida_mm": _mm(u.medida_mm),
            "medida_cm_edificio": round(u.medida_mm * a_cm_edificio, 1),
            "largo_m": round(u.largo_m, 3),
            "prevista": bool(u.prevista),
            "pieza_que_ajusta": u.pieza_que_ajusta,
            "punto": _punto(u.punto),
            "segmento": [_punto(u.segmento[0]), _punto(u.segmento[1])],
        })
        for pid in (u.pieza_a, u.pieza_b):
            if pid in piezas:
                piezas[pid]["uniones"] += 1
                piezas[pid]["estado"] = _peor(piezas[pid]["estado"], u.estado)
                if u.estado != "ok":
                    piezas[pid]["problemas"] += 1

    # --- problemas: las uniones que fallan, agrupadas por causa --------------------
    pendientes = [i for i, u in enumerate(uniones) if u.estado != "ok"]
    problemas: List[Dict] = []

    def problema(tipo: str, indices: List[int], comun: Optional[str]) -> Dict:
        us = [uniones[i] for i in indices]
        peor = max(us, key=lambda x: x.medida_mm)
        estado = "ok"
        for x in us:
            estado = _peor(estado, x.estado)
        nombres: List[str] = []
        for x in us:
            for pid in (x.pieza_a, x.pieza_b):
                if pid not in nombres:
                    nombres.append(pid)
        if comun is not None and len(us) > 1:
            otras = [n for n in nombres if n != comun]
            titulo = (
                f"{comun}: {len(otras)} piezas {_VERBO_GRUPO[tipo]} "
                f"(hasta {peor.medida_mm:.1f} mm)"
            )
        else:
            titulo = _titulo(tipo, nombres, peor.medida_mm)
        return {
            "tipo": tipo,
            "estado": estado,
            "titulo": titulo,
            "detalle": _EXPLICACION[tipo].format(t=espesor_mm)
            + f" En el edificio son {peor.medida_mm * a_cm_edificio:.0f} cm.",
            "cantidad": len(us),
            "peor_mm": _mm(peor.medida_mm),
            "peor_cm_edificio": round(peor.medida_mm * a_cm_edificio, 1),
            "piezas": nombres,
            "pieza_comun": comun if len(us) > 1 else None,
            "punto": _punto(peor.punto),
            "uniones": list(indices),
            # Con varias uniones no hay UNA acción que las arregle todas salvo las que
            # actúan sobre la pieza común; las acciones finas van por unión.
            "acciones": (
                _acciones(peor, area_por_pieza, cede_por_par, categoria)
                if len(us) == 1 else _acciones_de_grupo(tipo, comun, us, piezas)
            ),
        }

    por_tipo: Dict[str, List[int]] = defaultdict(list)
    for i in pendientes:
        por_tipo[uniones[i].tipo].append(i)

    for tipo, indices in por_tipo.items():
        restantes = set(indices)
        while restantes:
            cuenta: Dict[str, int] = defaultdict(int)
            for i in restantes:
                cuenta[uniones[i].pieza_a] += 1
                cuenta[uniones[i].pieza_b] += 1
            comun, veces = max(cuenta.items(), key=lambda kv: (kv[1], kv[0]))
            if veces < MINIMO_PARA_AGRUPAR:
                break
            grupo = sorted(i for i in restantes if comun in (uniones[i].pieza_a, uniones[i].pieza_b))
            problemas.append(problema(tipo, grupo, comun))
            restantes -= set(grupo)
        for i in sorted(restantes):
            problemas.append(problema(tipo, [i], None))

    # --- piezas más finas que la placa --------------------------------------------
    limite = PLATE_THICKNESS_M * escala
    for p in panels:
        if p.id not in piezas:
            continue
        chico = min(p.width_m, p.height_m)
        if chico >= limite - 1e-9:
            continue
        mm = chico / escala * 1000.0
        piezas[p.id]["estado"] = _peor(piezas[p.id]["estado"], "revisar")
        piezas[p.id]["problemas"] += 1
        o = (p.frame or {}).get("origin") or {"x": 0.0, "y": 0.0, "z": 0.0}
        problemas.append({
            "tipo": "pieza_fina",
            "estado": "revisar",
            "titulo": f"{p.id} mide {mm:.1f} mm de ancho: menos que la placa",
            "detalle": _EXPLICACION["pieza_fina"]
            + f" En el edificio son {chico * 100:.0f} cm.",
            "cantidad": 1,
            "peor_mm": _mm(mm),
            "peor_cm_edificio": round(chico * 100.0, 1),
            "piezas": [p.id],
            "pieza_comun": None,
            "punto": {"x": round(o["x"], 4), "y": round(o["y"], 4), "z": round(o["z"], 4)},
            "uniones": [],
            "acciones": [
                {
                    "tipo": "descartar",
                    "etiqueta": f"No cortar {p.id}",
                    "aplicar": {"overrides": {str(p.source_group_id): "discard"}},
                },
                {
                    "tipo": "escala_mas_fina",
                    "etiqueta": "Probar una escala más fina",
                    "aplicar": None,
                },
            ],
        })

    problemas.sort(key=lambda x: (_ORDEN_ESTADO[x["estado"]], -x["cantidad"], -x["peor_mm"]))
    for i, pr in enumerate(problemas):
        pr["id"] = i

    def contar(estado: str, donde) -> int:
        return sum(1 for x in donde if x["estado"] == estado)

    veredicto = "listo"
    if any(pr["estado"] == "no_arma" for pr in problemas):
        veredicto = "no_arma"
    elif problemas:
        veredicto = "revisar"

    return {
        "version": VERSION,
        "escala": escala,
        "espesor_placa_mm": espesor_mm,
        "tolerancias_mm": {"ok": TOL_OK_MM, "revisar": TOL_REVISAR_MM},
        "veredicto": veredicto,
        "resumen": {
            "piezas": len(piezas),
            "piezas_ok": contar("ok", piezas.values()),
            "piezas_revisar": contar("revisar", piezas.values()),
            "piezas_no_arma": contar("no_arma", piezas.values()),
            "uniones": len(filas),
            "uniones_ok": contar("ok", filas),
            "uniones_revisar": contar("revisar", filas),
            "uniones_no_arma": contar("no_arma", filas),
            "problemas": len(problemas),
        },
        "piezas": piezas,
        "problemas": problemas,
        "uniones": filas,
    }


def _acciones_de_grupo(tipo: str, comun: Optional[str], us: List[Union], piezas: Dict) -> List[Dict]:
    """Acciones que valen para TODO un grupo de uniones contra una pieza común."""
    out: List[Dict] = []
    if comun is not None and comun in piezas and tipo in ("choque", "superpuestas"):
        out.append({
            "tipo": "descartar",
            "etiqueta": f"No cortar {comun} (resuelve las {len(us)} de una vez)",
            "aplicar": {"overrides": {str(piezas[comun]["grupo"]): "discard"}},
        })
    out.append({
        "tipo": "escala_mas_fina",
        "etiqueta": "Probar una escala más fina (la placa ocupa menos edificio)",
        "aplicar": None,
    })
    out.append({
        "tipo": "revisar_modelo",
        "etiqueta": "Corregir el modelo 3D en esta zona",
        "aplicar": None,
    })
    return out
