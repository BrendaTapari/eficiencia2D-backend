"""
Reporte de verificación: lo que una persona necesita para decidir si manda a cortar.

Por qué existe
--------------
`verificacion_uniones` dice qué pasa en cada encuentro entre dos piezas. Es la verdad, pero
no es legible: una maqueta de 300 piezas tiene más de 600 uniones. Quien la va a armar no
puede revisar 600 renglones, y menos con una entrega encima.

Este módulo convierte esa lista en algo que se puede mirar:

1. **Un veredicto** que cuenta sólo lo que impide armar. Las uniones que se tocan de canto
   son un tema de TERMINACIÓN: van en una lista aparte y no ensucian el titular. Antes
   contaban como "para revisar" y eran más de la mitad del amarillo (81 de 142 uniones con
   aviso en un modelo real), con lo que el titular decía "144 piezas para revisar" cuando
   las que de verdad no armaban eran muchas menos.
2. **Nudos, no piezas.** Los problemas se agrupan por LUGAR: una esquina, una T, un apoyo.
   Es la unidad en la que se arregla algo —cambiar quién cede mueve un solo extremo de dos
   piezas— y son varias veces menos que las piezas afectadas.
3. **Quién lo puede arreglar.** Cada problema dice si es una decisión de quien arma, un
   límite de la escala o algo a corregir en el modelo 3D. No se ofrece como arreglo lo que
   el resolvedor ya probó y no sirvió.
4. **Un detalle en planta de cada nudo**: las placas cortadas por ese lugar, con su
   espesor real, y lo que se pisa marcado. Es el dibujo que un arquitecto ya sabe leer.

Todas las medidas van en milímetros de PLANCHA —lo que se va a ver en la mano— y además en
centímetros de EDIFICIO, que es la unidad en la que piensa quien corrige el modelo 3D.

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
    _placas,
)

VERSION = 2

_ORDEN_ESTADO = {"no_arma": 0, "revisar": 1, "ok": 2}
# Con cuántas uniones del mismo tipo contra una misma pieza se arma un grupo.
MINIMO_PARA_AGRUPAR = 3
# Dos uniones con problema están en el mismo NUDO si comparten una pieza y están a menos
# de esto, en espesores de placa.
RADIO_DE_NUDO_PLACAS = 2.5
# Medio lado de la ventana del detalle en planta, en espesores de placa.
VENTANA_DETALLE_PLACAS = 5.0
# Escalas que se le pueden sugerir a alguien (denominadores usuales en arquitectura).
ESCALAS_USUALES = (10, 20, 25, 50, 75, 100, 125, 150, 200, 250, 500)

_ORDEN_QUIEN = {"decidir": 0, "escala": 1, "modelo": 2}


def _peor(a: str, b: str) -> str:
    return a if _ORDEN_ESTADO[a] <= _ORDEN_ESTADO[b] else b


def _mm(x: float) -> float:
    return round(x, 2)


def _punto(p) -> Dict[str, float]:
    return {"x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "z": round(float(p[2]), 4)}


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
        "otra, así que queda el canto a la vista o medio apoyo en el aire. Arma igual."
    ),
    "superpuestas": (
        "Son dos piezas paralelas que en el modelo están más cerca que el espesor de la "
        "placa: físicamente no hay lugar para las dos."
    ),
    "pieza_fina": (
        "La pieza mide menos que el espesor de la placa: no se puede armar parada."
    ),
}

_QUIEN_TEXTO = {
    "decidir": "Elegí vos: hay más de una forma de resolverlo.",
    "escala": "Es de la escala: con otra escala la placa representa otro espesor.",
    "modelo": (
        "No encontramos una forma automática de cerrarlo. Suele venir del modelo 3D: "
        "conviene revisarlo en esta zona."
    ),
}


def _escala_usual_hasta(maximo: float, actual: float) -> Optional[int]:
    """La escala usual más gruesa que no pasa de `maximo` y es más fina que la actual."""
    ok = [e for e in ESCALAS_USUALES if e <= maximo + 1e-9 and e < actual - 1e-9]
    return max(ok) if ok else None


def _escala_usual_desde(minimo: float, actual: float) -> Optional[int]:
    """La escala usual más fina que no baja de `minimo` y es más gruesa que la actual."""
    ok = [e for e in ESCALAS_USUALES if e >= minimo - 1e-9 and e > actual + 1e-9]
    return min(ok) if ok else None


def _clasificar(
    u: Union,
    escala: float,
    fijas: set,
    probadas: set,
    cede_por_par: Dict[Tuple[int, int], int],
    espesor_modelo: Dict[int, float],
    ambas_muro: bool,
) -> Tuple[str, Optional[int]]:
    """`(quién lo arregla, escala sugerida)` para una unión con problema.

    - "decidir": hay una acción que el usuario puede tomar y que cambia el resultado.
    - "escala":  el defecto sale de cuánto edificio representa la placa a esta escala.
    - "modelo":  ni lo uno ni lo otro; lo que queda es revisar el modelo 3D.
    """
    k = pair_key(u.grupo_a, u.grupo_b)
    if u.tipo == "superpuestas":
        # Dejan de pisarse cuando la placa representa menos que la separación real.
        separacion_mm = max(0.0, PLATE_THICKNESS_M * 1000.0 - u.medida_mm)
        tope = escala * separacion_mm / (PLATE_THICKNESS_M * 1000.0)
        return "decidir", _escala_usual_hasta(tope, escala)

    invertible = ambas_muro and k in cede_por_par
    if invertible and k in fijas:
        return "decidir", None      # la decidió el usuario: puede volver a decidirla
    if u.tipo == "luz":
        # Un muro más grueso que la placa deja luz por construcción: la pieza vecina
        # llega hasta la cara del muro del modelo y la placa, centrada, no llega ahí.
        placa = PLATE_THICKNESS_M * escala
        grueso = max(espesor_modelo.get(u.grupo_a, 0.0), espesor_modelo.get(u.grupo_b, 0.0))
        if grueso > placa * 1.25:
            return "escala", _escala_usual_desde(grueso / PLATE_THICKNESS_M * 0.8, escala)
    if invertible and k not in probadas:
        return "decidir", None
    return "modelo", None


def _acciones(
    u: Union,
    area_por_pieza: Dict[str, float],
    cede_por_par: Dict[Tuple[int, int], int],
    categoria: Dict[int, str],
    probadas: set,
    escala_sugerida: Optional[int],
) -> List[Dict]:
    """Qué se puede hacer con esta unión, ya como el cambio a mandar al backend.

    Cada acción trae `aplicar`, un objeto con las MISMAS claves que aceptan `/recompute`,
    `/nesting-preview` y `/generate`. El front lo fusiona con su estado y vuelve a pedir;
    no tiene que interpretar nada.

    No se ofrece invertir quién cede en una unión donde el resolvedor ya lo probó sin
    mejora: sería mandar a la persona a intentar algo que se sabe que no sirve.
    """
    out: List[Dict] = []
    chica = (
        u.pieza_a
        if area_por_pieza.get(u.pieza_a, 0.0) <= area_por_pieza.get(u.pieza_b, 0.0)
        else u.pieza_b
    )
    gid_chica = u.grupo_a if chica == u.pieza_a else u.grupo_b
    ambas_muro = categoria.get(u.grupo_a) == "wall" and categoria.get(u.grupo_b) == "wall"
    k = pair_key(u.grupo_a, u.grupo_b)

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
    else:
        cede = cede_por_par.get(k)
        if ambas_muro and cede is not None and k not in probadas:
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
        "tipo": "cambiar_escala",
        "etiqueta": (
            f"Probar a escala 1:{escala_sugerida}" if escala_sugerida
            else "Probar otra escala"
        ),
        "aplicar": {"scale_denom": escala_sugerida} if escala_sugerida else None,
    })
    out.append({
        "tipo": "revisar_modelo",
        "etiqueta": "Corregir el modelo 3D en esta zona",
        "aplicar": None,
    })
    return out


def _acciones_de_grupo(
    tipo: str, comun: Optional[str], us: List[Union], piezas: Dict,
    escala_sugerida: Optional[int],
) -> List[Dict]:
    """Acciones que valen para TODO un grupo de uniones contra una pieza común."""
    out: List[Dict] = []
    if comun is not None and comun in piezas and tipo in ("choque", "superpuestas"):
        out.append({
            "tipo": "descartar",
            "etiqueta": f"No cortar {comun} (resuelve las {len(us)} de una vez)",
            "aplicar": {"overrides": {str(piezas[comun]["grupo"]): "discard"}},
        })
    out.append({
        "tipo": "cambiar_escala",
        "etiqueta": (
            f"Probar a escala 1:{escala_sugerida}" if escala_sugerida
            else "Probar otra escala"
        ),
        "aplicar": {"scale_denom": escala_sugerida} if escala_sugerida else None,
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


# ---------------------------------------------------------------------------
# Detalle en planta de un nudo
# ---------------------------------------------------------------------------


def _detalle(punto, tau, placas, estado_de: Dict[str, str], espesor: float) -> Optional[Dict]:
    """Corte de las placas por un plano perpendicular a la unión, alrededor de `punto`.

    Si la unión es vertical (dos muros) el corte es una PLANTA; si es horizontal (un muro
    contra una losa) es un CORTE en altura. Cada placa que pasa por ahí sale como el
    rectángulo de su sección, con su espesor real, en coordenadas 2D centradas en el nudo.
    Lo que dos placas comparten se devuelve aparte: es lo que hay que pintar de rojo.
    """
    import numpy as np
    from shapely.geometry import LineString, Polygon, box

    tau = np.asarray(tau, dtype=float)
    n = float(np.linalg.norm(tau))
    if n < 1e-9:
        return None
    tau = tau / n
    p0 = np.asarray(punto, dtype=float)
    if abs(tau[1]) > 0.7:
        vista = "planta"
        a = np.array([1.0, 0.0, 0.0])
    else:
        vista = "corte"
        a = np.cross(np.array([0.0, 1.0, 0.0]), tau)
    a = a - (a @ tau) * tau
    na = float(np.linalg.norm(a))
    if na < 1e-9:
        return None
    a = a / na
    b = np.cross(tau, a)
    if vista == "corte" and b[1] < 0:
        b = -b                      # arriba en el dibujo es arriba en el edificio
    if vista == "planta" and b[2] < 0:
        b = -b

    R = VENTANA_DETALLE_PLACAS * espesor
    ventana = box(-R, -R, R, R)
    h = espesor / 2.0
    secciones: List[Tuple[str, int, object]] = []
    for pl in placas:
        if abs(float(pl.n @ tau)) > 0.2:
            continue                # la placa no contiene la dirección de la unión
        # Descarte barato: el plano de la placa pasa lejos del nudo.
        if abs(float(pl.n @ p0) - pl.d) > R + h:
            continue
        ku, kv = float(pl.u @ tau), float(pl.v @ tau)
        c = float((p0 - pl.o) @ tau)
        if abs(ku) < 1e-9 and abs(kv) < 1e-9:
            continue
        # Recta del plano de corte dentro del marco de la placa: ku·x + kv·y = c.
        minx, miny, maxx, maxy = pl.poly.bounds
        if abs(kv) >= abs(ku):
            recta = LineString([
                (minx - 1.0, (c - ku * (minx - 1.0)) / kv),
                (maxx + 1.0, (c - ku * (maxx + 1.0)) / kv),
            ])
        else:
            recta = LineString([
                ((c - kv * (miny - 1.0)) / ku, miny - 1.0),
                ((c - kv * (maxy + 1.0)) / ku, maxy + 1.0),
            ])
        try:
            corte = recta.intersection(pl.poly)
        except Exception:
            continue
        n2 = np.array([float(pl.n @ a), float(pl.n @ b)])
        ln = float(np.linalg.norm(n2))
        if ln < 1e-9:
            continue
        n2 = n2 / ln
        for g in getattr(corte, "geoms", [corte]):
            cs = list(getattr(g, "coords", []))
            if len(cs) < 2:
                continue
            ext = []
            for (x, y) in (cs[0], cs[-1]):
                w = pl.o + pl.u * x + pl.v * y - p0
                ext.append(np.array([float(w @ a), float(w @ b)]))
            if float(np.linalg.norm(ext[1] - ext[0])) < 1e-9:
                continue
            rect = Polygon([
                tuple(ext[0] + n2 * h), tuple(ext[1] + n2 * h),
                tuple(ext[1] - n2 * h), tuple(ext[0] - n2 * h),
            ])
            try:
                rect = rect.intersection(ventana)
            except Exception:
                continue
            if rect.is_empty or rect.geom_type != "Polygon" or rect.area <= 1e-12:
                continue
            secciones.append((pl.id, pl.gid, rect))
    if not secciones:
        return None

    def coords(poly) -> List[List[float]]:
        return [[round(x, 4), round(y, 4)] for x, y in list(poly.exterior.coords)[:-1]]

    solapes: List[Dict] = []
    minimo = (0.05 * espesor) ** 2
    for i in range(len(secciones)):
        for j in range(i + 1, len(secciones)):
            if secciones[i][0] == secciones[j][0]:
                continue
            try:
                inter = secciones[i][2].intersection(secciones[j][2])
            except Exception:
                continue
            for g in getattr(inter, "geoms", [inter]):
                if g.geom_type == "Polygon" and g.area > minimo:
                    solapes.append({
                        "piezas": [secciones[i][0], secciones[j][0]],
                        "poligono": coords(g),
                    })

    return {
        "vista": vista,
        "ventana_m": round(R, 4),
        "espesor_m": round(espesor, 4),
        "ejes": {
            "a": {"x": round(float(a[0]), 4), "y": round(float(a[1]), 4), "z": round(float(a[2]), 4)},
            "b": {"x": round(float(b[0]), 4), "y": round(float(b[1]), 4), "z": round(float(b[2]), 4)},
        },
        "placas": [
            {
                "pieza": pid, "grupo": gid,
                "estado": estado_de.get(pid, "ok"),
                "poligono": coords(rect),
            }
            for pid, gid, rect in secciones
        ],
        "solapes": solapes,
    }


def _direccion(u: Union, placa_de: Dict[str, object]):
    """Dirección de la unión: a lo largo de su segmento, o vertical/horizontal según cómo
    estén paradas las placas cuando el segmento es un punto (placas paralelas)."""
    import numpy as np

    e0 = np.asarray(u.segmento[0], dtype=float)
    e1 = np.asarray(u.segmento[1], dtype=float)
    d = e1 - e0
    if float(np.linalg.norm(d)) > 1e-9:
        return d
    pl = placa_de.get(u.pieza_a)
    if pl is None or abs(float(pl.n[1])) < 0.5:
        return np.array([0.0, 1.0, 0.0])        # placas verticales: se mira en planta
    return pl.u                                  # placas horizontales: corte en altura


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def construir_reporte(
    panels: List[Panel],
    uniones: List[Union],
    scale_denom: float,
    cede_por_par: Optional[Dict[Tuple[int, int], int]] = None,
    categoria_por_grupo: Optional[Dict[int, str]] = None,
    resolucion: Optional[Dict] = None,
    pares_decididos_por_el_usuario: Optional[set] = None,
    espesor_modelo: Optional[Dict[int, float]] = None,
) -> Dict:
    """Reporte serializable (JSON) de la verificación. Ver `CONTRATO_verificacion.md`."""
    escala = scale_denom or 1.0
    cede_por_par = cede_por_par or {}
    categoria = dict(categoria_por_grupo or {})
    espesor_modelo = espesor_modelo or {}
    fijas = set(pares_decididos_por_el_usuario or ())
    probadas = {
        pair_key(a, b) for a, b in (resolucion or {}).get("probadas_sin_mejora", [])
    }
    espesor_mm = PLATE_THICKNESS_M * 1000.0
    espesor = PLATE_THICKNESS_M * escala
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
            "terminaciones": 0,
        }
        area_por_pieza[p.id] = float((p.frame or {}).get("area_m2") or p.width_m * p.height_m)
        categoria.setdefault(p.source_group_id, p.category)

    # --- uniones -----------------------------------------------------------------
    # Tocarse de canto es un tema de terminación, no de armado: en el reporte esa unión
    # vale "ok" y lleva `terminacion: true`.
    def es_terminacion(u: Union) -> bool:
        return u.tipo == "sin_cubrir"

    def estado_de_union(u: Union) -> str:
        return "ok" if es_terminacion(u) else u.estado

    clasif: Dict[int, Tuple[str, Optional[int]]] = {}
    filas: List[Dict] = []
    for i, u in enumerate(uniones):
        estado = estado_de_union(u)
        quien = None
        if estado != "ok":
            ambas_muro = (
                categoria.get(u.grupo_a) == "wall" and categoria.get(u.grupo_b) == "wall"
            )
            clasif[i] = _clasificar(
                u, escala, fijas, probadas, cede_por_par, espesor_modelo, ambas_muro
            )
            quien = clasif[i][0]
        filas.append({
            "id": i,
            "pieza_a": u.pieza_a,
            "pieza_b": u.pieza_b,
            "grupo_a": u.grupo_a,
            "grupo_b": u.grupo_b,
            "tipo": u.tipo,
            "estado": estado,
            "terminacion": es_terminacion(u),
            "quien": quien,
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
                piezas[pid]["estado"] = _peor(piezas[pid]["estado"], estado)
                if estado != "ok":
                    piezas[pid]["problemas"] += 1
                elif es_terminacion(u):
                    piezas[pid]["terminaciones"] += 1

    # --- agrupar por causa -----------------------------------------------------------
    def entrada(tipo: str, indices: List[int], comun: Optional[str]) -> Dict:
        us = [uniones[i] for i in indices]
        peor = max(us, key=lambda x: x.medida_mm)
        i_peor = indices[us.index(peor)]
        estado = "ok"
        for x in us:
            estado = _peor(estado, estado_de_union(x))
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
        quienes = [clasif[i][0] for i in indices if i in clasif]
        quien = min(quienes, key=lambda q: _ORDEN_QUIEN[q]) if quienes else None
        sugerida = clasif.get(i_peor, (None, None))[1]
        return {
            "tipo": tipo,
            "estado": estado,
            "quien": quien,
            "quien_texto": _QUIEN_TEXTO.get(quien) if quien else None,
            "escala_sugerida": sugerida,
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
            "acciones": (
                _acciones(peor, area_por_pieza, cede_por_par, categoria, probadas, sugerida)
                if len(us) == 1
                else _acciones_de_grupo(tipo, comun, us, piezas, sugerida)
            ),
        }

    def agrupar(indices: List[int]) -> List[Dict]:
        out: List[Dict] = []
        por_tipo: Dict[str, List[int]] = defaultdict(list)
        for i in indices:
            por_tipo[uniones[i].tipo].append(i)
        for tipo, idx in por_tipo.items():
            restantes = set(idx)
            while restantes:
                cuenta: Dict[str, int] = defaultdict(int)
                for i in restantes:
                    cuenta[uniones[i].pieza_a] += 1
                    cuenta[uniones[i].pieza_b] += 1
                comun, veces = max(cuenta.items(), key=lambda kv: (kv[1], kv[0]))
                if veces < MINIMO_PARA_AGRUPAR:
                    break
                grupo = sorted(
                    i for i in restantes
                    if comun in (uniones[i].pieza_a, uniones[i].pieza_b)
                )
                out.append(entrada(tipo, grupo, comun))
                restantes -= set(grupo)
            for i in sorted(restantes):
                out.append(entrada(tipo, [i], None))
        return out

    con_problema = [i for i, u in enumerate(uniones) if estado_de_union(u) != "ok"]
    de_terminacion = [i for i, u in enumerate(uniones) if es_terminacion(u)]
    problemas = agrupar(con_problema)
    terminaciones = agrupar(de_terminacion)

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
        sugerida = _escala_usual_hasta(chico / PLATE_THICKNESS_M, escala)
        problemas.append({
            "tipo": "pieza_fina",
            "estado": "revisar",
            "quien": "escala" if sugerida else "modelo",
            "quien_texto": _QUIEN_TEXTO["escala" if sugerida else "modelo"],
            "escala_sugerida": sugerida,
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
                    "tipo": "cambiar_escala",
                    "etiqueta": (
                        f"Probar a escala 1:{sugerida}" if sugerida else "Probar otra escala"
                    ),
                    "aplicar": {"scale_denom": sugerida} if sugerida else None,
                },
            ],
        })

    problemas.sort(key=lambda x: (_ORDEN_ESTADO[x["estado"]], -x["cantidad"], -x["peor_mm"]))
    for i, pr in enumerate(problemas):
        pr["id"] = i
    terminaciones.sort(key=lambda x: (-x["cantidad"], -x["peor_mm"]))
    for i, pr in enumerate(terminaciones):
        pr["id"] = i

    # --- nudos: los problemas por LUGAR ---------------------------------------------
    nudos = _nudos(
        panels, uniones, con_problema, de_terminacion, problemas, piezas,
        clasif, espesor, a_cm_edificio, estado_de_union,
    )

    def contar(estado: str, donde) -> int:
        return sum(1 for x in donde if x["estado"] == estado)

    veredicto = "listo"
    if any(pr["estado"] == "no_arma" for pr in problemas):
        veredicto = "no_arma"
    elif problemas:
        veredicto = "revisar"

    invertidas = list((resolucion or {}).get("invertidas", []))
    por_quien = {q: sum(1 for n in nudos if n["quien"] == q) for q in _ORDEN_QUIEN}

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
            "nudos": len(nudos),
            "nudos_por_quien": por_quien,
            "terminaciones": len(de_terminacion),
            "resueltas_solas": len(invertidas),
        },
        "piezas": piezas,
        "nudos": nudos,
        "problemas": problemas,
        "terminaciones": terminaciones,
        "uniones": filas,
        # Decisiones de quién cede que tomó el resolvedor en lugar de la sugerida. Son
        # las que se cortan: el selector de "quién cede" tiene que mostrar ÉSTAS.
        "decisiones_automaticas": invertidas,
    }


def _nudos(
    panels, uniones, con_problema, de_terminacion, problemas, piezas,
    clasif, espesor, a_cm_edificio, estado_de_union,
) -> List[Dict]:
    """Los problemas agrupados por lugar, cada uno con su detalle en planta."""
    import numpy as np

    if not con_problema:
        return []
    placas = _placas(panels)
    placa_de = {pl.id: pl for pl in placas}
    radio = RADIO_DE_NUDO_PLACAS * espesor
    pts = {i: np.asarray(uniones[i].punto, dtype=float) for i in con_problema + de_terminacion}

    padre = {i: i for i in con_problema}

    def raiz(i: int) -> int:
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for x in range(len(con_problema)):
        i = con_problema[x]
        pi = {uniones[i].pieza_a, uniones[i].pieza_b}
        for y in range(x + 1, len(con_problema)):
            j = con_problema[y]
            if not pi & {uniones[j].pieza_a, uniones[j].pieza_b}:
                continue
            if float(np.linalg.norm(pts[i] - pts[j])) <= radio:
                padre[raiz(i)] = raiz(j)

    grupos: Dict[int, List[int]] = defaultdict(list)
    for i in con_problema:
        grupos[raiz(i)].append(i)

    problema_de_union: Dict[int, int] = {}
    for pr in problemas:
        for i in pr["uniones"]:
            problema_de_union[i] = pr["id"]
    acciones_de: Dict[int, List[Dict]] = {pr["id"]: pr["acciones"] for pr in problemas}

    out: List[Dict] = []
    for indices in grupos.values():
        us = [uniones[i] for i in indices]
        peor_i = max(indices, key=lambda i: (-_ORDEN_ESTADO[estado_de_union(uniones[i])], uniones[i].medida_mm))
        peor = uniones[peor_i]
        estado = "ok"
        for u in us:
            estado = _peor(estado, estado_de_union(u))
        nombres: List[str] = []
        for u in us:
            for pid in (u.pieza_a, u.pieza_b):
                if pid not in nombres:
                    nombres.append(pid)
        centro = np.mean([pts[i] for i in indices], axis=0)
        # Terminaciones del mismo lugar: contexto, no problema.
        cerca = [
            i for i in de_terminacion
            if {uniones[i].pieza_a, uniones[i].pieza_b} & set(nombres)
            and float(np.linalg.norm(pts[i] - centro)) <= radio
        ]
        quienes = [clasif[i][0] for i in indices if i in clasif]
        quien = min(quienes, key=lambda q: _ORDEN_QUIEN[q]) if quienes else "modelo"
        tipos = sorted({u.tipo for u in us})
        estado_pieza = {n: piezas[n]["estado"] for n in nombres if n in piezas}
        detalle = _detalle(
            centro, _direccion(peor, placa_de), placas,
            {pl.id: (estado_pieza.get(pl.id, "contexto")) for pl in placas},
            espesor,
        )
        # Acciones: las de la unión peor si es un problema suelto; si pertenece a un
        # grupo por causa, las de ese grupo.
        pid = problema_de_union.get(peor_i)
        acciones = acciones_de.get(pid, []) if pid is not None else []
        out.append({
            "estado": estado,
            "quien": quien,
            "quien_texto": _QUIEN_TEXTO[quien],
            "titulo": (
                " · ".join(nombres[:4]) + ("…" if len(nombres) > 4 else "")
                + f": {_VERBO[peor.tipo]} ({peor.medida_mm:.1f} mm)"
            ),
            "tipos": tipos,
            "peor_mm": _mm(peor.medida_mm),
            "peor_cm_edificio": round(peor.medida_mm * a_cm_edificio, 1),
            "piezas": nombres,
            "punto": _punto(centro),
            "uniones": sorted(indices),
            "terminaciones": sorted(cerca),
            "problemas": sorted({problema_de_union[i] for i in indices if i in problema_de_union}),
            "acciones": acciones,
            "detalle": detalle,
        })

    out.sort(key=lambda n: (
        _ORDEN_ESTADO[n["estado"]], _ORDEN_QUIEN[n["quien"]], -n["peor_mm"],
    ))
    for i, n in enumerate(out):
        n["id"] = i
    return out
