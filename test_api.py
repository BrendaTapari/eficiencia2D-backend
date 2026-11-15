"""
Pruebas DE PUNTA A PUNTA: levantan la API, suben un modelo y tocan botones.

Por qué existen
---------------
Las demás pruebas ejercitan el cálculo llamando a las funciones. Ninguna hacía un pedido
HTTP, y por eso llegó al usuario un problema que ninguna podía ver: cada botón del
instructivo tardaba entre 16 y 43 segundos y devolvía hasta 121 MB, el front cortaba la
espera ("el backend no respondió a tiempo") y el cambio nunca aparecía. El cálculo, medido
suelto, tardaba 7 segundos; nadie había medido el pedido.

Acá se mide el pedido. Las pruebas FALLAN si después de un clic la respuesta tarda más que
el presupuesto o pesa más que el máximo, y comprueban que la memoria entre pedidos (que es
lo que lo hace rápido) no cambia NINGÚN resultado.

No necesitan base de datos: los pedidos por `file_id` no la tocan.
"""

import os
import pickle
import time

os.environ.setdefault("DATABASE_URL", "postgresql://u:p@127.0.0.1:59999/x")

import pytest
from fastapi.testclient import TestClient

from api import sesiones
from api.routes import uploads
from main import app

AQUI = os.path.dirname(os.path.abspath(__file__))
CHICO = os.path.join(AQUI, "demo.obj")
GRANDE = os.path.join(AQUI, "Prueba app.obj")

# Presupuesto después de un clic, sobre el modelo grande (295 piezas a 1:100). Medido:
# 0 s si el efecto de ese botón ya se había previsto, 3 a 5 s si no. El presupuesto es el
# criterio de aceptación acordado, no la medición: deja margen para una máquina más lenta.
SEGUNDOS_POR_CLIC = 8.0
MB_POR_RESPUESTA = 3.0


@pytest.fixture(scope="module")
def cliente():
    # Sin `with`: así no corre el arranque de la aplicación, que es lo que se conecta a
    # la base de datos.
    yield TestClient(app)
    sesiones.olvidar_todo()


def _subir(cliente, ruta):
    with open(ruta, "rb") as fh:
        r = cliente.post(
            "/api/upload",
            files={"file": (os.path.basename(ruta), fh, "application/octet-stream")},
        )
    assert r.status_code == 200, r.text[:300]
    return r.json()["file_id"]


def _base(fid, ruta, escala=100.0):
    return {
        "file_id": fid, "original_filename": os.path.basename(ruta),
        "scale_denom": escala, "min_area_m2": 1.0,
    }


def _verificar(cliente, cuerpo):
    t = time.perf_counter()
    r = cliente.post("/api/verificacion", json=cuerpo)
    dt = time.perf_counter() - t
    assert r.status_code == 200, r.text[:300]
    return r.json(), dt, len(r.content) / 1e6


def _acciones(reporte):
    """Acciones de los nudos "para decidir" que cambian decisiones, sin repetir."""
    vistas, out = set(), []
    for n in reporte["nudos"]:
        if n["quien"] != "decidir":
            continue
        for a in n["acciones"]:
            if a.get("aplicar") and "scale_denom" not in a["aplicar"] and a["id"] not in vistas:
                vistas.add(a["id"])
                out.append(a)
    return out


def _con(cuerpo, accion):
    """Lo que hace el front con `accion.aplicar` (CONTRATO_verificacion.md §Acciones)."""
    nuevo = dict(cuerpo)
    ap = accion["aplicar"]
    if "overrides" in ap:
        nuevo["overrides"] = {**nuevo.get("overrides", {}), **ap["overrides"]}
    if "merges" in ap:
        nuevo["merges"] = nuevo.get("merges", []) + ap["merges"]
    if "wall_wall_decisions_by_pair" in ap:
        pares = [
            d for d in nuevo.get("wall_wall_decisions_by_pair", [])
            if {d["group_a"], d["group_b"]} != {
                ap["wall_wall_decisions_by_pair"][0]["group_a"],
                ap["wall_wall_decisions_by_pair"][0]["group_b"],
            }
        ]
        nuevo["wall_wall_decisions_by_pair"] = pares + ap["wall_wall_decisions_by_pair"]
    return nuevo


def _esperar_efectos(cliente, cuerpo, maximo=600.0):
    t = time.perf_counter()
    while True:
        e = cliente.post("/api/verificacion/efectos", json=cuerpo).json()
        if e["listo"]:
            return e
        assert time.perf_counter() - t < maximo, f"efectos sin terminar: {e['calculadas']}/{e['total']}"
        time.sleep(0.1)


def _en_plancha(nesting_preview):
    """Etiquetas de todas las piezas que /nesting-preview manda a cortar."""
    ids = set()
    for clave in ("wall_nesting", "floor_nesting"):
        for hoja in nesting_preview[clave]["sheets"]:
            ids |= {p["panel"]["id"] for p in hoja["panels"]}
        ids |= {p["id"] for p in nesting_preview[clave].get("unplaced", [])}
    return ids


def _sin_lo_que_cambia_siempre(respuesta):
    return {k: v for k, v in respuesta.items() if k not in ("servidor", "efectos")}


# ---------------------------------------------------------------------------
# El pedido de una acción
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chico(cliente):
    fid = _subir(cliente, CHICO)
    return fid, _base(fid, CHICO)


def test_verificacion_devuelve_solo_lo_que_cambia_con_una_decision(cliente, chico):
    _, base = chico
    j, _, _ = _verificar(cliente, base)
    assert set(j) == {
        "efectos", "servidor", "verificacion", "placements", "panel_id_by_group",
        "absorbed_groups", "warnings", "plate_thickness_m", "plate_joints",
    }
    assert j["verificacion"]["veredicto"] in ("listo", "revisar", "no_arma")
    # Ni caras del modelo ni planchas: eso es de /upload y /nesting-preview.
    assert "faces_packed" not in j and "wall_nesting" not in j


def test_verificacion_coincide_con_nesting_preview(cliente, chico):
    """Los dos pedidos tienen que describir LAS MISMAS piezas: uno es lo que el usuario
    revisa y el otro lo que se corta."""
    _, base = chico
    v, _, _ = _verificar(cliente, base)
    n = cliente.post("/api/nesting-preview", json=base).json()
    assert v["verificacion"] == n["verificacion"]
    assert v["placements"] == n["placements"]
    assert v["absorbed_groups"] == n["absorbed_groups"]
    assert set(v["panel_id_by_group"].values()) <= _en_plancha(n)


def test_cada_accion_tiene_id_y_el_clic_cambia_el_reporte(cliente, chico):
    _, base = chico
    j, _, _ = _verificar(cliente, base)
    acciones = _acciones(j["verificacion"])
    assert acciones, "el modelo de prueba tiene que ofrecer al menos una acción"
    assert all(a["id"] for a in acciones)
    assert len({a["id"] for a in acciones}) == len(acciones)
    cambiaron = 0
    for a in acciones:
        j2, _, _ = _verificar(cliente, _con(base, a))
        cambiaron += j2["verificacion"]["resumen"] != j["verificacion"]["resumen"]
    assert cambiaron, "ninguna acción cambió nada: el botón no hace lo que dice"


def test_la_memoria_entre_pedidos_no_cambia_ningun_resultado(cliente, chico):
    """Lo más delicado de todo esto: si se reusa una pieza que había que recalcular, el
    resultado sale mal y nadie se entera. Cada estado se pide con memoria (después de
    haber pasado por otros estados) y de cero, y tiene que dar EXACTAMENTE lo mismo."""
    fid, base = chico
    j, _, _ = _verificar(cliente, base)
    acciones = _acciones(j["verificacion"])
    estados = [base] + [_con(base, a) for a in acciones]
    if len(acciones) >= 2:
        estados.append(_con(_con(base, acciones[0]), acciones[-1]))
    con_memoria = [_sin_lo_que_cambia_siempre(_verificar(cliente, e)[0]) for e in estados]
    for e, esperado in zip(estados, con_memoria):
        sesiones.olvidar_todo()
        de_cero = _sin_lo_que_cambia_siempre(_verificar(cliente, e)[0])
        assert de_cero == esperado, f"cambia según la memoria: {e}"


def test_las_etiquetas_no_se_corren_al_descartar_o_fusionar(cliente, chico):
    """Si el usuario descarta A14, la que estaba mirando como A15 tiene que seguir
    llamándose A15 — en el visor Y en la plancha."""
    _, base = chico
    sesiones.olvidar_todo()
    j, _, _ = _verificar(cliente, base)
    antes = j["panel_id_by_group"]
    for a in _acciones(j["verificacion"]):
        cuerpo = _con(base, a)
        j2, _, _ = _verificar(cliente, cuerpo)
        despues = j2["panel_id_by_group"]
        movidas = {g: (antes[g], despues[g]) for g in despues if g in antes and antes[g] != despues[g]}
        assert not movidas, f"{a['etiqueta']}: cambiaron de etiqueta {movidas}"
        assert len(set(despues.values())) == len(despues), "dos piezas con la misma etiqueta"
        n = cliente.post("/api/nesting-preview", json=cuerpo).json()
        assert set(despues.values()) <= _en_plancha(n), "la plancha usa otras etiquetas"


def test_el_efecto_de_cada_boton_se_calcula_solo_y_el_clic_es_inmediato(cliente, chico):
    _, base = chico
    sesiones.olvidar_todo()
    j, _, _ = _verificar(cliente, base)
    acciones = _acciones(j["verificacion"])
    e = _esperar_efectos(cliente, base)
    assert set(e["efectos"]) == {a["id"] for a in acciones}
    for a in acciones:
        ef = e["efectos"][a["id"]]
        assert ef["resultado"] in ("mejora", "igual", "empeora"), ef
        assert ef["texto"]
        j2, dt, _ = _verificar(cliente, _con(base, a))
        assert j2["servidor"]["ya_calculada"], "el clic recalculó algo que ya estaba previsto"
        assert dt < 1.0
        # El efecto anunciado es el que después se ve.
        r0, r1 = j["verificacion"]["resumen"], j2["verificacion"]["resumen"]
        assert ef["uniones_no_arma"] == r1["uniones_no_arma"] - r0["uniones_no_arma"]
        assert ef["nudos"] == r1["nudos"] - r0["nudos"]


def test_un_pedido_no_modifica_el_modelo_en_memoria(cliente, chico):
    """El modelo leído se comparte entre pedidos. Si algún paso lo modificara, el pedido
    siguiente trabajaría sobre un modelo distinto del que se subió."""
    fid, base = chico
    sesiones.olvidar_todo()
    modelo = uploads.load_phase1_cache(fid)
    assert uploads.load_phase1_cache(fid) is modelo, "no quedó en memoria"
    antes = pickle.dumps(modelo)
    j, _, _ = _verificar(cliente, base)
    for a in _acciones(j["verificacion"]):
        _verificar(cliente, _con(base, a))
    cliente.post("/api/nesting-preview", json=base)
    cliente.post("/api/recompute", json=base)
    r = cliente.post("/api/generate", json=base)
    assert r.status_code == 200, r.text[:300]
    assert pickle.dumps(modelo) == antes


def test_los_efectos_de_un_modelo_desconocido_no_crean_nada(cliente):
    r = cliente.post("/api/verificacion/efectos", json=_base("no-existe", CHICO))
    assert r.status_code == 200 and r.json()["sin_sesion"] is True


# ---------------------------------------------------------------------------
# Presupuesto de tiempo y de tamaño, sobre el modelo grande
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(GRANDE), reason="falta el modelo grande")
def test_presupuesto_despues_de_un_clic_en_el_modelo_grande(cliente):
    """El criterio de aceptación: después de un clic la respuesta llega en menos de 8 s y
    pesa menos de 3 MB. Se prueba el PEOR caso: clic apenas llega la primera respuesta,
    sin darle tiempo al cálculo en segundo plano a prever nada."""
    sesiones.olvidar_todo()
    fid = _subir(cliente, GRANDE)
    base = _base(fid, GRANDE)
    j, _, mb = _verificar(cliente, base)
    assert mb < MB_POR_RESPUESTA, f"primera respuesta: {mb:.1f} MB"
    acciones = _acciones(j["verificacion"])
    assert len(acciones) >= 4
    # Un descarte y una fusión, de nudos distintos, encadenados.
    descartar = next(a for a in acciones if a["tipo"] == "descartar")
    fusionar = next(a for a in reversed(acciones) if a["tipo"] == "fusionar")
    estado = base
    for a in (descartar, fusionar):
        estado = _con(estado, a)
        j2, dt, mb = _verificar(cliente, estado)
        print(f"clic {a['tipo']}: {dt:.1f} s, servidor {j2['servidor']}")
        assert dt < SEGUNDOS_POR_CLIC, f"{a['etiqueta']}: {dt:.1f} s ({j2['servidor']})"
        assert mb < MB_POR_RESPUESTA, f"{a['etiqueta']}: {mb:.1f} MB"
        assert j2["verificacion"]["resumen"] != j["verificacion"]["resumen"]
    # Volver atrás es inmediato.
    _, dt, _ = _verificar(cliente, base)
    assert dt < 1.0, f"volver al estado anterior: {dt:.1f} s"
    # Y el resultado con memoria es el mismo que de cero.
    con_memoria = _sin_lo_que_cambia_siempre(j2)
    sesiones.olvidar_todo()
    de_cero = _sin_lo_que_cambia_siempre(_verificar(cliente, estado)[0])
    assert de_cero == con_memoria
