"""
Una placa por lugar: grupos paralelos que, A LA ESCALA DE CORTE, son la misma pieza.

Por qué existe
--------------
El modelo describe un edificio y la maqueta se arma con placas de 3 mm. Entre las dos
cosas hay una conversión que depende de la escala y que nadie hacía: 3 mm de MDF son 15 cm
de edificio a 1:50 y 30 cm a 1:100. Dos superficies paralelas del modelo que están más
cerca que eso —las dos pieles de un muro de 20 cm, una hoja de puerta apoyada contra su
tabique, un revestimiento— NO son dos piezas: son la misma placa, porque físicamente no
hay lugar para dos.

El clasificador ya une "gemelas finas" (`are_thin_twins`), pero con un criterio del MODELO
(centroides, razón de tamaños, 40 cm fijos) y sin conocer la escala, así que se le escapan
todas las que no se parecen lo suficiente. Medido antes de este módulo, sobre los dos
modelos del repo y a 1:50 y 1:100: de 120 choques entre piezas, 104 eran entre placas
PARALELAS, o sea esto. El verificador los rotulaba "no_detectada" —"el sistema no vio la
unión"— pero no hay ninguna unión que ver: no son dos piezas que se encuentran, es una
pieza contada dos veces.

Criterio
--------
Dos grupos de la misma categoría se funden si, y sólo si:

1. son paralelos (`COS_PARALELAS`);
2. sus planos medios están a menos de MEDIA placa. Es exactamente la condición con la que
   `assembly_verify` declara un choque (el plano medio de una cae dentro del volumen de la
   otra), así que este módulo y el verificador no pueden discrepar sobre qué es un solape;
3. sus contornos, vistos en el mismo plano, se pisan en una fracción apreciable de la más
   chica (`FRACCION_MINIMA`). Dos tramos coplanares que sólo se TOCAN de canto no entran:
   son dos piezas legítimas, una al lado de la otra.

No se decide nada más acá. La fusión en sí la hace `apply_merges`, que ya existía para las
fusiones manuales del usuario y trae sus guardas (no fundir a través de una losa, no fundir
planos distintos) y recalcula juntas y ajustes sobre la topología nueva. Por eso esto corre
ANTES de resolver encastres: los vecinos se resuelven contra la placa que de verdad se va a
cortar, y no contra una piel que después desaparece.

Qué NO hace
-----------
No modifica el modelo ni la clasificación que ve el usuario: es una decisión de
FABRICACIÓN, vale para una escala y se recalcula en cada una. A 1:20 la placa son 6 cm y
las dos pieles de un muro de 20 cm vuelven a ser dos piezas, que es lo correcto.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from core.group_classifier import GeometryGroup
from core.services.plate_intersect import PLATE_THICKNESS_M
from core.services.recorte_csg import (
    Placa,
    construir_placa,
    distancia_entre_planos,
    proyeccion_paralela,
)
from core.services.types import Face3D

# Fracción del área de la pieza MÁS CHICA que tiene que quedar tapada por la otra para
# considerarlas la misma placa. Por debajo de esto es un solape marginal (dos muros
# desfasados que se montan en una punta) y fundirlos inventaría una pieza que no existe;
# ese caso se deja al verificador, que lo informa como choque.
FRACCION_MINIMA = 0.2

# Segundo criterio, que NO depende de la escala: las dos PIELES de un mismo muro macizo.
# Espalda con espalda, a no más que el espesor de un muro, y la más chica tapada en su
# mayor parte por la más grande.
#
# El clasificador ya une gemelas (`are_thin_twins`), pero compara CENTROIDES: exige que
# estén alineados dentro de medio largo de la más chica. Un muro largo cuya piel interior
# el modelo trae partida en tramos —uno por ambiente, cortada donde llega cada tabique—
# tiene cada tramo con el centroide lejos del centro del muro, y no lo empareja. Cada tramo
# quedaba como una pieza propia montada media placa sobre el muro, y —peor— los tabiques
# registraban su junta contra ESE tramo y no contra la placa que de verdad se corta, así
# que el encuentro en T con el muro figuraba como "no detectado". Acá se mira el solape
# real de los contornos, que es lo que define que sean el mismo muro.
ESPESOR_MAX_MURO_M = 0.40      # igual que THIN_WALL_THRESHOLD del clasificador
FRACCION_MISMA_PARED = 0.5

# Margen bajo la media placa, en metros de PLANCHA. Igual a la tolerancia del verificador:
# una separación que él no contaría como choque tampoco se funde acá.
TOL_PLANCHA_M = 0.0002


def _espalda_con_espalda(a: GeometryGroup, b: GeometryGroup) -> bool:
    """Las normales del MODELO miran en sentidos opuestos y cada superficie queda DETRÁS
    de la otra: son las dos caras de un sólido, no dos paredes enfrentadas a través de un
    pasillo angosto (ésas se miran de frente)."""
    na, nb = a.representative_normal, b.representative_normal
    if na.x * nb.x + na.y * nb.y + na.z * nb.z > -0.9:
        return False
    dx = b.centroid.x - a.centroid.x
    dy = b.centroid.y - a.centroid.y
    dz = b.centroid.z - a.centroid.z
    return (dx * na.x + dy * na.y + dz * na.z) < 0 and (dx * nb.x + dy * nb.y + dz * nb.z) > 0


def detectar_placas_superpuestas(
    groups: List[GeometryGroup],
    faces: List[Face3D],
    scale_denom: float,
    categoria: Callable[[GeometryGroup], str],
    normales: Optional[Dict[int, object]] = None,
) -> List[List[int]]:
    """Conjuntos de ids de grupo que a esta escala ocupan la misma placa.

    Devuelve listas listas para pasarle a `apply_merges`. Vacío si no hay ninguna.
    """
    escala = scale_denom or 1.0
    limite = max(0.0, (PLATE_THICKNESS_M / 2.0 - TOL_PLANCHA_M) * escala)

    placas: List[Tuple[GeometryGroup, str, Placa]] = []
    for g in groups:
        cat = categoria(g)
        if cat == "discard":
            continue
        pl = construir_placa(
            g, faces, (normales or {}).get(g.id, g.representative_normal)
        )
        if pl is not None and pl.poly.area > 1e-9:
            placas.append((g, cat, pl))

    padre = {g.id: g.id for g, _c, _p in placas}

    def raiz(x: int) -> int:
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    hubo = False
    for i in range(len(placas)):
        gi, ci, pi = placas[i]
        for j in range(i + 1, len(placas)):
            gj, cj, pj = placas[j]
            if ci != cj or raiz(gi.id) == raiz(gj.id):
                continue
            sep = distancia_entre_planos(pi, pj)
            if sep is None:
                continue
            if sep < limite:
                fraccion = FRACCION_MINIMA
            elif sep <= ESPESOR_MAX_MURO_M and _espalda_con_espalda(gi, gj):
                fraccion = FRACCION_MISMA_PARED
            else:
                continue
            # PENDIENTE, medido: cuando una de las dos pieles NO quedó partida al nivel
            # de un entrepiso (la losa no llega a su plano) y la otra sí, fundirlas da una
            # pieza que cruza la losa sin ranura y pisa al tramo siguiente: 7 de los 13
            # choques que quedan en `Prueba app.obj` a 1:100. Se probó NO fundir ese par y
            # es peor (13 -> 16 choques, 11 -> 14 a 1:50): quedan las dos pieles montadas.
            # El arreglo de fondo es partir esa piel en `mesh_splitter`, no acá.
            otra = proyeccion_paralela(pi, pj)
            if otra is None:
                continue
            try:
                comun = pi.poly.intersection(otra).area
            except Exception:
                continue
            if comun >= fraccion * min(pi.poly.area, pj.poly.area):
                padre[raiz(gi.id)] = raiz(gj.id)
                hubo = True

    if not hubo:
        return []
    conjuntos: Dict[int, List[int]] = {}
    for g, _c, _p in placas:
        conjuntos.setdefault(raiz(g.id), []).append(g.id)
    return [sorted(ids) for ids in conjuntos.values() if len(ids) > 1]
