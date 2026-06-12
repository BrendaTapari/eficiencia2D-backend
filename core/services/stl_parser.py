import time
import logging
from typing import Any, Dict

import numpy as np
import trimesh

from core.services.types import IndexedFace3D, Vec3

logger = logging.getLogger("eficiencia2d.pipeline")


def _guess_unit_scale(vertices: np.ndarray) -> float:
    """
    Heurística de unidades → metros (igual que el guessUnitScale del front).
    STL no guarda unidades y suele venir en mm; el pipeline trabaja en metros
    (umbrales como min_real_area=1.0 m², grosores de 0.40m, etc.), así que un
    modelo en mm rompería la clasificación si no se escala.
    """
    if len(vertices) == 0:
        return 1.0
    span = float((vertices.max(axis=0) - vertices.min(axis=0)).max())
    if span <= 0:
        return 1.0
    if span <= 100:
        return 1.0          # ya en metros
    if span <= 1000:
        return 0.01         # centímetros → metros
    if span <= 50000:
        return 0.001        # milímetros → metros
    return 20.0 / span      # desconocido muy grande → normalizar a ~20m


def parse_stl(file_path: str) -> Dict[str, Any]:
    """
    Parser STL (ASCII o binario) vía trimesh. Devuelve la MISMA estructura que
    parse_obj: lista de IndexedFace3D con vertices, normal, inner_loops=[] y
    vertex_indices (pool de vértices soldado por trimesh → conectividad exacta).
    """
    t0 = time.perf_counter()
    warnings: list = []

    # process=True suelda vértices duplicados y elimina caras degeneradas.
    loaded = trimesh.load(file_path, file_type="stl", process=True)
    # Algunos STL con múltiples sólidos cargan como Scene; los unificamos.
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)

    faces_arr = getattr(loaded, "faces", None)
    if faces_arr is None or len(faces_arr) == 0:
        return {"faces": [], "warnings": ["STL sin geometría válida (0 caras)."]}

    V = np.asarray(loaded.vertices, dtype=float)
    F = np.asarray(loaded.faces)
    N = np.asarray(loaded.face_normals, dtype=float)

    scale = _guess_unit_scale(V)
    if scale != 1.0:
        V = V * scale
        warnings.append(f"STL escalado x{scale:g} (heurística de unidades a metros).")

    faces = []
    for i in range(len(F)):
        a, b, c = int(F[i, 0]), int(F[i, 1]), int(F[i, 2])
        va, vb, vc = V[a], V[b], V[c]
        n = N[i]
        faces.append(
            IndexedFace3D(
                vertices=[
                    Vec3(float(va[0]), float(va[1]), float(va[2])),
                    Vec3(float(vb[0]), float(vb[1]), float(vb[2])),
                    Vec3(float(vc[0]), float(vc[1]), float(vc[2])),
                ],
                normal=Vec3(float(n[0]), float(n[1]), float(n[2])),
                inner_loops=[],
                vertex_indices=[a, b, c],
            )
        )

    ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"[stl_parser] {len(V):,} vértices, {len(faces):,} caras, escala x{scale:g} "
        f"-- {ms:.1f} ms"
    )
    return {"faces": faces, "warnings": warnings}
