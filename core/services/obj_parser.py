import io
import time
import logging
from typing import Any, List, Dict, Iterable, Union

from core.services.types import IndexedFace3D, Vec3, sub, cross, normalize

logger = logging.getLogger("eficiencia2d.pipeline")


def parse_obj(source: Union[str, Iterable[str]]) -> Dict[str, Any]:
    """
    Parser OBJ optimizado v2:
    - Acepta un string completo o un iterable de líneas (file handle / generador),
      permitiendo parsear en streaming sin materializar todo el archivo en RAM.
    - Fast-path para 'v' y 'f' (los tokens mas comunes)
    - Produce IndexedFace3D con indices originales del OBJ
    """
    t_start = time.perf_counter()

    vertices: List[Vec3] = []
    normals: List[Vec3] = []
    faces: List[IndexedFace3D] = []
    warnings: List[str] = []

    line_count = 0
    skip_count = 0

    # Si es un string: splitlines() (rápido, sin copias en CPython, ya sin saltos).
    # Si es un iterable (file handle): iterar línea a línea y limpiar el salto de
    # línea por iteración — la RAM solo mantiene la línea actual.
    if isinstance(source, str):
        line_iter: Iterable[str] = source.splitlines()
        strip_newlines = False
    else:
        line_iter = source
        strip_newlines = True

    for line in line_iter:
        line_count += 1
        if strip_newlines:
            line = line.rstrip("\r\n")
        # Saltar lineas vacias y comentarios rapido
        if not line:
            continue
        c0 = line[0]
        if c0 == '#' or c0 == 'm' or c0 == 'u' or c0 == 'g' or c0 == 's' or c0 == 'o':
            continue

        # Encontrar el primer espacio para separar prefix del resto
        sp = line.find(' ')
        if sp == -1:
            continue

        prefix = line[:sp]
        rest = line[sp + 1:]

        if prefix == 'v':
            # Fast split de coordenadas (solo 3 floats)
            # Usar partition en vez de split para evitar lista
            p1, _, r1 = rest.partition(' ')
            p2, _, p3 = r1.partition(' ')
            try:
                vertices.append(Vec3(float(p1), float(p2), float(p3.split()[0] if ' ' in p3 else p3)))
            except (ValueError, IndexError):
                skip_count += 1

        elif prefix == 'f':
            # Parsear cara - split rapido
            parts = rest.split()
            n_parts = len(parts)
            if n_parts < 3:
                continue

            face_verts: List[Vec3] = []
            face_norms: List[Vec3] = []
            face_v_indices: List[int] = []
            valid = True
            nv = len(vertices)
            nn = len(normals)

            for p in parts:
                # Manejo de formato v/vt/vn, v//vn o simplemente v
                slash = p.find('/')
                if slash == -1:
                    # Solo indice de vertice
                    try:
                        v_idx = int(p)
                    except ValueError:
                        valid = False
                        break
                    n_idx_val = None
                else:
                    try:
                        v_idx = int(p[:slash])
                    except ValueError:
                        valid = False
                        break
                    # Buscar segundo slash para normal
                    slash2 = p.find('/', slash + 1)
                    if slash2 != -1 and slash2 + 1 < len(p):
                        try:
                            n_idx_val = int(p[slash2 + 1:])
                        except ValueError:
                            n_idx_val = None
                    else:
                        n_idx_val = None

                # Convertir a 0-indexed con soporte de indices negativos
                if v_idx < 0:
                    v_idx = nv + v_idx
                else:
                    v_idx -= 1

                if v_idx < 0 or v_idx >= nv:
                    valid = False
                    break

                face_verts.append(vertices[v_idx])
                face_v_indices.append(v_idx)

                if n_idx_val is not None:
                    if n_idx_val < 0:
                        n_idx_val = nn + n_idx_val
                    else:
                        n_idx_val -= 1
                    if 0 <= n_idx_val < nn:
                        face_norms.append(normals[n_idx_val])

            if not valid or len(face_verts) < 3:
                skip_count += 1
                continue

            # Calcular normal
            if face_norms:
                normal = face_norms[0]
            else:
                v0, v1, v2 = face_verts[0], face_verts[1], face_verts[2]
                e1x, e1y, e1z = v1.x - v0.x, v1.y - v0.y, v1.z - v0.z
                e2x, e2y, e2z = v2.x - v0.x, v2.y - v0.y, v2.z - v0.z
                cx = e1y * e2z - e1z * e2y
                cy = e1z * e2x - e1x * e2z
                cz = e1x * e2y - e1y * e2x
                length = (cx*cx + cy*cy + cz*cz) ** 0.5
                if length < 1e-12:
                    normal = Vec3(0.0, 0.0, 0.0)
                else:
                    normal = Vec3(cx / length, cy / length, cz / length)

            faces.append(
                IndexedFace3D(
                    vertices=face_verts,
                    normal=normal,
                    inner_loops=[],
                    vertex_indices=face_v_indices,
                )
            )

        elif prefix == 'vn':
            p1, _, r1 = rest.partition(' ')
            p2, _, p3 = r1.partition(' ')
            try:
                normals.append(Vec3(float(p1), float(p2), float(p3.split()[0] if ' ' in p3 else p3)))
            except (ValueError, IndexError):
                skip_count += 1

        # 'vt' (UV coords) se ignoran intencionalmente - no los necesitamos

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"[obj_parser] {line_count:,} lineas -> {len(vertices):,} vertices, "
        f"{len(normals):,} normales, {len(faces):,} caras, {skip_count} omitidas -- "
        f"{elapsed_ms:.1f} ms"
    )

    if skip_count > 0:
        warnings.append(f"Se omitieron {skip_count} caras/vertices con datos invalidos.")

    return {"faces": faces, "warnings": warnings}
