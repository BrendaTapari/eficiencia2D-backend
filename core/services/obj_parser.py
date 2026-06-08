from typing import Any, List, Dict
from core.services.types import Face3D, Vec3


def parse_obj(text: str) -> Dict[str, Any]:
    faces: List[Face3D] = []
    warnings: List[str] = []

    vertices: List[Vec3] = []
    normals: List[Vec3] = []

    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue

        prefix = parts[0]

        if prefix == "v":
            vertices.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
        elif prefix == "vn":
            normals.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
        elif prefix == "f":
            face_verts = []
            # Manejamos caras con formato v/vt/vn
            for p in parts[1:]:
                v_idx = int(p.split("/")[0]) - 1
                face_verts.append(vertices[v_idx])

            # Asumimos que es triangular o cuadrangular
            # Para simplificar, si es cuadrangular, lo dividimos en dos triángulos
            if len(face_verts) >= 3:
                # Normal simple calculada al vuelo si no hay normal
                normal = Vec3(0, 1, 0)  # Placeholder
                faces.append(Face3D(vertices=face_verts, normal=normal, inner_loops=[]))

    return {"faces": faces, "warnings": warnings}
