from typing import Any, List, Dict
from core.services.types import Face3D, Vec3, sub, cross, normalize


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
            face_norms = []
            # Manejamos caras con formato v/vt/vn
            for p in parts[1:]:
                subparts = p.split("/")
                v_idx = int(subparts[0]) - 1
                face_verts.append(vertices[v_idx])
                if len(subparts) >= 3 and subparts[2]:
                    n_idx = int(subparts[2]) - 1
                    face_norms.append(normals[n_idx])

            # Asumimos que es triangular o cuadrangular
            if len(face_verts) >= 3:
                if face_norms:
                    # Usar la primera normal como representativa
                    normal = face_norms[0]
                else:
                    # Calcular la normal mediante producto cruz
                    e1 = sub(face_verts[1], face_verts[0])
                    e2 = sub(face_verts[2], face_verts[0])
                    normal = normalize(cross(e1, e2))
                
                faces.append(Face3D(vertices=face_verts, normal=normal, inner_loops=[]))

    return {"faces": faces, "warnings": warnings}
