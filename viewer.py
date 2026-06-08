import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from core.pipeline import parse_pipeline
from core.services.obj_parser import parse_obj

def visualize_groups(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    
    parsed = parse_obj(text)
    # Ejecutamos el pipeline
    result = parse_pipeline(file_path, parsed["faces"], parsed["warnings"])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {"wall": "red", "floor": "blue", "discard": "gray"}
    
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')

    for group in result.groups:
        color = colors.get(group.category, "gray")
        for fi in group.face_indices:
            face = result.faces[fi]
            # Crear polígono para matplotlib
            poly = [[(v.x, v.y, v.z) for v in face.vertices]]
            ax.add_collection3d(Poly3DCollection(poly, color=color, alpha=0.5, edgecolor='k'))
            
            # Recopilar límites
            for v in face.vertices:
                if v.x < min_x: min_x = v.x
                if v.x > max_x: max_x = v.x
                if v.y < min_y: min_y = v.y
                if v.y > max_y: max_y = v.y
                if v.z < min_z: min_z = v.z
                if v.z > max_z: max_z = v.z
            
    # Auto-escala básico (Matplotlib 3D no auto-escala collections automáticamente)
    max_range = max(max_x - min_x, max_y - min_y, max_z - min_z) / 2.0
    mid_x = (max_x + min_x) * 0.5
    mid_y = (max_y + min_y) * 0.5
    mid_z = (max_z + min_z) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Vista previa de Clasificación: Rojo=Pared, Azul=Piso, Gris=Descarte")
    plt.show()

if __name__ == "__main__":
    visualize_groups("demo.obj")