import trimesh
import numpy as np

def cortar_modelo_en_z(ruta_archivo: str, altura_z: float):
    """
    Carga un modelo 3D y realiza un corte transversal en la altura Z especificada.
    Retorna un objeto Path2D con los contornos del corte.
    """
    print(f"Cargando modelo desde: {ruta_archivo}...")
    
    # 1. Cargar la malla 3D
    # force='mesh' asegura que si es un archivo complejo (ej. una escena con múltiples objetos), 
    # se aplane todo a una sola malla procesable.
    malla = trimesh.load(ruta_archivo, force='mesh')
    
    print(f"Modelo cargado. Vértices: {len(malla.vertices)}, Caras: {len(malla.faces)}")
    
    # 2. Definir el plano de corte
    # El origen es el punto (0, 0, Z)
    origen_plano = [0.0, 0.0, altura_z]
    # La normal es el vector que apunta hacia arriba [X, Y, Z], indicando que el plano es horizontal
    normal_plano = [0.0, 0.0, 1.0] 
    
    print(f"Realizando corte en Z = {altura_z}...")
    
    # 3. Ejecutar la intersección matemática
    corte_3d = malla.section(plane_origin=origen_plano, plane_normal=normal_plano)
    
    if corte_3d is None:
        raise ValueError(f"El plano en Z={altura_z} no intersecta con el modelo.")
        
    # 4. Proyectar el corte 3D a un plano 2D (aplastarlo a X, Y)
    # trimesh hace esto automáticamente usando la matriz de transformación del plano
    corte_2d, transformacion = corte_3d.to_planar()
    
    print(f"Corte exitoso. Entidades 2D generadas: {len(corte_2d.entities)}")
    
    return corte_2d