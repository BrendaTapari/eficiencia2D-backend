import trimesh
import os
from core.geometry import cortar_modelo_en_z

# Aseguramos que la carpeta temporal exista
os.makedirs("temp/uploads", exist_ok=True)

# OPCIÓN A: Usar un modelo real
# Si tenés a mano algún .stl de los que usás en tu laminador para imprimir en 3D, 
# podés copiarlo a la carpeta 'temp/uploads/' y descomentar la siguiente línea:
# ruta_prueba = "temp/uploads/mi_modelo.stl"

# OPCIÓN B: Generar un modelo de prueba por código (una "habitación" cuadrada)
print("Generando modelo 3D de prueba...")
# Creamos una caja de 10x10 y 5 de altura
malla_prueba = trimesh.creation.box(extents=[10, 10, 5])
# La movemos para que descanse sobre el piso (Z=0)
malla_prueba.apply_translation([0, 0, 2.5]) 

ruta_prueba = "temp/uploads/caja_prueba.stl"
malla_prueba.export(ruta_prueba)

print("\n--- Iniciando prueba del motor matemático ---")
try:
    # Vamos a cortar la caja justo a la mitad de su altura (Z = 2.5)
    resultado_2d = cortar_modelo_en_z(ruta_prueba, altura_z=2.5)
    
    print("\n¡Corte exitoso!")
    print(f"Líneas/Polígonos generados: {len(resultado_2d.entities)}")
    
    # Esto abrirá una ventana emergente mostrando el contorno 2D del corte
    print("Abriendo visor 2D (cerrá la ventana para finalizar el script)...")
    resultado_2d.show()
    
except Exception as e:
    print(f"\n❌ Error durante la prueba: {e}")