# main.py
from api.main import app
from core.pipeline import parse_pipeline
from core.services.obj_parser import parse_obj


def run_test():
    file_name = "demo.obj"
    print(f"[1/3] Leyendo archivo: {file_name}")
    with open(file_name, "r") as f:
        text = f.read()

    print("[2/3] Parseando OBJ...")
    # 1. Parsear el OBJ
    parsed = parse_obj(text)
    print(f"    -> Caras: {len(parsed['faces'])} | Avisos: {len(parsed['warnings'])}")

    print("[3/3] Ejecutando pipeline...")
    # 2. Correr el pipeline con las caras obtenidas
    # (Asegúrate de importar parse_pipeline de tu nuevo pipeline.py)
    result = parse_pipeline(file_name, parsed["faces"], parsed["warnings"])

    print(f"Éxito: Se procesaron {len(result.groups)} grupos estructurales.")

    wall_count = sum(1 for g in result.groups if g.category == "wall")
    floor_count = sum(1 for g in result.groups if g.category == "floor")
    discard_count = sum(1 for g in result.groups if g.category == "discard")

    print(
        "Resumen por categoría: "
        f"paredes={wall_count}, pisos={floor_count}, descartes={discard_count}"
    )
    print(f"Pisos procesados antes del split: {result.pre_split_face_count}")
    print(f"Uniones detectadas: {len(result.joints)}")
    print(f"Ajustes calculados: {len(result.adjustments)}")

    print("Detalle de grupos:")
    for g in result.groups:
        print(f"-> {g.label}: {g.total_area:.2f} m² [{g.category}]")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
