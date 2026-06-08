# main.py
from core.pipeline import parse_pipeline
from core.services.obj_parser import parse_obj

def run_test():
    file_name = "demo.obj"
    with open(file_name, "r") as f:
        text = f.read()
    
    # 1. Parsear el OBJ
    parsed = parse_obj(text)
    
    # 2. Correr el pipeline con las caras obtenidas
    # (Asegúrate de importar parse_pipeline de tu nuevo pipeline.py)
    result = parse_pipeline(file_name, parsed["faces"], parsed["warnings"])
    
    print(f"Éxito: Se procesaron {len(result.groups)} grupos estructurales.")
    for g in result.groups:
        print(f"-> {g.label}: {g.total_area:.2f} m²")
    

if __name__ == "__main__":
    run_test()