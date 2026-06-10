# -*- coding: utf-8 -*-
"""
benchmark.py - Script de benchmark para medir el rendimiento del pipeline
con el archivo 'Prueba app.obj' (50 MB).

Uso:
    python benchmark.py
    python benchmark.py "Prueba app.obj"
"""

import sys
import time
import os
import io
import logging

# Forzar stdout/stderr a UTF-8 en Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configurar logging detallado
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# Intentar importar psutil para métricas de memoria
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
    print("[AVISO] psutil no instalado — no se medirá memoria. Instalar con: pip install psutil")


def get_memory_mb() -> float:
    if not HAVE_PSUTIL:
        return 0.0
    import os
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def run_benchmark(file_path: str):
    print("=" * 70)
    print(f"  BENCHMARK — Eficiencia2D Pipeline")
    print(f"  Archivo: {file_path}")
    file_size_mb = os.path.getsize(file_path) / 1024 / 1024
    print(f"  Tamaño:  {file_size_mb:.2f} MB")
    print("=" * 70)

    mem_inicial = get_memory_mb()
    print(f"\n  Memoria inicial: {mem_inicial:.0f} MB")

    # --- Importaciones ---
    print("\n[1/5] Importando módulos del pipeline...")
    t_import = time.perf_counter()
    from core.services.obj_parser import parse_obj
    from core.pipeline import parse_pipeline
    print(f"  -> importado en {(time.perf_counter()-t_import)*1000:.0f} ms")

    # --- Lectura del archivo ---
    print(f"\n[2/5] Leyendo archivo ({file_size_mb:.2f} MB)...")
    t0 = time.perf_counter()
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    t_read = time.perf_counter() - t0
    mem_tras_lectura = get_memory_mb()
    print(f"  -> leido en {t_read*1000:.0f} ms")
    print(f"  -> memoria: {mem_tras_lectura:.0f} MB (D +{mem_tras_lectura - mem_inicial:.0f} MB)")
    print(f"  -> caracteres en string: {len(text):,}")

    # --- Parseo OBJ ---
    print(f"\n[3/5] Parseando OBJ...")
    t0 = time.perf_counter()
    parsed = parse_obj(text)
    t_parse = time.perf_counter() - t0
    del text  # Liberar RAM
    mem_tras_parse = get_memory_mb()
    faces = parsed["faces"]
    print(f"  -> parseado en {t_parse*1000:.0f} ms ({t_parse:.2f}s)")
    print(f"  -> caras:     {len(faces):,}")
    print(f"  -> memoria:   {mem_tras_parse:.0f} MB (D +{mem_tras_parse - mem_inicial:.0f} MB)")

    # --- Pipeline completo ---
    print(f"\n[4/5] Ejecutando pipeline completo...")
    t0 = time.perf_counter()
    result = parse_pipeline(os.path.basename(file_path), faces, parsed["warnings"])
    t_pipeline = time.perf_counter() - t0
    mem_tras_pipeline = get_memory_mb()

    print(f"\n  -> pipeline en {t_pipeline:.2f}s")
    print(f"  -> memoria:   {mem_tras_pipeline:.0f} MB (D +{mem_tras_pipeline - mem_inicial:.0f} MB)")

    # --- Resumen de resultados ---
    print(f"\n[5/5] Resultados del pipeline:")
    wall_count = sum(1 for g in result.groups if g.category == "wall")
    floor_count = sum(1 for g in result.groups if g.category == "floor")
    discard_count = sum(1 for g in result.groups if g.category == "discard")
    print(f"  Grupos:      {len(result.groups)}")
    print(f"    Paredes:   {wall_count}")
    print(f"    Pisos:     {floor_count}")
    print(f"    Descarte:  {discard_count}")
    print(f"  Joints:      {len(result.joints)}")
    print(f"  Ajustes:     {len(result.adjustments)}")
    print(f"  Wall-Wall:   {len(result.wall_wall_joints)}")
    print(f"  Caras final: {len(result.faces):,}")

    # --- Timing detallado (si está disponible en el resultado) ---
    if result.timing:
        print(f"\n{'─'*70}")
        print(f"  TIMING DETALLADO:")
        print(f"{'─'*70}")
        for step in result.timing.get("steps", []):
            bar_len = int(step.get("pct", 0) / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            extras = {k: v for k, v in step.items() if k not in ("name", "ms", "pct")}
            extra_str = ("  →  " + ", ".join(f"{k}={v}" for k, v in extras.items())) if extras else ""
            print(f"  {bar} {step['pct']:5.1f}%%  [{step['ms']:>8.1f} ms]  {step['name']}{extra_str}")

    # --- Resumen final ---
    total_time = t_read + t_parse + t_pipeline
    print(f"\n{'='*70}")
    print(f"  TIEMPOS TOTALES:")
    print(f"    Lectura archivo:  {t_read*1000:>8.0f} ms")
    print(f"    Parseo OBJ:       {t_parse*1000:>8.0f} ms")
    print(f"    Pipeline:         {t_pipeline*1000:>8.0f} ms")
    print(f"    --------------------------------")
    if total_time >= 60:
        print(f"    TOTAL:            {int(total_time//60)}m {total_time%60:.1f}s")
    else:
        print(f"    TOTAL:            {total_time:.2f}s")
    print(f"{'='*70}\n")

    if total_time > 60:
        print(f"[LENTO] {total_time:.0f}s - revisar timing detallado arriba")
    elif total_time > 10:
        print(f"[MODERADO] {total_time:.0f}s")
    else:
        print(f"[RAPIDO] {total_time:.2f}s")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "Prueba app.obj"

    if not os.path.exists(target):
        print(f"ERROR: No se encontró el archivo '{target}'")
        print(f"Archivos .obj disponibles en el directorio actual:")
        for f in os.listdir("."):
            if f.endswith(".obj"):
                size = os.path.getsize(f) / 1024 / 1024
                print(f"  {f}  ({size:.2f} MB)")
        sys.exit(1)

    run_benchmark(target)
