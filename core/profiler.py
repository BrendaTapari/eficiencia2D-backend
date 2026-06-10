"""
profiler.py — Sistema de instrumentación de timing para el pipeline de Eficiencia2D.

Uso:
    from core.profiler import PipelineTimer, timed

    timer = PipelineTimer("upload")
    with timer.step("parse_obj"):
        result = parse_obj(text)
    timer.report()
"""

import time
import logging
import functools
from contextlib import contextmanager
from typing import List, Dict, Optional

# Configurar logger con formato detallado
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("eficiencia2d.pipeline")


class StepResult:
    def __init__(self, name: str, elapsed_ms: float, metadata: Optional[Dict] = None):
        self.name = name
        self.elapsed_ms = elapsed_ms
        self.metadata = metadata or {}

    def __repr__(self):
        meta_str = ""
        if self.metadata:
            meta_str = "  →  " + ", ".join(f"{k}={v}" for k, v in self.metadata.items())
        return f"  [{self.elapsed_ms:>8.1f} ms]  {self.name}{meta_str}"


class PipelineTimer:
    """
    Registra el tiempo de cada paso del pipeline y muestra un reporte final.

    Ejemplo:
        timer = PipelineTimer("upload_pipeline")
        with timer.step("parse_obj", file_size_mb=50.9):
            parsed = parse_obj(text)
        timer.report()
    """

    def __init__(self, pipeline_name: str):
        self.name = pipeline_name
        self.steps: List[StepResult] = []
        self._start = time.perf_counter()
        logger.info(f"=== Pipeline '{pipeline_name}' iniciado ===")

    @contextmanager
    def step(self, step_name: str, **metadata):
        """Context manager que mide el tiempo de un bloque de código."""
        logger.debug(f"|  >> {step_name}...")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            result = StepResult(step_name, elapsed_ms, metadata)
            self.steps.append(result)

            # Log inmediato del paso completado
            if elapsed_ms > 5000:
                level = logging.WARNING
                prefix = "[LENTO]"
            elif elapsed_ms > 1000:
                level = logging.WARNING
                prefix = "[ATENCION]"
            else:
                level = logging.DEBUG
                prefix = "ok"

            meta_parts = [f"{k}={v}" for k, v in metadata.items()]
            meta_str = ("  ->  " + ", ".join(meta_parts)) if meta_parts else ""
            logger.log(
                level,
                f"|  {prefix} {step_name}: {elapsed_ms:.1f} ms{meta_str}",
            )

    def report(self) -> Dict:
        """Imprime el reporte completo y retorna los datos como dict."""
        total_ms = (time.perf_counter() - self._start) * 1000
        total_s = total_ms / 1000

        logger.info(f"=== Reporte pipeline '{self.name}' ===")
        for step in self.steps:
            pct = (step.elapsed_ms / total_ms * 100) if total_ms > 0 else 0
            bar_len = int(pct / 5)  # barra de 20 chars max
            bar = "#" * bar_len + "." * (20 - bar_len)
            meta_str = ""
            if step.metadata:
                meta_str = "  ->  " + ", ".join(
                    f"{k}={v}" for k, v in step.metadata.items()
                )
            logger.info(
                f"| {bar} {pct:5.1f}%  [{step.elapsed_ms:>8.1f} ms]  {step.name}{meta_str}"
            )

        if total_s >= 60:
            total_human = f"{int(total_s // 60)}m {total_s % 60:.1f}s"
        else:
            total_human = f"{total_s:.2f}s"

        logger.info(f"=== TOTAL: {total_human} ({total_ms:.0f} ms) ===")

        return {
            "pipeline": self.name,
            "total_ms": round(total_ms, 1),
            "steps": [
                {
                    "name": s.name,
                    "ms": round(s.elapsed_ms, 1),
                    "pct": round(s.elapsed_ms / total_ms * 100, 1) if total_ms > 0 else 0,
                    **s.metadata,
                }
                for s in self.steps
            ],
        }


def timed(label: Optional[str] = None):
    """
    Decorador para medir el tiempo de una función individual.

    Uso:
        @timed("classify_all_faces")
        def classify_all_faces(faces):
            ...
    """
    def decorator(fn):
        fn_label = label or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms > 100:
                logger.debug(f"[timed] {fn_label}: {elapsed_ms:.1f} ms")
            return result

        return wrapper
    return decorator


def log_memory(label: str = ""):
    """Log del uso de memoria actual (si psutil está disponible)."""
    try:
        import psutil, os
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        logger.debug(f"[memoria] {label}: {mem_mb:.1f} MB RSS")
    except ImportError:
        pass  # psutil opcional
