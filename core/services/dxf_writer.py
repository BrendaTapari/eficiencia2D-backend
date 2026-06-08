from typing import Dict, List
from core.services.types import Facade, FloorPlan

# ============================================================================
# DXF Writer
#
# Genera archivos DXF compatibles con AutoCAD (AC1009 / R12).
# Protocolo de 4 capas para corte láser:
#   - CUT_EXTERIOR   (ACI 1 / rojo)   — contornos exteriores
#   - ENGRAVE_VECTOR (ACI 5 / azul)   — títulos, anotaciones
#   - ENGRAVE_RASTER (ACI 7 / blanco) — cotas/dimensiones
#   - CUT_INTERIOR   (ACI 3 / verde)  — cortes interiores
# ============================================================================

# Definiciones de capas
LAYERS = [
    {"name": "CUT_EXTERIOR", "aci": "1"},  # rojo
    {"name": "ENGRAVE_VECTOR", "aci": "5"},  # azul
    {"name": "ENGRAVE_RASTER", "aci": "7"},  # blanco
    {"name": "CUT_INTERIOR", "aci": "3"},  # verde
]

# Diccionario para buscar rápidamente el color por nombre de capa
LAYER_ACI: Dict[str, str] = {layer["name"]: layer["aci"] for layer in LAYERS}


def dxf_header() -> str:
    lines = [
        "0",
        "SECTION",
        "2",
        "HEADER",
        "9",
        "$ACADVER",
        "1",
        "AC1009",
        "9",
        "$INSUNITS",
        "70",
        "6",
        "0",
        "ENDSEC",
    ]
    return "\n".join(lines) + "\n"


def dxf_tables() -> str:
    lines = [
        "0",
        "SECTION",
        "2",
        "TABLES",
        # Tabla LTYPE
        "0",
        "TABLE",
        "2",
        "LTYPE",
        "70",
        "1",
        "0",
        "LTYPE",
        "2",
        "CONTINUOUS",
        "70",
        "0",
        "3",
        "Solid line",
        "72",
        "65",
        "73",
        "0",
        "40",
        "0.0",
        "0",
        "ENDTAB",
        # Tabla LAYER
        "0",
        "TABLE",
        "2",
        "LAYER",
        "70",
        str(len(LAYERS)),
    ]

    for layer in LAYERS:
        lines.extend(
            [
                "0",
                "LAYER",
                "2",
                layer["name"],
                "70",
                "0",
                "62",
                layer["aci"],
                "6",
                "CONTINUOUS",
            ]
        )

    lines.extend(["0", "ENDTAB", "0", "ENDSEC"])
    return "\n".join(lines) + "\n"


def dxf_line(x1: float, y1: float, x2: float, y2: float, layer: str) -> str:
    aci = LAYER_ACI.get(layer, "7")
    lines = [
        "0",
        "LINE",
        "8",
        layer,
        "62",
        aci,
        "10",
        str(x1),
        "20",
        str(y1),
        "11",
        str(x2),
        "21",
        str(y2),
    ]
    return "\n".join(lines) + "\n"


def dxf_text(x: float, y: float, h: float, text: str, layer: str) -> str:
    aci = LAYER_ACI.get(layer, "7")
    lines = [
        "0",
        "TEXT",
        "8",
        layer,
        "62",
        aci,
        "10",
        str(x),
        "20",
        str(y),
        "40",
        str(h),
        "1",
        text,
        "72",
        "1",
        "11",
        str(x),
        "21",
        str(y),
    ]
    return "\n".join(lines) + "\n"


def dxf_arc(
    cx: float,
    cy: float,
    radius: float,
    start_angle: float,
    end_angle: float,
    layer: str,
) -> str:
    aci = LAYER_ACI.get(layer, "7")
    lines = [
        "0",
        "ARC",
        "8",
        layer,
        "62",
        aci,
        "10",
        str(cx),
        "20",
        str(cy),
        "40",
        str(radius),
        "50",
        str(start_angle),
        "51",
        str(end_angle),
    ]
    return "\n".join(lines) + "\n"


def dxf_footer() -> str:
    return "0\nENDSEC\n0\nEOF\n"


# ============================================================================
# Generadores Finales
# ============================================================================


def generate_facade_dxf(facade: Facade, scale_denom: float) -> str:
    s = 1.0 / scale_denom
    text_h = 0.003

    # Usamos una lista para recolectar todas las entidades (más rápido y eficiente en RAM)
    out: List[str] = [dxf_header(), dxf_tables(), "0\nSECTION\n2\nENTITIES\n"]

    for poly in facade.polygons:
        verts = poly.vertices
        n_verts = len(verts)
        if n_verts < 2:
            continue

        if n_verts == 2:
            out.append(
                dxf_line(
                    verts[0].x * s,
                    verts[0].y * s,
                    verts[1].x * s,
                    verts[1].y * s,
                    "CUT_EXTERIOR",
                )
            )
        else:
            for i in range(n_verts):
                j = (i + 1) % n_verts
                out.append(
                    dxf_line(
                        verts[i].x * s,
                        verts[i].y * s,
                        verts[j].x * s,
                        verts[j].y * s,
                        "CUT_EXTERIOR",
                    )
                )

    # Título
    out.append(
        dxf_text(
            facade.width * 0.5 * s,
            (facade.height + 0.5) * s,
            text_h * 1.5,
            facade.label,
            "ENGRAVE_VECTOR",
        )
    )

    # Cotas (Usamos f-strings para formatear a 2 decimales equivalente a .toFixed(2))
    out.append(
        dxf_text(
            facade.width * 0.5 * s,
            -0.4 * s,
            text_h,
            f"{facade.width:.2f} m",
            "ENGRAVE_RASTER",
        )
    )
    out.append(
        dxf_text(
            (facade.width + 0.3) * s,
            facade.height * 0.5 * s,
            text_h,
            f"{facade.height:.2f} m",
            "ENGRAVE_RASTER",
        )
    )

    out.append(dxf_footer())

    # Unimos todo el array de strings de una sola vez
    return "".join(out)


def generate_floor_plan_dxf(plan: FloorPlan, scale_denom: float) -> str:
    s = 1.0 / scale_denom
    text_h = 0.003

    out: List[str] = [dxf_header(), dxf_tables(), "0\nSECTION\n2\nENTITIES\n"]

    # --- Segmentos de muro (Capa CUT_EXTERIOR) ---
    for seg in plan.segments:
        out.append(
            dxf_line(seg.a.x * s, seg.a.y * s, seg.b.x * s, seg.b.y * s, "CUT_EXTERIOR")
        )

    # --- Símbolos de puertas (Capa CUT_INTERIOR) ---
    if plan.doors:
        for door in plan.doors:
            # Línea de la hoja de la puerta
            out.append(
                dxf_line(
                    door.hinge.x * s,
                    door.hinge.y * s,
                    door.leaf_end.x * s,
                    door.leaf_end.y * s,
                    "CUT_INTERIOR",
                )
            )

            # Arco de apertura
            out.append(
                dxf_arc(
                    door.hinge.x * s,
                    door.hinge.y * s,
                    door.width * s,
                    door.start_angle,
                    door.end_angle,
                    "CUT_INTERIOR",
                )
            )

    # Título
    out.append(
        dxf_text(
            plan.width * 0.5 * s,
            (plan.height + 0.5) * s,
            text_h * 1.5,
            plan.label,
            "ENGRAVE_VECTOR",
        )
    )

    # Cotas
    out.append(
        dxf_text(
            plan.width * 0.5 * s,
            -0.4 * s,
            text_h,
            f"{plan.width:.2f} m",
            "ENGRAVE_RASTER",
        )
    )
    out.append(
        dxf_text(
            (plan.width + 0.3) * s,
            plan.height * 0.5 * s,
            text_h,
            f"{plan.height:.2f} m",
            "ENGRAVE_RASTER",
        )
    )

    out.append(dxf_footer())
    return "".join(out)
