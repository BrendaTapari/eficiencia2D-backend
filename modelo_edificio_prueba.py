"""
Edificio de prueba con aberturas, armado en memoria.

Dos plantas, 10 x 8 m, muros y losas de 15 cm, con puertas, ventanas, un tabique con puerta
en cada planta, un balcón y un hueco de escalera en el entrepiso. Existe una versión
correcta y variantes con UN cambio cada una (`ERRORES`).

También una casa simple sin aberturas (`casa`, `ERRORES_CASA`).

Lo usan `test_aberturas.py`, `test_huecos.py` y, para generar los .obj y abrirlos en la
aplicación:

    python modelo_edificio_prueba.py carpeta_de_salida

Por qué existe: los modelos del corpus no tenían ventanas vacías (traen vidrios y
carpinterías modelados en el vano), y por eso nadie vio que TODA abertura cerrada se perdía
al recortar las piezas. Este modelo tiene la respuesta conocida: 10 ventanas y un hueco de
escalera, con medidas elegidas a mano.
"""
import sys

T = 0.15; LX = 10.0; LZ = 8.0; H = 2.70
Y_PB = (0.0, H); Y_EP = (H, H + T); Y_PA = (H + T, 2 * H + T); Y_TE = (2 * H + T, 2 * H + 2 * T)


class Obj:
    def __init__(self):
        self.vmap, self.verts, self.faces = {}, [], []

    def vid(self, p):
        k = tuple(round(c, 6) for c in p)
        if k not in self.vmap:
            self.vmap[k] = len(self.verts) + 1
            self.verts.append(k)
        return self.vmap[k]

    def quad(self, pts, afuera):
        a, b, c = pts[0], pts[1], pts[2]
        u = [b[i] - a[i] for i in range(3)]; v = [c[i] - a[i] for i in range(3)]
        n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        if sum(n[i] * afuera[i] for i in range(3)) < 0:
            pts = list(reversed(pts))
        self.faces.append(tuple(self.vid(p) for p in pts))

    def solido(self, mapa, s, v, t, huecos=()):
        """Placa en coordenadas locales (s, v, t) con huecos rectangulares (sa, sb, va, vb)
        que la atraviesan en t. `mapa(s, v, t) -> (x, y, z)`. Es un sólido cerrado: caras
        de las dos pieles, cantos exteriores y los cantos de cada hueco (jambas, dintel,
        antepecho)."""
        S = sorted({s[0], s[1], *[c for h in huecos for c in h[:2] if s[0] < c < s[1]]})
        V = sorted({v[0], v[1], *[c for h in huecos for c in h[2:] if v[0] < c < v[1]]})

        def lleno(i, j):
            if not (0 <= i < len(S) - 1 and 0 <= j < len(V) - 1):
                return False
            cs, cv = (S[i] + S[i + 1]) / 2, (V[j] + V[j + 1]) / 2
            return not any(h[0] < cs < h[1] and h[2] < cv < h[3] for h in huecos)

        def d(ds, dv, dt):
            o = mapa(0, 0, 0); p = mapa(ds, dv, dt)
            return tuple(p[k] - o[k] for k in range(3))

        for i in range(len(S) - 1):
            for j in range(len(V) - 1):
                if not lleno(i, j):
                    continue
                s0, s1, v0, v1 = S[i], S[i + 1], V[j], V[j + 1]
                for tt, sg in ((t[0], -1), (t[1], 1)):
                    self.quad([mapa(s0, v0, tt), mapa(s1, v0, tt), mapa(s1, v1, tt), mapa(s0, v1, tt)], d(0, 0, sg))
                if not lleno(i - 1, j):
                    self.quad([mapa(s0, v0, t[0]), mapa(s0, v1, t[0]), mapa(s0, v1, t[1]), mapa(s0, v0, t[1])], d(-1, 0, 0))
                if not lleno(i + 1, j):
                    self.quad([mapa(s1, v0, t[0]), mapa(s1, v1, t[0]), mapa(s1, v1, t[1]), mapa(s1, v0, t[1])], d(1, 0, 0))
                if not lleno(i, j - 1):
                    self.quad([mapa(s0, v0, t[0]), mapa(s1, v0, t[0]), mapa(s1, v0, t[1]), mapa(s0, v0, t[1])], d(0, -1, 0))
                if not lleno(i, j + 1):
                    self.quad([mapa(s0, v1, t[0]), mapa(s1, v1, t[0]), mapa(s1, v1, t[1]), mapa(s0, v1, t[1])], d(0, 1, 0))

    def muro_x(self, x0, x1, y, zc0, zc1, huecos=()):     # muro a lo largo de X, espesor en Z
        self.solido(lambda s, v, t: (s, v, t), (x0, x1), y, (zc0, zc1),
                    [(a, b, y[0] + c, y[0] + e) for a, b, c, e in huecos])

    def muro_z(self, z0, z1, y, xc0, xc1, huecos=()):     # muro a lo largo de Z, espesor en X
        self.solido(lambda s, v, t: (t, v, s), (z0, z1), y, (xc0, xc1),
                    [(a, b, y[0] + c, y[0] + e) for a, b, c, e in huecos])

    def losa(self, x0, x1, z0, z1, y, huecos=()):
        self.solido(lambda s, v, t: (s, t, v), (x0, x1), (z0, z1), y, huecos)

    def text(self, cabecera):
        return cabecera + "\n".join(
            [f"v {x} {y} {z}" for x, y, z in self.verts]
            + [f"f {' '.join(map(str, f))}" for f in self.faces]) + "\n"


ERRORES = {
    None: "sin error",
    "esquina_abierta": "muro este de planta alta empieza 10 cm despues del muro norte (esquina abierta)",
    "tabique_no_llega": "tabique de planta baja termina 8 cm antes del muro sur",
    "muro_desplazado": "muro oeste de planta alta corrido 10 cm hacia adentro respecto del de abajo",
    "ventana_en_esquina": "ventana norte de planta alta llega hasta 5 cm de la esquina (corta el encuentro con el muro este)",
    "muro_sobre_escalera": "hueco de escalera debajo del tabique de planta alta (el tabique queda en el aire)",
    "losa_corta": "entrepiso 20 cm corto del lado sur (el muro sur de planta alta no apoya)",
    "muro_duplicado": "muro este de planta baja duplicado, corrido 4 cm hacia adentro",
    "tabique_atraviesa_losa": "tabique de planta baja 30 cm mas alto (se mete en el entrepiso)",
    "con_vidrios": "SIN ERROR: igual a la correcta pero con vidrios de 1 cm modelados en las ventanas",
}


def edificio(err=None):
    o = Obj()
    # Losas
    o.losa(0, LX, 0, LZ, (-T, 0))                                        # platea
    hueco_esc = (7.5, 9.7, 3.4, 5.6) if err == "muro_sobre_escalera" else (7.5, 9.7, 5.0, 7.7)
    o.losa(0, LX, 0, LZ - (0.20 if err == "losa_corta" else 0), Y_EP, [hueco_esc])   # entrepiso
    o.losa(0, LX, 0, LZ, Y_TE)                                           # techo
    # Planta baja
    o.muro_x(0, LX, Y_PB, 0, T, [(4.4, 5.4, 0, 2.1), (1.2, 2.7, 0.9, 2.1), (7.0, 8.8, 0.9, 2.1)])   # norte
    o.muro_x(0, LX, Y_PB, LZ - T, LZ, [(3.0, 5.5, 0, 2.2), (7.5, 8.7, 0.9, 2.1)])                   # sur
    o.muro_z(T, LZ - T, Y_PB, 0, T, [(3.0, 5.0, 0.9, 2.1)])                                         # oeste
    o.muro_z(T, LZ - T, Y_PB, LX - T, LX, [(2.0, 3.2, 1.0, 2.1)])                                   # este
    if err == "muro_duplicado":
        o.muro_z(T, LZ - T, Y_PB, LX - T - 0.04, LX - 0.04, [(2.0, 3.2, 1.0, 2.1)])
    y_tab = (Y_PB[0], Y_PB[1] + (0.30 if err == "tabique_atraviesa_losa" else 0))
    z_fin = LZ - T - (0.08 if err == "tabique_no_llega" else 0)
    o.muro_z(T, z_fin, y_tab, 6.0 - T / 2, 6.0 + T / 2, [(3.5, 4.4, 0, 2.0)])                      # tabique PB
    # Planta alta
    v_n = (7.2, 9.95 if err == "ventana_en_esquina" else 8.7, 0.9, 2.1)
    o.muro_x(0, LX, Y_PA, 0, T, [(1.2, 2.7, 0.9, 2.1), (4.2, 5.7, 0.9, 2.1), v_n])                  # norte
    o.muro_x(0, LX, Y_PA, LZ - T, LZ, [(2.0, 4.0, 0.9, 2.1), (6.0, 7.0, 0, 2.1)])                   # sur (balcon)
    dx = 0.10 if err == "muro_desplazado" else 0
    o.muro_z(T, LZ - T, Y_PA, dx, T + dx, [(2.0, 3.5, 0.9, 2.1)])                                   # oeste
    o.muro_z(T + (0.10 if err == "esquina_abierta" else 0), LZ - T, Y_PA, LX - T, LX)              # este (ciego)
    o.muro_x(T, LX - T, Y_PA, 4.0 - T / 2, 4.0 + T / 2, [(2.5, 3.4, 0, 2.0)])                      # tabique PA
    if err == "con_vidrios":
        for (a, b, c, e) in [(1.2, 2.7, 0.9, 2.1), (7.0, 8.8, 0.9, 2.1)]:
            o.muro_x(a, b, (c, e), T / 2 - 0.005, T / 2 + 0.005)
        for (a, b, c, e) in [(1.2, 2.7, 0.9, 2.1), (4.2, 5.7, 0.9, 2.1), (7.2, 8.7, 0.9, 2.1)]:
            o.muro_x(a, b, (Y_PA[0] + c, Y_PA[0] + e), T / 2 - 0.005, T / 2 + 0.005)
    return o


# ---------------------------------------------------------------------------
# Casa simple de dos plantas, sin aberturas: 8 x 6 m, muros y losas de 20 cm.
# Fue el primer modelo con errores a propósito. Los mismos errores que el edificio, más
# algunos propios.
# ---------------------------------------------------------------------------

CASA_T, CASA_L, CASA_F, CASA_H = 0.20, 8.0, 6.0, 2.8
ERRORES_CASA = {
    None: "sin error",
    "corto25": "muro este de planta alta 25 cm corto (no llega al muro norte)",
    "corto5": "muro este de planta alta 5 cm corto",
    "pasado": "muro este de planta alta 30 cm largo (atraviesa el muro norte y asoma)",
    "duplicado": "muro oeste de planta baja duplicado, corrido 5 cm hacia adentro",
    "flotante": "muro oeste de planta alta levantado 30 cm (no apoya en el entrepiso)",
    "losa_chica": "entrepiso 40 cm corto del lado este (no llega al muro)",
    "tabique_alto": "tabique interior 60 cm mas alto (atraviesa el entrepiso)",
    "corrido": "muro sur de planta alta corrido 15 cm hacia afuera (vuela sobre el de abajo)",
}


def casa(err=None) -> "Obj":
    T, L, F, H = CASA_T, CASA_L, CASA_F, CASA_H
    o = Obj()

    def caja(x0, x1, y0, y1, z0, z1):
        o.solido(lambda s, v, t: (s, v, t), (x0, x1), (y0, y1), (z0, z1))

    caja(0, L, -T, 0, 0, F)
    caja(0, L - (0.4 if err == "losa_chica" else 0), H, H + T, 0, F)
    caja(0, L, 2 * H + T, 2 * H + 2 * T, 0, F)
    for y0, y1, pl in ((0, H, "PB"), (H + T, 2 * H + T, "PA")):
        caja(0, L, y0, y1, 0, T)
        d = 0.15 if (pl == "PA" and err == "corrido") else 0
        caja(0, L, y0, y1, F - T + d, F + d)
        yy0 = y0 + 0.3 if (pl == "PA" and err == "flotante") else y0
        caja(0, T, yy0, y1, T, F - T)
        z0 = T
        if pl == "PA" and err == "corto25":
            z0 = T + 0.25
        if pl == "PA" and err == "corto5":
            z0 = T + 0.05
        if pl == "PA" and err == "pasado":
            z0 = -0.30
        caja(L - T, L, y0, y1, z0, F - T)
        if pl == "PB" and err == "duplicado":
            caja(T + 0.05, 2 * T + 0.05, y0, y1, T, F - T)
    caja(4.0 - T / 2, 4.0 + T / 2, 0, H + (0.6 if err == "tabique_alto" else 0), T, F - T)
    return o


def texto_casa(err=None) -> str:
    return casa(err).text(
        "# Casa de dos plantas 8 x 6 m, muros y losas de 20 cm. Eje vertical: Y\n"
        f"# {ERRORES_CASA[err]}\n"
    )


# Aberturas CERRADAS del edificio correcto (las puertas tocan el borde: son mordidas, no
# huecos). (ancho, alto) en metros.
VENTANAS = {
    "PB": [(1.5, 1.2), (1.8, 1.2), (1.2, 1.2), (2.0, 1.2), (1.2, 1.1)],
    "PA": [(1.5, 1.2), (1.5, 1.2), (1.5, 1.2), (2.0, 1.2), (1.5, 1.2)],
}
HUECO_DE_ESCALERA = (2.2, 2.7)
# De esas ventanas, las que la variante "con_vidrios" trae con un vidrio modelado adentro.
VENTANAS_CON_VIDRIO = 5


def texto(err=None) -> str:
    return edificio(err).text(
        "# Edificio de dos plantas 10 x 8 m, muros y losas de 15 cm. Eje vertical: Y\n"
        f"# {ERRORES[err]}\n"
    )


if __name__ == "__main__":
    import os

    destino = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(destino, exist_ok=True)
    for err in ERRORES:
        nombre = "edificio2_ok.obj" if err is None else f"edificio2_{err}.obj"
        with open(os.path.join(destino, nombre), "w") as fh:
            fh.write(texto(err))
        print(nombre)
    for err in ERRORES_CASA:
        nombre = "casa_ok.obj" if err is None else f"casa_error_{err}.obj"
        with open(os.path.join(destino, nombre), "w") as fh:
            fh.write(texto_casa(err))
        print(nombre)
