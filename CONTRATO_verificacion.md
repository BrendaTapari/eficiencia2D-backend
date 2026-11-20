# Contrato backend → front: verificación de la maqueta (versión 2)

Estado del backend: implementado y con tests (`test_verificacion.py`, `test_corpus.py`
y, de punta a punta por HTTP, `test_api.py`).

**Lo primero que hay que hacer está en §0.**

**Compatibilidad con la versión 1:** no se quitó ni se renombró ninguna clave. Un front que
ya lee `resumen`, `piezas`, `problemas` y `uniones` sigue funcionando; va a ver menos
problemas, porque las uniones que sólo se tocan de canto dejaron de contar (§2). Lo nuevo
es `nudos`, `terminaciones`, `quien` y `decisiones_automaticas`.

Un solo cambio de valor: la acción de tipo `escala_mas_fina` ahora se llama
`cambiar_escala` y puede traer la escala a probar (§3). Si el front tiene un caso para el
nombre viejo, hay que renombrarlo.

## Qué cambia en la pantalla

| Antes (v1) | Ahora (v2) |
|---|---|
| Titular: "151 de 295 piezas verificadas · 144 para revisar" | Titular con lo que **impide armar**. En ese mismo modelo: 206 de 295, y 27 uniones resueltas solas. |
| Lista de problemas por pieza o por unión | Lista de **nudos**: un lugar, con las 2 a 4 piezas que se encuentran ahí. 50 nudos en ese modelo. |
| Cada problema con los mismos botones | Cada nudo dice **quién lo arregla**: decidir, escala o modelo. |
| Sólo vista 3D | Cada nudo trae su **detalle en planta** listo para dibujar en 2D. |
| "Se tocan de canto" mezclado con lo demás | Sección aparte de **terminaciones**. |

---

## 0. CAMBIO URGENTE: qué pedir después de cada botón

**Después de que el usuario toca una acción, pedir `POST /api/verificacion`. No pedir
`/recompute` ni `/nesting-preview`.** Es lo que hacía aparecer "El backend no respondió a
tiempo": esos dos pedidos tardaban 16 a 43 s y devolvían hasta 121 MB.

| Momento | Pedido |
|---|---|
| Se carga el modelo o cambia eje / área mínima / divisiones | `/upload`, `/recompute` (como hoy) |
| Se abre el instructivo, o cambia la escala | `/verificacion` |
| **El usuario toca una acción (fusionar, no cortar, invertir, deshacer)** | **`/verificacion`** |
| Se abre la pantalla de planchas | `/nesting-preview` (como hoy) |
| Generar | `/generate` (como hoy) |

Medido de punta a punta sobre un modelo de 295 piezas a 1:100:

| Pedido | Tiempo | Tamaño |
|---|---|---|
| `/verificacion`, primera vez para ese modelo y escala | 9 a 19 s | 1 MB |
| `/verificacion` después de un clic cuyo efecto ya estaba previsto (§3.2) | **menos de 0.3 s** | 1 MB |
| `/verificacion` después de un clic todavía no previsto | 4 a 8 s | 1 MB |
| `/verificacion` volviendo a un estado anterior (Deshacer) | menos de 0.3 s | 1 MB |

**Subir el tiempo de espera del front a 60 s** para `/verificacion`, `/nesting-preview`,
`/recompute` y `/generate`. El primer pedido de un modelo grande sigue tardando.

### 0.1 `POST /api/verificacion`

**Cuerpo:** exactamente el mismo que `/nesting-preview` (`file_id` o `proyecto_id`,
`scale_denom`, `min_area_m2`, `axis`, `splits`, `overrides`, `merges`,
`wall_wall_decisions`, `wall_wall_decisions_by_pair`, `marks`, `user_cuts`, `flex`,
`mark_lines`, `ribs`, `columns`…). Los campos de plancha se ignoran.

**Respuesta:**

```jsonc
{
  "verificacion": { … },            // §2. Reemplazar la anterior entera.
  "placements": { "245": { … } },   // piezas YA RECORTADAS, igual que /nesting-preview
  "panel_id_by_group": { "245": "A84" },   // etiqueta de cada grupo (ESTABLE, §0.2)
  "absorbed_groups": { "301": 245 },
  "warnings": [ "…" ],
  "plate_thickness_m": 0.3,
  "plate_joints": [ … ],
  "efectos": { "listo": false, "calculadas": 3, "total": 18, "efectos": { … } },  // §3.2
  "servidor": { "ms": 4715, "ya_calculada": false, "espera_ms": 473 }   // diagnóstico
}
```

No trae caras del modelo ni planchas: las caras no cambian con una acción (usar las de
`/upload`) y la plancha se pide cuando el usuario va a esa pantalla.

`verificacion`, `placements` y `absorbed_groups` son **idénticos** a los que devolvería
`/nesting-preview` para el mismo cuerpo. Hay una prueba automática que lo comprueba.

### 0.2 Las etiquetas ya no se renumeran

Una pieza conserva su etiqueta aunque el usuario fusione o descarte otras: si descarta
`A14`, la `A15` sigue siendo `A15` (queda un hueco en la numeración). Vale para
`/verificacion`, `/nesting-preview` y `/generate`: la etiqueta que el usuario ve en el
instructivo es la que sale grabada en la plancha.

- Una pieza que nace de una fusión conserva la etiqueta de una de las dos; la otra
  etiqueta desaparece.
- Una pieza que no existía antes (un descartado que el usuario rescata) recibe un número
  nuevo, después del último.
- Las etiquetas **sí** cambian con la escala, el eje, el área mínima y las divisiones.
- Después de una acción usar `panel_id_by_group` de `/verificacion`. El de `/recompute`
  no conoce esta numeración: no mezclarlos.

### 0.3 Lo que se cambia en el visor 3D tiene que llegar al instructivo

El instructivo refleja las fusiones, reclasificaciones y descartes hechos en el visor
**sólo si el pedido los trae**. `/verificacion`, `/nesting-preview`, `/generate` y
`/recompute` tienen que recibir siempre el MISMO estado de revisión:

- `merges`: todas las fusiones hechas en el visor, `[[id, id, …], …]`.
- `overrides`: todas las reclasificaciones, `{"id": "wall" | "floor" | "discard"}`. Un
  descartado que el usuario pasa a pared va como `"wall"`, no alcanza con quitarlo.
- `wall_wall_decisions_by_pair`.

Para comprobarlo sin adivinar, `/verificacion` devuelve cuánto recibió:

```jsonc
"servidor": { "recibido": { "overrides": 1, "merges": 1, "decisiones": 0 }, … }
```

Si el usuario fusionó algo en el visor y `recibido.merges` es 0, el front no lo mandó.

**Piezas fusionadas.** Cuando se fusionan cuatro tramos, la pieza resultante conserva el
id de UNO (el de mayor área). Los otros tres no tienen entrada en `placements`: figuran en
`absorbed_groups` apuntando al que quedó.

```jsonc
"placements":      { "180": { … } },                       // la pieza fusionada, entera
"absorbed_groups": { "179": 180, "236": 180, "181": 180 }  // están DENTRO de la 180
```

`absorbed_groups` ahora incluye las fusiones del usuario además de las automáticas por
escala (antes sólo las automáticas: los tramos fusionados "desaparecían"). Regla para
dibujar un grupo `g` en el instructivo:

1. Si está en `placements` → dibujar ese placement.
2. Si está en `absorbed_groups` → no dibujar nada propio: ya está dentro del placement
   del grupo indicado. Al seleccionarlo, resaltar ese.
3. Si no está en ninguno → no se corta (descartado, o más chico que el mínimo).

**`/recompute` con `scale_denom`** devuelve en `topology` exactamente los mismos
`placements`, `panel_id_by_group` y `absorbed_groups` que `/verificacion` para el mismo
estado (antes los calculaba aparte y el visor podía llamar A13 a la pieza que la plancha
llama A14). `topology.groups` sigue listando los grupos del modelo; para saber qué se
corta, mirar `placements` y `absorbed_groups`.

Una pieza fusionada puede estar formada por dos o tres contornos que no se tocan (tramos
separados por un vano): se corta como varias partes con una sola etiqueta.

### 0.4 Aberturas: ventanas abiertas y ventanas cerradas

Las aberturas cerradas del modelo (ventanas, huecos de escalera) ahora salen **cortadas**
en la pieza. Antes se perdían: `placements[..].outline` y la plancha traían el muro
macizo. El front no tiene que cambiar nada para esto: los contornos ya traen las aristas
con `hole: true`, sólo que ahora llegan.

Excepción, con aviso: si el modelo tiene algo **dentro** del vano (un vidrio, una
carpintería, los paños de un frente vidriado), ese elemento ocupa la misma placa que el
muro y la abertura sale **cerrada**. Llega en `warnings`:

> 5 abertura(s) salen CERRADAS: el modelo tiene algo dentro del vano (vidrios,
> carpinterías, paños) y eso ocupa la misma placa que el muro. Para que salgan abiertas,
> descartá esos elementos en el visor o quitalos del modelo. Pared Vertical - Norte #194: 4

Conviene mostrar este aviso en el instructivo, no sólo en la pantalla de planchas: es la
diferencia entre una maqueta con ventanas y una sin ventanas.

---

## 1. Dónde llega

`POST /verificacion` → clave `verificacion` (también viene en `/nesting-preview`).

- Es un objeto (§2) o `null`. `null` significa **"no se pudo verificar"**: estado gris,
  nunca verde.
- Depende de la escala y de cada fusión, descarte o decisión: hay que volver a pedirlo.
- El resultado depende **sólo** del estado enviado, no del orden en que el usuario tocó
  los botones ni de lo que se pidió antes.
- Cambiar de escala, o reiniciar el servidor, vuelve a ser un primer pedido (lento).
- Mientras espera, **no dejar los botones grises sin explicación** (ver §3.1).

`assembly_warnings` e `interferencias` siguen viniendo por compatibilidad y **no hay que
mostrarlos**.

---

## 2. Forma de `verificacion`

```jsonc
{
  "version": 2,
  "escala": 100.0,
  "espesor_placa_mm": 3.0,
  "tolerancias_mm": { "ok": 0.2, "revisar": 0.75 },

  "veredicto": "no_arma",            // "listo" | "revisar" | "no_arma"

  "resumen": {
    "piezas": 29, "piezas_ok": 15, "piezas_revisar": 7, "piezas_no_arma": 7,
    "uniones": 69, "uniones_ok": 63, "uniones_revisar": 2, "uniones_no_arma": 4,
    "problemas": 10,
    "nudos": 6,
    "nudos_por_quien": { "decidir": 3, "escala": 0, "modelo": 3 },
    "terminaciones": 10,
    "resueltas_solas": 10
  },

  "piezas":        { "A9": { ... } },   // §2.1
  "nudos":         [ { ... } ],         // §2.2  ← la lista principal a mostrar
  "problemas":     [ { ... } ],         // §2.3  los mismos problemas, agrupados por causa
  "terminaciones": [ { ... } ],         // §2.4  avisos que no impiden armar
  "uniones":       [ { ... } ],         // §2.5  todas las uniones medidas
  "decisiones_automaticas": [ { ... } ] // §4
}
```

### Estados

| Valor | Significado | Color sugerido |
|---|---|---|
| `ok` | Desvío ≤ 0.2 mm de plancha, o sólo un aviso de terminación | verde |
| `revisar` | Entre 0.2 y 0.75 mm | amarillo |
| `no_arma` | Más de 0.75 mm: las piezas se pisan o no se tocan | rojo |

`veredicto` es el peor estado entre los problemas; `listo` si no hay ninguno. Las
terminaciones no lo cambian.

Titular sugerido, con los campos de `resumen`:

> **206 de 295 piezas verificadas** · 50 lugares para mirar (9 para decidir) · 27 uniones
> resueltas automáticamente · 68 avisos de terminación

### 2.1 `piezas`

Clave: etiqueta de la pieza, la misma que `panel_id` en `placements`.

```json
"A9": { "grupo": 180, "categoria": "wall", "estado": "no_arma",
        "uniones": 5, "problemas": 2, "terminaciones": 0 }
```

- `grupo`: id de grupo. Es la clave de `placements` (como string) y de `groups[].id`.
- `estado`: el peor de sus uniones. Con esto se pinta la pieza en el visor.
- Un grupo que no aparece acá no tiene pieza propia a esta escala (`absorbed_groups`) o
  está descartado. No pintarlo.

### 2.2 `nudos` — la lista principal

Un nudo es un **lugar** donde algo falla: una esquina, una T, un apoyo. Vienen ordenados:
primero los que no arman y, dentro de eso, primero los que se resuelven con una decisión.

```json
{
  "id": 3,
  "estado": "no_arma",
  "quien": "modelo",
  "quien_texto": "No encontramos una forma automática de cerrarlo. Suele venir del modelo 3D: conviene revisarlo en esta zona.",
  "titulo": "A9 · A12: se pisan (0.8 mm)",
  "tipos": ["choque"],
  "peor_mm": 0.8,
  "peor_cm_edificio": 8.0,
  "piezas": ["A9", "A12"],
  "grupos": [180, 245],
  "punto": { "x": 5.1393, "y": 0.6452, "z": 2.946 },
  "uniones": [38],
  "terminaciones": [],
  "problemas": [3],
  "acciones": [
    { "tipo": "descartar", "etiqueta": "No cortar A12", "aplicar": { "overrides": { "245": "discard" } } },
    { "tipo": "cambiar_escala", "etiqueta": "Probar otra escala", "aplicar": null },
    { "tipo": "revisar_modelo", "etiqueta": "Corregir el modelo 3D en esta zona", "aplicar": null }
  ],
  "detalle": { ... }       // §2.2.1
}
```

| Campo | Uso |
|---|---|
| `titulo`, `quien_texto` | Texto listo para mostrar. |
| `quien` | `decidir`, `escala` o `modelo`. Ver tabla de abajo. Sirve para separar la lista en tres bloques. |
| `piezas` | Las piezas con problema en este lugar. Al abrir el nudo: mostrar **sólo éstas** en el 3D, en su posición, y ocultar el resto. |
| `grupos` | Ids de grupo de esas piezas. **Las etiquetas (`A43`) se renumeran** cuando una pieza se fusiona o se descarta; los ids no. Para reconocer el mismo nudo antes y después de una acción, comparar `grupos`, no `piezas`. |
| `punto` | Centro del nudo, en el espacio de `faces_packed` (Y arriba). Llevar la cámara acá. |
| `uniones` | Índices en `verificacion.uniones` de lo que falla acá. |
| `terminaciones` | Índices de uniones de canto en el mismo lugar. Son contexto, no problema. |
| `problemas` | Índices en `verificacion.problemas` relacionados. |
| `peor_mm` / `peor_cm_edificio` | La medida en plancha y en edificio. |
| `acciones` | Botones (§3). |
| `detalle` | Dibujo 2D del nudo (§2.2.1). Puede ser `null` en casos raros: mostrar sólo el 3D. |

`quien`:

| Valor | Qué significa | Qué mostrar |
|---|---|---|
| `decidir` | Hay una acción que el usuario puede tomar y que cambia el resultado (fusionar, descartar, invertir quién cede). | Los botones de `acciones`, arriba de todo. Es lo primero que conviene resolver. |
| `escala` | El defecto sale de cuánto edificio representa la placa a esta escala (muros más gruesos que la placa, piezas más finas que ella). | Si viene `escala_sugerida` en el problema, el botón "Probar a 1:N". |
| `modelo` | El backend ya probó lo que podía y no cierra. | `peor_cm_edificio` y las piezas resaltadas, para pasarle a quien modela. Sin botones de arreglo automático. |

Un nudo es `decidir` si alguna de sus uniones lo es.

#### 2.2.1 `detalle`: el dibujo 2D del nudo

Corte de las placas por un plano perpendicular a la unión, alrededor del nudo. Es para
dibujar en un SVG o un canvas 2D, **sin** depender de la cámara 3D.

```jsonc
{
  "vista": "planta",          // "planta" (unión vertical) | "corte" (unión horizontal)
  "ventana_m": 1.5,           // el dibujo va de -1.5 a +1.5 en los dos ejes
  "espesor_m": 0.3,           // espesor de la placa en metros de edificio
  "ejes": {                   // direcciones del mundo de los ejes del dibujo
    "a": { "x": 1.0, "y": 0.0, "z": 0.0 },
    "b": { "x": 0.0, "y": 0.0, "z": 1.0 }
  },
  "placas": [
    { "pieza": "A9",  "grupo": 180, "estado": "no_arma",
      "poligono": [[-0.0021, -0.15], [-0.0021, 0.15], [1.5, 0.15], [1.5, -0.15]],
      "centro": [0.7489, 0.0] },
    { "pieza": "A11", "grupo": 179, "estado": "contexto",
      "poligono": [[-1.5, -0.0699], [-1.5, 0.2301], [-0.15, 0.2301], [-0.15, -0.0699]],
      "centro": [-0.825, 0.0801] }
  ],
  "solapes": [
    { "piezas": ["A9", "A12"],
      "poligono": [[-0.0021, -0.0699], [0.15, -0.0699], [0.15, -0.15], [-0.0021, -0.15]] }
  ],
  "cotas": [
    { "union": 38, "tipo": "choque", "piezas": ["A9", "A12"],
      "mm": 0.8, "cm_edificio": 8.0, "texto": "0.8 mm",
      "desde": [0.0739, -0.15], "hasta": [0.0739, -0.0699] }
  ]
}
```

Cómo dibujarlo:

- Coordenadas `[a, b]` en **metros de edificio**, con el origen en el centro del nudo. El
  área visible es el cuadrado de lado `2 × ventana_m`.
- `placas`: un polígono relleno por cada placa que pasa por ahí, con su espesor real.
  Color según `estado`; `"contexto"` es una pieza vecina que no tiene problema en este
  nudo: dibujarla en gris.
- `solapes`: lo que dos placas ocupan a la vez. Dibujarlo encima, en rojo o rayado. Es lo
  que hace visible un choque de un vistazo.
- Una luz se ve sola: es el hueco entre dos rectángulos que deberían tocarse.
- `vista: "planta"` se mira desde arriba. En `vista: "corte"` el eje `b` apunta hacia
  arriba del edificio.
- Para rotular en milímetros de plancha: `mm = metros × 1000 / escala`.
- `placas[].centro`: dónde escribir la etiqueta de la pieza. Está dentro de su polígono y
  el backend ya lo corre cuando dos placas montadas caerían en el mismo punto. Usarlo tal
  cual en vez de calcular el centro.
- `cotas`: una por cada unión con problema que se ve en este corte. Dibujar una línea de
  cota de `desde` a `hasta` y un globo con `texto` al lado. Es lo que dice CUÁNTO.
  - `texto` y `mm` salen de la medición exacta de la unión, no de medir el dibujo: no
    recalcularlos a partir de `desde` y `hasta`.
  - En un `choque` o en dos placas montadas, la línea marca el ancho de lo compartido. En
    una `luz`, va de una placa a la otra.
  - Si `desde` y `hasta` coinciden (en este corte las placas se tocan y la luz está en
    otro punto de la unión), dibujar sólo el globo en ese punto.
  - `union` es el índice en `verificacion.uniones`; `cm_edificio` es la misma medida para
    quien corrige el modelo.
- Las piezas vecinas que apenas rozan la ventana no se mandan: lo que llega se puede
  dibujar y rotular sin filtrar nada.

### 2.3 `problemas`

Los mismos problemas, agrupados **por causa** en vez de por lugar (30 muros contra la misma
losa son un solo problema). Misma forma que en la versión 1, con tres campos nuevos:
`quien`, `quien_texto` y `escala_sugerida`.

```json
{ "id": 2, "tipo": "superpuestas", "estado": "no_arma",
  "quien": "decidir", "quien_texto": "Elegí vos: hay más de una forma de resolverlo.",
  "escala_sugerida": null,
  "titulo": "A11 y A22 están montadas una sobre otra (1.4 mm)",
  "detalle": "Son dos piezas paralelas que en el modelo están más cerca que el espesor de la placa: físicamente no hay lugar para las dos. En el edificio son 14 cm.",
  "cantidad": 1, "peor_mm": 1.45, "peor_cm_edificio": 14.5,
  "piezas": ["A11", "A22"], "pieza_comun": null,
  "punto": { "x": 1.1304, "y": 2.0932, "z": 3.1038 },
  "uniones": [50], "acciones": [ ... ] }
```

Usar `nudos` como lista principal. `problemas` sirve para una vista alternativa "por
causa" y para los problemas que no son de un lugar: `pieza_fina` sólo aparece acá.

Tipos de problema: `choque`, `luz`, `superpuestas`, `pieza_fina`.

### 2.4 `terminaciones`

Uniones donde las piezas se tocan sólo de canto. **Arman igual**; queda un canto a la vista
o medio apoyo en el aire. Misma forma que un problema, con `tipo: "sin_cubrir"` y
`estado: "ok"`.

```json
{ "id": 0, "tipo": "sin_cubrir", "estado": "ok",
  "titulo": "A11: 3 piezas la tocan sólo de canto (hasta 3.0 mm)",
  "cantidad": 3, "piezas": ["A11", "A12", "A23", "B5"], "uniones": [44, 46, 48], ... }
```

Mostrarlas en una sección plegada, debajo de los nudos, con un texto del estilo "Avisos de
terminación: no impiden armar". No pintan las piezas de amarillo.

### 2.5 `uniones`

Todas las uniones medidas, incluidas las que encajan. Campos nuevos respecto de v1:
`terminacion` y `quien`.

```json
{ "id": 50, "pieza_a": "A11", "pieza_b": "A22", "grupo_a": 179, "grupo_b": 211,
  "tipo": "superpuestas", "estado": "no_arma", "terminacion": false, "quien": "decidir",
  "medida_mm": 1.45, "medida_cm_edificio": 14.5, "largo_m": 1.525,
  "prevista": false, "pieza_que_ajusta": "A22",
  "punto": { "x": 1.1304, "y": 2.0932, "z": 3.1038 },
  "segmento": [ { "x": ..., "y": ..., "z": ... }, { "x": ..., "y": ..., "z": ... } ] }
```

- `tipo`: `encaja`, `choque`, `luz`, `sin_cubrir`, `superpuestas`.
- `terminacion: true` sólo en `sin_cubrir`; esas uniones llevan `estado: "ok"`.
- `quien`: `null` si la unión está bien.
- `segmento`: los extremos del tramo donde ocurre, en el mundo. Dibujarlo como línea
  gruesa del color del estado.

---

### 2.6 `huecos` — lo que el modelo deja abierto en la maqueta

**Nuevo.** Lista aparte, con su cuenta en `resumen.huecos`. **No cambia el veredicto**: un
modelo con huecos puede dar "listo".

Por qué existe: el veredicto contesta "¿las piezas encajan tal como las une el modelo?".
Si el modelo tiene un muro que no llega a la esquina, para el veredicto esas dos piezas no
tienen por qué tocarse (podría ser un vano) y da "listo". En la maqueta queda el hueco.
Esto lo avisa.

```jsonc
{
  "id": 0,
  "tipo": "hueco",
  "piezas": ["A3", "A5"],        // las dos piezas que casi se tocan
  "grupos": [84, 89],
  "medida_cm_edificio": 10.0,     // separación entre las dos
  "medida_mm": 2.0,               // la misma, en la maqueta
  "largo_cm_edificio": 277.5,     // a lo largo de cuánto
  "punto": {"x": 9.85, "y": 4.2, "z": 0.2},   // dónde, para la cámara
  "texto": "A3 y A5 casi se tocan: queda un hueco de 10 cm (2.0 mm en la maqueta) a lo largo de 278 cm.",
  "pregunta": "¿Tienen que encontrarse? Si es un error del modelo (un muro corto, una esquina abierta, un muro que no apoya), corregilo en el modelo: en la maqueta queda ese hueco."
}
```

Ordenados del más grande al más chico (separación × largo).

Cómo mostrarlo: una sección propia, **después** de los nudos, con el título "Huecos que
quedan en la maqueta" y una línea de explicación: "Puede ser diseño (un vano, una junta) o
un error del modelo. Revisalos antes de cortar". Al tocar uno: el 3D con las dos piezas y
la cámara en `punto`. No hay acciones: se arregla en el modelo.

Qué se avisa y qué no (medido con dos edificios armados con 16 errores a propósito):

| Caso | ¿Se avisa? |
|---|---|
| Modelo correcto | No (0 avisos en todos los modelos de prueba, a 1:50 y 1:100) |
| Muro corto, esquina abierta, tabique que no llega, muro levantado, muro corrido | Sí, si el hueco sobrevive en las piezas |
| Hueco más chico que lo que cubre la placa a esa escala (5 cm a 1:100) | No: en la maqueta no queda hueco |
| Error que el motor corrige solo (piel duplicada, losa unos cm corta) | No: la maqueta sale bien |
| Error que las piezas reproducen tal cual y encajan (un muro corrido hacia adentro, un tabique sobre el hueco de la escalera) | No: no se puede saber si era la idea |
| Hueco de menos de 50 cm de largo | No: son los encuentros de esquina |

En un modelo grande real (`Prueba app.obj`, 295 piezas) da unos 100 avisos: no todos se
pudieron revisar a mano. Conviene mostrarlos plegados con la cuenta en el título.

---

## 3. Acciones

Cada acción tiene `tipo`, `etiqueta` (texto del botón), `aplicar` e `id`.

`id` es un texto estable (`"3f9a1c0b7e2d"`) que identifica **el cambio**: la misma acción
tiene el mismo `id` en cualquier nudo y en cualquier pedido. Es `null` en las acciones
informativas. Sirve para buscar su efecto previsto (§3.2).

`aplicar` trae **las mismas claves que ya se mandan** a `/verificacion`, `/nesting-preview`
y `/generate`. Aplicar es fusionar ese objeto con el estado de la revisión y pedir
`/verificacion` (§0):

| Clave en `aplicar` | Cómo se fusiona |
|---|---|
| `merges: [[a, b]]` | Agregar el conjunto a `merges`. |
| `overrides: {"245": "discard"}` | Mezclar en `overrides` (claves: id de grupo como string). |
| `wall_wall_decisions_by_pair: [{group_a, group_b, yield_group_id}]` | Agregar a la lista; si ya hay una entrada para ese par, reemplazarla. |
| `scale_denom: 75` | **Nuevo.** Cambiar la escala de la revisión a ese valor. |

Tipos de acción:

| `tipo` | `aplicar` | Qué hace |
|---|---|---|
| `fusionar` | `merges` | Las dos piezas se cortan como una. |
| `descartar` | `overrides` | No se corta esa pieza. |
| `invertir_cede` | `wall_wall_decisions_by_pair` | Se acorta la otra pieza. **Sólo se ofrece si el backend no lo probó ya**: si no aparece, es que invertir no mejora. |
| `cambiar_escala` | `scale_denom` o `null` | Con valor: botón "Probar a 1:N". Con `null`: llevar al selector de escala. (En v1 se llamaba `escala_mas_fina`; ya no se emite.) |
| `revisar_modelo` | `null` | Informativa: mostrar la medida en cm y las piezas. No modifica nada. |

Las acciones que llegan son las que **sirven**:

- `fusionar` sólo se ofrece si la fusión se va a aplicar. (Antes se ofrecía para tramos
  separados por una losa, el backend la rechazaba y el problema volvía igual.)
- `invertir_cede` sólo se ofrece en uniones que el backend **no** evaluó. Si no aparece,
  invertir ya se probó y no mejora, o es una unión que el backend ya dejó invertida.
- Una acción puede igual no mejorar el conjunto, o empeorarlo. El backend ahora la
  **prueba en segundo plano** y avisa qué va a pasar antes de que el usuario la toque
  (§3.2).

### 3.1 Mientras se recalcula

Aplicar una acción dispara un pedido que puede tardar varios segundos (§0). En ese lapso:

1. Marcar **el botón tocado** con un indicador de progreso ("Recalculando…").
2. Deshabilitar los demás botones de acción, pero con el mismo indicador visible en la
   cabecera de la lista: que se entienda que el sistema está trabajando y no trabado.
3. Si el pedido **falla o vence**, volver a habilitar todo y mostrar el error. No dejar
   los botones grises.
4. Cuando llega la respuesta, **reemplazar** `verificacion` entera. El nudo puede haber
   desaparecido, cambiado de estado o seguir igual.
5. Si el nudo sigue igual (mismos `grupos`, mismo `estado`), decirlo: "No cambió nada con
   esa acción". Es mejor que dejar a la persona mirando una pantalla idéntica.

Las etiquetas ya **no** se renumeran después de fusionar o descartar (§0.2).

### 3.2 Efecto previsto de cada botón

Apenas responde `/verificacion`, el backend empieza a calcular en segundo plano qué pasa
con cada acción de los nudos **"para decidir"** (no las de cambiar de escala). Dos
consecuencias:

1. **El clic es inmediato**: cuando el usuario toca un botón cuyo efecto ya se calculó,
   `/verificacion` responde en menos de 0.3 s (`servidor.ya_calculada: true`).
2. **El botón puede decir qué va a pasar** antes de tocarlo.

Los efectos llegan en `efectos` dentro de la respuesta de `/verificacion` (los que ya
estén) y completos en:

`POST /api/verificacion/efectos` — mismo cuerpo que `/verificacion`. No calcula nada,
responde al instante. Consultar cada 1 a 2 s hasta que `listo` sea `true`.

```jsonc
{
  "listo": false,          // true = ya están todos
  "calculadas": 7,
  "total": 18,
  "efectos": {
    "3f9a1c0b7e2d": {      // clave: accion.id
      "resultado": "mejora",        // "mejora" | "igual" | "empeora" | "desconocido"
      "este_lugar": "resuelto",     // "resuelto" | "sigue" | null
      "texto": "Resuelve este lugar y 1 otro más",   // listo para mostrar
      "veredicto": "no_arma",       // veredicto que quedaría
      "uniones_no_arma": -1,        // diferencias contra el estado actual
      "uniones_revisar": 0,
      "nudos": -2,
      "piezas_ok": 2
    }
  }
}
```

Si llega `"sin_sesion": true`, todavía no se pidió `/verificacion` para ese modelo (o el
servidor se reinició): pedirla.

Cómo mostrarlo, debajo o al lado de cada botón:

| `resultado` | Sugerencia |
|---|---|
| `mejora` | Texto en verde: "Resuelve este lugar". Si hay una sola acción que mejora, destacarla. |
| `igual` | Texto gris: "No cambia el resultado…". |
| `empeora` | Texto en rojo y botón atenuado (no deshabilitado): "Empeora: 2 uniones más no arman". |
| sin entrada todavía | Nada, o un punto de carga. El botón funciona igual. |

En el modelo de 295 piezas son 18 acciones y terminan de calcularse en unos 90 s; se van
completando de a una, en el orden de la lista de nudos. Si el usuario toca un botón antes,
el backend interrumpe lo que estaba previendo, atiende el clic y sigue con el estado nuevo.

Ejemplo real de ese modelo: en un nudo, "Cortarlas como una sola pieza" **empeora** (2
uniones más no arman) y "No cortar A176" **resuelve ese lugar y otro más**. Sin este dato
el usuario no tiene cómo saberlo.

### Antes y después

Cambiar quién cede sólo mueve un extremo de dos piezas, así que su efecto es local. Fusionar,
descartar y cambiar de escala afectan todo. Para que el usuario no tema "arreglo acá y rompo
allá", después de aplicar una acción comparar el `resumen` nuevo con el anterior y mostrarlo:

> Resolvió 3 lugares · no apareció ninguno nuevo

Alcanza con restar `resumen.nudos` y `resumen.uniones_no_arma` de antes y después (es lo
mismo que anticipa `efectos`). **Deshacer** es volver a mandar el estado anterior a
`/verificacion`: el backend lo tiene guardado y responde al instante.

---

## 4. Quién cede

### 4.1 Lo que el backend decide solo: `decisiones_automaticas`

Donde la decisión sugerida deja una unión pisándose o con luz, el backend prueba la
contraria y se queda con la que mide mejor. Nunca cambia una decisión del usuario.

```json
"decisiones_automaticas": [
  { "group_a": 174, "group_b": 232, "yield_group_id": 174 }
]
```

- Son las decisiones **que se cortan**, y difieren de `suggested_yield_group_id` de
  `wall_wall_joints`.
- El selector de "quién cede" tiene que mostrar, para cada unión, en este orden: la
  decisión del usuario si la hay; si no, la de `decisiones_automaticas`; si no, la sugerida.
- Conviene marcarlas ("resuelta automáticamente") para que se entienda por qué no coincide
  con la sugerencia.
- `resumen.resueltas_solas` es la cantidad.

### 4.2 Lo que decide el usuario: `wall_wall_decisions_by_pair`

Aceptado en `/recompute`, `/nesting-preview` y `/generate`:

```json
"wall_wall_decisions_by_pair": [
  { "group_a": 179, "group_b": 245, "yield_group_id": 245 }
]
```

- Reemplaza a `wall_wall_decisions` (por índice de junta), que se pierde con cada fusión,
  división o cambio de escala. El formato viejo se sigue aceptando; si llegan los dos para
  la misma unión, manda el del par.
- Una unión decidida por el usuario queda fija: el backend no la vuelve a tocar y, si
  falla, el nudo vuelve como `quien: "decidir"`.
- Una decisión que ya no corresponde a ninguna unión vuelve como texto en `warnings`.

---

## 5. Bloqueo al generar

Sin cambios respecto de v1. `POST /generate` acepta `bloquear_si_no_arma` y `forzar`
(ambos `false` por defecto). Si se pide bloquear y el veredicto es `no_arma`, responde
**HTTP 409**:

```json
{ "code": "ensamble_no_arma",
  "message": "La maqueta tiene uniones que no arman. Revisalas o generá igual enviando forzar=true.",
  "verificacion": { ... } }
```

Flujo: mandar siempre `bloquear_si_no_arma: true`; ante un 409 mostrar los nudos `no_arma`
con dos salidas, "Revisar" y "Generar igual" (reenvía con `forzar: true`).

---

## 6. Probeta de calibración

Sin cambios. `POST /probeta`, sin modelo. Devuelve `zip_base64`, `nesting` (misma forma que
`wall_nesting`) e `instrucciones` con `pasos` y `lectura` para mostrar tal cual.

Límite actual: el resultado de la probeta todavía no se puede cargar; espesor y holgura son
constantes del backend.

---

## 7. Otros campos que ya llegan

| Campo | Dónde | Qué es |
|---|---|---|
| `absorbed_groups` | `/verificacion`; `/nesting-preview`; `topology` de `/recompute` con `scale_denom` | `{ "id absorbido": id del grupo que lo incluye }`, por fusión automática **o del usuario** (§0.3). No dibujarlo aparte ni caer a la malla cruda. |
| `warnings` | `/nesting-preview` | Avisos de la generación a esa escala. |
| `panel_id_by_group` | `/verificacion` (usar éste después de una acción, §0.2); `/recompute` | Calculado a la escala pedida. Las etiquetas cambian con la escala. |

---

## 8. Qué construir, en orden

0. **Cambiar el pedido de las acciones a `/verificacion` y subir el tiempo de espera a
   60 s (§0).** Sin esto nada de lo demás se ve.
1. **Titular** con `resumen` (texto de §2).
2. **Lista de nudos** en tres bloques por `quien`: "Para decidir", "Depende de la escala",
   "Para corregir en el modelo". Cada fila: `titulo`, medida y botones.
3. **Ficha de nudo** al tocar una fila: a la izquierda el 3D con sólo `piezas` visibles y
   la cámara en `punto`; a la derecha el `detalle` en 2D. Debajo, `quien_texto` y
   `acciones`.
4. **Antes y después** al aplicar una acción, con Deshacer (§3), y el **efecto previsto**
   en cada botón (§3.2).
5. **Terminaciones** en sección plegada.
6. **Selector de quién cede** leyendo `decisiones_automaticas` (§4.1).
7. **Pintar las piezas** del visor general por `piezas[etiqueta].estado`.
8. Opcional: la plancha coloreada por estado como índice. Tocar una pieza en la plancha
   abre el primer nudo de `nudos` que la contenga.

---

## 9. Lo que el veredicto no cubre

- Mide con el espesor **nominal** de la placa. Si el MDF real es otro, lo dice la probeta.
- Los umbrales (0.2 y 0.75 mm) están puestos a criterio, sin calibrar contra maquetas
  cortadas.
- El resolvedor sólo prueba invertir quién cede. No prueba fusionar ni descartar: eso
  cambia qué piezas existen y es decisión de la persona.
- La memoria entre pedidos vive en el proceso del servidor. Con más de un proceso, o tras
  un reinicio, un pedido puede caer sin memoria y tardar como el primero. El resultado
  puede variar levemente entre un pedido con memoria y uno sin ella.
- `quien: "modelo"` quiere decir "no encontramos un arreglo automático". A veces la causa
  es el modelo y a veces un límite del procesamiento; el backend no los distingue todavía.
- `quien: "escala"` y `escala_sugerida` salen de una cuenta, no de haber probado esa
  escala: a la escala sugerida ese problema deja de existir, pero pueden aparecer otros.
- Un `listo` dice que las piezas encajan entre sí. No dice que el modelo 3D represente
  bien el edificio.
