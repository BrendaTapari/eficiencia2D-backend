# Contrato backend → front: verificación de la maqueta (versión 2)

Estado del backend: implementado y con tests (`test_verificacion.py`, `test_corpus.py`).

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

## 1. Dónde llega

`POST /nesting-preview` → clave `verificacion`.

- Es un objeto (§2) o `null`. `null` significa **"no se pudo verificar"**: estado gris,
  nunca verde.
- Depende de la escala y de cada fusión, descarte o decisión: hay que volver a pedirlo.
- **Tarda más que antes** en modelos grandes, porque ahora el backend prueba decisiones
  por su cuenta. En un modelo de 300 piezas: unos 20 s (antes 7). Mostrar un estado de
  "verificando" y no bloquear el resto de la pantalla.

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

> **206 de 295 piezas verificadas** · 50 lugares para mirar · 27 uniones resueltas
> automáticamente · 68 avisos de terminación

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
      "poligono": [[-0.0021, -0.15], [-0.0021, 0.15], [1.5, 0.15], [1.5, -0.15]] },
    { "pieza": "A11", "grupo": 179, "estado": "contexto",
      "poligono": [[-1.5, -0.0699], [-1.5, 0.2301], [-0.15, 0.2301], [-0.15, -0.0699]] }
  ],
  "solapes": [
    { "piezas": ["A9", "A12"],
      "poligono": [[-0.0021, -0.0699], [0.15, -0.0699], [0.15, -0.15], [-0.0021, -0.15]] }
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
- Escribir la etiqueta de cada pieza dentro de su rectángulo.

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

## 3. Acciones

Cada acción tiene `tipo`, `etiqueta` (texto del botón) y `aplicar`.

`aplicar` trae **las mismas claves que ya se mandan** a `/recompute`, `/nesting-preview` y
`/generate`. Aplicar es fusionar ese objeto con el estado de la revisión y volver a pedir:

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

### Antes y después

Cambiar quién cede sólo mueve un extremo de dos piezas, así que su efecto es local. Fusionar,
descartar y cambiar de escala afectan todo. Para que el usuario no tema "arreglo acá y rompo
allá", después de aplicar una acción comparar el `resumen` nuevo con el anterior y mostrarlo:

> Resolvió 3 lugares · no apareció ninguno nuevo

Alcanza con restar `resumen.nudos` y `resumen.uniones_no_arma` de antes y después. Guardar
el estado anterior permite ofrecer **Deshacer** sin pedir nada al backend.

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
| `absorbed_groups` | `/nesting-preview`; `topology` de `/recompute` con `scale_denom` | `{ "id absorbido": id del grupo que lo incluye }`. No dibujarlo aparte ni caer a la malla cruda. |
| `warnings` | `/nesting-preview` | Avisos de la generación a esa escala. |
| `panel_id_by_group` | `/recompute` | Calculado a la escala pedida. Las etiquetas cambian con la escala. |

---

## 8. Qué construir, en orden

1. **Titular** con `resumen` (texto de §2).
2. **Lista de nudos** en tres bloques por `quien`: "Para decidir", "Depende de la escala",
   "Para corregir en el modelo". Cada fila: `titulo`, medida y botones.
3. **Ficha de nudo** al tocar una fila: a la izquierda el 3D con sólo `piezas` visibles y
   la cámara en `punto`; a la derecha el `detalle` en 2D. Debajo, `quien_texto` y
   `acciones`.
4. **Antes y después** al aplicar una acción, con Deshacer (§3).
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
- `quien: "modelo"` quiere decir "no encontramos un arreglo automático". A veces la causa
  es el modelo y a veces un límite del procesamiento; el backend no los distingue todavía.
- `quien: "escala"` y `escala_sugerida` salen de una cuenta, no de haber probado esa
  escala: a la escala sugerida ese problema deja de existir, pero pueden aparecer otros.
- Un `listo` dice que las piezas encajan entre sí. No dice que el modelo 3D represente
  bien el edificio.
