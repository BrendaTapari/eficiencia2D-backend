# Auditoría y propuesta: verificación interactiva

Fecha: 14/11. Para aprobar o rechazar por ítem (sección 6). Fecha límite: 18/11.

> **Estado al 15/11: aprobado e implementado en el backend** (A, B, C, E, G y H). Ver la
> sección 8 al final. D quedó para después del 18/11 y F es sólo de front.

## 1. Qué pasó

Cada vez que tocás un botón, el front vuelve a pedirle al backend el estado completo del
modelo. Ese pedido tarda más de lo que el front espera, y por eso aparece "El backend no
respondió a tiempo" y el cambio nunca se ve.

Medí el recorrido real, de punta a punta, sobre el modelo de tus capturas
(`Prueba app.obj`, 295 piezas, 1:100), levantando la API acá:

| Pedido | Tiempo | Tamaño de la respuesta |
|---|---|---|
| `/upload` | 33 s | 121 MB |
| `/nesting-preview`, primera vez | 26 s | 1.9 MB |
| `/nesting-preview`, después de un clic | 16 s | 1.9 MB |
| `/recompute` con escala | 27 s | 121 MB |

Si después de un clic el front pide `/recompute` y `/nesting-preview` (no puedo ver su
código, pero es lo que hace su flujo normal cuando cambian fusiones o descartes), son
**43 segundos** como mínimo, porque el servidor atiende de a un pedido por vez.

### Mi error

Te dije que después de un clic tardaba "entre 7 y 10 segundos". Ese número era el del
cálculo interno, medido llamando a la función directamente. Nunca había levantado la API
para medir el pedido real. La diferencia son 9 segundos por pedido que no vi, más
`/recompute` entero.

## 2. En qué se van los 16 segundos de un clic

Desglose de `/nesting-preview` después de una acción (tiempos reales aproximados):

| Etapa | Tiempo | ¿Hace falta en cada clic? |
|---|---|---|
| Leer el modelo desde el disco | 5 s | No. Es el mismo modelo que en el pedido anterior. |
| Rearmar las 295 piezas | 5 s | En parte. Con "no cortar A262" cambian 45; las otras 250 dan idéntico. |
| Fusiones y detección de juntas | 1.2 s | En parte. |
| Medir las uniones | 1 s | Sí, pero ya es incremental. |
| Verificador viejo por muestreo | 0.7 s | No. Está obsoleto; se calcula y el front no debería usarlo. |
| Acomodar las piezas en planchas | 1.5 s | No. Al usuario que está resolviendo un nudo no le importa la plancha. |
| Armar y comprimir la respuesta | 1 s | En parte. |

Y `/recompute` (27 s):

| Etapa | Tiempo | ¿Hace falta en cada clic? |
|---|---|---|
| Leer el modelo desde el disco | 5 s | No. |
| Armar las piezas **dos veces** (una para dibujarlas, otra para las etiquetas) | 14 s | No. Es el mismo cálculo que ya hizo `/nesting-preview`. |
| Volver a codificar las 560.000 caras del modelo | 6 s | No. Las caras no cambian con una acción. |
| Comprimir y enviar 121 MB | 2 s | No. |

Conclusión: de los 43 segundos, lo que de verdad cambia con un clic se puede calcular en
unos 3.

## 3. Auditoría: lo que encontré, por gravedad

### Bloquean el uso

1. **Tiempo de respuesta después de una acción** (sección 2). Es el problema que viste.
2. **No hay un pedido pensado para "apliqué una acción"**. Se reusan los dos pedidos más
   pesados del sistema, que devuelven todo: geometría, planchas, etiquetas.
3. **El servidor atiende un pedido por vez.** Los cálculos corren dentro del hilo que
   atiende la red. Mientras calcula, cualquier otro pedido espera; dos pestañas o dos
   usuarios se bloquean entre sí.
4. **No hay pruebas de punta a punta.** Las 206 pruebas ejercitan el cálculo, ninguna el
   pedido HTTP. Por eso este problema llegó hasta vos. Es una falla de proceso mía.

### Hacen que no se pueda confiar

5. **Una acción puede empeorar y el botón no avisa.** De 6 fusiones que probé, 5 mejoran y
   1 deja el modelo peor (24 → 26 uniones que no arman). El backend no prueba la acción
   antes de ofrecerla.
6. **"Para corregir en el modelo" es un cajón.** Son 41 de 50 nudos en ese modelo. Quiere
   decir "no encontré arreglo automático", y no distingue si la causa es el modelo o un
   límite del procesamiento. Al usuario no le da nada para hacer.
7. **Las etiquetas se renumeran** al fusionar o descartar (`A84` pasa a ser `A83`). El
   usuario pierde la referencia de lo que estaba mirando.
8. **La memoria entre pedidos vive en el proceso.** Se pierde al reiniciar el servidor, y
   con `--reload` se reinicia cada vez que cambia un archivo.

### Deuda conocida, no urgente para el 18/11

9. La probeta no se puede cargar: espesor y holgura son constantes del código.
10. Los umbrales (0.2 y 0.75 mm) no están calibrados contra maquetas cortadas.
11. Quedan muros sin partir al nivel del entrepiso (pendiente en `mesh_splitter`).
12. `/upload` devuelve 121 MB. Es lento, pero pasa una sola vez por modelo.

## 4. Propuesta

Cambio de enfoque: **una acción del usuario no es "recalcular el modelo", es un cambio
chico sobre un modelo que el servidor ya tiene armado.**

### A. Un pedido propio para las acciones — `POST /api/verificacion`

Recibe el mismo estado de revisión que ya manda el front. Devuelve sólo lo que cambia con
una decisión: `verificacion`, `placements`, `panel_id_by_group`, `absorbed_groups` y
`warnings`. Sin caras, sin planchas, sin el verificador viejo.

El front lo llama después de cada acción **en lugar de** `/recompute` y
`/nesting-preview`. Los otros dos quedan para lo que son: cargar el modelo y armar las
planchas.

### B. El modelo queda en memoria

El servidor conserva en memoria los últimos modelos usados en vez de releerlos del disco
en cada pedido. Ahorra 5 segundos por pedido en los tres endpoints.

### C. Recalcular sólo lo afectado

Cada pieza recuerda con qué datos se calculó; si esos datos no cambiaron, se reusa.

Lo probé en un prototipo fuera del repo, sobre el mismo modelo y con resultado idéntico al
cálculo completo:

| Acción | Hoy | Prototipo | Piezas recalculadas |
|---|---|---|---|
| No cortar una pieza | 6.0 s | 3.3 s | 45 de 294 |
| Fusionar dos piezas | 9.5 s | 6.2 s | 164 de 588 (arma dos veces) |

Esos tiempos son del cálculo. Sumando A y B, el pedido completo después de un clic debería
quedar en **4 a 7 segundos** en este modelo. Es una estimación a partir de mediciones
separadas; el número real lo voy a tener cuando esté armado.

Repitiendo el mismo pedido, sin recalcular ninguna pieza, el prototipo tarda 3.1 s: son
fusiones, juntas y proyecciones que hoy se rehacen siempre. Se pueden bajar con el mismo método, pero no lo
medí.

### D. El pedido no puede vencer — trabajo en segundo plano

Aunque quede rápido, el primer cálculo de un modelo grande va a seguir tardando 15 a 25
segundos. Para que **nunca** aparezca "no respondió a tiempo":

- El pedido responde enseguida con un identificador.
- El front consulta el avance cada segundo y muestra una barra con la etapa ("Armando
  piezas… 180 de 295").
- El cálculo corre fuera del hilo de red, así que otros pedidos no se bloquean.

Es el cambio más grande para el front. Alternativa más barata: dejar el pedido como está y
que el front suba su tiempo de espera a 60 segundos. Resuelve el síntoma, no el bloqueo.

### E. Cada botón dice qué va a pasar antes de tocarlo

Después de calcular el reporte, el servidor prueba en segundo plano cada acción de los
nudos "para decidir" (en ese modelo son 9 nudos, unas 20 acciones) y guarda el resultado.
El botón pasa a decir:

> **Cortarlas como una sola pieza** — resuelve este lugar
> **No cortar A263** — resuelve este lugar y otro más
> ~~Cortarlas como una sola pieza~~ — empeora: aparecen 2 problemas nuevos

Y al tocarlo el resultado ya está calculado: la respuesta es inmediata.

Resuelve el hallazgo 5 y cambia la sensación de la herramienta. Depende de C; sin C serían
2 a 3 minutos de cálculo en segundo plano, y con C alrededor de uno.

### F. Decisiones en lote

Permitir marcar una decisión en varios nudos y aplicarlas juntas con un solo cálculo. Es
un cambio sólo de front; el backend ya acepta varias fusiones y descartes en un pedido.
Sirve como red de seguridad si E no llega.

### G. Etiquetas que no cambian

Que una pieza conserve su etiqueta aunque otras desaparezcan (queda un hueco en la
numeración en vez de correrse todo). Resuelve el hallazgo 7.

Riesgo: las etiquetas también van grabadas en la plancha. Hay que mantener una sola
numeración para las dos cosas.

### H. Pruebas de punta a punta con presupuesto de tiempo

Pruebas que levantan la API, suben un modelo y ejecutan una acción, y que **fallan si el
pedido supera un tiempo máximo**. Es lo que hubiera detectado esto antes de que lo vieras.

## 5. Plan hasta el 18/11

| Orden | Qué | Dónde | Depende de |
|---|---|---|---|
| 1 | H: pruebas de punta a punta | backend | — |
| 2 | B: modelo en memoria | backend | — |
| 3 | A: `POST /api/verificacion` + contrato | backend, y front para usarlo | — |
| 4 | C: recalcular sólo lo afectado | backend | — |
| 5 | G: etiquetas estables | backend | — |
| 6 | E: efecto previsto de cada botón | backend, y front para mostrarlo | C |
| 7 | D: trabajo en segundo plano con avance | backend y front | A |
| 8 | F: decisiones en lote | front | — |

Los pasos 1 a 5 son de backend y los puedo hacer en esta sesión en cuanto apruebes. El
resultado visible depende de que el front cambie a qué pedido llama (paso 3): sin eso,
nada de lo demás se nota.

### Criterio de aceptación que propongo

Sobre `Prueba app.obj` a 1:100, medido de punta a punta con la API levantada:

- Después de un clic, la respuesta llega en **menos de 8 segundos**.
- Ningún pedido de acción devuelve más de **3 MB**.
- Una prueba automática falla si se supera cualquiera de las dos.
- Tocar un botón nunca deja la pantalla sin cambio y sin explicación.

## 6. Lo que necesito que decidas

| # | Decisión | Mi recomendación |
|---|---|---|
| 1 | ¿Apruebo A, B, C y H (pedido propio, memoria, recálculo parcial, pruebas)? | Sí. Es el arreglo del problema que viste. |
| 2 | ¿D completo (segundo plano con avance) o sólo subir el tiempo de espera del front? | Subir el tiempo de espera ahora; D después del 18/11. D toca mucho front. |
| 3 | ¿E (efecto previsto en cada botón)? | Sí, después de C. Es lo que más cambia la experiencia. |
| 4 | ¿G (etiquetas estables)? | Sí. |
| 5 | ¿F (lote)? | Sólo si E no llega. |

## 7. Lo que no puedo prometer

- **No puedo ver ni tocar el front.** Todo lo que dependa de él necesita que alguien lo
  implemente o que conectes ese repo a la sesión. Si el front no cambia de pedido, el
  problema sigue igual aunque el backend quede rápido.
- **No sé cuál es el tiempo de espera del front** ni si después de un clic pide uno o dos
  endpoints. El diagnóstico de los 43 segundos supone dos.
- **Los tiempos son de este entorno.** Tu notebook puede ser más rápida o más lenta.
- **El primer cálculo de un modelo grande va a seguir tardando** 15 a 25 segundos. Lo que
  baja es cada clic posterior.
- **C es el cambio más delicado**: reusar una pieza que debía recalcularse daría un
  resultado mal sin ningún aviso. Va con una prueba que compara, en cada modelo, el
  resultado parcial contra el completo.

## 8. Resultado (15/11)

Hecho en el backend, con la batería completa en verde (215 pruebas, 9 de ellas de punta a
punta por HTTP):

| Ítem | Qué quedó |
|---|---|
| A | `POST /api/verificacion`: responde sólo lo que cambia con una decisión (1 MB). |
| B | El modelo queda en memoria entre pedidos (`MODELOS_EN_MEMORIA`, 1 por defecto). |
| C | Después de una acción se recalculan sólo las piezas afectadas. |
| E | El efecto de cada botón se calcula en segundo plano; el clic previsto es inmediato. `POST /api/verificacion/efectos`. |
| G | Las etiquetas no se renumeran al fusionar o descartar; la plancha usa las mismas. |
| H | `test_api.py`: sube modelos, toca botones y falla si un clic tarda más de 8 s o pesa más de 3 MB. |

Medido de punta a punta, mismo modelo (`Prueba app.obj`, 295 piezas, 1:100):

| Pedido | Antes | Ahora |
|---|---|---|
| Después de un clic (lo que hacía el front: `/recompute` + `/nesting-preview`) | 43 s, 123 MB | — |
| `/verificacion` después de un clic ya previsto | — | menos de 0.3 s, 1 MB |
| `/verificacion` después de un clic no previsto | — | 4 a 8 s, 1 MB |
| `/verificacion` volviendo atrás (deshacer) | — | menos de 0.3 s |
| `/nesting-preview` después de un clic | 16 s | 5 s |
| `/recompute` | 27 s | 18 a 23 s |

### Lo que las pruebas nuevas encontraron en mi propio código

Las pruebas de punta a punta se pagaron solas. Detectaron cuatro errores que con las
pruebas anteriores no se veían, todos corregidos:

1. **Un cálculo interrumpido se guardaba como bueno.** Al cancelar el trabajo en segundo
   plano, un `except` genérico lo dejaba terminar con un resultado a medias.
2. **Dos piezas fusionadas distintas se confundían.** La memoria reconocía las piezas por
   número, y dos fusiones distintas reciben el mismo número. Ahora se reconocen por su
   geometría.
3. **El resultado dependía del orden de los clics.** La "pista" que aceleraba al
   resolvedor (ya estaba en producción) podía dejar 2 piezas verificadas menos según de
   qué estado se viniera. La quité: ahora el resultado depende sólo de lo que el usuario
   decidió, y la velocidad sale del recálculo parcial.
4. **Dos cálculos a la vez se pisaban.** Una variable compartida hacía que un cálculo
   leyera las caras del modelo del otro. Ya existía; con el trabajo en segundo plano se
   hubiera manifestado. Ahora es por hilo.

### Lo que NO quedó resuelto

- **El front tiene que cambiar a `/verificacion`** (CONTRATO_verificacion.md §0). Sin eso
  el usuario sigue viendo lo mismo que antes.
- **El clic no previsto está cerca del límite.** 4 a 8 s en este entorno; una fusión
  encadenada inmediatamente después de otra acción llegó a 7.9 s con la máquina cargada.
  En una máquina más lenta puede pasar de 8 s. El tiempo de espera del front tiene que
  subir igual.
- **La memoria es del proceso.** Se pierde al reiniciar, y si el servidor corre con varios
  procesos o instancias, un pedido puede caer en uno que no tiene la sesión: responde bien
  pero lento. Para que rinda, un solo proceso (o afinidad de sesión).
- **Memoria RAM:** con el modelo grande el proceso llegó a 1.4 a 1.6 GB (el pico es la
  subida, que ya era así). Si la instancia tiene menos, bajar `SESIONES_EN_MEMORIA` a 1.
- **`/upload` y `/recompute` siguen devolviendo 121 MB.** Son una vez por modelo.
- **D (avance en segundo plano), el cajón "para corregir en el modelo", la probeta y la
  calibración de umbrales** siguen pendientes, como estaba previsto.
