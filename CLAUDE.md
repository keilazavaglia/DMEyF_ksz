# Contexto general

Soy estudiante de una maestría en ciencias de datos. Mi nivel de Python es cero: estoy empezando.
Trabajo en Windows, con notebooks de Jupyter (.ipynb) dentro de Visual Studio Code.
Mi carpeta de trabajo es `py`, en el Escritorio. Cada materia tiene su propia subcarpeta.

# Cómo quiero que me ayudes

- Respondé siempre en español.
- Sos mi asistente y mi maestro: quiero entender el código, no solo tenerlo.
- Antes de darme código, explicame la idea en pocas líneas y con un ejemplo simple.
- Comentá el código para que pueda leerlo y entender qué hace cada parte.
- Si hay un error, explicame por qué ocurre y cómo se arregla, no solo la corrección.
- Cuando sea posible, hacé cambios pequeños y decime qué cambió, en vez de reescribir archivos enteros.
- Si algo de mi pedido es ambiguo, preguntame antes de hacer mucho trabajo.
- Tengo poco tiempo: priorizá lo eficiente. Explicaciones cortas, no un curso completo.
- Decidí vos quién corre cada cosa y avisame: corro yo lo nuevo o lo que conviene entender
  (en versión corta); corrés vos (Claude) lo repetitivo o largo (semillas, Optuna,
  comparaciones), y me mostrás el código y un resumen.
- **Siempre avisame antes de correr o modificar algo** (qué vas a hacer y cuánto puede
  tardar) y esperá mi OK. Leer archivos que te nombré no necesita aviso.

# Cuidado del consumo

- Podés listar nombres de archivos y carpetas, pero abrí solo los que te nombre; no explores todo el proyecto sin necesidad.
- Los notebooks pesan mucho por las salidas guardadas: si necesitás leer uno, pedime que limpie las salidas primero.
- Al terminar una sesión, si te lo pido, actualizá la sección "Estado actual" de este archivo.

# Objetivo actual

- Primera competencia: predecir qué clientes reciben estímulo de retención en **202108**.
- **Cierre: domingo 11/10/2026, 23:59 (hora Buenos Aires).** Detalles en `CONSIGNA_COMPETENCIA.md`.
- Las clases (carpetas 2 a 7) no se entregan: son herramientas para la competencia.
- Datos: **usar `py/competencia_01.parquet`** (`pd.read_parquet`), creado el 07/10/2026 a partir
  de `competencia_01.csv` (original, no tocar; queda de respaldo). 983.061 filas, 155 columnas.
  Cambios respecto del CSV:
  - `clase_ternaria` recalculada con la definición estricta (BAJA+1 = no está el mes siguiente;
    BAJA+2 = está el siguiente pero no el subsiguiente). 29 filas pasaron de CONTINUA a BAJA
    (clientes que se van y vuelven). Coincide con los números de los compañeros.
  - `ccajas_depositos` = NA solo en 202105 (estaba todo en cero; recomendación del profesor).
- Diccionario de variables: `py/diccionario.ods`.
- Resumen teórico de las clases: `py/RESUMEN_TEORICO.md`.
- Top 40 de variables ya seleccionadas en la clase 5: `clusters/variables_seleccionadas.txt`.
- Ganancia (z601): +1.072.500 por cada BAJA+2 estimulado, -27.500 por cada estímulo a un no BAJA+2.
  Umbral 0,025 válido solo si se entrena con BAJA+2 sola; si se juntan BAJA+1 y BAJA+2,
  cortar por cantidad de clientes (top N), no por probabilidad.

# Estado actual

## z301 (incertidumbre): corrido y con resultados válidos
- Semilla 501419: Optuna 100 ensayos, mejor 305,9 M (max_depth=8, min_samples_leaf=10,
  min_samples_split 182-197, pegado al límite 200). Ganancia en mayo: 193 M.
- Semilla 623353 (StratifiedShuffleSplit de 20): de 15 pares, 3 inversiones de ranking
  sobre 10 comparables (30%). Diferencias < 0,5 M en validación: modelos indistinguibles.
- Lección: el leaderboard público es ruidoso; usar varias semillas.
- Errores conocidos (no invalidan resultados): "/0.3" duplicado en las medias impresas,
  textos "Abril" que en realidad son mayo, model_base/model_ale sin random_state.

## Errores conocidos en otros notebooks
- z401: el archivo "_1000.sav" guarda el modelo de 100 árboles.
- z402: los select no agregan columnas; usar create or replace table antes del COPY.
- Optuna con load_if_exists=True suma ensayos: cambiar el nombre del estudio al cambiar la semilla.
- z601: **falta `bagging_freq`** → bagging_fraction no hace nada (LightGBM lo apaga con
  bagging_freq=0). Si se usa bagging_fraction, poner `bagging_freq = 1` (fijo; subirlo si va lento).
- z601: el entrenamiento usa 250 árboles fijos con `#best_iter` comentado. En el modelo final
  usar el best_iter real (`study.best_trial.user_attrs["best_iter"]`).
- z601: la celda `top_n = 3` reentrena los 3 mejores trials y se elige a mano (el 1° de Optuna
  no siempre es el mejor). Conviene adoptarlo.
- Juntar clases (BAJA+1 + BAJA+2): la ganancia personalizada dentro de LightGBM solo ve 0/1 y no
  distingue BAJA+1 (costo) de BAJA+2 (ganancia). Salidas: pesos (`weight`) distintos por clase
  (un hack), o entrenar/predecir a mano (sin `lgb.cv`) y calcular la ganancia afuera con
  clase_ternaria; para early stopping usar AUC. (Con la receta del profesor, métrica AUC y
  ganancia calculada afuera como en z002, el problema no aparece.)

## Indicaciones de la tutoría (05/10/2026) — SEGUIR ESTO
- **No sacar `numero_de_cliente`**: se usa como variable (z002 todavía lo saca: corregir).
- **No sacar marzo** del entrenamiento.
- **Feature engineering**: llegar a entre **1000 y 1500 atributos** (lags, deltas, rankings, etc.).
- **Optuna**: 30 a 50 trials.
- **Muchas semillas** para eliminar la variabilidad.
- **Undersampling**: entrenar con una parte de los CONTINUA (no con todos).
- **Elegir N**: graficar la curva de ganancia de cada semilla (tienen forma de parábola),
  promediarlas y tomar el máximo de la curva promedio como cantidad a enviar.
- **Ensamble de semillas**: entrenar el mismo modelo con varias semillas, promediar las
  probabilidades por cliente y mandar el top N → un solo CSV (sirve para la entrega final).

## Recomendaciones del profesor (Gustavo Denicolay) — PARA REVISAR
Texto casi literal. Revisar juntas qué adoptamos y qué no.

- **1.1 Data Quality**: `ccajas_depositos` está rota en 202105 (todos ceros).
  Solución: asignar NA en ccajas_depositos **solo para 202105**.
- **1.2 Data Drifting**: tbd.
- **1.3 FE intra mes**: variables a eliminar = (vacío); variables manuales a crear = (vacío).
- **1.4 FE histórico**: es superador agregar **lags y delta lags de orden 1 y 2**.
- **1.5 FE hojas de Random Forest**: tbd.
- **1.6 FE canaritos asesinos**: tbd.
- **1.7 Training strategy**:
  - **Sin undersampling** de CONTINUA durante la optimización bayesiana ("son muy pocos
    meses"; el undersampling recién hace falta en la Segunda Competencia).
  - Métrica de LightGBM: **AUC**.
  - Hiperparámetros **fijos**: max_bin = 31, **min_data_in_leaf = 0** (muy importante),
    feature_fraction = 0.50, feature_fraction_bynode = 0.20, **learning_rate = 0.005** (muy importante).
  - Hiperparámetros a optimizar (bayesiana): num_iterations, num_leaves, min_sum_hessian_in_leaf.
- **2.1 Final training**: entrenar **varios modelos (10+) con hiperparámetros distintos**
  en 202103, 202104, 202105, 202106, sin undersampling.
- **2.2 Scoring**: aplicar esos modelos a 202108 y **ensamblar promediando probabilidades
  por numero_de_cliente**.
- **2.3 Classification**: probar distintos cortes (10000, 10500, 11000, etc.) contra el
  pseudo Kaggle (el bot de Zulip).
- **Plataforma**: recomienda Google Cloud ya para esta competencia (no Windows/Mac).

### Decisiones tomadas (07/10), sobre las diferencias tutoría / profesor
- **Seguimos el Plan A (profesor)**: 1 modelo tarda ~30 s (1 mes) / ~2,2 min (4 meses) en esta compu.
  No hace falta Google Cloud.
- **Sin undersampling** (rápido igual; los hiperparámetros de Optuna son para datos completos).
- **Atributos**: originales + lags y deltas de orden 1 y 2 = 761 variables. No se fue a 1000-1500.
- **Ensamble**: hiperparámetros distintos (mejores trials) **y** varias semillas.
- **Elegir N**: curva promedio en validación + Montecarlo + cortes contra el bot → **N ≈ 12.000**.
- Validación: **holdout temporal** (train 202104, valid 202106), no cross validation; el ruido se
  cubre con 5 semillas (Montecarlo de semillas).

## Target por mes (parquet, clase_ternaria recalculada)
| foto_mes | BAJA+1 | BAJA+2 | CONTINUA | sin target | % BAJA+2 |
|---|---|---|---|---|---|
| 202103 | 1019 | 960  | 160921 | 0      | 0,59 |
| 202104 | 964  | 1139 | 161181 | 0      | 0,70 |
| 202105 | 1143 | 870  | 161755 | 0      | 0,53 |
| 202106 | 874  | 1098 | 162142 | 0      | 0,67 |
| 202107 | 1103 | 0    | 0      | 163245 | —    |
| 202108 | 0    | 0    | 0      | 164647 | —    |

## Experimentos (todo en `0_exploracion/resultados_experimentos.csv`)
Validación: train 202104, valid 202106, solo BAJA+2. Ganancia máx en validación (millones).
- **z001** exploración. **z002** modelo base (lee el parquet, usa numero_de_cliente): 335,8 M.
- **z003** lags/deltas: con 5 semillas, base 336,6 → **+ orden 1: 347,2** (gana 5/5).
  Orden 2: empate con orden 1 (con train 202104, lag2 queda todo NaN). Se usan orden 1 y 2.
- **z004** receta del profesor (1 modelo): 383,6 M (vs 350,2 M receta vieja, misma semilla).
- **z005** Optuna 40 trials (estudio `z005_profesor_s501419`, SQLite en `C:/Users/keila/optuna/z005.db`,
  fuera de OneDrive) + revalidación de los 10 mejores con 5 semillas.
  - El 1° de Optuna (trial 6, 401,8 M) era suerte: con 5 semillas 377,5 ± 15,5.
  - **Mejor real: trial 34** (num_leaves 137, min_sum_hessian 9,76, 538 árboles): **399,6 ± 2,3**.
  - Zona buena: hessian 5-10, 60-240 hojas. Archivos: `z005_optuna_trials.csv`, `z005_top10_5semillas.csv`.
- **z006** elegir N: curva promedio de 5 semillas (trial 34) plana entre ~12.000 y 15.500;
  ensamble de 8 trials: máx en 11.628 (399,6 M). Montecarlo (200 muestras del 25%):
  **ruido del público ± 12 M** con el mismo archivo; mejor N 11.500-12.000.
- **z007** drift (variables en pesos → ranking por mes): **empeora −19 M** (0/5 semillas). Descartado.

## Submits (Zulip, bot @competencia-uno) — carpeta `submits/`
Convención: **el número del submit va adelante** en el nombre del archivo y en el título del bot.
Todos los archivos de un mismo submit llevan el mismo número; cada submit nuevo, +1.
**REGLA: cada submit necesita su código completo (corre de principio a fin y genera exactamente el archivo enviado),
guardado y subido a GitHub.** Nada de scripts sueltos.

### Submits 33-37 (11/10): ens100 juntar + variables BC_limpio (z027) — `33-37_ens100juntarBC_completo.ipynb` (4,4 h)
Igual que el 32 + 54 variables nuevas (815). Un archivo por corte: **33 = 12.000** (coincide 96,6% con el 32), 34 = 11.500,
35 = 11.000, 36 = 10.500, 37 = 12.500. Probabilidades en `33_ens100_probs.parquet` y `probs_33/`.
**Público (10/10, 23:09):** BC vs juntar viejo (18-28): 10.500 **105,22** vs 103,51 · 11.000 104,06 vs 102,08 · 11.500 101,50 vs 102,99 ·
12.000 100,02 vs 102,11 · 12.500 98,95 vs 100,90. Promedio de los 5 cortes: 101,95 vs 102,32 → empate; BC mejora en N chicos y
empeora en N grandes (diferencias ±2 M, dentro del ruido). **Último enviado: 37 (98,95) → hoy es la entrega vigente: reenviar la elegida.**
Comparación 32 vs 36: comparten 10.459 clientes (99,6% del 36); el 36 es casi el 32 recortado → la diferencia es el N, no las variables.
**Seguro: reenviar `32_ens100juntar_12000.csv` sin cambios con título `38_ens100juntar_12000` (38 = copia del 32; código = notebook del 32).**
(**Enviado el 10/10 → entrega vigente: 38.**) Recomendación de N: 12.000 (conservador); 11.500 si se le da peso al público
(un N más chico le gana a 12.000 solo si p(agosto como mayo) > ~70%).

### z028 HECHO (11/10, 02:03) — Optuna juntar con **5 semillas por trial** (cambio pedido por Keila; 5 h 20 min)
**Regla: pasan 0/10** (falla solo la ganancia máx en junio: empate, la ref llega a su máximo en N ≈ 14.800).
**Pero en N fijo ≈ 12.000 (el que se manda) ganan TODOS: junio +17,5 a +25,6 M en 10/10; mayo +1 a +7 M en 7-9/10**;
mayo ganancia máx +3 a +7 M (9-10/10). Usan ~1.000-1.650 árboles (ref 538) → con juntar convienen más árboles.
Candidatos para submit nuevo: trials 16, 18, 34, 15, 21, 36, 32 (`z028_revalidacion.csv`). **Pendiente: Keila decide** si se
hace el submit 39-43 igual (cambiar la regla después de ver resultados) o queda el 38. ~4,5 min por modelo con 4 meses.
**Keila decidió (11/10) hacer el submit igual → `submits/39-43_ens70juntarZ028_completo.ipynb`** (7 trials × 10 semillas del 32
= 70 modelos, ~4,9 min c/u, ~5,7 h; lanzado 07:45, termina ~13:30). 39 = 12.000, 40 = 11.500, 41 = 11.000, 42 = 10.500,
43 = 12.500. Al terminar: verificar CSV, reentrenar 1 modelo en proceso nuevo y comparar (`np.array_equal`), commit + push con salidas.
Keila manda al bot y compara contra 32/38; si no convence, reenviar el 32 como último submit.
**Entrega a la cátedra (consigna):** link a carpeta de GitHub con notebooks + instrucciones para replicar exacto la predicción final.
Hecho: `submits/datos_crear_parquet.ipynb` (verificado idéntico), `submits/requirements.txt`, sección "Cómo replicar" en
`submits/README.md` (falta completar cuál es el notebook final). Repo: https://github.com/keilazavaglia/DMEyF_ksz
Windows (Control de aplicaciones) bloquea los .pyd de sqlalchemy → Optuna con SQLite falla. Se corrió con una copia de
sqlalchemy sin .pyd en `PYTHONPATH` (alternativa permanente: renombrar los .pyd del .venv, con OK).

### (Plan original de z028, ya ejecutado)
Los hiperparámetros del 32 (z005) se buscaron con **solo BAJA+2**; juntar se adoptó después sin volver a buscar.
1. **Correr `0_exploracion/z028_optuna_juntar.ipynb`** (ya armado, sin correr; ~2 h 45 min) con nbclient desde `.venv`
   (cwd = `0_exploracion`, `timeout=None`; avance en `z028_progreso.txt`). Repite z005 tal cual (40 trials, semilla 501419,
   hojas 8-256, hessian 0,001-10, early stopping AUC 200, tope 3000, train 04 → valid 06, 761 variables) con **target juntar**.
   Revalida los 10 mejores con **10 semillas** en junio (04→06) y mayo (03→05) contra el trial 34 juntar (`probs_z026/ref_*`).
   **Regla (fijada antes): pasa si gana ≥ 8/10 y mejora N fijo en los dos cortes. Si pasan ≥ 3 → ensamble nuevo** (hasta 10, por promedio de los dos cortes).
2. Si pasan ≥ 3: notebook completo `submits/39-.._completo.ipynb` = copia del notebook del 32 cambiando solo `TRIALS`
   (hojas, hessian y best_iter de los que pasan) y la carpeta de caché; 10 semillas por trial (las del 32); un CSV por N
   (12.000, 11.500, 11.000, 10.500, 12.500), un número de submit por archivo. ~4,5 h. Commit + push.
3. Si pasan < 3: el 32 (38) queda como entrega. Anotar en z028 "Conclusiones" y acá.
**El último archivo enviado es la entrega:** si se prefiere el 32, reenviarlo al final con número nuevo.

### Estado del código por submit (11/10)
- **Completo: 33-37** (`33-37_ens100juntarBC_completo.ipynb`, corre de principio a fin y genera los 5 CSV).
- **Completo y verificado: 32** (`32_ens100juntar_12000_completo.ipynb`; reproduce el CSV byte a byte).
- Con notebook propio (no re-verificado): 0-17 (un notebook por submit o grupo) y 29/31 (`29_lag3.ipynb` genera ambos).
- **SIN código completo: 18-28 y 30** (cortes y variantes del ens100 juntar, 26 solo150, 27 mezcla, 30 sin nuevos):
  se hicieron con scripts sueltos combinando predicciones de `14-15_ens50_juntar.ipynb` y `16-17_distribucion_semillas.ipynb`
  (y de `12_ens100_z005.ipynb` para 26-27). **Si alguno de estos se usa, primero armar su `<submit>_completo.ipynb`
  y hacer la prueba del 32:**
  1. Un solo notebook de principio a fin (datos → lags → entrenar → ensamble → CSV), hiperparámetros escritos en el notebook.
  2. Reproducir el Dataset igual que el original: LightGBM arma los bines con la semilla del PRIMER modelo de cada tanda
     y los reusa (en el 32: un Dataset por grupo de semillas, armado con la primera semilla del grupo).
  3. Prueba ampliada: reentrenar 1 modelo por trial (cubriendo todas las semillas) con el código del notebook y comparar
     contra las predicciones guardadas (`np.array_equal`) → tienen que dar idénticas.
  4. Copiar las predicciones originales a la carpeta de caché del notebook, correrlo y verificar que el CSV es igual
     byte a byte al enviado. Commit y push.
| N° | Qué | Público |
|---|---|---|
| 0 | base_0: modelo base, 1 semilla, top 10.000 | 90,83 |
| 1 | LagsSemillas01: lags, receta vieja, 5 semillas (5 archivos), top 11.500 | 92,67 ± 3,90 |
| 2 | LagsEnsamble01: ensamble de esas 5 semillas, top 11.500 | 97,07 |
| 3-11 | OptunaEns: ensamble 8 trials z005 × 1 semilla, cortes 10.000 a 14.000 | máx **99,11** (11.500 y 12.000) |
Curva del público (3-11): 10.000 94,46 · 10.500 94,63 · 11.000 97,82 · 11.500 99,11 · 12.000 99,11 ·
12.500 96,42 · 13.000 95,18 · 13.500 93,72 · 14.000 94,63.
| 12 | 12_Ens100z005_12000: 10 trials z005 × 10 semillas (100 modelos), top 12.000 | 94,82 |
| 13 | 13_Ens15z012_12000: 5 trials z012 × 3 semillas (15 modelos), top 12.000; coincide 90% con el 12 | 93,20 |
**Entrega vigente: 13_Ens15z012_12000 (93,20).** Ojo: z012 (+35 M en validación) no se vio en el público;
sus hiperparámetros se eligieron y revalidaron en 202106 → **z017 confirmó sobreajuste**: en 202105 (mes no visto)
z005 trial 34 le gana a z012 trial 26 por 12-16 M en 5/5 semillas. **Volver a z005.** Todo lo elegido en 202106
debe confirmarse en otro mes.
- **z019: juntar BAJA+1 + BAJA+2 también gana en mayo** (+21,5 y +17,8 M, 5/5 semillas) → mejora REAL.
- **z018: modelo común en 5 cortes**: mayo es más difícil (27-29% del ideal vs ~34% junio); el mejor N por BAJA+2
  es ~12-14 en junio y ~9-10 en mayo. Sumar marzo al train ayuda un poco.
- **z020: sacar ccajas_depositos no cambia nada** (+0,1 / −0,7 / −2,6 M). Mayo no es difícil por esa variable.
- **Submits 14 y 15** (`submits/14-15_ens50_juntar.ipynb`): z005 × 5 semillas con target BAJA+1 + BAJA+2,
  top 13.000 y 12.000. Coincide 91,8% con el 12. **Público: 14 = 99,36; 15 = 101,92 (el mejor hasta ahora).**
- **Submits 16 y 17** (`submits/16-17_distribucion_semillas.ipynb`, 08/10): semillas nuevas 270131, 417293, 583117,
  692347, 851071; 10 trials z005 × 5 semillas, 1 archivo por semilla (promedio de 10 trials), top 12.000.
  **16 juntar: 101,45 ± 0,53; 17 solo BAJA+2: 95,63 ± 1,09 → juntar +5,8 M, no es suerte. SE ADOPTA JUNTAR.**
  Archivos de una misma versión coinciden 98%; juntar vs solo 92%. Predicciones en `submits/probs_16-17/`.
  **Ojo: el último submit (17) es de varios archivos → NO vale como entrega final; reenviar un archivo único.**
  Entrega vigente (último archivo único): 15_Ens50juntar_12000 (101,92).
- **Submits 18-28** (08/10, sin entrenar; "ens100 juntar" = 50 modelos del 15 + 50 del 16, probabilidades en
  `submits/28_ens100_juntar_probs.parquet`):
  18-25 = ens100 juntar con N 10.000 / 10.500 / 11.000 / 11.500 / 12.500 / 13.000 / 13.500 / 14.000;
  26 = 150 modelos solo BAJA+2 (12 + 17), top 12.000; 27 = mezcla juntar + solo (promedio de rankings), top 12.000;
  **28 = ens100 juntar top 12.000 (coincide 99,5% con el 15).**
  Público curva juntar: 10.000 100,21 · 10.500 **103,51** · 11.000 102,08 · 11.500 102,99 · 12.000 102,11 ·
  12.500 100,90 · 13.000 98,48 · 13.500 98,20 · 14.000 97,93. Meseta 10.500-12.000 (~102-103,5), cae desde 12.500.
  26 solo150 = 94,90; 27 mezcla = 98,15 → juntar puro es lo mejor.
  ~~Entrega vigente: 28~~ (ver submits 29-32 abajo).
- **Submits 29-32 (11/10, madrugada)**:
  | N° | Qué | Público |
  |---|---|---|
  | 29 | 10 trials z005 × semilla 270131, juntar, **con lag 3** (`29_lag3.ipynb`, 31 min) | 103,15 |
  | 30 | ens100 juntar **sin clientes con antigüedad ≤ 3 meses** (46 reemplazados), top 12.000 | 102,00 |
  | 31 | control del 29: mismo modelo **sin lag 3** (= semilla 270131 del 16) | 101,17 |
  | 32 | **copia exacta del 28** (ens100 juntar, top 12.000) | **102,11** |
  Lag 3: +2,0 M vs control, < umbral de 5 M (ruido; el lag 3 solo existe para 202106 en el train) → no se reentrena.
  Sin nuevos: −0,1 M; en validación (preds de z024) mayo +0,8 / junio −1,8 (el modelo acertó 4 de 5 BAJA+2 nuevos) → no.
  El 32 = 28 confirma que el bot es determinístico.
  **ENTREGA FINAL VIGENTE: 32_ens100juntar_12000 (102,11), subido último el 11/10.** Para cambiar a 11.500: reenviar el 21.

## z021: curva promedio de N para juntar (método del tutor) — HECHO 08/10
`0_exploracion/z021_curva_n_juntar.ipynb` (2,9 min): trial 34 z005, target juntar, 5 semillas (las del 16-17), dos cortes:
train 202104 → valid 202106 y train 202103 → valid 202105. Resultados en `z021_curva_n_juntar.csv`.
- **Mejor N: junio 13.900-15.100 (prom. 15.142); mayo 6.700-8.650 (prom. 6.802).** El N depende de cuántas BAJA+2 tiene
  el mes (mayo 870, junio 1098). Promedio general: máx en 13.968 pero casi plano de 10.000 a 14.000 (lo arrastra junio).
- Pérdida vs óptimo de cada mes (junio / mayo, M): 10.000 −48/−10 · 11.000 −41/−16 · 12.000 −30/−20 ·
  **12.500 −25/−25** · 13.000 −17/−31 · 14.000 −2/−43. → **12.000-12.500 es lo que menos arriesga.**
- Junto con el público (meseta 10.500-12.000): **recomendación = quedarse con 12.000 (submit 28).** Decisión final: Keila.
- Pendiente menor: el gráfico `z021_curva_n_juntar.png` sale aplastado (falta `ax.set_ylim`).
- Julio (202107) no se puede evaluar con el bot (el bot solo mira 202108); julio ya entra en los lags de 202108.
- Correr notebooks desde afuera: usar el Python de `py/.venv/Scripts/python.exe` (el `python` del sistema no tiene nbclient).
  El .venv no tiene openpyxl: para leer .xlsx, leer el XML del zip (o instalarlo con OK).

Notebooks de submit: `modelo_base.ipynb`, `modelo_lags_semillas.ipynb`, `modelo_optuna_ensamble.ipynb`,
`modelo_final_noche.ipynb`. Correr notebooks desde afuera: Windows bloquea `jupyter nbconvert`;
usar `nbclient` desde Python.

## Corriendo la noche del 07/10
`submits/modelo_final_noche.ipynb`: revalida 4 trials más, toma los **10 mejores trials × 10 semillas
= 100 modelos** (train 202103-202106), ~4 h. Guarda cada predicción en `submits/probs_noche/`
(si se corta, volver a correr y retoma). Genera `12_final_ens100_11500.csv` y `13_final_ens100_12000.csv`.

## z022-z024 (09 al 11/10) — ninguna variable nueva se adopta: la receta queda igual
- **z022 importancia** (1 modelo trial 34 juntar, train 202103-202106; `z022_importancia.csv`): top 10 todas originales
  (ctrx_quarter 5,2%, mpasivos_margen, cpayroll_trx, mcuentas_saldo, mcaja_ahorro, mpayroll, Visa/débito). Gain por tipo:
  originales 73%, lags 16,8%, deltas 10,1%. 167 de 761 variables con gain 0. numero_de_cliente puesto 94.
- **z023 préstamos y calidad** (`z023_calidad.csv`): con préstamo se van MENOS en 03-05 (BAJA+1 0,1-0,2% vs 0,7% sin préstamo),
  pero hay una **ola de salidas en agosto** (BAJA+2 junio = BAJA+1 julio): top 10% de monto BAJA+2 0,75% en junio (antes 0,10-0,18);
  3+ préstamos 1,60%. Calidad: Master/Visa_fultimo_cierre = 0 en el 69% de 202106 (artefacto de fecha); ccajas_depositos ya conocido.
  Exposición del submit 28: 6,8% de los 12.000 en top 10% monto o 3+ préstamos vs 12,1% de la cartera → chica.
- **z024 pruebas** (`z024_pruebas_noche.ipynb`, `z024_resultados.csv`; trial 34 juntar, 10 semillas, cortes 04→06 y 03→05):
  regla definida antes = gana ≥ 8/10 en los dos cortes y mejora N fijo (préstamos/fechas: decide mayo). **Ninguna pasa.**
  a sin préstamos: mayo −23,5 M (0/10), junio +38 (la ola) → se quedan. b sin fechas: ~0. a+b: como a.
  **c solo BAJA+2: −20,8 mayo / −18,7 junio, 0/10 en ambos, p=0,002 → juntar confirmado por 3ª vez.**
  d pasó a cero / e rankings / f tendencias: empate o peor.
- Ojo metodológico: el corte de mayo entrena con 202103 (primer mes) → sin lags; flags y tendencias no se pueden evaluar ahí.

## z025 exploración (10/10) — ¿quiénes se van? (`0_exploracion/z025_exploracion.ipynb`, 12 s)
- % BAJA+2 por rango, 202103-202106 (prom. 0,62%). Fuertes: **sin sueldo 2,0× (con sueldo 0,14×)**, **sin tarjeta/límite 0 5,1×**,
  saldo negativo/cero ~3×, antigüedad 1-5 años ~1,5× (más de 10 años 0,7×). Edad en U, moderada (65+ 1,33×).
- Saldo/límite y consumo/límite: BAJA+2 con mediana ≈ 0 en **todos** los rangos de límite → el cociente sirve parejo.
- Transferencias recibidas NO incluyen el sueldo (m_ingresos_total OK). **`cmobile_app_trx` vale solo 0/1**
  (marca de "usa la app", no cantidad) → c_trx_digitales = home banking + call center, sin la app.
- Excel actualizado (`nuevas_variables_bajas.xlsx`, versión anterior en `nuevas_variables_bajas_antes_z025.xlsx`): nueva `sin_tarjeta` (prioridad alta);
  tiene_payroll y m_patrimonio_banco prioridad baja; r_saldo_limite / r_consumo_limite (z013, probadas solo en grupo) → reprobar solas.
- **Próximo paso:** probar las variables de prioridad alta (sin_tarjeta, r_saldo_limite, r_consumo_limite) de a una,
  con 5-10 semillas y los dos cortes (04→06 y 03→05), contra la referencia (trial 34 juntar). Sin commit todavía.

## z026 nuevas variables (11/10) — `0_exploracion/z026_nuevas_variables.ipynb` (40 min), Excel `nuevas_variables_bajas_final.xlsx`
- Grupos A tarjetas (+ sin_tarjeta), B plata, C uso/fidelidad; trial 34, 10 semillas, dos cortes, regla de z024. **Ninguno pasa.**
- A: junio −4,0 M (2/10), mayo +2,7 (9/10). **B y C ganan claro en junio** (+3,8/+4,4 máx, **+10,7/+12,3 en N fijo**, 8/10)
  pero empatan o pierden en mayo (−0,8/−0,9), donde el train 202103 no tiene historia → candidatos a reprobar con un corte con historia.
- Juntar le gana a solo BAJA+2 en 9-10/10 en todas las versiones (4ª confirmación). Segmentos: ningún grupo tapa a otro (≤ 1,5 pts).
- Gain: nuevas = A 7%, B 20%, C 14%; fuertes m_ingresos_total, c_servicios_usados, m_egresos_total, r_uso_vs_tenencia, c_tarjetas_trx.
  Gain 0: m_adelantos_total, c_cheques_rechazados y todos los lag2/delta2 nuevos; sin_tarjeta casi 0. Receta sin cambios.
- **z027** (`z027_bc_limpio.ipynb`): B+C "limpio" (54 vars, sin gain 0 ni lag2/delta2) y C limpio (20), + corte jun2m (train 03+04 → 06).
  Regla: ≥ 8/10 y N fijo mejor en junio y jun2m; mayo N fijo ≥ −2 M. **BC_limpio: junio +18,3 M N fijo (10/10), jun2m +11,0 (9/10),
  mayo −2,09 → no pasa por 0,09 M.** C limpio no alcanza. Se armó igual el submit con BC_limpio para verlo en el bot (submits 33-37).
## Próximos pasos (08 al 11/10)
1. Revisar el resultado de la noche; mandar el **12** y después el **13** (12.000 último).
2. Opcional: probar si el modelo final necesita más árboles al entrenar con 4 meses
   (validar train 202103-202104 → valid 202106 con best_iter ×1 vs ×1,5).
3. **HECHO (11/10)**: entrega final = **32_ens100juntar_12000** (copia del 28, 102,11), subido último. Si se cambia a 11.500,
   reenviar `21_ens100juntar_11500.csv` como último antes de las 23:59. Si no, no tocar nada más.
4. **PENDIENTE: revisar el N final por la diferencia entre meses** (ver z021): el mejor N cambia mucho según el mes
   (mayo ~7.000, junio ~15.000) porque depende de cuántas BAJA+2 tiene el mes, y no sabemos a cuál se parece agosto.
   Hoy la recomendación es 12.000 (menor pérdida en el peor caso + meseta del público 10.500-12.000), pero revisarlo
   con Keila antes de la entrega final.
   **Observación de Keila:** en el público dieron más los N chicos (10.500 = 103,51; 11.500 = 102,99; 12.000 = 102,11;
   desde 12.500 cae). Sugiere que agosto se parece más a mayo (pocas BAJA+2) → considerar bajar a **11.500**.
   Cuidado: el público es 25% y dentro de la meseta las diferencias son de ±1,5 M; lo firme es la caída desde 12.500.
   **¿10.500 u 11.500?** El pico de 10.500 (103,51) probablemente es ruido: zigzag 10.500 → 11.000 (102,08) → 11.500
   (102,99). Pérdida vs óptimo (z021): 10.500 = −14 M si agosto es como mayo / −44 M si es como junio; 11.500 = −18 / −35.
   → 11.500 arriesga menos (pierde 4 M si es mayo, gana 9 M si es junio) y está a 0,5 M del máximo público.
   10.500 no es mala elección, pero es apostar más a que agosto se parece a mayo.
   Archivos ya hechos (100 modelos juntar): `submits/19_ens100juntar_10500.csv`, `21_ens100juntar_11500.csv`,
   `28_ens100juntar_12000.csv` (vigente). Para cambiar la entrega: reenviar el elegido como ÚLTIMO submit.

## Resultados de la noche del 07 al 08/10 — DECIDIR CON KEILA qué se adopta
Ensamble final `submits/modelo_final_noche.ipynb`: listo (10 trials × 10 semillas = 100 modelos, 4 h).
Archivos `12_final_ens100_11500.csv` y `13_final_ens100_12000.csv` (todavía NO enviados). Coincide 98,4% con
el ensamble de 8 (top 12.000). Trials: 34, 36, 27, 29, 37, 31, 20, 9, 28, 39.

Pruebas (todas contra el trial 34 = 399,6 M, una modificación cada una; detalle en la celda "Conclusiones"
de cada notebook y en `resultados_experimentos.csv`):
| Notebook | Prueba | Resultado | Veredicto |
|---|---|---|---|
| z012_optuna_rango | hessian 1-200, hojas 16-512 | 434,7-438,2 M en 202106 | **sobreajuste a 202106** (z017: −12 a −16 M en 202105) |
| z009_juntar_bajas | BAJA+1 + BAJA+2 | **421,8 M**, mejor N ~14.700; top 12.000 empata | **+22 M: candidato** (cambia el N) |
| z010_arboles | 2 meses train, ×1/×1,5/×2 árboles | 405,8 / 400,5 / 396,7 | más árboles no suma |
| z011_ajustar_drift | sin fechas / sueldo ×2/3 / depósitos / Visa suavizado | +1,6 / −0,6 / 0,0 / −2,2 | drift casi no afecta |
| z013_variables_manuales | ~20 variables manuales | −2,7 M | descartado |
| z014_tendencias | media 3 meses + cociente | +2,5 M (ruido), top 12.000 pierde | no claro |
| z015_canaritos | sacar las que pierden con 50 canaritos | −24 M (quedaron 38 de 761 variables) | descartado (criterio muy agresivo) |
| z016_undersampling | 100/50/20% CONTINUA | −3,5 / −7,3 M; 20% es 3,3× más rápido | aprendizaje para 2da competencia |

Para decidir: (1) rehacer Optuna/ensamble con z012 (subir el tope de árboles, porque quedó en 3000);
(2) probar z009 combinado con z012 (y con Optuna sobre ese target); (3) "sin fechas" es opcional (+1,6).
Antes de combinar: probar la combinación con 5-10 semillas.

## Ideas pendientes
- 8: Optuna con 2 meses de train (depende de z010). 10: segundo corte temporal 202103 → 202105.
  12: hojas de Random Forest (1.5 del profesor).
- **Clusters como ayuda al modelo (idea de Keila, 08/10)**: definir grupos de clientes con sentido de negocio
  y usarlos para "darle un toque" al patrón general (subir o bajar la probabilidad según el grupo).
  - Cuidado: los clusters de la clase 5 (`clusters/clusters_asignacion.csv`) son **solo de BAJA+2** → usarlos
    como variable es fuga del target. Hay que agrupar a **todos** los clientes, **sin mirar el target**.
  - Opciones: (a) grupos por reglas de negocio (cobra sueldo o no, VIP, cantidad de productos, edad/antigüedad);
    (b) KMeans sobre variables clave de cada mes (ver si sklearn corre; Windows lo bloqueaba);
    (c) usar el grupo como variable, o ajustar el corte/probabilidad por grupo, o un modelo por grupo.
  - Probar como siempre: una sola modificación, 5-10 semillas, contra la referencia.

## HECHO (08/10, resultado arriba en Submits): submits 16 y 17 — ¿el 15 es confiable? → Sí, juntar gana.
Objetivo: ver si "juntar BAJA+1 + BAJA+2" (submit 15, 101,92) es realmente mejor que "solo BAJA+2" (submit 12, 94,82)
o si fue suerte del público. Se comparan **distribuciones** en el bot (varios archivos en un submit → media ± desvío).
1. Semillas NUEVAS (elegidas por Keila que sean bien distintas): **270131, 417293, 583117, 692347, 851071**.
2. Entrenar con esas 5 semillas los 10 trials de `0_exploracion/z005_top10_final.csv`, train 202103-202106,
   **dos versiones**: target BAJA+1 + BAJA+2 (como `submits/14-15_ens50_juntar.ipynb`) y solo BAJA+2
   (como `submits/12_ens100_z005.ipynb`). 2 × 50 modelos, ~2,2 min cada uno → **~4 h**. Guardar cada predicción
   en `submits/probs_16-17/` para poder retomar si se corta.
3. Por cada semilla, promediar sus 10 trials → 1 archivo top 12.000. Notebook: `submits/16-17_distribucion_semillas.ipynb`.
   - **Submit 16**: 5 archivos de juntar (`16_juntar_s<semilla>_12000.csv`), todos en el mismo submit.
   - **Submit 17**: 5 archivos de solo BAJA+2 (`17_solo_s<semilla>_12000.csv`), todos en el mismo submit.
4. Comparar media ± desvío. Ojo: el público es siempre el mismo 25%; la evidencia fuerte de juntar es que ganó
   en validación en junio (z009) y en mayo (z019).
5. **Después** (cuando Keila decida): volver a mandar un archivo único como entrega vigente. Un submit de varios
   archivos NO vale como entrega final.
Lanzar el entrenamiento desde el chat nuevo (los procesos dependen del chat que los lanza). Avisar y esperar OK antes.
