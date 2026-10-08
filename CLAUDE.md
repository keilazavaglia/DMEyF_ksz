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

Notebooks de submit: `modelo_base.ipynb`, `modelo_lags_semillas.ipynb`, `modelo_optuna_ensamble.ipynb`,
`modelo_final_noche.ipynb`. Correr notebooks desde afuera: Windows bloquea `jupyter nbconvert`;
usar `nbclient` desde Python.

## Corriendo la noche del 07/10
`submits/modelo_final_noche.ipynb`: revalida 4 trials más, toma los **10 mejores trials × 10 semillas
= 100 modelos** (train 202103-202106), ~4 h. Guarda cada predicción en `submits/probs_noche/`
(si se corta, volver a correr y retoma). Genera `12_final_ens100_11500.csv` y `13_final_ens100_12000.csv`.

## Próximos pasos (08 al 11/10)
1. Revisar el resultado de la noche; mandar el **12** y después el **13** (12.000 último).
2. Opcional: probar si el modelo final necesita más árboles al entrenar con 4 meses
   (validar train 202103-202104 → valid 202106 con best_iter ×1 vs ×1,5).
3. **Sáb 10**: elegir el archivo único final. **Dom 11**: reenviarlo antes de las 23:59.

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
