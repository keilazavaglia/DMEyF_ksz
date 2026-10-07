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
- Datos: `py/competencia_01.csv` (155 columnas; target `clase_ternaria`).
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

## Indicaciones de la tutoría (05/10/2026) — SEGUIR ESTO
- **No sacar `numero_de_cliente`**: se usa como variable (z002 todavía lo saca: corregir).
- **No sacar marzo** del entrenamiento.
- **Feature engineering**: llegar a entre **1000 y 1500 atributos** (lags, deltas, rankings, etc.).
- **Optuna**: 30 a 50 trials.
- **Muchas semillas** para eliminar la variabilidad.
- **Undersampling**: entrenar con una parte de los CONTINUA (no con todos).
- **Elegir N**: graficar la curva de ganancia de cada semilla (tienen forma de parábola),
  promediarlas y tomar el máximo de la curva promedio como cantidad a enviar.

## Plan (sesión del 05/10/2026)
1. Exploración: `0_exploracion/z001_exploracion.ipynb`  ← hecho
   (meses 202103-202108, target completo hasta 202106; ccajas_depositos rota en 202105;
   Finiciomora ~100% nulos = sin mora, no descartar; drift por aguinaldo/inflación)
2. Modelo base LightGBM: `0_exploracion/z002_modelo_base.ipynb` (versión A: solo BAJA+2,
   train 202104, valid 202106)  ← hecho. Resultados en `0_exploracion/resultados_experimentos.csv`:
   - [202104]: ganancia máx 323,8 M (N=14184), AUC 0,879. Corte 0,025: 298,2 M (6677 clientes).
   - [202103, 202104]: ganancia máx 357,0 M (N=11700), AUC 0,891. Corte 0,025: 322,1 M (7327).
   - Prob. en el corte óptimo < 0,025 → el modelo subestima; cortar por cantidad.
   - ctrx_quarter es la variable más importante (15%).

## Flujo eficiente (06 al 11/10) — VOS = Keila, CLAUDE = lo corre Claude
Idea: UN pipeline reutilizable (`pipeline.py`) con opciones que se prenden de a una.
Cada mejora se valida en 202106 con ~5 semillas y se anota en resultados_experimentos.csv.
Solo se adopta si sube la ganancia de la curva promedio.

- **Mar 06**
  1. CLAUDE: convertir el CSV a parquet (carga más rápida, menos RAM).
  2. VOS: en z002, volver a incluir numero_de_cliente y correr una vez.
  3. CLAUDE arma las celdas, VOS corrés: **primer submit simple** (train 202103-202106,
     predecir 202108, top 12.000). Objetivo: comprobar el formato, no ganar.
- **Mié 07**
  4. CLAUDE: escribir `pipeline.py` (carga → atributos → undersampling → entrenar
     N semillas → curva de ganancia promedio → CSVs de 202108). VOS: leerlo, versión corta.
  5. CLAUDE corre comparaciones con 5 semillas: clase A vs B (BAJA+1+BAJA+2) y
     undersampling (todos los CONTINUA vs 10-20%).
- **Jue 08**
  6. CLAUDE: feature engineering automático sobre todas las variables (lags 1-3,
     deltas 1-3, rankings por mes de los montos) → ~1000-1500 atributos. Con DuckDB
     por la memoria. VOS: mirar el código de un lag para entenderlo.
  7. CLAUDE: comparar contra el pipeline sin atributos nuevos. Submit con varias semillas.
- **Vie 09**
  8. CLAUDE: Optuna 30-50 trials con undersampling (de fondo, puede ser de noche).
- **Sáb 10**
  9. CLAUDE: modelo final (train 202103-202106), 20 semillas, curva promedio → N.
     Submit con los 20 CSV (media y desvío del bot).
  10. Armar el **archivo único final**: promedio de probabilidades de las 20 semillas,
      top N. Enviarlo ya el sábado.
- **Dom 11** (margen por si algo falla)
  11. Reenviar el submit de UN solo archivo antes de las 23:59 (mejor de día).

Si falta tiempo, recortar en este orden: Optuna → parte de los atributos → clase B.
Nunca recortar: el submit simple (paso 3) ni el archivo único final (pasos 10-11).

## Pendiente técnico
- Instalar nbformat e ipywidgets en el .venv (gráficos de Optuna).


