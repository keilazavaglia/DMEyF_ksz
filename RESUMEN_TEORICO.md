# Resumen teórico – Primera competencia

Orden: por clase. Al final, qué priorizar para la competencia.
Cada concepto tiene: **qué es**, un **ejemplo simple** y, cuando aplica, **para qué sirve en la competencia**.

---

## 0. El problema en una frase

Tenemos clientes de un banco, mes a mes (`foto_mes`). Queremos elegir a quién mandarle un estímulo
de retención en **202108**, para ganar la mayor cantidad de plata posible.

- `clase_ternaria`: qué pasó con el cliente.
  - **CONTINUA**: sigue en el banco.
  - **BAJA+1**: se va el mes siguiente (ya es tarde para retenerlo).
  - **BAJA+2**: se va dentro de dos meses → **es a quien queremos encontrar**.
- Los últimos meses del dataset no tienen target completo (todavía no pasó el tiempo para saberlo).

---

## 2. Árboles de decisión (z201)

### Árbol de decisión
- **Qué es:** un modelo que hace preguntas de sí/no sobre las variables y va dividiendo a los clientes en grupos.
- **Ejemplo:** "¿ctrx_quarter < 10?" → sí: grupo con muchas bajas; no: grupo con pocas bajas. Después
  vuelve a preguntar dentro de cada grupo.
- Cada grupo final se llama **hoja**. La predicción es la proporción de bajas en esa hoja.

### Hiperparámetros
Son "perillas" que elegís vos antes de entrenar (el modelo no las aprende).
- `max_depth`: cuántas preguntas seguidas como máximo. Más profundo = más detallado = más riesgo de memorizar.
- `min_samples_split`: cuántos clientes necesita un grupo para poder dividirse.
- `min_samples_leaf`: cuántos clientes como mínimo debe tener cada hoja.

### Sobreajuste (overfitting)
- **Qué es:** el modelo aprende de memoria los datos de entrenamiento, incluido el ruido, y le va mal con datos nuevos.
- **Ejemplo:** un árbol con hojas de 1 cliente acierta el 100% en marzo y falla en mayo.
- Los hiperparámetros de arriba sirven para frenarlo.

### Matriz de confusión
Tabla que cruza lo que el modelo predijo con lo que pasó en realidad.

|                  | Predijo baja | Predijo no baja |
|------------------|--------------|-----------------|
| **Fue baja**     | Verdadero positivo (TP) | Falso negativo (FN) |
| **No fue baja**  | Falso positivo (FP)     | Verdadero negativo (TN) |

- **TPR** (tasa de verdaderos positivos) = TP / (TP + FN): de los que se fueron, cuántos encontré.
- **FPR** (tasa de falsos positivos) = FP / (FP + TN): de los que se quedaban, a cuántos les avisé de más.

### Curva ROC y AUC
- **ROC:** grafica TPR contra FPR para todos los umbrales posibles.
- **AUC:** el área bajo esa curva. 0,5 = azar; 1 = perfecto.
- **Ejemplo:** AUC 0,9 significa que si tomás un cliente que se fue y uno que se quedó, el 90% de las
  veces el modelo le da más probabilidad al que se fue.

### Importancia de variables
- Cuánto usó el modelo cada variable para hacer cortes y cuánto mejoró con ellas.
- Sirve para ver qué variables importan (ojo: no dice si la relación es positiva o negativa).

---

## 3. Sobre la incertidumbre (z301)

### Entrenar en un mes, probar en otro
- **Idea:** simular el futuro. Entrenás con marzo (202103) y probás con mayo (202105), que el modelo nunca vio.
- Es lo más parecido a lo que vamos a hacer de verdad: entrenar con el pasado y predecir 202108.

### Validación con particiones (ShuffleSplit)
- **Qué es:** partir los datos muchas veces al azar en entrenamiento/prueba y promediar los resultados.
- **Ejemplo:** 5 particiones dan 300, 310, 295, 305, 290 M → media 300 M. Un solo número podría haber
  sido 290 o 310 por suerte.
- **StratifiedShuffleSplit:** igual, pero cada partición mantiene la misma proporción de bajas.
  Importa porque las bajas son muy pocas.

### Escalar la ganancia
- Si evaluás con el 30% de los datos, la ganancia sale ~30% de la total. Para comparar, se divide por 0,3.
- **Cuidado:** dividir dos veces infla el número (el error que encontramos en z301).

### Semilla (seed / random_state)
- **Qué es:** un número que fija el azar para que el resultado se repita igual.
- **Ejemplo:** misma semilla → mismas particiones → mismo resultado. Sin semilla, cada corrida da distinto.

### Optuna
- **Qué es:** una librería que busca buenos hiperparámetros de forma inteligente (prueba, ve qué funcionó
  y propone combinaciones mejores).
- **Estudio:** la búsqueda completa. **Ensayo (trial):** una combinación probada.
- **Espacio de búsqueda:** los rangos permitidos (ej: `max_depth` entre 3 y 12).
  Si el mejor valor queda pegado al borde, conviene ampliar el rango.
- `load_if_exists=True`: si el estudio ya existe, lo continúa y **suma** ensayos.

### Leaderboard público y privado
- **Público:** una parte del mes futuro (~25%) que te muestran mientras competís.
- **Privado:** el resto. Es el que define el resultado final.
- El público es **ruidoso**: con pocos datos, la suerte pesa mucho.

### Maldición del ganador
- **Qué es:** si probás muchos modelos y elegís el que mejor dio en el público, probablemente elegiste
  el que tuvo más suerte, no el mejor.
- **Ejemplo:** tirás 20 monedas 10 veces cada una; la que sacó 8 caras no está cargada, tuvo suerte.

### Inversión de ranking
- Dos modelos con diferencia mínima en validación pueden invertir su orden en el futuro.
- **Conclusión:** diferencias más chicas que el ruido no significan nada. Mejor elegir configuraciones robustas.

---

## 4.1 Fuego contra fuego (z401, Random Forest)

### Bagging / Random Forest
- **Qué es:** entrenar muchos árboles, cada uno con una muestra al azar de clientes y de variables,
  y promediar sus predicciones.
- **Idea del nombre:** combatir la inestabilidad del azar con más azar. Cada árbol se equivoca distinto
  y los errores se compensan al promediar.
- **Ejemplo:** preguntarle la hora a 100 personas con relojes un poco desajustados; el promedio es más preciso.

### Hiperparámetros propios
- `n_estimators`: cantidad de árboles. Más árboles = más estable (y más lento).
- `max_features`: cuántas variables al azar puede mirar cada corte. Menos variables = árboles más distintos entre sí.

### OOB (out of bag)
- Cada árbol se entrena con una muestra que deja afuera ~37% de los clientes.
- Esos clientes que no vio sirven para validarlo "gratis", sin separar un conjunto de prueba.

### Hojas como información
- Se puede registrar en qué hoja cae cada cliente en cada árbol.
- Dos clientes que caen seguido en las mismas hojas se parecen → base de los clusters de la clase 5.

---

## 4.2 Feature Engineering en SQL (z402)

### Feature engineering
- **Qué es:** crear variables nuevas a partir de las que ya existen, para darle más información al modelo.
- **Ejemplo:** el saldo de un mes dice poco; saber que **bajó a la mitad** respecto del mes anterior dice mucho.

### DuckDB
- Permite escribir SQL directamente sobre el CSV desde Python. Rápido para tablas grandes.

### Funciones de ventana
- `OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)`: "para cada cliente, mirando sus meses en orden".
- Permiten calcular cosas de un cliente usando sus otros meses.

### Lags y deltas
- **Lag:** el valor del mes anterior. `lag1` = 1 mes atrás, `lag2` = 2 meses atrás.
- **Delta:** valor actual menos el lag. `delta1 = valor - lag1`.
- **Ejemplo:** `mpayroll` = 0 este mes y `mpayroll_lag1` = 500.000 → dejó de cobrar el sueldo en el banco.
  Señal fuerte de baja.

### Rankings y cocientes
- **Ranking (`percent_rank`):** posición relativa del cliente dentro de su mes (0 a 1). Ayuda contra la inflación.
- **Cociente:** dividir dos variables (ej: consumo / límite de la tarjeta).

### Guardar los cambios
- Un `SELECT` solo **muestra**. Para que las columnas nuevas queden, hay que usar `CREATE OR REPLACE TABLE ... AS SELECT ...`.

---

## 6. Clases con clase (z601, LightGBM)

### Gradient boosting
- **Qué es:** árboles construidos **uno detrás del otro**; cada árbol nuevo intenta corregir los errores
  de los anteriores.
- **Diferencia con Random Forest:** RF arma árboles independientes y promedia; boosting los arma en
  cadena y suma. Boosting suele ser más preciso.
- **Ejemplo:** un estudiante que en cada examen de práctica estudia solo lo que le salió mal en el anterior.

### LightGBM: hiperparámetros principales
- `num_leaves`: cantidad máxima de hojas por árbol (complejidad).
- `learning_rate`: cuánto aporta cada árbol. Más chico = aprende más despacio pero más fino (necesita más árboles).
- `num_boost_round` / `num_iterations`: cantidad de árboles.
- `min_data_in_leaf`: mínimo de clientes por hoja (frena el sobreajuste).
- `feature_fraction`: proporción de variables al azar por árbol.
- `bagging_fraction`: proporción de clientes al azar por árbol.
- `seed`: semilla.

### La ganancia (CLAVE)
- Estimular a un **BAJA+2**: **+1.072.500**.
- Estimular a cualquier otro: **−27.500**.
- **Umbral de equilibrio:** 27.500 / (1.072.500 + 27.500) = **0,025** = 1 de cada 40.
  Conviene estimular si P(BAJA+2) > 0,025.

### Clase binaria vs. binaria2
- **binaria:** positivo = solo BAJA+2. El modelo estima P(BAJA+2) → el corte 0,025 es válido.
- **binaria2:** positivo = BAJA+1 + BAJA+2. Más ejemplos para aprender (BAJA+1 se parece mucho a BAJA+2).
- **Problema:** ahora el modelo estima P(BAJA+1 o BAJA+2), que es más alta. Con corte 0,025 se estimula de más
  y la ganancia baja.
- **Solución:** cortar por **cantidad**, no por probabilidad (ver siguiente).

### Ganancia acumulada y corte óptimo
1. Ordenar a los clientes de mayor a menor probabilidad.
2. Ir sumando la ganancia de cada uno (+1.072.500 si es BAJA+2, −27.500 si no).
3. La suma sube, llega a un máximo y después baja. El **máximo** marca cuántos clientes estimular (top N).
- La consigna pide entre 8.000 y 15.000 clientes: es exactamente un corte por cantidad.

### Métricas extra
- **KS (Kolmogorov-Smirnov):** máxima distancia entre las distribuciones de probabilidad de los que se van
  y los que se quedan. Más alto = separa mejor. Marca un punto de corte.
- **Lift:** cuántas veces más bajas hay en el grupo elegido que en el total.
  Ejemplo: si en general hay 0,5% de bajas y en el top 10.000 hay 5%, el lift es 10.
- **Captura:** qué porcentaje de todas las bajas quedó dentro del grupo elegido.

### (Anti)discriminación
- Revisar que el modelo no trate distinto a grupos de personas por variables sensibles (edad, sexo, etc.)
  de forma injusta.

### Semillerío
- Entrenar el **mismo** modelo con distintas semillas.
- Cada semilla da un resultado algo distinto; el promedio (y el desvío) muestra el rendimiento real.
- En la competencia: un CSV por semilla, hasta 20 por submit; el bot devuelve media y desvío.

---

## 7. Drifting y calidad de datos (z701)

### Controles de calidad
- **Nulos por mes:** una variable que en un mes está 100% vacía probablemente se cargó mal.
- **Ceros por mes:** igual. Si en un mes todos tienen 0, es un error de datos, no un comportamiento real.
- **Qué hacer:** no usar esa variable en ese mes (o en todo el modelo).

### Tipos de drift
- **Data drift:** cambia la distribución de una variable.
  Ejemplo: por la inflación, el saldo promedio pasa de 1.000 en marzo a 1.500 en agosto.
- **Concept drift:** cambia la relación entre las variables y la baja.
  Ejemplo: antes, tener poco saldo indicaba baja; después de un cambio de política del banco, ya no.

### PSI (Population Stability Index)
- **Qué es:** un número que mide cuánto cambió la distribución de una variable entre dos meses.
- **Cómo se calcula:** se parte la variable en tramos (bins) y se compara qué porcentaje de clientes
  cae en cada tramo en cada mes.
  `PSI = Σ (% actual − % referencia) × ln(% actual / % referencia)`
- **Interpretación habitual:**
  - < 0,1: estable.
  - 0,1 a 0,25: cambio moderado.
  - > 0,25: cambio fuerte → revisar la variable.

### Target dentro de los bins
- Además de ver si la variable cambió, mirar si la proporción de bajas en cada tramo se mantiene.
  Si cambia, es concept drift.

### Puntos de corte del modelo
- Revisar en qué valores corta LightGBM. Si corta en zonas donde la variable tuvo mucho drift,
  esos cortes pueden no servir para el mes futuro.

### Mirando el futuro
- Comparar el mes a predecir (202108) con los de entrenamiento. Cuanto más distinto sea, menos confiable
  es usar los cortes aprendidos.
- **Soluciones típicas:** usar rankings dentro de cada mes en lugar de montos, ajustar por inflación,
  o sacar variables muy inestables.

---

## Qué priorizar para la competencia

1. **Ganancia y corte por cantidad** (clase 6): define cómo elegir a quién mandar.
2. **LightGBM** (clase 6): es el modelo final.
3. **Lags y deltas** (clase 4.2): lo que más suele mejorar la ganancia.
4. **Validación y semillas** (clase 3): para decidir sin engañarse con el leaderboard público.
5. **Nulos, ceros y PSI** (clase 7): para no entrenar con variables rotas.
6. Árboles y Random Forest (clases 2 y 4.1): base conceptual; no se usan directamente en el modelo final.
