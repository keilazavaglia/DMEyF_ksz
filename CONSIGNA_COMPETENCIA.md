# Consigna – Primera Competencia

## Fecha límite
- **Domingo 11 de octubre de 2026, 23:59:59 (hora Buenos Aires).**
- Vale como entrega el **ÚLTIMO submit** recibido antes de esa hora.
- Si no hay ningún submit, no se participa de la competencia.

## Objetivo
Predecir qué clientes deben recibir estímulo en la **Campaña Proactiva de Retención de Clientes**, para el mes futuro **202108**.

## Cómo se entrega (NO es por Kaggle)
Mensaje privado por **Zulip** al bot `@competencia-uno`, con:

1. **Cuerpo del mensaje:**
   ```
   submit NombreDelSubmit
   ```
   - `submit` siempre en minúsculas.
   - `NombreDelSubmit`: sin espacios ni caracteres raros, **distinto en cada submit** (ej: `PredSemillas01`).
2. **Uno o varios archivos adjuntos** (máximo 20 por submit).

## Formato obligatorio de cada archivo .csv
- Extensión **.csv**.
- Nombre distinto dentro del mismo mensaje (recomendado: nombre nunca usado antes).
- **Sin encabezado** (ningún nombre de columna).
- **Una sola columna**: solo los `numero_de_cliente` a los que se les envía estímulo.
- Entre **8.000 y 15.000 filas** aprox.
- Los IDs deben ser **enteros, NUNCA en notación científica**.
  - Ojo con `48000000` → puede salir como `4.8e+07` y el submit falla.
  - También puede fallar `31100000`.

Ejemplo de contenido:
```
42491091
21368282
32107031
48000000
```

### Exportación segura en Python (pandas)
```python
ids = df_pred.loc[df_pred["enviar"] == 1, "numero_de_cliente"].astype("int64")
ids.to_csv("pred_seed_01.csv", index=False, header=False)
```
Verificar después abriendo el archivo como texto que no haya `e+` en ninguna línea.

## Límites de envío
- Máximo **13 submits por día**, cada uno con hasta **20 archivos**.
- El contador se reinicia a las **21:00 hora Argentina**.

## Respuesta del bot
- **Public Gain mean:** ganancia promedio de los archivos en el Public Leaderboard (~25% de los datos), sin escalar, en millones de pesos.
- **Public gain std dev:** desvío estándar de esas ganancias.

## Otros comandos al bot
- `list submits` → ver mis submits anteriores (solo los propios).
- `help` → ayuda.

## Estrategia contra el overfitting (concepto clave)
- En el modelo final con **LightGBM**, entrenar **varios modelos idénticos cambiando solo el hiperparámetro `seed`**.
- Cada modelo predice sobre **202108** → un archivo .csv por semilla (hasta 20).
- Subir todos en un mismo submit: el bot devuelve media y desvío, lo que reduce mucho (no elimina) el overfitting al Public Leaderboard ("maldición del ganador").

## Entrega final (MUY IMPORTANTE)
- Al final elegir **un submit previo con UN SOLO archivo adjunto** y **volver a enviarlo** para que sea el último antes del cierre.

## Extra
- Algunas veces por semana se comparte el Public Leaderboard global para compararse con los compañeros.
