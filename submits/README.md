# Submits — Primera competencia (202108)

## Cómo replicar la predicción final (para la cátedra)

**Entrega final: submit 44 → `submits/44_ens70juntarZ028_12000_completo.ipynb` genera `44_ens70juntarZ028_12000.csv`**
(70 modelos LightGBM: 7 trials de z028 × 10 semillas, target BAJA+1 + BAJA+2, top 12.000; público 104,69).

1. **Entorno:** Python 3.14.7 y los paquetes de `requirements.txt` (`pip install -r submits/requirements.txt`).
   Versiones clave: lightgbm 4.7.0, pandas 3.0.6, numpy 2.5.3, pyarrow 25.0.1. Se corrió en Windows 11.
2. **Datos:** copiar `competencia_01.csv` (el de la cátedra) en la carpeta raíz del repo (al lado de `submits/`).
3. **Correr `submits/datos_crear_parquet.ipynb`** (~1 min): crea `competencia_01.parquet` (clase_ternaria estricta y
   `ccajas_depositos` = NA en 202105). Verificado: idéntico al parquet usado en todos los submits.
4. **Correr `submits/44_ens70juntarZ028_12000_completo.ipynb`** desde la carpeta `submits/` (de principio a fin,
   sin parámetros; ~6,7 h desde cero).
   Entrena todos los modelos, guarda cada predicción en su carpeta `probs_*/` (si se corta, al volver a correr retoma)
   y escribe el CSV enviado. La última celda informa filas, únicos y coincidencia con otros submits.

Los hiperparámetros y semillas están escritos en cada notebook. LightGBM se corrió con todos los núcleos (default);
el submit 32 se reprodujo byte a byte en esta máquina.

**Regla:** cada submit (o tanda de cortes) tiene su notebook con el número adelante. Los notebooks de esta
carpeta **solo generan archivos para el bot**; lo experimental está en `0_exploracion/`.
Un notebook ya enviado no se modifica: si cambia algo, se hace uno nuevo con el número siguiente.
Los CSV y las probabilidades (`probs_*/`, `*.parquet`) no se suben a GitHub.

| N° | Título en el bot | Notebook | Archivo | Modelo | Público |
|---|---|---|---|---|---|
| 0 | base_0 | `00_modelo_base` | `0_modelo_base_todos_meses_10000.csv` | LightGBM base, 1 semilla, top 10.000 | 90,83 |
| 1 | LagsSemillas01 | `01-02_lags_semillas` | `1_lags_s*_11500.csv` (5) | lags 1 y 2, receta vieja, 5 semillas | 92,67 ± 3,90 |
| 2 | LagsEnsamble01 | `01-02_lags_semillas` | `2_lags_ensamble5_11500.csv` | ensamble de esas 5 semillas | 97,07 |
| 3-11 | N_OptunaEns_corte | `03-11_optuna_ens8` | `N_optuna_ens8_corte.csv` | receta del profesor, 8 trials z005 × 1 semilla | máx 99,11 (11.500 y 12.000) |
| 12 | 12_Ens100z005_12000 | `12_ens100_z005` | `12_ens100_12000.csv` | 10 trials z005 × 10 semillas = 100 modelos | 94,82 |
| 13 | 13_Ens15z012_12000 | `13_ens15_z012` | `13_ens15z012_12000.csv` | 5 trials z012 × 3 semillas = 15 modelos | 93,20 |
| 14 | 14_Ens50juntar_13000 | `14-15_ens50_juntar` | `14_ens50juntar_13000.csv` | 10 trials z005 × 5 semillas, target BAJA+1 + BAJA+2 | 99,36 |
| 15 | 15_Ens50juntar_12000 | `14-15_ens50_juntar` | `15_ens50juntar_12000.csv` | ídem, top 12.000 | **101,92** |

Cortes del 3 al 11 (público): 10.000 94,46 · 10.500 94,63 · 11.000 97,82 · 11.500 99,11 · 12.000 99,11 ·
12.500 96,42 · 13.000 95,18 · 13.500 93,72 · 14.000 94,63.

Submits 16-38: ver CLAUDE.md (sección Submits). Los dos candidatos a entrega final:

| N° | Título en el bot | Notebook | Archivo | Modelo | Público |
|---|---|---|---|---|---|
| 32 / 38 | 32_ens100juntar_12000 / 38 (copia) | `32_ens100juntar_12000_completo` | `32_ens100juntar_12000.csv` | 10 trials z005 × 10 semillas, target juntar | 102,11 |
| 39 | 39_ens70juntarZ028_12000 | `39-43_ens70juntarZ028_completo` | `39_ens70juntarZ028_12000.csv` | 7 trials z028 × 10 semillas, target juntar | **104,69** |
| 40 | 40_ens70juntarZ028_11500 | ídem | `40_ens70juntarZ028_11500.csv` | ídem, top 11.500 | 104,61 |
| 41 | 41_ens70juntarZ028_11000 | ídem | `41_ens70juntarZ028_11000.csv` | ídem, top 11.000 | 104,80 |
| 42 | 42_ens70juntarZ028_10500 | ídem | `42_ens70juntarZ028_10500.csv` | ídem, top 10.500 | 103,62 |
| **44** | **44_ens70juntarZ028_12000** | `44_ens70juntarZ028_12000_completo` | `44_ens70juntarZ028_12000.csv` | **copia exacta del 39 → ENTREGA FINAL** | 104,69 |
