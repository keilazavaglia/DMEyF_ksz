# Submits — Primera competencia (202108)

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
| 12 | 12_Ens100z005_12000 | `12_ens100_z005` | `12_ens100_12000.csv` | 10 trials z005 × 10 semillas = 100 modelos | (pendiente) |
| 13 | 13_Ens15z012_12000 | `13_ens15_z012` | `13_ens15z012_12000.csv` | 5 trials z012 × 3 semillas = 15 modelos | (pendiente) |

Cortes del 3 al 11 (público): 10.000 94,46 · 10.500 94,63 · 11.000 97,82 · 11.500 99,11 · 12.000 99,11 ·
12.500 96,42 · 13.000 95,18 · 13.500 93,72 · 14.000 94,63.
