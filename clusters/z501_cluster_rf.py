"""Perfiles de clientes BAJA+2 con Random Forest: genera clusters_tendencias.pdf.

Requisitos: Python 3.12+ y
    pip install duckdb pandas numpy scikit-learn matplotlib

Uso:
    python z501_cluster_rf.py --csv competencia_01.csv --evaluar-k 2-8 --k 4

Opciones (todas con default):
    --out clusters_tendencias.pdf   archivo de salida (al lado se escriben los CSV de
                                    asignación, perfil y atributos definitorios)
    --n-trees 300                   árboles del Random Forest
    --min-samples-leaf 50           tamaño mínimo de hoja
    --k 5                           cantidad de clusters (máx. 8)
    --evaluar-k 2-8                 métricas para elegir k → k_eval.csv / k_eval.png
    --top-n 3                       atributos definitorios por cluster
    --seed 214363                   semilla (muestreo, RF y KMeans)
    --banda ic95|desvio|iqr         ancho de la banda en los gráficos
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score,
)

ID_COL = "numero_de_cliente"
MES_COL = "foto_mes"
TARGET_COL = "clase_ternaria"
GRUPO_COL = "grupo"  # 1 = cliente BAJA+2, 0 = cliente fiel
CLUSTER_COL = "cluster"
NON_FEATURE_COLS = (ID_COL, MES_COL, TARGET_COL, GRUPO_COL)
RAIZ_COL = "_sin_split"  # hojas sin split (árbol de un solo nodo): no entran al clustering

COLORES_CLUSTER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7b5cc4", "#8c6d3f", "#00a3b4"]
COLOR_FIELES = "#9aa19c"  # referencia: clientes que se quedan
INK, INK_2, GRID = "#14181a", "#4b534e", "#d7dbd3"

BANDAS = {
    "ic95": ("media", "IC 95% de la media (± 1,96 · desvío / √n)"),
    "desvio": ("media", "± 1 desvío estándar"),
    "iqr": ("mediana", "Q25–Q75"),
}

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 1. Datos: todos los BAJA+2 con su historia + igual cantidad de fieles
# --------------------------------------------------------------------------- #

_QUERY_MUESTRA = """
WITH raw AS (
    SELECT * REPLACE (CAST({id} AS BIGINT) AS {id})
    FROM read_csv('{csv}', sample_size = -1)
),
-- meses de estudio: sólo los que tienen BAJA+2 en el target. Los últimos meses del
-- dataset (sin futuro de 2 meses) no tienen BAJA+2 y no se usan.
meses_ok AS (
    SELECT DISTINCT {mes} FROM raw WHERE {target} = 'BAJA+2'
),
-- churn: mes en que el cliente fue BAJA+2 (si fue más de una vez, la primera)
churn AS (
    SELECT {id}, MIN({mes}) AS mes_baja2 FROM raw WHERE {target} = 'BAJA+2' GROUP BY {id}
),
n_meses AS (
    SELECT COUNT(DISTINCT {mes}) AS n FROM raw
),
-- fieles: presentes todos los meses y nunca BAJA+1/BAJA+2. ORDER BY hash(...) es un
-- shuffle determinístico por seed, así el LIMIT es un muestreo reproducible.
fieles AS (
    SELECT {id}
    FROM raw
    GROUP BY {id}
    HAVING COUNT(*) = (SELECT n FROM n_meses)
       AND SUM(CASE WHEN {target} IN ('BAJA+1', 'BAJA+2') THEN 1 ELSE 0 END) = 0
    ORDER BY hash({id} + {seed})
    LIMIT (SELECT COUNT(*) FROM churn)
)
-- BAJA+2: historia hasta el mes BAJA+2 inclusive. El mes BAJA+1 posterior se descarta:
-- ahí el cliente ya está vaciando todo y el bosque aprendería "ya se fue", no "se va a ir".
SELECT raw.*, 1 AS {grupo}
FROM raw JOIN churn USING ({id})
WHERE raw.{mes} <= churn.mes_baja2 AND raw.{mes} IN (SELECT {mes} FROM meses_ok)
UNION ALL
SELECT raw.*, 0 AS {grupo}
FROM raw JOIN fieles USING ({id})
WHERE raw.{mes} IN (SELECT {mes} FROM meses_ok)
ORDER BY {id}, {mes}
"""


def cargar_muestra(csv_path: Path, seed: int) -> pd.DataFrame:
    query = _QUERY_MUESTRA.format(
        csv=csv_path.as_posix(), id=ID_COL, mes=MES_COL, target=TARGET_COL, grupo=GRUPO_COL, seed=seed
    )
    df = duckdb.connect().execute(query).df()
    df[TARGET_COL] = df[TARGET_COL].astype("string")
    return df


VARS_DELTA = ["mcuentas_saldo", "mcaja_ahorro", "mpayroll", "ctrx_quarter", "tarjetas_consumo",
              "chomebanking_transacciones", "mrentabilidad"]


def agregar_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Total Visa + Master y variación contra el mes anterior (_delta1) de algunas variables clave:
    el bosque ve cada mes por separado, así las caídas pasan a ser una variable más."""
    df = df.sort_values([ID_COL, MES_COL]).copy()
    df["tarjetas_trx"] = df[["ctarjeta_visa_transacciones", "ctarjeta_master_transacciones"]].sum(axis=1)
    df["tarjetas_consumo"] = df[["mtarjeta_visa_consumo", "mtarjeta_master_consumo"]].sum(axis=1)
    for c in VARS_DELTA:
        df[f"{c}_delta1"] = df.groupby(ID_COL)[c].diff()
    return df


def leer_vars(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def columnas_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]


# --------------------------------------------------------------------------- #
# 2. Random Forest sobre filas cliente-mes; target = "el cliente es BAJA+2"
# --------------------------------------------------------------------------- #

def entrenar_rf(X: np.ndarray, y: np.ndarray, n_trees: int, min_samples_leaf: int, seed: int) -> RandomForestClassifier:
    # min_samples_leaf alto: hojas que sean poblaciones de clientes, no filas memorizadas.
    # class_weight compensa que un fiel aporta 4 filas y un BAJA+2 entre 1 y 4.
    rf = RandomForestClassifier(
        n_estimators=n_trees, min_samples_leaf=min_samples_leaf, max_features="sqrt",
        class_weight="balanced", oob_score=True, n_jobs=-1, random_state=seed,
    )
    rf.fit(X, y)
    return rf


# --------------------------------------------------------------------------- #
# 3. Atributo del último split de cada hoja + conteo por cliente
# --------------------------------------------------------------------------- #

def atributo_de_hoja(estimator) -> np.ndarray:
    """Para cada nodo, el atributo con el que se hizo el último split antes de llegar
    a él (el atributo de su padre). En las hojas es "el atributo que define la hoja".
    -1 para la raíz."""
    t = estimator.tree_
    mapa = np.full(t.node_count, -1, dtype=np.int64)
    internos = np.flatnonzero(t.children_left != -1)
    mapa[t.children_left[internos]] = t.feature[internos]
    mapa[t.children_right[internos]] = t.feature[internos]
    return mapa


def contar_hojas_por_id(rf: RandomForestClassifier, X: np.ndarray, ids: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """C[id, a] = cantidad de pares (mes, árbol) en que una fila del id cayó en una
    hoja cuyo último split fue el atributo a. Cada fila suma n_meses(id) × n_trees."""
    hojas = rf.apply(X)  # (n_filas, n_trees)
    n_feat = len(feature_names)
    codigos = np.empty_like(hojas)
    for j, est in enumerate(rf.estimators_):
        codigos[:, j] = atributo_de_hoja(est)[hojas[:, j]]
    codigos[codigos < 0] = n_feat

    idx, ids_unicos = pd.factorize(ids)
    C = np.zeros((len(ids_unicos), n_feat + 1), dtype=np.int64)
    np.add.at(C, (np.repeat(idx, hojas.shape[1]), codigos.ravel()), 1)
    return pd.DataFrame(C, index=pd.Index(ids_unicos, name=ID_COL), columns=feature_names + [RAIZ_COL])


def normalizar(C: pd.DataFrame) -> pd.DataFrame:
    """Proporciones por fila (cada cliente suma 1), sin atributos que nunca definieron una hoja."""
    P = C.drop(columns=[RAIZ_COL], errors="ignore")
    P = P.loc[:, P.sum(axis=0) > 0]
    return P.div(P.sum(axis=1), axis=0)


# --------------------------------------------------------------------------- #
# 4. Clustering sobre el perfil completo de hojas (sólo BAJA+2)
# --------------------------------------------------------------------------- #

def clusterizar(P: pd.DataFrame, k: int, seed: int) -> pd.Series:
    """La distancia entre dos clientes es la distancia entre sus perfiles de hojas:
    qué tan parecido los trató el bosque. KMeans sólo corta ese espacio en k."""
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(P.values)
    orden = pd.Series(km.labels_).value_counts().index.tolist()  # cluster_1 = el más grande
    remap = {viejo: nuevo + 1 for nuevo, viejo in enumerate(orden)}
    return pd.Series([remap[l] for l in km.labels_], index=P.index, name=CLUSTER_COL)


def evaluar_k(P: pd.DataFrame, ks: list[int], seed: int, n_boot: int = 10) -> pd.DataFrame:
    """Métricas para elegir k. silueta y calinski: más alto mejor; davies: más bajo mejor;
    estabilidad = ARI medio entre el clustering completo y KMeans sobre submuestras del 80%
    (cerca de 1 = los grupos no dependen de qué clientes entraron); min_pct = % del cluster más chico."""
    X = P.values
    rng = np.random.default_rng(seed)
    filas = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        aris = []
        for _ in range(n_boot):
            idx = rng.choice(len(X), size=int(0.8 * len(X)), replace=False)
            sub = KMeans(n_clusters=k, n_init=5, random_state=int(rng.integers(1e9))).fit(X[idx])
            aris.append(adjusted_rand_score(km.labels_[idx], sub.labels_))
        filas.append({
            "k": k, "silueta": silhouette_score(X, km.labels_, random_state=seed),
            "calinski": calinski_harabasz_score(X, km.labels_), "davies": davies_bouldin_score(X, km.labels_),
            "estabilidad": float(np.mean(aris)), "inercia": km.inertia_,
            "min_pct": 100 * np.bincount(km.labels_).min() / len(X),
        })
    return pd.DataFrame(filas).set_index("k")


def grafico_eval_k(ev: pd.DataFrame, path: Path) -> None:
    cols = [("silueta", "Silueta (↑)"), ("calinski", "Calinski-Harabasz (↑)"), ("davies", "Davies-Bouldin (↓)"),
            ("estabilidad", "Estabilidad ARI (↑)"), ("inercia", "Inercia / codo (↓)"), ("min_pct", "% cluster más chico")]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (c, t) in zip(axs.ravel(), cols):
        ax.plot(ev.index, ev[c], marker="o", color=COLORES_CLUSTER[0], linewidth=2)
        ax.set_title(t, color=INK, fontsize=11, loc="left")
        ax.set_xticks(ev.index)
        _estilo(ax)
    fig.suptitle("Elección de k (KMeans sobre perfiles de hojas de clientes BAJA+2)", color=INK, fontsize=14, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Caracterización: share y lift por cluster; tendencias mensuales
# --------------------------------------------------------------------------- #

def caracterizar_clusters(P: pd.DataFrame, labels: pd.Series, top_n: int, share_min: float) -> pd.DataFrame:
    """Top-n atributos por cluster según lift = share_cluster / share_global."""
    share_global = P.mean(axis=0)
    filas = []
    for c, Pc in P.groupby(labels):
        share_c = Pc.mean(axis=0)
        tabla = pd.DataFrame({"share_cluster": share_c, "share_global": share_global,
                              "lift": share_c / share_global.replace(0, np.nan)})
        tabla = tabla[tabla["share_cluster"] >= share_min].sort_values("lift", ascending=False).head(top_n)
        for rank, (attr, r) in enumerate(tabla.iterrows(), start=1):
            filas.append({CLUSTER_COL: c, "rank": rank, "atributo": attr, **r.to_dict()})
    return pd.DataFrame(filas)


def tendencias_por_cluster(df_churn: pd.DataFrame, atributos: list[str], meses: list[int], banda: str) -> pd.DataFrame:
    """Por (cluster, mes): centro, banda y n de clientes presentes, por atributo.
    La banda se recorta al rango observado del atributo."""
    d = df_churn[df_churn[MES_COL].isin(meses)]
    g = d.groupby([CLUSTER_COL, MES_COL])[atributos]
    n = g.count()
    if banda == "iqr":
        centro, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
    else:
        centro, sd = g.mean(), g.std()
        ancho = sd if banda == "desvio" else 1.96 * sd / np.sqrt(n)
        lo, hi = centro - ancho, centro + ancho
    lo = lo.clip(lower=d[atributos].min(), axis=1)
    hi = hi.clip(upper=d[atributos].max(), axis=1)
    return pd.concat({"centro": centro, "lo": lo, "hi": hi, "n": n}, axis=1)


VARS_NEGOCIO = [
    "cliente_edad", "cliente_antiguedad", "cliente_vip", "active_quarter", "cproductos", "mrentabilidad_annual",
    "ctrx_quarter", "mcuentas_saldo", "cpayroll_trx", "mpayroll", "ctarjeta_visa", "mtarjeta_visa_consumo",
    "Visa_status", "Visa_delinquency", "cprestamos_personales", "mprestamos_personales", "cplazo_fijo",
    "ccomisiones_mantenimiento", "mcomisiones_mantenimiento", "chomebanking_transacciones", "cmobile_app_trx",
    "ccuenta_debitos_automaticos", "cpagomiscuentas",
]


def perfil_clusters(df: pd.DataFrame, labels: pd.Series, feats: list[str]) -> pd.DataFrame:
    """Foto de cada cluster en el mes en que fue BAJA+2 (el momento en que hay que actuar)
    frente a los fieles en los mismos meses. Variables binarias / flags: % con valor > 0;
    el resto: mediana."""
    churn = df[df[GRUPO_COL] == 1]
    foto = churn[churn[TARGET_COL] == "BAJA+2"].merge(labels, left_on=ID_COL, right_index=True)
    fieles = df[df[GRUPO_COL] == 0]
    flags = [f for f in feats if df[f].dropna().nunique() <= 2]
    def resumir(d: pd.DataFrame) -> pd.Series:
        med = d[feats].median()
        pct = (d[flags] > 0).mean() * 100
        med[flags] = pct
        return med
    cols = {f"cluster_{c}": resumir(g) for c, g in foto.groupby(CLUSTER_COL)}
    cols["fieles"] = resumir(fieles)
    out = pd.DataFrame(cols)
    out.insert(0, "medida", np.where(out.index.isin(flags), "% con valor > 0", "mediana"))
    return out


def _pagina_perfil(pdf: PdfPages, perfil: pd.DataFrame, attrs_def: pd.DataFrame, tam: dict) -> None:
    definitorios = attrs_def["atributo"].unique().tolist()
    filas = [v for v in VARS_NEGOCIO if v in perfil.index] + [v for v in definitorios if v not in VARS_NEGOCIO]
    tabla_df = perfil.loc[filas]
    def fmt(v: float) -> str:
        if pd.isna(v):
            return "–"
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.1f}"
    celdas = [[fmt(v) for v in r] for r in tabla_df.drop(columns="medida").values]
    etiquetas = [f"{v}  ({'%' if m.startswith('%') else 'med'})" + ("  ★" if v in definitorios else "")
                 for v, m in zip(tabla_df.index, tabla_df["medida"])]
    cols = [f"{c}\n(n={tam[int(c.split('_')[1])]:,})" if c.startswith("cluster_") else c
            for c in tabla_df.columns if c != "medida"]
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.04, 0.96, "Ficha de cada cluster en el mes BAJA+2 vs. clientes fieles", fontsize=16, color=INK, weight="bold", va="top")
    fig.text(0.04, 0.915, "med = mediana · % = porcentaje de clientes con el atributo > 0 · ★ = atributo definitorio de algún cluster",
             fontsize=9, color=INK_2, va="top")
    ax = fig.add_axes([0.26, 0.02, 0.72, 0.86])
    ax.axis("off")
    t = ax.table(cellText=celdas, rowLabels=etiquetas, colLabels=cols, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1, 1.0 if len(filas) > 30 else 1.25)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0 and c >= 0:
            cell.set_text_props(color="white", weight="bold")
            cell.set_facecolor(COLOR_FIELES if c == len(cols) - 1 else COLORES_CLUSTER[c])
            cell.set_height(cell.get_height() * 1.6)
    pdf.savefig(fig)
    plt.close(fig)


def orden_atributos(atributos_def: pd.DataFrame, feats: list[str]) -> list[str]:
    """Primero los definitorios (en orden de cluster y rank), después el resto por abecedario."""
    primero = atributos_def.sort_values([CLUSTER_COL, "rank"])["atributo"].unique().tolist()
    return primero + sorted((f for f in feats if f not in primero), key=str.lower)


# --------------------------------------------------------------------------- #
# 6. PDF: página de resumen + una página apaisada por atributo
# --------------------------------------------------------------------------- #

def _estilo(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _pagina_resumen(pdf: PdfPages, ctx: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    texto = [
        f"Muestra: {ctx['n_churn']:,} clientes BAJA+2 (toda su historia) + {ctx['n_fieles']:,} fieles.",
        f"Random Forest: {ctx['n_trees']} árboles, min_samples_leaf={ctx['min_samples_leaf']}, OOB accuracy {ctx['oob']:.3f}.",
        f"KMeans k={ctx['k']} sobre la matriz id × atributo (proporción de hojas definidas por cada atributo).",
        "",
        "Tamaño de cada cluster: " + ", ".join(f"cluster_{c}: {n:,}" for c, n in ctx["tam"].items()),
        "",
        "Cómo leer cada página:",
        "  • Título = atributo. Primero los definitorios (top-n por lift de cada cluster, en orden de cluster),",
        "    después todos los demás en orden alfabético.",
        f"  • Eje X = mes calendario ({ctx['meses'][0]}–{ctx['meses'][-1]}); se excluyen {', '.join(map(str, ctx['meses_excluidos']))}"
        " porque no tienen BAJA+2 (falta el futuro de 2 meses).",
        "  • De cada BAJA+2 se usa su historia hasta el mes en que es BAJA+2 (el mes BAJA+1 posterior no entra).",
        f"  • Línea = {ctx['centro']} del atributo entre los clientes del cluster presentes ese mes;",
        "    línea gris punteada = misma medida en los clientes fieles (referencia de 'cliente sano').",
        f"    banda = {ctx['banda_desc']}, recortada al rango observado del atributo.",
        "  • Los clusters se achican mes a mes (tabla de n al pie): los clientes se van dando de baja.",
        "  • Si un atributo define a un cluster, su línea debería despegarse claramente de las otras.",
    ]
    fig.text(0.06, 0.92, "Clusters de clientes BAJA+2 — tendencia mensual por atributo",
             fontsize=18, color=INK, weight="bold", va="top")
    fig.text(0.06, 0.84, "\n".join(texto), fontsize=11, color=INK_2, va="top", linespacing=1.6)
    pdf.savefig(fig)
    plt.close(fig)


def pdf_tendencias(tend: pd.DataFrame, atributos: list[str], atributos_def: pd.DataFrame, lift_all: pd.DataFrame,
                   meses: list[int], path: Path, ctx: dict) -> None:
    todos = sorted(tend.index.get_level_values(CLUSTER_COL).unique())  # 0 = fieles (referencia)
    clusters = [c for c in todos if c != 0]
    with PdfPages(path) as pdf:
        _pagina_resumen(pdf, ctx)
        _pagina_perfil(pdf, ctx["perfil"], atributos_def, ctx["tam"])
        for attr in atributos:
            define = atributos_def[atributos_def["atributo"] == attr].sort_values("lift", ascending=False)
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_axes([0.07, 0.30, 0.90, 0.54])
            x = np.arange(len(meses))
            if 0 in todos:
                s = tend.xs(0, level=CLUSTER_COL)
                ax.plot(x, s["centro"][attr].reindex(meses).values, color=COLOR_FIELES, linewidth=2, linestyle="--",
                        marker="s", markersize=5, label="fieles (referencia)")
            for i, c in enumerate(clusters):
                s = tend.xs(c, level=CLUSTER_COL)
                centro = s["centro"][attr].reindex(meses).values
                lo, hi = s["lo"][attr].reindex(meses).values, s["hi"][attr].reindex(meses).values
                ax.fill_between(x, lo, hi, color=COLORES_CLUSTER[i], alpha=0.12, linewidth=0)
                ax.plot(x, centro, color=COLORES_CLUSTER[i], linewidth=2.2, marker="o", markersize=6, label=f"cluster_{c}")
            ax.set_xticks(x, meses)
            ax.set_xlim(-0.3, len(meses) - 0.7)
            _estilo(ax)
            ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(todos))
            ax.set_ylabel(attr, color=INK_2)

            fig.text(0.07, 0.945, attr, fontsize=20, color=INK, weight="bold")
            if len(define):
                quien = "; ".join(f"cluster_{r[CLUSTER_COL]} (lift {r['lift']:.2f}, rank {r['rank']})" for _, r in define.iterrows())
                rol = f"Define a {quien}."
            elif attr in lift_all.columns:
                c_max = lift_all[attr].idxmax()
                rol = f"No es definitorio de ningún cluster (mayor lift: cluster_{c_max}, {lift_all.loc[c_max, attr]:.2f})."
            else:
                rol = "Nunca definió una hoja del bosque."
            fig.text(0.07, 0.925, f"{rol}\nLínea = {ctx['centro']} mensual entre los clientes del cluster presentes ese mes; "
                     f"banda = {ctx['banda_desc']}, recortada al rango observado.", fontsize=10, color=INK_2, va="top", linespacing=1.4)

            # tabla de n por cluster y mes: hace visible el achicamiento de los grupos
            n_tab = tend["n"][attr].unstack(MES_COL).reindex(index=clusters, columns=meses).fillna(0).astype(int)
            ax_t = fig.add_axes([0.07, 0.05, 0.90, 0.17])
            ax_t.axis("off")
            tabla = ax_t.table(cellText=n_tab.values, rowLabels=[f"cluster_{c}" for c in clusters],
                               colLabels=[str(m) for m in meses], loc="center", cellLoc="center")
            tabla.auto_set_font_size(False)
            tabla.set_fontsize(8.5)
            tabla.scale(1, 1.15)
            for (r, _c), cell in tabla.get_celld().items():
                cell.set_edgecolor(GRID)
                cell.get_text().set_color(INK_2 if r == 0 else INK)
            ax_t.set_title("n de clientes del cluster presentes en cada mes", loc="left", fontsize=9, color=INK_2)
            pdf.savefig(fig)
            plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("competencia_01.csv"))
    p.add_argument("--out", type=Path, default=Path("clusters_tendencias.pdf"))
    p.add_argument("--n-trees", type=int, default=300)
    p.add_argument("--min-samples-leaf", type=int, default=50)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--evaluar-k", type=str, default=None, metavar="2-8",
                   help="evalúa este rango de k (métricas + k_eval.png) antes de clusterizar")
    p.add_argument("--vars", type=Path, default=None,
                   help="txt con una variable por línea (ej. variables_seleccionadas.txt de z502): sólo esas entran al RF")
    p.add_argument("--derivadas", action="store_true", help="agrega totales Visa+Master y variaciones mensuales")
    p.add_argument("--comparar", type=Path, default=None,
                   help="clusters_asignacion.csv de otra corrida: imprime la tabla cruzada y el ARI")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--seed", type=int, default=214363)
    p.add_argument("--banda", choices=list(BANDAS), default="ic95")
    return p.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # consola de Windows (cp1252) no imprime → · etc.
    args = parse_args()
    if args.k > len(COLORES_CLUSTER):
        raise SystemExit(f"--k máximo {len(COLORES_CLUSTER)}")
    if not args.csv.exists():
        raise SystemExit(f"no encuentro {args.csv}")

    log(f"leyendo {args.csv.name}")
    df = cargar_muestra(args.csv, args.seed)
    meses = sorted(df[MES_COL].unique().tolist())  # la query ya dejó sólo meses con BAJA+2
    meses_excluidos = sorted(set(duckdb.connect().execute(
        f"SELECT DISTINCT {MES_COL} FROM read_csv('{args.csv.as_posix()}', sample_size = -1)").df()[MES_COL]) - set(meses))
    log(f"meses usados: {meses} · excluidos (sin BAJA+2): {meses_excluidos}")
    if args.derivadas or args.vars:
        df = agregar_derivadas(df)
    feats_todas = columnas_features(df)
    feats = feats_todas
    if args.vars:
        pedidas = leer_vars(args.vars)
        faltan = [v for v in pedidas if v not in feats_todas]
        if faltan:
            log(f"variables de {args.vars.name} que no existen (se ignoran): {faltan}")
        feats = [v for v in pedidas if v in feats_todas]
        log(f"usando {len(feats)} variables de {args.vars.name}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    por_grupo = df.groupby(GRUPO_COL)[ID_COL].agg(ids="nunique", filas="size")
    log(f"muestra: {len(df):,} filas × {len(feats)} features · "
        + " · ".join(f"grupo {g}: {r.ids:,} ids / {r.filas:,} filas" for g, r in por_grupo.iterrows()))

    X = df[feats].to_numpy(dtype=np.float32)
    y = df[GRUPO_COL].to_numpy()
    ids = df[ID_COL].to_numpy()

    log(f"entrenando RF ({args.n_trees} árboles, min_samples_leaf={args.min_samples_leaf})")
    rf = entrenar_rf(X, y, args.n_trees, args.min_samples_leaf, args.seed)
    log(f"OOB accuracy {rf.oob_score_:.4f}")

    log("contando hojas por cliente")
    P = normalizar(contar_hojas_por_id(rf, X, ids, feats))
    ids_churn = pd.Index(df.loc[df[GRUPO_COL] == 1, ID_COL].unique())
    P_churn = P.loc[ids_churn]

    if args.evaluar_k:
        k_min, k_max = map(int, args.evaluar_k.split("-"))
        log(f"evaluando k = {k_min}..{k_max}")
        ev = evaluar_k(P_churn, list(range(k_min, k_max + 1)), args.seed)
        log("métricas por k:\n" + ev.round(3).to_string())
        ev.to_csv(args.out.with_name("k_eval.csv"))
        grafico_eval_k(ev, args.out.with_name("k_eval.png"))

    labels = clusterizar(P_churn, args.k, args.seed)
    tam = labels.value_counts().sort_index()
    log("clusters: " + ", ".join(f"cluster_{c}={n:,}" for c, n in tam.items()))

    attrs_def = caracterizar_clusters(P_churn, labels, args.top_n, share_min=0.5 / P_churn.shape[1])
    for c, g in attrs_def.groupby(CLUSTER_COL):
        log(f"  cluster_{c}: " + ", ".join(f"{r.atributo} (lift {r.lift:.1f})" for r in g.itertuples()))
    lift_all = P_churn.groupby(labels).mean() / P_churn.mean(axis=0)

    df_churn = df[df[GRUPO_COL] == 1].merge(labels, left_on=ID_COL, right_index=True)
    df_fieles = df[df[GRUPO_COL] == 0].assign(**{CLUSTER_COL: 0})
    atributos_pdf = orden_atributos(attrs_def, feats)
    tend = tendencias_por_cluster(pd.concat([df_churn, df_fieles]), atributos_pdf, meses, args.banda)

    if args.comparar:
        prev = pd.read_csv(args.comparar, index_col=0)["cluster"]
        comunes = labels.index.intersection(prev.index)
        log(f"comparación con {args.comparar} (ARI {adjusted_rand_score(prev[comunes], labels[comunes]):.3f}; "
            "1 = mismos grupos):\n" + pd.crosstab(prev[comunes], labels[comunes], rownames=["antes"], colnames=["ahora"]).to_string())

    perfil = perfil_clusters(df, labels, feats_todas)  # la ficha siempre con todas las variables
    perfil.to_csv(args.out.with_name("clusters_perfil.csv"), float_format="%.2f")
    labels.rename_axis(ID_COL).to_csv(args.out.with_name("clusters_asignacion.csv"))
    attrs_def.to_csv(args.out.with_name("clusters_atributos_definitorios.csv"), index=False, float_format="%.4f")

    ctx = dict(
        n_trees=args.n_trees, min_samples_leaf=args.min_samples_leaf, k=args.k, oob=rf.oob_score_,
        n_churn=int(por_grupo.loc[1, "ids"]), n_fieles=int(por_grupo.loc[0, "ids"]),
        meses=meses, meses_excluidos=meses_excluidos, tam=tam.to_dict(), perfil=perfil,
        centro=BANDAS[args.banda][0], banda_desc=BANDAS[args.banda][1],
    )
    log(f"generando PDF ({len(atributos_pdf) + 1} páginas)")
    pdf_tendencias(tend, atributos_pdf, attrs_def, lift_all, meses, args.out, ctx)
    log(f"listo → {args.out}")


if __name__ == "__main__":
    main()
