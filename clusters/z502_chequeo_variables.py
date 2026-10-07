"""Chequeo de variables para los clusters BAJA+2.

1. Relevancia global: qué variables separan a los BAJA+2 (en su mes BAJA+2) de los fieles.
   Combina la importancia del Random Forest con un efecto por rangos (AUC - 0,5), que capta
   diferencias de montos aunque todos tengan algo y no lo afectan los valores extremos.
2. Por cluster (si existe la asignación): efecto de cada variable cluster vs. fieles y marca
   las relevantes que no están en el guion del video.
3. Propone variables para clusterizar: las más relevantes, descartando casi duplicados
   (correlación de Spearman > --max-corr con una ya elegida).

Uso:
    python z502_chequeo_variables.py --asignacion clusters_asignacion.csv --n-vars 40
Salidas (en --dir): chequeo_variables.csv, chequeo_por_cluster.csv, chequeo_variables.pdf,
    variables_seleccionadas.txt (sirve directo para z501 --vars)

Cómo leer el efecto: P(cliente del grupo > cliente fiel) - 0,5.
    +0,30 → el grupo tiene un valor más alto que un fiel en el 80% de las comparaciones.
    -0,30 → más bajo en el 80%. Cerca de 0 → no se diferencia.
    Valores vacíos (ej. variables de tarjeta sin tarjeta) cuentan como el mínimo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import z501_cluster_rf as z

# variables que se usan en guion_video_clusters.md (tabla resumen + fichas)
GUION_VARS = {
    "cliente_antiguedad", "ctrx_quarter", "active_quarter", "mpayroll", "cpayroll_trx", "mcuentas_saldo",
    "mcaja_ahorro", "ctarjeta_visa_transacciones", "mtarjeta_visa_consumo", "mttarjeta_visa_debitos_automaticos",
    "Visa_status", "ctarjeta_visa", "Visa_mfinanciacion_limite", "Visa_fultimo_cierre", "Visa_delinquency",
    "cprestamos_personales", "mprestamos_personales", "cdescubierto_preacordado", "mcomisiones_mantenimiento",
    "cmobile_app_trx", "chomebanking_transacciones", "mrentabilidad_annual", "cproductos", "matm", "internet",
    "mpasivos_margen",
}


def efecto(x_grupo: np.ndarray, x_ref: np.ndarray) -> float:
    """AUC de Mann-Whitney - 0,5, con empates promediados."""
    n1, n2 = len(x_grupo), len(x_ref)
    r = pd.Series(np.concatenate([x_grupo, x_ref])).rank().to_numpy()
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2) - 0.5


def rellenar(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """NaN → por debajo del mínimo de la variable (sin producto = el valor más bajo)."""
    return df[feats].fillna(df[feats].min() - 1).fillna(0)


def seleccionar(score: pd.Series, X: pd.DataFrame, n: int, max_corr: float) -> list[str]:
    corr = X[score.index].rank().corr().abs()  # Spearman
    elegidas: list[str] = []
    for v in score.sort_values(ascending=False).index:
        if all(corr.loc[v, e] <= max_corr for e in elegidas):
            elegidas.append(v)
        if len(elegidas) == n:
            break
    return elegidas


def pdf_chequeo(glob: pd.DataFrame, por_c: pd.DataFrame | None, path: Path, n_top: int = 30) -> None:
    with PdfPages(path) as pdf:
        top = glob.head(n_top).iloc[::-1]
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        colores = [z.COLORES_CLUSTER[0] if s else z.COLOR_FIELES for s in top["seleccionada"]]
        ax.barh(top.index, top["score"], color=colores)
        ax.set_yticks(range(len(top)), [f"{v}{'  ★' if v in GUION_VARS else ''}" for v in top.index], fontsize=8)
        for i, (_, r) in enumerate(top.iterrows()):
            ax.text(r["score"] + 0.005, i, f"efecto {r['efecto']:+.2f}", va="center", fontsize=7, color=z.INK_2)
        z._estilo(ax)
        ax.set_title(f"Top {n_top} variables que separan BAJA+2 de fieles  (azul = seleccionada para clusterizar, "
                     "gris = descartada por casi duplicada, ★ = está en el guion)", loc="left", fontsize=10, color=z.INK)
        ax.set_xlabel("score = promedio de percentiles de importancia RF y |efecto|", color=z.INK_2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if por_c is None:
            return
        piv = por_c.pivot(index="variable", columns="cluster", values="efecto")
        filas = []
        for c in piv.columns:
            filas += piv[c].abs().sort_values(ascending=False).head(10).index.tolist()
        piv = piv.loc[list(dict.fromkeys(filas))]
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        ax.set_xticks(range(piv.shape[1]), [f"cluster_{c}" for c in piv.columns])
        ax.set_yticks(range(len(piv)), [f"{v}{'  ★' if v in GUION_VARS else ''}" for v in piv.index], fontsize=7.5)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.3 else z.INK)
        fig.colorbar(im, ax=ax, shrink=0.4, label="efecto vs. fieles")
        ax.set_title("Top 10 variables de cada cluster vs. fieles (en el mes BAJA+2)\n"
                     "rojo = más alto que un fiel · azul = más bajo · ★ = está en el guion", loc="left", fontsize=10, color=z.INK)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("competencia_01.csv"))
    p.add_argument("--asignacion", type=Path, default=Path("clusters_asignacion.csv"))
    p.add_argument("--n-vars", type=int, default=40)
    p.add_argument("--max-corr", type=float, default=0.9)
    p.add_argument("--dir", type=Path, default=Path("."))
    p.add_argument("--seed", type=int, default=214363)
    args = p.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)

    z.log(f"leyendo {args.csv.name}")
    df = z.agregar_derivadas(z.cargar_muestra(args.csv, args.seed))
    feats = z.columnas_features(df)
    X = rellenar(df, feats)
    es_foto = (df[z.TARGET_COL] == "BAJA+2").to_numpy()  # BAJA+2 en su mes BAJA+2
    es_fiel = (df[z.GRUPO_COL] == 0).to_numpy()

    z.log(f"RF de importancia sobre {len(feats)} variables")
    rf = z.entrenar_rf(df[feats].to_numpy(dtype=np.float32), df[z.GRUPO_COL].to_numpy(), 300, 50, args.seed)
    glob = pd.DataFrame({
        "importancia_rf": rf.feature_importances_,
        "efecto": [efecto(X.loc[es_foto, f].to_numpy(), X.loc[es_fiel, f].to_numpy()) for f in feats],
        "mediana_baja2": df.loc[es_foto, feats].median().to_numpy(),
        "mediana_fieles": df.loc[es_fiel, feats].median().to_numpy(),
        "pct_mayor0_baja2": (df.loc[es_foto, feats] > 0).mean().to_numpy() * 100,
        "pct_mayor0_fieles": (df.loc[es_fiel, feats] > 0).mean().to_numpy() * 100,
    }, index=pd.Index(feats, name="variable"))
    glob["score"] = (glob["importancia_rf"].rank(pct=True) + glob["efecto"].abs().rank(pct=True)) / 2
    glob = glob.sort_values("score", ascending=False)
    elegidas = seleccionar(glob["score"], X, args.n_vars, args.max_corr)
    glob["seleccionada"] = glob.index.isin(elegidas)
    glob["en_guion"] = glob.index.isin(GUION_VARS)
    glob.round(4).to_csv(args.dir / "chequeo_variables.csv")
    (args.dir / "variables_seleccionadas.txt").write_text(
        f"# top {args.n_vars} de z502 (sin pares con Spearman > {args.max_corr})\n" + "\n".join(elegidas) + "\n", encoding="utf-8")

    print("\nTOP 25 variables (BAJA+2 vs fieles):  ★ = en guion, ✓ = seleccionada")
    for v, r in glob.head(25).iterrows():
        print(f"  {'✓' if r.seleccionada else ' '} {'★' if r.en_guion else ' '} {v:38s} efecto {r.efecto:+.2f}  "
              f"mediana {r.mediana_baja2:>10,.0f} vs {r.mediana_fieles:>10,.0f}")
    fuera = glob.head(40).query("not en_guion").index.tolist()
    print(f"\nEntre las 40 más relevantes, NO están en el guion: {', '.join(fuera)}")

    por_c = None
    if args.asignacion.exists():
        lab = pd.read_csv(args.asignacion, index_col=0)["cluster"]
        cl = df[z.ID_COL].map(lab).to_numpy()
        filas = []
        for c in sorted(lab.unique()):
            m = es_foto & (cl == c)
            for f in feats:
                filas.append({"cluster": c, "variable": f,
                              "efecto": efecto(X.loc[m, f].to_numpy(), X.loc[es_fiel, f].to_numpy()),
                              "mediana_cluster": df.loc[m, f].median(), "mediana_fieles": glob.loc[f, "mediana_fieles"],
                              "pct_mayor0_cluster": (df.loc[m, f] > 0).mean() * 100,
                              "pct_mayor0_fieles": glob.loc[f, "pct_mayor0_fieles"], "en_guion": f in GUION_VARS})
        por_c = pd.DataFrame(filas)
        por_c.round(4).to_csv(args.dir / "chequeo_por_cluster.csv", index=False)
        print(f"\nPor cluster ({args.asignacion.name}): top 8 vs fieles  (★ = en guion)")
        for c, g in por_c.groupby("cluster"):
            g = g.reindex(g["efecto"].abs().sort_values(ascending=False).index).head(8)
            print(f"  cluster_{c}: " + " · ".join(f"{'★' if r.en_guion else ''}{r.variable} {r.efecto:+.2f}" for r in g.itertuples()))

    pdf_chequeo(glob, por_c, args.dir / "chequeo_variables.pdf")
    z.log(f"listo → {args.dir / 'chequeo_variables.pdf'}")


if __name__ == "__main__":
    main()
