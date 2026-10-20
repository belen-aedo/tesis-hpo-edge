"""
report.py - Etapa 5: tablas LaTeX para la tesis.

Convierte los resultados de evaluate.py y edge_benchmark.py en archivos .tex que
el documento incluye con \\input{}. Asi ningun numero del texto se escribe a mano.
Si una tabla todavia no tiene datos, se genera con guiones y con la marca
[Pendiente].

Genera en outputs/tables/:
    tab_mock_performance.tex   <- outputs/results/final_evaluation.csv
    tab_mock_hardware.tex      <- outputs/results/edge_benchmark.csv
    tab_mock_balance.tex       <- outputs/results/balance.csv

USO
    python src/report.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import load_config, outputs_dir, results_dir

DASH = "---"
NAMES = {"iforest": "iForest", "lof": "LOF", "autoencoder": "AE"}
METRIC_ES = {
    "por_defecto": "Por defecto",
    "silhouette": "Silhouette",
    "davies_bouldin": "Davies--Bould.",
    "reconstruction": "Error recons.",
    "stability": "Estabilidad",
    "consistency": "Consistencia",
    "oracle": "Or\\'aculo (MCC)",
    "oracle_q": "Or\\'aculo (MCC, $q$ libre)",
}


def _wrap(caption: str, label: str, colspec: str, header: str, body: str,
          size: str = "\\scriptsize", pending: bool = False) -> str:
    tag = "[Pendiente] " if pending else ""
    return f"""% Generado automaticamente por src/report.py -- no editar a mano
\\begin{{table}}[t]
\\caption{{{tag}{caption}}}
\\label{{{label}}}
\\centering
{size}
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{{colspec}}}
\\toprule
{header}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def _pending_rows(cfg, n_dash: int, with_metric: bool = True) -> str:
    if not with_metric:
        return "\n".join(f"{NAMES[d]} & " + " & ".join([DASH] * n_dash) + " \\\\"
                         for d in cfg["detectors"])
    return "\n".join(f"{NAMES[d]} & {METRIC_ES.get(m, m)} & " + " & ".join([DASH] * n_dash) + " \\\\"
                     for d, ms in cfg["detectors"].items() for m in ["por_defecto"] + list(ms))


def table_performance(cfg, path: Path) -> None:
    src = results_dir(cfg) / "final_evaluation.csv"
    header = ("\\textbf{Detector} & \\textbf{Criterio} & \\textbf{MCC} & \\textbf{BACC} & "
              "\\textbf{$F_1$} & \\textbf{Recall} & \\textbf{FPR} & \\textbf{FNR} \\\\")
    cols = "llcccccc"
    cap = "Capacidad de deteccion sobre el conjunto de prueba aislado."
    if not src.exists():
        path.write_text(_wrap(cap + " Requiere ejecutar optimize.py y evaluate.py.",
                              "tab:mock_performance", cols, header, _pending_rows(cfg, 6),
                              pending=True), encoding="utf-8")
        return
    df = pd.read_csv(src)
    df = df[df["config"] != "linea_base_nativa"]
    agg = (df.groupby(["detector", "metric"])[["MCC", "BACC", "F1", "Recall", "FPR", "FNR"]]
             .mean().reset_index())
    rows = [f"{NAMES.get(r['detector'], r['detector'])} & {METRIC_ES.get(r['metric'], r['metric'])} & "
            f"{r['MCC']:.3f} & {r['BACC']:.3f} & {r['F1']:.3f} & {r['Recall']:.3f} & "
            f"{r['FPR']:.3f} & {r['FNR']:.3f} \\\\" for _, r in agg.iterrows()]
    path.write_text(_wrap(cap + " Promedio sobre las semillas ejecutadas.", "tab:mock_performance",
                          cols, header, "\n".join(rows)), encoding="utf-8")


def table_hardware(cfg, path: Path) -> None:
    src = results_dir(cfg) / "edge_benchmark.csv"
    header = ("\\textbf{Detector} & \\textbf{Criterio} & \\textbf{$t$ med.} & "
              "\\textbf{$t$ p95} & \\textbf{CPU} & \\textbf{RAM} \\\\\n"
              " & & \\textbf{(ms)} & \\textbf{(ms)} & \\textbf{(\\%)} & \\textbf{(MB)} \\\\")
    cols = "llcccc"
    cap = "Consumo de recursos medido sobre la Raspberry Pi~5."
    if not src.exists():
        path.write_text(_wrap(cap + " Requiere ejecutar edge\\_benchmark.py.", "tab:mock_hardware",
                              cols, header, _pending_rows(cfg, 4), pending=True), encoding="utf-8")
        return
    df = pd.read_csv(src)
    sel = results_dir(cfg) / "selected_models.csv"
    keys = ["detector", "metric", "seed", "rank"]
    if sel.exists() and all(k in df.columns for k in keys):   # solo las configuraciones elegidas
        df = df.merge(pd.read_csv(sel)[keys], on=keys, how="inner")
    elif sel.exists():
        print("AVISO: edge_benchmark.csv es de una version anterior; vuelva a correr edge_benchmark.py")
    agg = (df.groupby(["detector", "metric"])
             [["latency_median_ms", "latency_p95_ms", "cpu_util_at_rate", "ram_peak_mb"]]
             .mean().reset_index())
    rows = [f"{NAMES.get(r['detector'], r['detector'])} & {METRIC_ES.get(r['metric'], r['metric'])} & "
            f"{r['latency_median_ms']:.2f} & {r['latency_p95_ms']:.2f} & "
            f"{r['cpu_util_at_rate']:.1f} & {r['ram_peak_mb']:.1f} \\\\" for _, r in agg.iterrows()]
    path.write_text(_wrap(cap, "tab:mock_hardware", cols, header, "\n".join(rows)), encoding="utf-8")


def table_balance(cfg, path: Path) -> None:
    src = results_dir(cfg) / "balance.csv"
    header = ("\\textbf{Detector} & \\textbf{Mejor criterio} & \\textbf{MCC base} & "
              "\\textbf{MCC opt.} & \\textbf{$\\Delta$MCC} \\\\")
    cols = "llccc"
    cap = "Equilibrio deteccion/costo: mejor criterio por detector frente a la linea base."
    if not src.exists():
        path.write_text(_wrap(cap + " Requiere ejecutar evaluate.py.", "tab:mock_balance", cols,
                              header, _pending_rows(cfg, 4, with_metric=False), pending=True),
                        encoding="utf-8")
        return
    df = pd.read_csv(src)
    rows = [f"{NAMES.get(r['detector'], r['detector'])} & "
            f"{METRIC_ES.get(r['mejor_metrica'], r['mejor_metrica'])} & "
            f"{r['mcc_linea_base']:.3f} & {r['mcc_medio']:.3f} & {r['delta_mcc']:+.3f} \\\\"
            for _, r in df.iterrows()]
    path.write_text(_wrap(cap, "tab:mock_balance", cols, header, "\n".join(rows)), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = outputs_dir(cfg) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table_performance(cfg, out / "tab_mock_performance.tex")
    table_hardware(cfg, out / "tab_mock_hardware.tex")
    table_balance(cfg, out / "tab_mock_balance.tex")
    for f in sorted(out.glob("*.tex")):
        state = "PENDIENTE" if "[Pendiente]" in f.read_text(encoding="utf-8") else "con datos"
        print(f"  {f.name:30s} {state}")
    print(f"\nTablas en {out}")


if __name__ == "__main__":
    main()
