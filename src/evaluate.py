"""
evaluate.py - Etapa 4: evaluacion final sobre test_final.

Es el unico script que puede abrir test_final y sus etiquetas. Se corre una vez,
cuando las configuraciones ya estan elegidas. Antes de abrir el archivo revisa
que su hash (SHA-256) coincida con el que guardo prepare_splits.py.

Como se elige la configuracion (sin etiquetas)
----------------------------------------------
Para cada combinacion de detector, metrica y semilla, optimize.py deja las k
mejores configuraciones, ordenadas segun su propia funcion objetivo. Aqui se toma
la primera que cumple los limites del borde medidos en la Raspberry (latencia,
CPU y RAM; ver edge_benchmark.csv). El MCC no se usa para elegir, solo se
reporta: elegir con el MCC del test seria usar las etiquetas de prueba.
Si todavia no existe edge_benchmark.csv, se usa la de rango 0 y se avisa.

Escenarios que se comparan en test_final (todos con los mismos datos)
    linea_base  hiperparametros por defecto + umbral en el percentil q
    optimizada  elegida con un criterio no supervisado (silhouette, davies_bouldin, ...)
    oraculo     MCC maximizado con las etiquetas de test_oracle (cota superior)
    (linea_base_nativa: la decision propia de la libreria, solo como referencia)

Metricas: la principal es el MCC con el umbral q. Como apoyo tambien se reportan
el MCC con otros q, el MCC ajustado a otra proporcion de ataques, AUROC y AUPRC
(que no dependen del umbral) y el recall por tipo de ataque.

Salidas en outputs/results/: final_evaluation.csv, selected_models.csv,
balance.csv, balance_por_metrica.csv (con la distancia al oraculo),
recall_por_ataque.csv, pareto_mcc_latencia.csv, hypothesis_check.json, resumen.md
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

from common import (load_config, load_labeled, models_dir, ranking_metrics, results_dir,
                    save_json, sha256_of, split_path, splits_dir, supervised_metrics)

EDGE_COLS = {"detector", "metric", "seed", "rank", "cumple_latencia", "cumple_cpu", "cumple_ram"}


def verify_integrity(cfg) -> None:
    mp = splits_dir(cfg) / "manifest.json"
    if not mp.exists():
        print("AVISO: no hay manifiesto; se omite la verificacion de integridad.")
        return
    man = json.loads(mp.read_text(encoding="utf-8"))
    for name in ("test_oracle", "test_final"):
        exp = man.get("sha256", {}).get(name)
        if exp and exp != sha256_of(split_path(cfg, name)):
            raise SystemExit(f"El hash de {name} NO coincide con el manifiesto: el bloque fue "
                             "modificado; resultados no validos. Repita prepare_splits.py.")
    print("Integridad de los bloques de prueba verificada (SHA-256).")


def select_entries(index: list[dict], edge: pd.DataFrame | None) -> list[dict]:
    """Elige un modelo por cada (detector, metrica, semilla), sin mirar etiquetas."""
    groups = defaultdict(list)
    for e in index:
        groups[(e["detector"], e["metric"], e["seed"])].append(e)
    chosen = []
    for key, items in sorted(groups.items()):
        items.sort(key=lambda e: e["rank"])
        pick, feasible = items[0], None
        if edge is not None and key[1] != "por_defecto":
            feasible = False
            for e in items:
                r = edge[(edge.detector == e["detector"]) & (edge.metric == e["metric"]) &
                         (edge.seed == e["seed"]) & (edge["rank"] == e["rank"])]
                if len(r) and bool(r.iloc[0][["cumple_latencia", "cumple_cpu", "cumple_ram"]].all()):
                    pick, feasible = e, True
                    break
        chosen.append({**pick, "selected_feasible": feasible})
    return chosen


def evaluate_entry(entry, mdir, X, y, cfg):
    with open(mdir / entry["file"], "rb") as fh:
        det = pickle.load(fh)
    s = det.scores(X)
    pred = (s > det.threshold_).astype(int)
    row = supervised_metrics(y, pred, cfg["split"]["report_prevalences"])
    row.update(ranking_metrics(y, s))
    for q in cfg["threshold"]["sensitivity_q"]:
        m = supervised_metrics(y, (s > det.threshold_for(q)).astype(int))
        tag = f"q{q * 100:g}"
        row[f"MCC_{tag}"], row[f"FPR_{tag}"], row[f"Recall_{tag}"] = m["MCC"], m["FPR"], m["Recall"]
    return row, det, pred


def summarize(df: pd.DataFrame, cfg) -> list[dict]:
    out = results_dir(cfg)
    base = df[df.config == "linea_base"].set_index(["detector", "seed"])
    detail = []
    for det_name in df.detector.unique():
        sub = df[(df.detector == det_name) & (df.config.isin(["optimizada", "oraculo"]))]
        for metric, g in sub.groupby("metric"):
            g = g.set_index("seed")
            seeds = [s for s in g.index if (det_name, s) in base.index]
            if not seeds:
                continue
            d = np.array([g.loc[s, "MCC"] - base.loc[(det_name, s), "MCC"] for s in seeds])
            n = len(d)
            se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            detail.append({"detector": det_name, "metric": metric, "config": g.config.iloc[0],
                           "n_semillas": n, "mcc_medio": g.loc[seeds, "MCC"].mean(),
                           "mcc_std": g.loc[seeds, "MCC"].std(ddof=1) if n > 1 else 0.0,
                           "mcc_base_medio": np.mean([base.loc[(det_name, s), "MCC"] for s in seeds]),
                           "delta_mcc": d.mean(), "delta_se": se,
                           "auroc_medio": g.loc[seeds, "AUROC"].mean(),
                           "fpr_medio": g.loc[seeds, "FPR"].mean(),
                           "recall_medio": g.loc[seeds, "Recall"].mean()})
    det_df = pd.DataFrame(detail)
    if not det_df.empty:     # distancia al oraculo = MCC medio del oraculo - MCC medio del criterio
        orc_mcc = det_df[det_df.metric == "oracle"].set_index("detector")["mcc_medio"]
        det_df["dist_oraculo"] = det_df.detector.map(orc_mcc) - det_df.mcc_medio
    det_df.to_csv(out / "balance_por_metrica.csv", index=False)

    balance, hyp = [], {}
    for det_name, g in det_df.groupby("detector"):
        un = g[g.config == "optimizada"]
        if un.empty:
            continue
        ranked = un.sort_values("mcc_medio", ascending=False)
        best = ranked.iloc[0]
        orc = g[g.metric == "oracle"]
        orc_q = g[g.metric == "oracle_q"]
        se2 = 2 * best.delta_se if np.isfinite(best.delta_se) else 0.0
        if int(best.n_semillas) < 2:      # con 1 semilla no hay error estandar, asi que no se puede concluir
            estado = "sin evidencia (1 semilla)"
        else:
            estado = ("mejora" if best.delta_mcc - se2 > 0 else
                      "empeora" if best.delta_mcc + se2 < 0 else "mantiene")
        runner = ranked.iloc[1] if len(ranked) > 1 else None
        margin = float(best.mcc_medio - runner.mcc_medio) if runner is not None else float("nan")
        pooled = float(np.nanmean([best.mcc_std, runner.mcc_std])) if runner is not None else float("nan")
        balance.append({
            "detector": det_name, "mcc_linea_base": round(best.mcc_base_medio, 4),
            "mejor_metrica": best.metric, "mcc_medio": round(best.mcc_medio, 4),
            "mcc_std": round(best.mcc_std, 4), "n_semillas": int(best.n_semillas),
            "delta_mcc": round(best.delta_mcc, 4),
            "mcc_oraculo": round(float(orc.mcc_medio.iloc[0]), 4) if len(orc) else None,
            "mcc_oraculo_q": round(float(orc_q.mcc_medio.iloc[0]), 4) if len(orc_q) else None,
            "dist_oraculo": round(float(orc.mcc_medio.iloc[0] - best.mcc_medio), 4) if len(orc) else None,
            "estado_vs_base": estado})
        hyp[det_name] = {"mejor_metrica": best.metric, "estado_vs_base": estado,
                         "margen_sobre_segunda": margin, "dispersion_semillas": pooled,
                         "separable": bool(int(best.n_semillas) >= 2 and np.isfinite(margin) and margin > pooled)}
    if balance:
        pd.DataFrame(balance).to_csv(out / "balance.csv", index=False)
        winners = {d: v["mejor_metrica"] for d, v in hyp.items()}
        soportada = (all(v["estado_vs_base"] != "empeora" for v in hyp.values())
                     and len(set(winners.values())) > 1
                     and all(v["separable"] for v in hyp.values()))
        save_json({"hipotesis": "la mejor metrica difiere entre detectores",
                   "criterio": ("soportada solo si (1) ninguna mejor metrica empeora la linea "
                                "base, (2) las ganadoras difieren y (3) el margen sobre la "
                                "segunda supera la dispersion entre semillas"),
                   "por_detector": hyp, "metricas_ganadoras": winners,
                   "difieren": len(set(winners.values())) > 1, "hipotesis_soportada": soportada},
                  out / "hypothesis_check.json")
        print("\nMejor metrica no supervisada por detector (MCC en test_final):")
        for b in balance:
            orc = f" | oraculo {b['mcc_oraculo']:.4f}" if b["mcc_oraculo"] is not None else ""
            print(f"  {b['detector']:12s} -> {b['mejor_metrica']:15s} MCC {b['mcc_medio']:.4f} "
                  f"(base {b['mcc_linea_base']:.4f}, delta {b['delta_mcc']:+.4f}, "
                  f"{b['estado_vs_base']}){orc}")
        print(f"\nHipotesis soportada: {soportada}")
    return balance


def pareto_front(cfg, df: pd.DataFrame, edge: pd.DataFrame | None) -> None:
    """Frente de Pareto entre MCC (mas alto es mejor) y latencia (mas baja es mejor),
    por detector, con todas las configuraciones evaluadas (linea base, optimizadas y
    oraculo, cada semilla por separado). Solo se reporta: no participa en ninguna
    eleccion. La latencia es el p95 medido en el borde; si aun no hay
    edge_benchmark.csv se usa la mediana medida en el PC y se indica en la columna
    fuente_latencia."""
    keys = ["detector", "metric", "seed", "rank"]
    d = df[df.config.isin(["linea_base", "optimizada", "oraculo"])].copy()
    if edge is not None:
        d = d.merge(edge[keys + ["latency_p95_ms"]].drop_duplicates(keys), on=keys, how="left")
        d["latencia_ms"], d["fuente_latencia"] = d["latency_p95_ms"], "borde_p95"
    else:
        d["latencia_ms"], d["fuente_latencia"] = d["latency_dev_ms"], "pc_mediana"
    d = d.dropna(subset=["latencia_ms"])
    if d.empty:
        print("AVISO: no hay latencias para calcular el frente de Pareto.")
        return
    d["en_frente"] = False
    for _, g in d.groupby("detector"):
        for i, r in g.iterrows():
            mejor_o_igual = (g.MCC >= r.MCC) & (g.latencia_ms <= r.latencia_ms)
            estricto = (g.MCC > r.MCC) | (g.latencia_ms < r.latencia_ms)
            d.loc[i, "en_frente"] = not bool((mejor_o_igual & estricto).any())
    cols = ["detector", "config", "metric", "seed", "rank", "MCC", "latencia_ms",
            "fuente_latencia", "en_frente"]
    d[cols].sort_values(["detector", "latencia_ms"]).to_csv(
        results_dir(cfg) / "pareto_mcc_latencia.csv", index=False)
    print(f"Frente de Pareto MCC/latencia ({d.fuente_latencia.iloc[0]}): "
          f"{int(d.en_frente.sum())} de {len(d)} configuraciones no dominadas.")


def write_summary(cfg, balance: list[dict], df: pd.DataFrame, y_te) -> None:
    lines = ["# Resumen de la evaluacion final", "",
             f"test_final: {len(y_te):,} flujos, {y_te.mean():.1%} ataques.", "",
             "| Detector | Mejor criterio | MCC base | MCC optimizado | Delta | Oraculo | Estado |",
             "|---|---|---|---|---|---|---|"]
    for b in balance:
        orc = "-" if b["mcc_oraculo"] is None else f"{b['mcc_oraculo']:.3f}"
        lines.append(f"| {b['detector']} | {b['mejor_metrica']} | {b['mcc_linea_base']:.3f} | "
                     f"{b['mcc_medio']:.3f} | {b['delta_mcc']:+.3f} | {orc} | {b['estado_vs_base']} |")
    lines += ["", "MCC = media sobre las semillas. Estado: 'mejora' solo si el delta supera 2 errores "
              "estandar entre semillas. El MCC con tanto ataque en el test es optimista: mire tambien "
              "MCC_pi05 / MCC_pi10 y AUROC en final_evaluation.csv."]
    (results_dir(cfg) / "resumen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    verify_integrity(cfg)
    X_te, y_te, types_te = load_labeled(cfg, "test_final")
    if y_te is None:
        raise SystemExit("test_final no contiene y_true.")
    print(f"test_final: {X_te.shape} | ataques {int(y_te.sum()):,} ({y_te.mean():.1%})")

    mdir = models_dir(cfg)
    index = json.loads((mdir / "index.json").read_text(encoding="utf-8"))
    edge_path = results_dir(cfg) / "edge_benchmark.csv"
    edge = pd.read_csv(edge_path) if edge_path.exists() else None
    if edge is not None and not EDGE_COLS.issubset(edge.columns):
        print("AVISO: edge_benchmark.csv es de una version anterior (faltan columnas "
              f"{sorted(EDGE_COLS - set(edge.columns))}). Se ignora: borre outputs/results/"
              "edge_benchmark.csv y vuelva a correr edge_benchmark.py.")
        edge = None
    if edge is None:
        print("AVISO: no hay edge_benchmark.csv valido; se usa la config de rango 0 (sin filtro de borde).")
    elif "platform" in edge and not edge["platform"].astype(str).str.contains("Raspberry").any():
        print("AVISO: edge_benchmark.csv NO fue medido en una Raspberry Pi.")

    chosen = select_entries(index, edge)
    rows, per_type = [], []
    for e in chosen:
        m, det, pred = evaluate_entry(e, mdir, X_te, y_te, cfg)
        rows.append({"detector": e["detector"], "config": e["config"], "metric": e["metric"],
                     "seed": e["seed"], "rank": e["rank"], "objective_value": e["value"],
                     "q": e["q"], "selected_feasible": e["selected_feasible"],
                     "latency_dev_ms": e.get("latency_dev_ms"), **m})
        print(f"  {e['detector']:12s} {e['metric']:15s} seed={e['seed']} rank={e['rank']} "
              f"MCC={m['MCC']:.4f} AUROC={m['AUROC']:.4f}")
        if types_te is not None:                       # recall por tipo de ataque
            for t in sorted(set(types_te) - {"normal"}):
                mk = types_te == t
                per_type.append({"detector": e["detector"], "config": e["config"],
                                 "metric": e["metric"], "seed": e["seed"], "attack_type": t,
                                 "n": int(mk.sum()), "recall": float(pred[mk].mean())})
        if e["config"] == "linea_base":                  # como referencia, tambien la decision propia de la libreria
            mn = supervised_metrics(y_te, det.predict_native(X_te), cfg["split"]["report_prevalences"])
            mn.update(ranking_metrics(y_te, det.scores(X_te)))
            rows.append({"detector": e["detector"], "config": "linea_base_nativa",
                         "metric": "por_defecto_nativo", "seed": e["seed"], "rank": 0,
                         "objective_value": None, "q": None, "selected_feasible": None, **mn})

    out = results_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "final_evaluation.csv", index=False)
    pd.DataFrame([{k: v for k, v in e.items() if k != "params"} for e in chosen]
                 ).to_csv(out / "selected_models.csv", index=False)
    if per_type:
        pd.DataFrame(per_type).groupby(["detector", "config", "metric", "attack_type"]
                                       ).agg(n=("n", "first"), recall=("recall", "mean")
                                             ).reset_index().to_csv(out / "recall_por_ataque.csv",
                                                                    index=False)
    balance = summarize(df, cfg)
    pareto_front(cfg, df, edge)
    write_summary(cfg, balance, df, y_te)
    print(f"\nResultados en {out}")


if __name__ == "__main__":
    main()