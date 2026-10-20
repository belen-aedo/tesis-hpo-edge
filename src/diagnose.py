"""
diagnose.py - revisa si cada metrica no supervisada sirve como guia.
Conviene correrlo antes de optimize.

Para cada detector se prueban N configuraciones al azar, y en cada una se mide:
  (a) cada funcion objetivo no supervisada, sobre la mitad `objetivo` de val;
  (b) que tan bien detecta de verdad (MCC y AUROC), sobre test_oracle.
Despues se calcula la correlacion de Spearman entre (a) y (b), con el signo
ajustado segun si la metrica se maximiza o se minimiza:

  rho > 0  -> la metrica ordena bien las configuraciones
  rho ~ 0  -> no dice nada: optimizarla es como optimizar ruido
  rho < 0  -> lleva a configuraciones peores

Solo usa etiquetas de test_oracle (nunca de test_final), y solo para este
diagnostico: no participa en ninguna eleccion de modelos. Los resultados tambien
sirven para el capitulo de resultados (relacion entre criterio y desempeno).

USO
    python src/diagnose.py --n 40
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import (load_config, load_split, ranking_metrics, results_dir, save_json,
                    split_val, supervised_metrics, train_for)
from detectors import DETECTORS
from objectives import DIRECTION, UNSUPERVISED, compute_objective
from search_space import RandomTrial, suggest_params


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=40, help="configuraciones por detector")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metrics", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    X_train, _ = load_split(cfg, "train")
    X_val, _ = load_split(cfg, "val")
    X_orc, y_orc = load_split(cfg, "test_oracle")
    X_cal, X_obj = split_val(X_val, cfg)
    metrics = args.metrics or list(UNSUPERVISED)

    recs = []
    for dname, Detector in DETECTORS.items():
        X_tr = train_for(cfg, dname, X_train)
        print(f"\n{dname}: {args.n} configuraciones aleatorias")
        for k in range(args.n):
            trial = RandomTrial(args.seed * 100000 + k)
            params = suggest_params(trial, dname, args.seed + k, cfg)
            if dname == "autoencoder":
                params["backend"] = cfg["autoencoder"]["backend"]
            try:
                det = Detector(**params).fit(X_tr).set_threshold(X_cal, cfg["threshold"]["q"])
            except Exception as exc:
                print("  [!]", exc)
                continue
            s = det.scores(X_orc)
            r = {"detector": dname, "trial": k, **trial.params,
                 "MCC": supervised_metrics(y_orc, (s > det.threshold_).astype(int))["MCC"],
                 **ranking_metrics(y_orc, s)}
            for m in metrics:
                try:
                    r[m] = compute_objective(m, det, X_obj, cfg, ctx={
                        "Detector": Detector, "params": params, "X_train": X_tr})
                except Exception:
                    r[m] = np.nan
            recs.append(r)

    df = pd.DataFrame(recs)
    out = results_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "diagnose_objectives.csv", index=False)

    rows = []
    print("\nCorrelacion de Spearman (signo ajustado: + = la metrica ordena bien)")
    print(f"{'detector':12s} {'metrica':15s} {'rho vs MCC':>11s} {'rho vs AUROC':>13s} {'n':>4s}")
    for dname, g in df.groupby("detector"):
        for m in metrics:
            if m not in g or g[m].nunique() < 3:
                continue
            sign = 1.0 if DIRECTION[m] == "maximize" else -1.0
            gg = g.dropna(subset=[m, "MCC", "AUROC"])
            if len(gg) < 3:
                continue
            r_mcc = sign * spearmanr(gg[m], gg["MCC"]).statistic
            r_auc = sign * spearmanr(gg[m], gg["AUROC"]).statistic
            rows.append({"detector": dname, "metric": m, "rho_mcc": r_mcc,
                         "rho_auroc": r_auc, "n": len(gg)})
            print(f"{dname:12s} {m:15s} {r_mcc:+11.3f} {r_auc:+13.3f} {len(gg):4d}")
    save_json(rows, out / "diagnose_objectives_summary.json")


if __name__ == "__main__":
    main()
