"""
optimize.py - Etapa 2: busqueda de hiperparametros con Optuna (TPE, un solo objetivo).

  * Escenarios no supervisados (silhouette, davies_bouldin, reconstruction,
    stability, consistency): solo usan train y val (trafico normal, sin
    etiquetas). En estos escenarios no se abre test_oracle, y test_final no se
    abre nunca en este script.
  * Escenario oraculo (oracle, oracle_q): maximiza el MCC usando las etiquetas de
    test_oracle (la parte del test que no entra en la evaluacion final). Sirve
    como cota superior; no es un metodo que se pueda usar sin etiquetas.
  * Linea base: hiperparametros por defecto, con la misma regla de umbral.

Limites del borde (obligatorios): un trial se descarta si la latencia por muestra
(pasando de a una muestra) en el PC supera max_latency_ms / dev_slowdown_factor,
o si el modelo pesa mas de max_model_kb. La medicion que vale es la de la
Raspberry Pi (edge_benchmark.py).

Salidas
  outputs/optuna/<detector>_<metrica>_seed<s>.json   (incluye el top-k)
  outputs/models/*.pkl                                (top-k reentrenados + linea base)
  outputs/models/index.json                           (lista de todos los modelos)
  outputs/models/env.json                             (versiones de las librerias)

USO
  python src/optimize.py --all                     # corrida completa (segun la config)
  python src/optimize.py --all --fast              # 25 trials, 1 semilla
  python src/optimize.py --detector lof --metric silhouette --trials 50
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import time

import numpy as np

from common import (load_config, load_split, mcc_from_rates, models_dir, outputs_dir,
                    save_json, set_seeds, split_val, subsample, train_for)
from detectors import DETECTORS, model_size_kb, tf_available
from objectives import DIRECTION, compute_objective, penalty_for
from search_space import RandomTrial, suggest_params


class PrunedTrial(Exception):
    """Trial descartado porque no cumple un limite del borde."""


def oracle_score(y, pred, prevalence=None) -> float:
    """MCC que usa el oraculo. Si se define `optimization.oracle_prevalence` (por
    ejemplo 0.10), se usa el MCC que se obtendria con esa proporcion de ataques
    (calculado desde TPR y FPR). Asi, un test con muchos ataques no premia marcar
    casi todo como ataque."""
    from sklearn.metrics import confusion_matrix, matthews_corrcoef
    if prevalence is None:
        return float(matthews_corrcoef(y, pred))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return mcc_from_rates(tp / max(tp + fn, 1), fp / max(fp + tn, 1), float(prevalence))


def measure_latency_ms(det, X: np.ndarray, n: int) -> float:
    """Mediana de la latencia por muestra, pasando de a una muestra (como llegaria
    un flujo real)."""
    n = min(n, len(X))
    for i in range(min(3, n)):                      # pasadas de calentamiento (no se miden)
        det.scores(X[i:i + 1])
    t = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter_ns()
        det.scores(X[i:i + 1])
        t[i] = (time.perf_counter_ns() - t0) / 1e6
    return float(np.median(t))


# --------------------------------------------------------------------------- #
# Estudio
# --------------------------------------------------------------------------- #
def _run_trials(objective, direction: str, seed: int, n_trials: int):
    """Devuelve (lista de trials completos, cada uno con value, params y user_attrs;
    cantidad de trials descartados)."""
    try:
        import optuna
    except ImportError:
        if os.environ.get("ALLOW_RANDOM_FALLBACK") != "1":
            raise SystemExit("Falta optuna: pip install optuna")
        print("  [!] Optuna no instalado: BUSQUEDA ALEATORIA (solo prueba de humo)")
        done = []
        for k in range(n_trials):
            tr = RandomTrial(seed * 1000 + k)
            try:
                v = objective(tr)
            except PrunedTrial:
                continue
            done.append({"value": v, "params": dict(tr.params), "user_attrs": dict(tr.user_attrs)})
        return done, n_trials - len(done)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def wrapped(trial):
        try:
            return objective(trial)
        except PrunedTrial:
            raise optuna.TrialPruned()

    study = optuna.create_study(direction=direction,
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.NopPruner())
    study.optimize(wrapped, n_trials=n_trials, catch=(Exception,))
    done = [{"value": t.value, "params": dict(t.params), "user_attrs": dict(t.user_attrs)}
            for t in study.trials if t.value is not None]
    return done, len(study.trials) - len(done)


def _fit_params(detector_name: str, p: dict, seed: int, cfg) -> dict:
    p = dict(p)
    if detector_name in ("iforest", "autoencoder"):
        p.setdefault("random_state", seed)
    if detector_name == "autoencoder":
        p.setdefault("backend", cfg["autoencoder"]["backend"])
    return p


def run_study(cfg, detector_name: str, metric: str, seed: int, n_trials: int, data: dict) -> dict:
    set_seeds(seed)
    opt, edge, th = cfg["optimization"], cfg["edge"], cfg["threshold"]
    Detector = DETECTORS[detector_name]
    direction = DIRECTION[metric]
    is_oracle = metric.startswith("oracle")
    budget = float(edge["max_latency_ms"]) / float(edge["dev_slowdown_factor"])
    max_kb = float(edge["max_model_kb"])
    X_tr = train_for(cfg, detector_name, data["train"])
    X_cal, X_obj = data["calib"], subsample(data["obj"], opt.get("max_val_rows"))
    q_fixed = float(th["q"])

    def objective(trial):
        params = suggest_params(trial, detector_name, seed, cfg)
        q = (trial.suggest_float("q", *th["oracle_q_range"]) if metric == "oracle_q" else q_fixed)
        params = _fit_params(detector_name, params, seed, cfg)
        try:
            det = Detector(**params).fit(X_tr)
            det.set_threshold(X_cal, q)
        except Exception as exc:
            trial.set_user_attr("error", str(exc)[:200])
            raise PrunedTrial()
        lat, kb = measure_latency_ms(det, X_obj, int(opt["latency_samples"])), model_size_kb(det)
        trial.set_user_attr("latency_ms", lat)
        trial.set_user_attr("model_kb", kb)
        if lat > budget or kb > max_kb:
            trial.set_user_attr("pruned_reason", "latency" if lat > budget else "model_size")
            raise PrunedTrial()
        if is_oracle:
            value = oracle_score(data["y_orc"], det.predict_outliers(data["X_orc"]),
                                 opt.get("oracle_prevalence"))
        else:
            value = compute_objective(metric, det, X_obj, cfg,
                                      ctx={"Detector": Detector, "params": params, "X_train": X_tr})
        return float(value) if np.isfinite(value) else penalty_for(metric)

    t0 = time.perf_counter()
    done, n_pruned = _run_trials(objective, direction, seed, n_trials)
    elapsed = time.perf_counter() - t0
    if not done:
        raise SystemExit(f"Ningun trial completo para {detector_name}/{metric}: revise "
                         "edge.max_latency_ms / dev_slowdown_factor o el espacio de busqueda.")

    done.sort(key=lambda t: t["value"], reverse=(direction == "maximize"))
    top = done[: int(opt["top_k"])]
    mdir = models_dir(cfg)
    mdir.mkdir(parents=True, exist_ok=True)
    entries = []
    for rank, t in enumerate(top):
        p = dict(t["params"])
        q = float(p.pop("q", q_fixed))
        fit_params = _fit_params(detector_name, p, seed, cfg)
        det = Detector(**fit_params).fit(X_tr).set_threshold(X_cal, q)
        fname = f"{detector_name}__{metric}__seed{seed}__rank{rank}.pkl"
        with open(mdir / fname, "wb") as fh:
            pickle.dump(det, fh, protocol=pickle.HIGHEST_PROTOCOL)
        entries.append({"detector": detector_name, "metric": metric, "seed": seed, "rank": rank,
                        "config": "oraculo" if is_oracle else "optimizada",
                        "value": t["value"], "params": fit_params, "q": q,
                        "latency_dev_ms": t["user_attrs"].get("latency_ms"),
                        "model_kb": t["user_attrs"].get("model_kb"), "file": fname})
    return {"detector": detector_name, "metric": metric, "direction": direction, "seed": seed,
            "n_trials": n_trials, "n_completed": len(done), "n_pruned": n_pruned,
            "best_value": top[0]["value"], "best_params": top[0]["params"],
            "best_latency_ms": top[0]["user_attrs"].get("latency_ms"),
            "dev_latency_budget_ms": budget, "q": q_fixed, "top_k": entries,
            "elapsed_s": round(elapsed, 2)}


def build_baseline(cfg, detector_name: str, seed: int, data: dict) -> list[dict]:
    """Linea base: hiperparametros por defecto, con la misma regla de umbral (q fijo)."""
    set_seeds(seed)
    Detector = DETECTORS[detector_name]
    X_tr = train_for(cfg, detector_name, data["train"])
    params = _fit_params(detector_name, {}, seed, cfg)
    det = Detector(**params).fit(X_tr).set_threshold(data["calib"], cfg["threshold"]["q"])
    mdir = models_dir(cfg)
    mdir.mkdir(parents=True, exist_ok=True)
    fname = f"{detector_name}__por_defecto__seed{seed}__rank0.pkl"
    with open(mdir / fname, "wb") as fh:
        pickle.dump(det, fh, protocol=pickle.HIGHEST_PROTOCOL)
    lat = measure_latency_ms(det, data["obj"], int(cfg["optimization"]["latency_samples"]))
    return [{"detector": detector_name, "metric": "por_defecto", "seed": seed, "rank": 0,
             "config": "linea_base", "value": None, "params": params,
             "q": float(cfg["threshold"]["q"]), "latency_dev_ms": lat,
             "model_kb": model_size_kb(det), "file": fname}]


def update_index(cfg, new_entries: list[dict]) -> None:
    path = models_dir(cfg) / "index.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    keys = {(e["detector"], e["metric"], e["seed"]) for e in new_entries}
    old = [e for e in old if (e["detector"], e["metric"], e["seed"]) not in keys]
    save_json(old + new_entries, path)


def write_env(cfg) -> None:
    """Guarda las versiones con que se entrenaron los modelos. La Raspberry necesita
    las mismas versiones de scikit-learn y NumPy para cargar los .pkl
    (edge_benchmark.py lo revisa)."""
    import sklearn
    import scipy
    env = {"python": platform.python_version(), "numpy": np.__version__,
           "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
           "tensorflow": None, "platform": platform.platform()}
    if tf_available():
        import tensorflow as tf
        env["tensorflow"] = tf.__version__
    save_json(env, models_dir(cfg) / "env.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--detector", choices=list(DETECTORS))
    ap.add_argument("--metric", choices=list(DIRECTION))
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--fast", action="store_true", help="25 trials y una semilla")
    ap.add_argument("--all", action="store_true",
                    help="todas las combinaciones de config['detectors'] + oraculo + linea base")
    ap.add_argument("--no-oracle", action="store_true", help="no abre ningun bloque etiquetado")
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    opt = cfg["optimization"]
    n_trials = args.trials or (25 if args.fast else opt["n_trials"])
    seeds = args.seeds if args.seeds is not None else ([42] if args.fast else opt["seeds"])

    # Aqui solo se leen train y val (sin etiquetas). test_final no se abre nunca en este script.
    X_train, _ = load_split(cfg, "train")
    X_val, _ = load_split(cfg, "val")
    X_cal, X_obj = split_val(X_val, cfg)
    data = {"train": X_train, "calib": X_cal, "obj": X_obj}
    print(f"train {X_train.shape} | calibracion {X_cal.shape} | objetivo {X_obj.shape} "
          f"| q={cfg['threshold']['q']}")
    if cfg["autoencoder"]["backend"] == "auto":
        print("autoencoder: " + ("Keras/TensorFlow" if tf_available() else
                                 "scikit-learn (TensorFlow no instalado)"))

    if args.all:
        combos = [(d, m) for d, ms in cfg["detectors"].items() for m in ms]
    else:
        if not (args.detector and args.metric):
            ap.error("Indique --detector y --metric, o use --all")
        combos = [(args.detector, args.metric)]
    want_oracle = (args.all and opt["oracle"] and not args.no_oracle) or \
                  any(m.startswith("oracle") for _, m in combos)
    if want_oracle and not args.no_oracle:
        X_orc, y_orc = load_split(cfg, "test_oracle")      # estas etiquetas las usa solo el oraculo
        data.update(X_orc=X_orc, y_orc=y_orc)
        if args.all:
            for d in cfg["detectors"]:
                combos += [(d, "oracle"), (d, "oracle_q")]
    elif args.no_oracle:
        combos = [(d, m) for d, m in combos if not m.startswith("oracle")]

    write_env(cfg)
    out_dir = outputs_dir(cfg) / "optuna"
    summaries = []
    for detector, metric in combos:
        for seed in seeds:
            tag = f"{detector}_{metric}_seed{seed}"
            print(f"\n=== {tag} ({n_trials} trials) ===")
            res = run_study(cfg, detector, metric, seed, n_trials, data)
            save_json(res, out_dir / f"{tag}.json")
            update_index(cfg, res["top_k"])
            summaries.append({k: v for k, v in res.items() if k != "top_k"})
            print(f"  mejor valor {res['best_value']:.4f} | completos {res['n_completed']} "
                  f"| descartados {res['n_pruned']} | {res['elapsed_s']}s")

    if args.all and not args.no_baseline:
        base_entries = []
        for detector in cfg["detectors"]:
            for seed in seeds:
                base_entries += build_baseline(cfg, detector, seed, data)
        update_index(cfg, base_entries)
        print(f"\nLinea base: {len(base_entries)} modelos")
    # Se agrega a lo que ya habia (corridas anteriores con otras semillas o detectores)
    all_path = out_dir / "all_studies.json"
    old = json.loads(all_path.read_text(encoding="utf-8")) if all_path.exists() else []
    keys = {(s["detector"], s["metric"], s["seed"]) for s in summaries}
    old = [s for s in old if (s["detector"], s["metric"], s["seed"]) not in keys]
    save_json(old + summaries, all_path)
    print(f"\nResultados en {out_dir} | modelos en {models_dir(cfg)}")


if __name__ == "__main__":
    main()
