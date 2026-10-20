"""
edge_benchmark.py - Etapa 3: cuanto le cuesta al dispositivo de borde correr cada modelo.

Este script esta hecho para correr en la Raspberry Pi 5. Los modelos se entrenan
en el PC; aqui solo se hacen predicciones (el autoencoder corre solo con NumPy,
asi que la Pi no necesita TensorFlow). En un PC solo sirve para probar que todo
funciona.

Que copiar a la Raspberry (la carpeta del proyecto, sin Datasets ni outputs/optuna):
    src/   config/   scripts/   run_all.py   requirements-pi.txt
    outputs/models/ (completa)   data/splits/edge_sample.csv

Cada modelo se mide en un proceso nuevo y con un solo hilo, para que:
  * la RAM medida sea la del modelo y no la de modelos cargados antes;
  * el uso de CPU no dependa de cuantos nucleos tenga la maquina.

Se registra (pasando de a una muestra, como llegaria un flujo real):
  latency_median_ms / latency_p95_ms / latency_mean_ms
  cpu_ms_per_inference   tiempo de CPU que usa cada prediccion
  cpu_percent            % de todos los nucleos con el modelo trabajando sin pausa (referencia)
  cpu_util_at_rate       % de la CPU total si llegan edge.flow_rate_per_s flujos por segundo
  ram_peak_mb            memoria maxima que uso el proceso
  ram_delta_mb           cuanto sube la RAM al cargar el modelo
  model_size_kb          tamano del archivo del modelo

Un modelo cumple si: la latencia (p95 o mediana, segun edge.latency_stat) es menor
que max_latency_ms, cpu_util_at_rate es menor que max_cpu_percent y ram_peak_mb
es menor que max_ram_mb.

USO (en la Raspberry)
    python src/edge_benchmark.py
"""

from __future__ import annotations

import os

# Un solo hilo para los calculos. Tiene que quedar fijado antes de importar numpy y scikit-learn.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse             # noqa: E402
import json                 # noqa: E402
import pickle               # noqa: E402
import subprocess           # noqa: E402
import sys                  # noqa: E402
import time                 # noqa: E402
from pathlib import Path    # noqa: E402

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
import psutil               # noqa: E402

from common import load_config, models_dir, results_dir, save_json, splits_dir   # noqa: E402


def device_info() -> dict:
    model = ""
    try:
        model = Path("/proc/device-tree/model").read_text().strip("\x00\n ")
    except Exception:
        pass
    return {"device_model": model or "desconocido",
            "is_raspberry_pi": "raspberry" in model.lower(),
            "cpu_count": psutil.cpu_count(),
            "total_ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2)}


def load_edge_sample(cfg) -> np.ndarray:
    """Lee flujos de val sin etiquetas (edge_sample.csv). Es un csv para que la Pi no
    necesite pyarrow."""
    p = splits_dir(cfg) / "edge_sample.csv"
    if not p.exists():
        raise SystemExit(f"Falta {p}. Ejecute prepare_splits.py o copie el archivo a la Raspberry.")
    return pd.read_csv(p).to_numpy(dtype=np.float32)


def worker(model_path: str, cfg_path: str | None, n_runs: int, n_warmup: int) -> dict:
    """Corre en un proceso aparte y devuelve las mediciones (main las imprime como JSON)."""
    cfg = load_config(cfg_path)
    X = load_edge_sample(cfg)
    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss
    with open(model_path, "rb") as fh:
        det = pickle.load(fh)
    rss_loaded = proc.memory_info().rss
    rng = np.random.default_rng(42)
    for _ in range(n_warmup):
        i = int(rng.integers(0, len(X)))
        det.scores(X[i:i + 1])
    lat = np.empty(n_runs)
    cpu0, wall0 = time.process_time(), time.perf_counter()
    for k in range(n_runs):
        i = int(rng.integers(0, len(X)))
        t0 = time.perf_counter_ns()
        det.scores(X[i:i + 1])
        lat[k] = (time.perf_counter_ns() - t0) / 1e6
    cpu, wall = time.process_time() - cpu0, time.perf_counter() - wall0
    try:                                           # Linux / Raspberry Pi
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0   # KB -> MB
    except ImportError:                            # Windows (solo para probar el flujo)
        mi = proc.memory_info()
        peak = getattr(mi, "peak_wset", mi.rss) / 1024 ** 2
    ncpu = psutil.cpu_count() or 1
    cpu_ms = cpu / n_runs * 1000.0
    rate = float(cfg["edge"]["flow_rate_per_s"])
    return {
        "latency_median_ms": round(float(np.median(lat)), 4),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 4),
        "latency_mean_ms": round(float(lat.mean()), 4),
        "cpu_ms_per_inference": round(cpu_ms, 5),
        "cpu_percent": round(100.0 * cpu / max(wall, 1e-9) / ncpu, 2),
        "cpu_util_at_rate": round(cpu_ms * rate / 1000.0 / ncpu * 100.0, 3),
        "ram_peak_mb": round(peak, 2),
        "ram_delta_mb": round((rss_loaded - rss0) / 1024 ** 2, 2),
        "model_size_kb": round(os.path.getsize(model_path) / 1024.0, 2),
        "n_runs": n_runs,
    }


def meets(cfg, r: dict) -> dict:
    e = cfg["edge"]
    lat = r["latency_p95_ms"] if e["latency_stat"] == "p95" else r["latency_median_ms"]
    return {"cumple_latencia": bool(lat < e["max_latency_ms"]),
            "cumple_cpu": bool(r["cpu_util_at_rate"] < e["max_cpu_percent"]),
            "cumple_ram": bool(r["ram_peak_mb"] < e["max_ram_mb"])}


def check_env(cfg) -> None:
    """Avisa si las versiones de las librerias no son las mismas que al entrenar (los
    .pkl podrian no cargar)."""
    p = models_dir(cfg) / "env.json"
    if not p.exists():
        return
    import scipy
    import sklearn
    train = json.loads(p.read_text(encoding="utf-8"))
    here = {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    for k, v in here.items():
        if train.get(k) and train[k] != v:
            print(f"[!] AVISO: {k} {v} aqui vs {train[k]} al entrenar. Instale la misma version "
                  "(pip install scikit-learn==...) o los modelos pueden no cargar.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    cfg = load_config(args.config)
    edge = cfg["edge"]
    n_runs = args.runs or edge["n_inference_runs"]
    n_warm = args.warmup if args.warmup is not None else edge["n_warmup_runs"]

    if args.worker:
        print("RESULT " + json.dumps(worker(args.worker, args.config, n_runs, n_warm)))
        return

    info = device_info()
    print(f"Plataforma: {info['device_model']} | {info['cpu_count']} nucleos | "
          f"{info['total_ram_gb']} GB RAM")
    if not info["is_raspberry_pi"]:
        print("[!] AVISO: este equipo NO es una Raspberry Pi. Los resultados sirven para "
              "probar el flujo, no para reportar costo en el borde.")
    check_env(cfg)

    index_path = models_dir(cfg) / "index.json"
    if not index_path.exists():
        raise SystemExit(f"No existe {index_path}: ejecute primero optimize.py.")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = "1"

    rows = []
    for k, e in enumerate(index):
        tag = f"{e['detector']}/{e['metric']}/seed{e['seed']}/rank{e['rank']}"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
               str(index_path.parent / e["file"]), "--runs", str(n_runs), "--warmup", str(n_warm)]
        if args.config:
            cmd += ["--config", args.config]
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if p.returncode != 0 or line is None:
            print(f"[{k + 1}/{len(index)}] {tag}  [!] {p.stderr.strip()[-300:]}")
            continue
        res = json.loads(line[7:])
        row = {k2: v for k2, v in e.items() if k2 not in ("params", "file")} | res | meets(cfg, res)
        row["platform"] = info["device_model"]
        rows.append(row)
        print(f"[{k + 1}/{len(index)}] {tag}: mediana {res['latency_median_ms']:.3f} ms | "
              f"p95 {res['latency_p95_ms']:.3f} | CPU@{edge['flow_rate_per_s']:.0f}/s "
              f"{res['cpu_util_at_rate']:.2f}% | RAM {res['ram_peak_mb']:.0f} MB")

    if not rows:
        raise SystemExit("No se pudo medir ningun modelo (vea los errores de arriba).")
    out = results_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "edge_benchmark.csv", index=False)
    save_json({**info, "max_latency_ms": edge["max_latency_ms"],
               "max_cpu_percent": edge["max_cpu_percent"], "max_ram_mb": edge["max_ram_mb"],
               "flow_rate_per_s": edge["flow_rate_per_s"], "latency_stat": edge["latency_stat"]},
              out / "edge_platform.json")
    print(f"\nResultados en {out / 'edge_benchmark.csv'}")


if __name__ == "__main__":
    main()
